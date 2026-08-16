"""
hopai.schema

The shape of a graph, declared in Python and readable by anything:
which node types exist, which properties each carries, and which edge
kinds connect which node types.

Two notations, each complete and symmetric on its own -- the same
parallel-notations pattern the traversal API uses (Python / JSON /
Cypher). Pick one per project, not per line:

    # Class notation: plain dataclasses (or pydantic v2 models) for
    # nodes AND edges. An edge class declares its endpoints as fields
    # annotated with node classes, the way ORM association objects do.
    @dataclass
    class Person:
        email: str                  # no default -> required
        age: Optional[int] = None   # optional, may be JSON null

    @dataclass
    class Company:
        name: str

    @dataclass
    class WorksAt:
        source: Person              # endpoint, not a property
        target: Company             # endpoint, not a property
        since: int                  # property

    graph.define_schema(nodes=[Person, Company], edges=[WorksAt])

    # Primitive notation: explicit wrappers, same shape on both sides.
    graph.define_schema(
        nodes=[NodeType("person", properties=[Property("email", "string", required=True)]),
               NodeType("company", properties=[Property("name", "string", required=True)])],
        edges=[EdgeType("works_at", source="person", target="company",
                        properties=[Property("since", "number")])],
    )

Whatever the input form, `graph.schema` returns the same canonical
dataclasses -- normalization is one-way, so the notations cannot drift.
Class names become snake_case type names (WorksAt -> works_at), matching
how the Cypher examples spell kinds.

HOW ANNOTATIONS MAP -- one deterministic table, reusing the six JSON
types the constraints module already speaks (JSON_TYPES). A field with
no default is required; `Optional[X]` is not required and additionally
permits an explicit JSON null:

    str        -> "string"      dict / dict[...]  -> "object"
    int, float -> "number"      list / list[...]  -> "array"
    bool       -> "boolean"     Optional[X]       -> X's type + "null", not required

Anything else -- datetime, Enum, a union that is not Optional -- is
refused with the explicit-Property rewrite named, never silently
coerced. A property may carry a SET of JSON types (JSON Schema's type
array); an edge type's identity is its (kind, source, target) triple,
so the same kind may connect several endpoint pairs.

WHAT A SCHEMA IS, AND IS NOT: define_schema() records a contract in
memory and never touches the database. Graph.enforce_schema() is the
separate, explicit step that compiles it to Postgres CHECK constraints
-- separate because ALTER TABLE ADD CONSTRAINT validates every EXISTING
row, which can fail on pre-schema data and should be a conscious moment,
not a side effect. Once enforced, every write path is validated by the
server itself -- add_nodes, merge_nodes, Cypher CREATE/MERGE, even SQL
from another service -- and violations surface as ConstraintViolation.

WHAT ENFORCEMENT COMPILES TO, per node type / edge kind:

    required properties   CHECK ((properties->>'type') IS DISTINCT FROM 'person'
                                 OR properties ?& array['email', ...])
    property JSON types   CHECK ((properties->>'type') IS DISTINCT FROM 'person'
                                 OR jsonb_typeof(properties->'age') IN ('null', 'number'))

The guard makes every check vacuously true for rows of OTHER types --
and for rows carrying no discriminator at all, so untyped rows pass by
construction. IS DISTINCT FROM, not <>, because <> is NULL for a missing
key and a CHECK passes on NULL either way -- DISTINCT just says what is
meant.

ENDPOINT TYPES are policed only on request: "works_at connects only
person -> company" needs a look at the endpoint nodes, which a CHECK
cannot contain -- enforce_schema(endpoints=True) does it with a
CONSTRAINT TRIGGER (still plain Postgres). It fires per edge write,
which is why it is an opt-in with the cost stated rather than the
default; it validates edges as they are written, and retyping a NODE
under existing edges is not re-checked. Still uncovered: two edge types
sharing a kind with DIFFERENT property schemas. Property enforcement is
keyed on the kind alone, so it refuses that shape outright rather than
enforcing the wrong merge of the two.

The node discriminator is the `type` property and the edge discriminator
is `kind` -- the convention the Cypher front end already uses for
labels. Nothing requires a row to carry them until the schema is
enforced.

INFERENCE -- when the graph grew chaotically and nobody declared
anything, the schema is sitting in the data and Postgres can compute it:

    inferred, report = graph.infer_schema()   # connects, scans, observes
    print(report)                             # untyped rows, conflicts, counts
    graph.define_schema(schema=inferred)      # the deliberate adoption step
    graph.enforce_schema()                    # now the server validates writes

An INFERRED schema is an observation; a DEFINED one is a contract. That
is why infer_schema() is a method that visibly connects and scans, never
a silent fallback inside `.schema`: blurring the two would let a
property that merely HAPPENS to be on all forty rows today get frozen
into a required-forever constraint by an enforce_schema() nobody meant
that way. Adoption is the caller's second, explicit step -- and the
report (per-type row counts, untyped rows, type conflicts) is what to
read before taking it.

Inference stays honest about what chaos looks like: a key holding both
42 and "42" infers the type SET, never a silent winner, and is listed in
the report; one kind observed across several endpoint pairs becomes
several EdgeTypes; rows carrying no discriminator (or an empty one)
cannot be invented into a type -- they are counted in the report and
pass every per-type CHECK by construction, consistent with enforcement.
An edge whose endpoint nodes carry no type has no expressible triple and
is counted as skipped.

Cost: one sequential scan per query -- the GIN index cannot serve an
all-keys aggregate. Meant for start-up, migration or exploration, not
per-request. When even one scan is too much,
infer_schema(sample_percent=5) reads TABLESAMPLE SYSTEM (5) instead:
every count becomes an estimate, a rare property or edge triple can be
missed entirely, and `required` means only "present on every SAMPLED
row" -- so the report carries `sampled` and its summary says estimates.
The endpoint-triple join samples the edges side only: sampling the node
side too would count an edge whose node happened to fall outside the
sample as endpoint-less, manufacturing skipped-edge noise the data does
not contain.

PERSISTENCE -- a schema declared on one handle is invisible to every
other process on the same database, so Graph.save_schema() upserts the
declared contract into a third table, hopai_schema(graph_id, document,
saved_at), and Graph.load_schema() on any handle reads it back and
adopts it. The table is metadata, never on the query path, and is
created lazily by the first save -- callers who never persist never see
it. The document is the to_json() rendering (lossless);
schema_from_document() is the inverse, and it routes everything through
the same constructors define_schema() uses, so a corrupted row raises
the real validation error instead of half-loading. Loading ADOPTS --
unlike an inferred schema, a saved one was explicitly declared a
contract by whoever saved it.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import enum as _enum
import hashlib
import re
import types as _pytypes
import typing
from dataclasses import dataclass
from typing import Optional, Union

from sqlalchemy import func, or_, text
from sqlalchemy.dialects import postgresql

from .constraints import JSON_TYPES, _compile_check, _slug, _Target

_ENDPOINTS = ("source", "target")

_ANNOTATION_TYPES = {str: "string", int: "number", float: "number", bool: "boolean",
                     dict: "object", list: "array"}
_ORIGIN_TYPES = {dict: "object", list: "array"}


# ---------------------------------------------------------------------
# The canonical model
# ---------------------------------------------------------------------

@dataclass
class Property:
    """One property of a node or edge type.

    json_type is a JSON type name or a set of them ({"number", "null"});
    it is normalized to a sorted tuple so the same shape written in any
    input form -- or any order -- compares equal.

    values     allowed values (an Enum's members, or spelled directly);
               enforced as an IN (...) CHECK and rendered as JSON
               Schema's "enum".
    format     JSON Schema format annotation ("date-time", "date");
               validated by the generated pydantic model, NOT by the
               database -- a format regex CHECK is deliberately absent.
    unique     no two rows of THIS type share the value: compiles to a
               partial unique index, the per-type uniqueness Postgres
               gives away. Never inferred -- observed distinctness is
               not declared uniqueness.
    properties a nested object schema, when json_type is "object". The
               representations recurse; enforcement checks only the
               top-level jsonb_typeof -- stated here, not silent."""
    name: str
    json_type: tuple
    required: bool = False
    unique: bool = False
    values: tuple = ()
    format: Optional[str] = None
    properties: tuple = ()

    def __init__(self, name: str, json_type, required: bool = False, unique: bool = False,
                 values=(), format: Optional[str] = None, properties=()):
        if not isinstance(name, str) or not name:
            raise TypeError(f"a property name must be a non-empty string, got {name!r}")
        type_names = (json_type,) if isinstance(json_type, str) else tuple(json_type)
        if not type_names:
            raise ValueError(f"Property({name!r}) needs at least one JSON type")
        for type_name in type_names:
            if type_name not in JSON_TYPES:
                raise ValueError(
                    f"Property({name!r}): json_type must be one of {JSON_TYPES}, got {type_name!r}")
        self.name = name
        self.json_type = tuple(sorted(set(type_names)))
        self.required = bool(required)
        self.unique = bool(unique)
        self.values = tuple(values)
        self.format = format
        self.properties = (_normalize_properties(properties, owner=f"Property({name!r})")
                           if properties else ())


@dataclass
class NodeType:
    """A node type: a name and its property schema.

    `properties` takes a list of Property or a dataclass/pydantic class
    used as a property bag -- the identical forms EdgeType takes."""
    name: str
    properties: tuple

    def __init__(self, name: str, properties=()):
        if not isinstance(name, str) or not name:
            raise TypeError(f"a node type name must be a non-empty string, got {name!r}")
        self.name = name
        self.properties = _normalize_properties(properties, owner=f"NodeType({name!r})")


@dataclass
class EdgeType:
    """An edge kind between two node types. Identity is the whole
    (kind, source, target) triple -- the same kind may legitimately
    connect several endpoint pairs, as two EdgeType entries."""
    kind: str
    source: str
    target: str
    properties: tuple

    def __init__(self, kind: str, source: str, target: str, properties=()):
        for label, value in (("kind", kind), ("source", source), ("target", target)):
            if not isinstance(value, str) or not value:
                raise TypeError(f"EdgeType {label} must be a non-empty string, got {value!r}")
        self.kind, self.source, self.target = kind, source, target
        self.properties = _normalize_properties(properties, owner=f"EdgeType({kind!r})")
        reserved = [p.name for p in self.properties if p.name in _ENDPOINTS]
        if reserved:
            raise TypeError(
                f"EdgeType({kind!r}): {reserved} cannot be edge properties -- 'source' and "
                f"'target' name the endpoints. Rename the property and declare it explicitly, "
                f"e.g. EdgeType({kind!r}, source=..., target=..., "
                f"properties=[Property('{reserved[0]}_node', 'string')])"
            )


@dataclass
class GraphSchema:
    """The canonical schema: what .schema returns regardless of which
    notation defined it. Validation happens here, so a GraphSchema that
    exists is a GraphSchema that is internally consistent."""
    node_types: tuple
    edge_types: tuple

    def __init__(self, node_types=(), edge_types=()):
        self.node_types = tuple(node_types)
        self.edge_types = tuple(edge_types)
        names = [nt.name for nt in self.node_types]
        duplicate_names = sorted({n for n in names if names.count(n) > 1})
        if duplicate_names:
            raise ValueError(
                f"duplicate node type name(s) {duplicate_names} -- declare each node type once")
        defined = set(names)
        for et in self.edge_types:
            for end in ("source", "target"):
                value = getattr(et, end)
                if value not in defined:
                    raise ValueError(
                        f"EdgeType({et.kind!r}) {end}={value!r} is not a defined node type -- "
                        f"defined node types are {sorted(defined)}"
                    )
        triples = [(et.kind, et.source, et.target) for et in self.edge_types]
        duplicate_triples = sorted({t for t in triples if triples.count(t) > 1})
        if duplicate_triples:
            raise ValueError(
                f"duplicate edge type(s) {duplicate_triples} -- an edge type's identity is its "
                f"(kind, source, target) triple; declare each triple once"
            )

    # -- representations ----------------------------------------------

    def to_json(self) -> dict:
        """JSON Schema vocabulary, json.dumps-clean -- the form that gets
        pasted into a system prompt or returned from a tool call. Nodes
        are keyed by name; edges are a list, because a kind alone is not
        an identity."""
        return {
            "nodes": {nt.name: _properties_json(nt.properties) for nt in self.node_types},
            "edges": [
                {"kind": et.kind, "source": et.source, "target": et.target,
                 "properties": _properties_json(et.properties)}
                for et in self.edge_types
            ],
        }

    def to_networkx(self):
        """The schema as a meta-graph: node types as nodes, edge types as
        edges, each carrying its JSON Schema spec as attributes.

        Always a MultiDiGraph: two kinds between the same pair of node
        types -- or one kind with two endpoint pairs -- are parallel
        edges, and DiGraph would silently collapse them, the same trap
        Subgraph.to_networkx documents."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "the networkx representation needs networkx -- pip install hopai[networkx]"
            ) from exc
        meta = nx.MultiDiGraph()
        for nt in self.node_types:
            meta.add_node(nt.name, **_properties_json(nt.properties))
        for et in self.edge_types:
            meta.add_edge(et.source, et.target, key=et.kind, kind=et.kind,
                          **_properties_json(et.properties))
        return meta

    def to_pydantic(self) -> dict:
        """Generated pydantic models, one per node type and per edge
        kind, so callers get real validation (Person(email=42) raises).

        Always GENERATED from the canonical form, never the class the
        caller passed in -- the representation must not depend on which
        notation defined the schema."""
        try:
            import pydantic
        except ImportError as exc:
            raise ImportError(
                "the pydantic representation needs pydantic v2 -- pip install hopai[pydantic]"
            ) from exc
        if int(pydantic.VERSION.split(".")[0]) < 2:
            # some other dependency in the environment can still pull in
            # v1 even though hopai[pydantic] pins v2 -- and v1's
            # create_model builds models with different semantics rather
            # than failing, which is worse.
            raise ImportError(
                f"the pydantic representation needs pydantic v2, found {pydantic.VERSION} "
                f"-- pip install hopai[pydantic]"
            )
        models = {}
        for nt in self.node_types:
            models[nt.name] = _pydantic_model(nt.name, nt.properties, pydantic.create_model)
        for kind, properties in _properties_per_kind(self.edge_types).items():
            if kind in models:
                raise ValueError(
                    f"{kind!r} is both a node type and an edge kind -- one dict of models "
                    f"cannot hold both; rename one of them"
                )
            models[kind] = _pydantic_model(kind, properties, pydantic.create_model)
        return models

    def to_mermaid(self) -> str:
        """The schema as a Mermaid flowchart -- GitHub, GitLab and most
        doc tooling render it natively, so this string inside a
        ```mermaid fence turns the schema into a picture in any PR
        description or README. Node labels carry the same bounded
        property bag as tool_summary() (* required, ! unique, capped
        with +N more); one arrow per (kind, source, target) triple, so
        parallel kinds stay parallel arrows -- the same
        no-silent-collapse rule to_networkx() documents."""
        ids: dict = {}
        for index, nt in enumerate(self.node_types):
            # _slug can collide ("works-at" and "works_at") or come back
            # empty; the index suffix keeps every id distinct and the
            # label keeps the real name
            base = _slug(nt.name) or "node"
            ids[nt.name] = base if base not in ids.values() else f"{base}_{index}"
        lines = ["flowchart LR"]
        for nt in self.node_types:
            bag = _property_bag(nt.properties)
            label = _mermaid_text(f"{nt.name} {bag}" if bag else nt.name)
            lines.append(f'    {ids[nt.name]}["{label}"]')
        for et in self.edge_types:
            kind = (et.kind if re.fullmatch(r"[A-Za-z0-9_]+", et.kind)
                    else f'"{_mermaid_text(et.kind)}"')
            lines.append(f"    {ids[et.source]} -- {kind} --> {ids[et.target]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Class notation -> canonical model
# ---------------------------------------------------------------------

def _is_pydantic_model(obj) -> bool:
    # Duck-typed on purpose: recognizing a pydantic model must not cost
    # hopai a pydantic import -- the caller who passed one already paid
    # it. model_fields is the v2 marker; v1 models fall through and are
    # refused as unsupported input.
    return isinstance(obj, type) and isinstance(getattr(obj, "model_fields", None), dict)


def _is_schema_class(obj) -> bool:
    return isinstance(obj, type) and (dataclasses.is_dataclass(obj) or _is_pydantic_model(obj))


def _type_name(cls: type) -> str:
    """CamelCase -> snake_case, matching how the Cypher examples spell
    kinds: WorksAt -> works_at, HTTPServer -> http_server. Anyone who
    dislikes a derived name uses the primitive notation, which is fully
    explicit."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", cls.__name__).lower()


def _class_fields(cls: type) -> list:
    """(name, resolved annotation, has_default) per field, one shape for
    both class kinds. get_type_hints() rather than raw __annotations__
    because `from __future__ import annotations` turns the latter into
    strings."""
    if _is_pydantic_model(cls):
        return [(name, field.annotation, not field.is_required())
                for name, field in cls.model_fields.items()]
    hints = typing.get_type_hints(cls)
    # hints[f.name] cannot miss: dataclass fields exist BECAUSE of an
    # entry in __annotations__, and get_type_hints returns every one.
    return [
        (f.name, hints[f.name],
         f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING)
        for f in dataclasses.fields(cls)
    ]


def _map_annotation(owner: str, field: str, annotation) -> tuple:
    """One annotation -> (Property kwargs, is_optional). Refuses anything
    outside the documented table -- a value silently stored as "we'll
    see" is the approximation this library exists not to make. Beyond
    the scalar table: a single-value-type Enum becomes its JSON type
    plus allowed values, datetime/date become "string" with a JSON
    Schema format, and a nested dataclass/pydantic model becomes an
    "object" with a nested property schema."""
    optional = False
    origin = typing.get_origin(annotation)
    if origin is Union or origin is _pytypes.UnionType:
        args = typing.get_args(annotation)
        non_null = [a for a in args if a is not type(None)]
        # ==1 alone decides: typing collapses single-member unions before
        # get_origin ever fires, so a union here has >= 2 args, and one
        # non-null member among them implies the other is None.
        if len(non_null) == 1:
            optional = True
            annotation = non_null[0]
            origin = typing.get_origin(annotation)
        else:
            names = ", ".join(getattr(a, "__name__", repr(a)) for a in args)
            raise TypeError(
                f"{owner}.{field}: a union of [{names}] has no single JSON type. Only "
                f"Optional[X] is mapped; spell any other union explicitly as a type set, "
                f"e.g. Property({field!r}, (\"number\", \"string\"))"
            )

    def spec(base: str, **extra) -> tuple:
        return {"json_type": (base, "null") if optional else (base,), **extra}, optional

    if isinstance(annotation, type) and issubclass(annotation, _enum.Enum):
        member_types = {type(member.value) for member in annotation}
        bases = {_ANNOTATION_TYPES.get(t) for t in member_types}
        if len(bases) != 1 or None in bases:
            raise TypeError(
                f"{owner}.{field}: enum {annotation.__name__} mixes value types -- allowed "
                f"values need ONE JSON type; spell it explicitly, e.g. "
                f"Property({field!r}, \"string\", values=(...))"
            )
        return spec(bases.pop(), values=tuple(member.value for member in annotation))
    if isinstance(annotation, type) and issubclass(annotation, _datetime.datetime):
        return spec("string", format="date-time")
    if isinstance(annotation, type) and issubclass(annotation, _datetime.date):
        return spec("string", format="date")
    if _is_schema_class(annotation):
        return spec("object", properties=_bag_properties(annotation))

    base = _ORIGIN_TYPES.get(origin) if origin is not None else _ANNOTATION_TYPES.get(annotation)
    if base is None:
        raise TypeError(
            f"{owner}.{field}: unsupported annotation {annotation!r}. The mapped annotations "
            f"are str, int, float, bool, dict, list, Optional[...] of those, single-type "
            f"Enums, datetime/date, and nested dataclass/pydantic classes; anything else "
            f"must say what it stores, e.g. Property({field!r}, \"string\") in the primitive "
            f"notation -- guessing a storage type silently is refused on purpose"
        )
    return spec(base)


def _bag_properties(cls: type, skip: tuple = ()) -> list:
    """A class used as a property bag -> [Property]. A field with no
    default is required; Optional[X] is not required and permits null."""
    properties = []
    for name, annotation, has_default in _class_fields(cls):
        if name in skip:
            continue
        kwargs, optional = _map_annotation(cls.__name__, name, annotation)
        properties.append(Property(name, required=not optional and not has_default, **kwargs))
    return properties


def _endpoint_annotations(cls: type) -> dict:
    """The source/target fields present on a class, with annotations.
    Presence of BOTH is what classifies a class as edge-shaped."""
    if _is_pydantic_model(cls):
        return {name: field.annotation for name, field in cls.model_fields.items()
                if name in _ENDPOINTS}
    field_names = {f.name for f in dataclasses.fields(cls)}
    present = [name for name in _ENDPOINTS if name in field_names]
    if not present:
        return {}
    hints = typing.get_type_hints(cls)
    return {name: hints[name] for name in present}


def _node_type_from_class(cls: type) -> NodeType:
    endpoints = _endpoint_annotations(cls)
    if len(endpoints) == len(_ENDPOINTS):
        raise TypeError(
            f"{cls.__name__} carries both 'source' and 'target' fields, which is the shape of "
            f"an edge class -- pass it in edges=[...], or drop the two fields if it really is "
            f"a node type"
        )
    return NodeType(_type_name(cls), properties=_bag_properties(cls))


def _edge_type_from_class(cls: type) -> EdgeType:
    endpoints = _endpoint_annotations(cls)
    missing = [name for name in _ENDPOINTS if name not in endpoints]
    if missing:
        raise TypeError(
            f"{cls.__name__}: an edge class declares its endpoints as fields 'source' and "
            f"'target' annotated with node classes -- missing {missing}. A class without them "
            f"is a node class and belongs in nodes=[...]"
        )
    for name, annotation in endpoints.items():
        if not _is_schema_class(annotation):
            raise TypeError(
                f"{cls.__name__}.{name} must be annotated with a node class (a dataclass or "
                f"pydantic model), got {annotation!r}"
            )
    return EdgeType(
        _type_name(cls),
        source=_type_name(endpoints["source"]),
        target=_type_name(endpoints["target"]),
        properties=_bag_properties(cls, skip=_ENDPOINTS),
    )


def _normalize_properties(properties, owner: str) -> tuple:
    if _is_schema_class(properties):
        normalized = _bag_properties(properties)
    else:
        if isinstance(properties, (str, Property)) or not hasattr(properties, "__iter__"):
            raise TypeError(
                f"{owner}: properties must be a list of Property(...) or a dataclass/pydantic "
                f"class used as a property bag, got {properties!r}"
            )
        normalized = list(properties)
        for prop in normalized:
            if not isinstance(prop, Property):
                raise TypeError(
                    f"{owner}: each property must be a Property(...), got {prop!r} -- "
                    f"e.g. Property('email', 'string', required=True)"
                )
    names = [p.name for p in normalized]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{owner}: duplicate property name(s) {duplicates}")
    return tuple(normalized)


def build_schema(nodes, edges) -> GraphSchema:
    """What Graph.define_schema() calls: normalize both notations into
    one validated GraphSchema."""
    node_types = []
    for entry in nodes or ():
        if isinstance(entry, NodeType):
            node_types.append(entry)
        elif _is_schema_class(entry):
            node_types.append(_node_type_from_class(entry))
        else:
            raise TypeError(
                f"nodes=[...] takes NodeType(...) entries or dataclass/pydantic classes, "
                f"got {entry!r}"
            )
    edge_types = []
    for entry in edges or ():
        if isinstance(entry, EdgeType):
            edge_types.append(entry)
        elif _is_schema_class(entry):
            edge_types.append(_edge_type_from_class(entry))
        else:
            raise TypeError(
                f"edges=[...] takes EdgeType(...) entries or dataclass/pydantic classes, "
                f"got {entry!r}"
            )
    return GraphSchema(node_types, edge_types)


def check_no_column_collisions(schema: GraphSchema, nodes_tbl, edges_tbl) -> None:
    """Refuse a declared property whose name is already a real column on
    THIS Graph's table -- the identity/graph column, a vec_* vector
    field, or an EXTRA COLUMN (models.py's "EXTENDING THE MODEL").
    `Property('user_id', 'number')` -- or a dataclass field of the same
    name, both normalize to the same thing -- on a table that already
    has a real `user_id` column would compile a CHECK on
    `properties->>'user_id'`, a JSONB key ingestion can never populate
    for that name: add_nodes()/merge_nodes() route it to the real
    column instead, every single time. Left uncaught, every insert of
    the type fails enforce_schema() forever, for a row that is
    perfectly correct -- worse than a refusal, because nothing points
    at why.

    Graph.define_schema() and Graph.load_schema() both call this --
    every path that adopts a schema, since a schema saved for one table
    can be load_schema()'d onto a handle over a different one. Not a
    GraphSchema.__init__ check: which columns collide depends on the
    TABLE, and a schema is deliberately reusable across tables (that is
    what save_schema()/load_schema() and infer_schema() are for).

    Nested properties (Property.properties) are not checked: a nested
    key compiles to `properties->'parent'->>'key'`, which cannot
    collide with a top-level table column no matter what it is named."""
    for nt in schema.node_types:
        collisions = sorted(p.name for p in nt.properties if p.name in nodes_tbl.c)
        if collisions:
            raise ValueError(
                f"NodeType({nt.name!r}): {collisions} already name real column(s) on "
                f"{nodes_tbl.name!r} -- add_nodes()/merge_nodes() write those by column, "
                f"never into `properties`, so declaring them as properties too would compile "
                f"a CHECK that can never see the value it is testing for. Rename the "
                f"field, or drop it from properties=/the dataclass"
            )
    for et in schema.edge_types:
        collisions = sorted(p.name for p in et.properties if p.name in edges_tbl.c)
        if collisions:
            raise ValueError(
                f"EdgeType({et.kind!r}): {collisions} already name real column(s) on "
                f"{edges_tbl.name!r} -- same reason as NodeType's version of this error: "
                f"rename the field, or drop it from properties=/the dataclass"
            )


# ---------------------------------------------------------------------
# Representations: shared pieces
# ---------------------------------------------------------------------

def _property_json(p: Property) -> dict:
    if p.properties:
        spec = _properties_json(p.properties)   # a nested object schema
        # the property's OWN type set, not the "object" _properties_json
        # assumes -- ("null", "object") must round-trip through
        # schema_from_document() as itself
        spec["type"] = p.json_type[0] if len(p.json_type) == 1 else list(p.json_type)
    else:
        spec = {"type": p.json_type[0] if len(p.json_type) == 1 else list(p.json_type)}
    if p.values:
        spec["enum"] = list(p.values)
    if p.format:
        spec["format"] = p.format
    if p.unique:
        spec["unique"] = True
    return spec


def _properties_json(properties: tuple) -> dict:
    spec = {
        "type": "object",
        "properties": {p.name: _property_json(p) for p in properties},
    }
    required = [p.name for p in properties if p.required]
    if required:
        spec["required"] = required
    return spec


def _mermaid_text(value: str) -> str:
    """Text safe inside a Mermaid double-quoted label: the one character
    that can terminate the label early is '"', and Mermaid's own escape
    for it is the #quot; entity."""
    return value.replace('"', "#quot;")


_PYTHON_TYPES = {"string": str, "number": float, "boolean": bool, "object": dict,
                 "array": list, "null": type(None)}


def _pydantic_model(name: str, properties: tuple, create_model):
    fields = {}
    for p in properties:
        if p.properties:
            annotation = _pydantic_model(f"{name}_{p.name}", p.properties, create_model)
        elif p.values:
            annotation = typing.Literal[tuple(p.values)]
        elif p.format == "date-time":
            annotation = _datetime.datetime
        elif p.format == "date":
            annotation = _datetime.date
        else:
            non_null = [_PYTHON_TYPES[t] for t in p.json_type if t != "null"]
            annotation = (non_null[0] if len(non_null) == 1
                          else Union[tuple(non_null)] if non_null else type(None))
        if "null" in p.json_type:
            annotation = Optional[annotation]
        # Not-required fields default to None WITHOUT widening the
        # annotation: absent is fine, an explicit null is only fine when
        # the schema says "null". (pydantic does not validate defaults.)
        fields[p.name] = (annotation, ... if p.required else None)
    model_name = "".join(part.capitalize() for part in name.split("_")) or name
    return create_model(model_name, **fields)


def _properties_per_kind(edge_types: tuple) -> dict:
    """Collapse edge types to one property schema per kind, refusing a
    kind whose endpoint pairs disagree. Both the pydantic representation
    and enforcement key on the kind alone (endpoints being unenforceable
    in a CHECK), so an inconsistent kind is refused rather than merged
    into something neither declaration said."""
    per_kind: dict = {}
    for et in edge_types:
        if et.kind in per_kind and per_kind[et.kind] != et.properties:
            raise ValueError(
                f"edge kind {et.kind!r} is declared with different property schemas for "
                f"different endpoint pairs; per-kind enforcement and models cannot represent "
                f"both -- unify the property schemas, or use two kinds"
            )
        per_kind.setdefault(et.kind, et.properties)
    return per_kind


# ---------------------------------------------------------------------
# Enforcement: schema -> CHECK constraints
# ---------------------------------------------------------------------

# Every CHECK owned by schema enforcement, on one table -- the working
# set enforce_schema() reconciles. The LIKE is escaped so '_' matches
# literally, and per-graph filtering happens in Python against
# schema_constraint_prefixes(), whose tokens are underscore-free.
SCHEMA_CHECKS = text("""
    SELECT conname FROM pg_constraint
    WHERE contype = 'c' AND conrelid = CAST(:table AS regclass)
      AND conname LIKE 'ck\\_schema\\_%' ESCAPE '\\'
""")


def _graph_token(graph: Optional[str]) -> str:
    """A fixed-charset token identifying the graph inside a constraint
    name. Reconciliation on re-enforce matches constraints by name
    prefix, so the token must never contain '_' -- a slugged graph name
    could, and would make one graph's prefix a prefix of another's.
    'default' (and the no-discriminator case) stay readable; anything
    else becomes a short stable hash."""
    if graph is None or graph == "default":
        return "default"
    return hashlib.sha256(graph.encode()).hexdigest()[:10]


def schema_constraint_prefixes(target: _Target) -> tuple:
    """The name prefixes marking a constraint as owned by schema
    enforcement of THIS graph on THIS table -- what enforce_schema()
    reconciles against."""
    token = _graph_token(target.graph)
    return (f"ck_schema_req_{token}_", f"ck_schema_typ_{token}_")


def _type_rules(target: _Target, discriminator: str, type_name: str,
                properties: tuple) -> list:
    """(name, unscoped boolean expression) pairs for one node type or
    edge kind: one presence rule covering every required property, plus
    one jsonb_typeof rule per property. Shared by DDL compilation and
    schema_violations(), so the dry-run and the CHECK can never disagree
    about what violates. Distinct req_/typ_ name prefixes, so a property
    named 'required' can never collide with the presence rule."""
    guard = target.properties_col[discriminator].astext.is_distinct_from(type_name)
    token = _graph_token(target.graph)
    rules = []
    required = tuple(p.name for p in properties if p.required)
    if required:
        name = f"ck_schema_req_{token}_{_slug(type_name)}"[:63]
        rules.append((name, or_(guard, target.properties_col.has_all(
            postgresql.array(required)))))
    for p in properties:
        name = f"ck_schema_typ_{token}_{_slug(type_name)}_{_slug(p.name)}"[:63]
        typeof = func.jsonb_typeof(target.properties_col[p.name])
        # jsonb_typeof of a MISSING key is NULL and a CHECK passes on
        # NULL, so this constrains the type of a value that is there --
        # presence is the req_ rule's job, same split as PropertyType vs
        # Required in constraints.py.
        expression = typeof == p.json_type[0] if len(p.json_type) == 1 else typeof.in_(p.json_type)
        rules.append((name, or_(guard, expression)))
        if p.values:
            # ->> text comparison, so allowed values compare by their
            # text rendering (a numeric enum of 1 matches '1'). Absent
            # keys are NULL and pass -- presence stays req_'s job.
            allowed = [v if isinstance(v, str) else str(v) for v in p.values]
            name = f"ck_schema_val_{token}_{_slug(type_name)}_{_slug(p.name)}"[:63]
            rules.append((name, or_(guard, target.properties_col[p.name].astext.in_(allowed))))
    return rules


def _type_uniques(target: _Target, discriminator: str, type_name: str,
                  properties: tuple) -> list:
    """(name, ddl) per unique property: the PARTIAL unique index --
    unique among rows of THIS type only, the constraint the README
    already markets. Compiled through the same Unique() machinery
    define_constraints() uses, so the index shape and any future merge
    conflict-target stay in step."""
    from .constraints import Unique, compile_constraint
    pairs = []
    token = _graph_token(target.graph)
    for p in properties:
        if not p.unique:
            continue
        name = f"uq_schema_{token}_{_slug(type_name)}_{_slug(p.name)}"[:63]
        _, _, ddl = compile_constraint(
            Unique(p.name, where={discriminator: type_name}, name=name), target)
        pairs.append((name, ddl))
    return pairs


def compile_node_uniques(schema: GraphSchema, target: _Target) -> list:
    return [pair for nt in schema.node_types
            for pair in _type_uniques(target, "type", nt.name, nt.properties)]


def compile_edge_uniques(schema: GraphSchema, target: _Target) -> list:
    return [pair for kind, properties in _properties_per_kind(schema.edge_types).items()
            for pair in _type_uniques(target, "kind", kind, properties)]


SCHEMA_UNIQUES = text("""
    SELECT indexname FROM pg_indexes
    WHERE tablename = :table AND indexname LIKE 'uq\\_schema\\_%' ESCAPE '\\'
""")


def _type_constraints(target: _Target, discriminator: str, type_name: str,
                      properties: tuple) -> list:
    return [(name, _compile_check(target.table, name, target.scope_check(expression)))
            for name, expression in _type_rules(target, discriminator, type_name, properties)]


def compile_node_constraints(schema: GraphSchema, target: _Target) -> list:
    return [pair for nt in schema.node_types
            for pair in _type_constraints(target, "type", nt.name, nt.properties)]


def compile_edge_constraints(schema: GraphSchema, target: _Target) -> list:
    return [pair for kind, properties in _properties_per_kind(schema.edge_types).items()
            for pair in _type_constraints(target, "kind", kind, properties)]


# ---------------------------------------------------------------------
# Endpoint-type enforcement: (kind, source, target) as a trigger
# ---------------------------------------------------------------------
# A CHECK cannot subquery nodes, so "works_at connects only person ->
# company" needs a CONSTRAINT TRIGGER -- still plain Postgres, still no
# extension. It fires per edge row, which is why endpoints=True is an
# explicit opt-in on enforce_schema() rather than the default.

def _quote(value: str) -> str:
    """A string literal for DDL. Same rationale as constraints._literal:
    DDL cannot carry bind parameters, and these values come from schema
    declarations written by a developer, not from ingested data."""
    return "'" + value.replace("'", "''") + "'"


def endpoint_names(graph) -> tuple:
    """(trigger name, function name) for this graph's endpoint trigger.
    The function name carries the table too: trigger names are scoped
    per table, but function names are schema-global, and two custom
    edge tables must not share one function."""
    token = _graph_token(graph.graph if graph.graph_col is not None else None)
    trigger = f"ck_schema_end_{token}"[:63]
    function = f"ck_schema_endf_{token}_{_slug(graph.edges_tbl.name)}"[:63]
    return trigger, function


def compile_endpoint_ddl(schema: GraphSchema, graph) -> list:
    """The function + trigger DDL policing declared (kind, source,
    target) triples on this graph's edges table. Untyped/undeclared
    kinds pass; a DECLARED kind must connect a declared pair, and an
    untyped endpoint cannot satisfy any triple -- the error says which.

    Raises with ERRCODE 23514 (check_violation) and the trigger's name
    as CONSTRAINT, so the driver's diagnostics carry it and the write
    path's existing translation surfaces a ConstraintViolation naming
    it, like every other schema rule."""
    trigger_name, function_name = endpoint_names(graph)
    et, nt = graph.edges_tbl, graph.nodes_tbl
    edges_q = f'"{et.schema}"."{et.name}"' if et.schema else f'"{et.name}"'
    nodes_q = f'"{nt.schema}"."{nt.name}"' if nt.schema else f'"{nt.name}"'
    function_q = f'"{et.schema}"."{function_name}"' if et.schema else f'"{function_name}"'

    triples = sorted((e.kind, e.source, e.target) for e in schema.edge_types)
    kinds = sorted({k for k, _, _ in triples})
    values = ", ".join(f"({_quote(k)}, {_quote(s)}, {_quote(t)})" for k, s, t in triples)
    declared = "; ".join(f"{k}: {s} -> {t}" for k, s, t in triples)

    graph_guard = ""
    node_scope = ""
    if graph.graph_col is not None:
        graph_guard = (f"  IF NEW.{graph.graph_col} IS DISTINCT FROM "
                       f"{_quote(graph.graph)} THEN RETURN NEW; END IF;\n")
        node_scope = f" AND n.{graph.graph_col} = NEW.{graph.graph_col}"

    function_ddl = f"""CREATE OR REPLACE FUNCTION {function_q}() RETURNS trigger AS $ck$
DECLARE
  edge_kind text; src_type text; dst_type text;
BEGIN
{graph_guard}  edge_kind := NEW.properties->>'kind';
  IF edge_kind IS NULL OR edge_kind NOT IN ({", ".join(_quote(k) for k in kinds)}) THEN
    RETURN NEW;  -- undeclared and kindless edges are not policed
  END IF;
  SELECT n.properties->>'type' INTO src_type FROM {nodes_q} n
    WHERE n.{graph.node_id_col} = NEW.{graph.edge_start_col}{node_scope};
  SELECT n.properties->>'type' INTO dst_type FROM {nodes_q} n
    WHERE n.{graph.node_id_col} = NEW.{graph.edge_end_col}{node_scope};
  IF (edge_kind, src_type, dst_type) IN (VALUES {values}) THEN
    RETURN NEW;
  END IF;
  -- the observed types and the declared triples go in DETAIL: the
  -- driver-error translation keeps diagnostics, not the primary text,
  -- and the DETAIL is what reaches ConstraintViolation's message
  RAISE EXCEPTION 'edge violates the declared endpoint types'
    USING ERRCODE = '23514', CONSTRAINT = {_quote(trigger_name)},
      DETAIL = format('%s connects %s -> %s, but the schema declares: '
                      {_quote(declared)},
                      edge_kind, coalesce(src_type, '(untyped)'),
                      coalesce(dst_type, '(untyped)'));
END $ck$ LANGUAGE plpgsql"""

    # No CREATE OR REPLACE for CONSTRAINT triggers, so idempotency is
    # the DROP/CREATE pair, run in enforce_schema()'s one transaction.
    return [
        function_ddl,
        f'DROP TRIGGER IF EXISTS "{trigger_name}" ON {edges_q}',
        f'CREATE CONSTRAINT TRIGGER "{trigger_name}" AFTER INSERT OR UPDATE ON {edges_q} '
        f'FOR EACH ROW EXECUTE FUNCTION {function_q}()',
    ]


ENDPOINT_TRIGGER_EXISTS = text("""
    SELECT 1 FROM pg_trigger
    WHERE tgname = :name AND tgrelid = CAST(:table AS regclass)
""")


# ---------------------------------------------------------------------
# Strict mode: the Cypher front end validated against the schema
# ---------------------------------------------------------------------
# With a schema defined, MATCH (a:persn) is a typo or a hallucination --
# unvalidated it silently matches nothing, the worst outcome for an LLM
# caller. Validation runs over the TRANSLATION OUTPUT (Start/Hop or
# operations), never inside the parser: the front ends stay front ends.

def _filter_vocabulary(filt, pairs: list, keys: set) -> None:
    """Collect (key, value) equality pairs and every referenced property
    key from a translated filter. The callable escape hatch is opaque on
    purpose and collects nothing."""
    from .filters import AND, BETWEEN, GT, GTE, LT, LTE, NOT, OR
    if filt is None:
        return
    if isinstance(filt, dict):
        for key, value in filt.items():
            pairs.append((key, value))
            keys.add(key)
    elif isinstance(filt, (AND, OR)):
        for term in filt.filters:
            _filter_vocabulary(term, pairs, keys)
    elif isinstance(filt, NOT):
        _filter_vocabulary(filt.filt, pairs, keys)
    elif isinstance(filt, (GT, GTE, LT, LTE, BETWEEN)):
        keys.add(filt.key)


def _check_vocabulary(filt, declared: dict, discriminator: str, what: str) -> None:
    from .cypher import CypherError
    pairs: list = []
    keys: set = set()
    _filter_vocabulary(filt, pairs, keys)
    named = [value for key, value in pairs if key == discriminator]
    flat = [v for value in named for v in (value if isinstance(value, list) else [value])]
    for name in flat:
        if name not in declared:
            raise CypherError(
                f"unknown {what} {name!r} -- the schema declares: "
                f"{', '.join(sorted(declared))}"
            )
    if not flat:
        # no discriminator, no verdict: an untyped pattern's properties
        # are legitimately outside the schema (documented limit)
        return
    allowed = set().union(*(declared[name] for name in flat)) | {discriminator}
    unknown = sorted(keys - allowed)
    if unknown:
        raise CypherError(
            f"unknown propert{'ies' if len(unknown) > 1 else 'y'} {unknown} for "
            f"{'/'.join(sorted(set(flat)))} -- the schema declares: "
            f"{', '.join(sorted(allowed - {discriminator}))}"
        )


def validate_traversal(schema: GraphSchema, start, hops,
                       node_label_key: Optional[str] = "type",
                       edge_type_key: Optional[str] = "kind") -> None:
    """Refuse a translated traversal whose labels, kinds or properties
    the schema does not declare -- CypherErrors listing the vocabulary,
    so a hallucinated label becomes an immediate fix instead of a
    silently empty result."""
    node_types = {nt.name: {p.name for p in nt.properties} for nt in schema.node_types}
    edge_kinds: dict = {}
    for et in schema.edge_types:
        edge_kinds.setdefault(et.kind, set()).update(p.name for p in et.properties)
    if node_label_key is not None:
        _check_vocabulary(start.where, node_types, node_label_key, "label")
    for hop in hops:
        if node_label_key is not None:
            _check_vocabulary(hop.where, node_types, node_label_key, "label")
        if edge_type_key is not None:
            _check_vocabulary(hop.via, edge_kinds, edge_type_key, "relationship kind")


def validate_operations(schema: GraphSchema, operations: list,
                        node_label_key: Optional[str] = "type",
                        edge_type_key: Optional[str] = "kind") -> None:
    """The write-side twin: every row a plan would write is a property
    dict, validated with the same vocabulary rules."""
    node_types = {nt.name: {p.name for p in nt.properties} for nt in schema.node_types}
    edge_kinds: dict = {}
    for et in schema.edge_types:
        edge_kinds.setdefault(et.kind, set()).update(p.name for p in et.properties)
    for op in operations:
        if op["op"] in ("create_nodes", "merge_nodes"):
            for row in op["rows"]:
                if node_label_key is not None:
                    _check_vocabulary(dict(row), node_types, node_label_key, "label")
        elif op["op"] in ("create_edges", "merge_edges"):
            for row in op["rows"]:
                if edge_type_key is not None:
                    _check_vocabulary(dict(row.get("properties", {})), edge_kinds,
                                      edge_type_key, "relationship kind")
        elif op["op"] == "match" and node_label_key is not None:
            _check_vocabulary(dict(op["where"]), node_types, node_label_key, "label")


def validate_mutations(schema: GraphSchema, operations: list,
                       node_label_key: Optional[str] = "type",
                       edge_type_key: Optional[str] = "kind") -> None:
    """The delete/update twin.

    A hallucinated label costs more here than on the read path. There it
    returns an empty subgraph, which at least looks like a result; a
    delete that matched nothing reports success, and "0 rows" is exactly
    what a correct delete of an already-clean graph reports too. The
    properties an update WRITES are checked as well, against the type
    its filter names -- the same rule validate_operations() applies to
    a created row."""
    node_types = {nt.name: {p.name for p in nt.properties} for nt in schema.node_types}
    edge_kinds: dict = {}
    for et in schema.edge_types:
        edge_kinds.setdefault(et.kind, set()).update(p.name for p in et.properties)

    for op in operations:
        edges = op["op"].endswith("_edges")
        declared = edge_kinds if edges else node_types
        key = edge_type_key if edges else node_label_key
        what = "relationship kind" if edges else "label"
        if key is not None:
            _check_vocabulary(op.get("where"), declared, key, what)
            written = {**(op.get("set") or {}), **dict.fromkeys(op.get("remove") or ())}
            for name in _discriminator_values(op.get("where"), key):
                _check_vocabulary({key: name, **written}, declared, key, what)
        if edges and node_label_key is not None:
            for side in ("start", "end"):
                _check_vocabulary(op.get(side), node_types, node_label_key, "label")


def _discriminator_values(filt, discriminator: str) -> list:
    """Every value a filter pins the discriminator to -- what a mutation
    names as the type it is changing, since `set` carries no label of
    its own."""
    pairs: list = []
    _filter_vocabulary(filt, pairs, set())
    named = [value for key, value in pairs if key == discriminator]
    return [v for value in named for v in (value if isinstance(value, list) else [value])]


# ---------------------------------------------------------------------
# The schema, summarized for a tool-calling model
# ---------------------------------------------------------------------

#: Properties listed per type in a tool description before "+N more"
#: takes over. A model needs the vocabulary, not an inventory -- and a
#: tool description is prompt budget someone else is paying for.
_TOOL_SUMMARY_PROPERTIES = 12


def _property_bag(properties: tuple) -> str:
    """'(email*, sku!, +3 more)' -- the bounded property list
    tool_summary() and to_mermaid() share, so the prompt form and the
    picture form can never disagree about markers: * required, ! unique,
    capped because both outputs are budget someone else is paying for."""
    shown = [p.name + ("*" if p.required else "") + ("!" if p.unique else "")
             for p in properties[:_TOOL_SUMMARY_PROPERTIES]]
    overflow = len(properties) - _TOOL_SUMMARY_PROPERTIES
    if overflow > 0:
        shown.append(f"+{overflow} more")
    return f"({', '.join(shown)})" if shown else ""


def tool_summary(schema: GraphSchema) -> str:
    """The declared schema as one compact paragraph for a tool
    description: what an agent can filter on, so it stops guessing
    labels and property names. `*` marks required, endpoint pairs are
    grouped per kind, and long property lists are capped -- bounded on
    purpose, never an unbounded dump."""
    nodes = "; ".join(f"{nt.name}{_property_bag(nt.properties)}" for nt in schema.node_types)
    per_kind: dict = {}
    for et in schema.edge_types:
        pairs, _ = per_kind.setdefault(et.kind, ([], et.properties))
        pairs.append(f"{et.source} -> {et.target}")
    edges = "; ".join(
        f"{kind}: {', '.join(pairs)}"
        + (f" {_property_bag(properties)}" if properties else "")
        for kind, (pairs, properties) in per_kind.items())
    parts = ["This graph's declared schema (properties in parentheses, "
             "* = required, ! = unique)."]
    if nodes:
        parts.append(f"Node types: {nodes}.")
    if edges:
        parts.append(f"Edge kinds (source -> target): {edges}.")
    return " ".join(parts)


# ---------------------------------------------------------------------
# Violations: the dry-run before enforcement
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RuleViolations:
    """Rows one schema rule would reject -- named exactly as the CHECK
    enforce_schema() would create, so the report reads as the work list
    for the enforcement that failed (or is about to)."""
    constraint: str
    table: str
    rows: int
    sample_ids: tuple


@dataclass
class SchemaViolations:
    """What enforce_schema() would reject, found by reading. Falsy when
    clean, so `if graph.schema_violations():` reads correctly."""
    rules: tuple

    def __bool__(self) -> bool:
        return bool(self.rules)

    def __str__(self) -> str:
        if not self.rules:
            return "no schema violations -- enforce_schema() would succeed"
        lines = [f"{sum(r.rows for r in self.rules)} row(s) violate "
                 f"{len(self.rules)} schema rule(s):"]
        for r in self.rules:
            ids = ", ".join(str(i) for i in r.sample_ids)
            more = "" if r.rows <= len(r.sample_ids) else ", ..."
            lines.append(f"  {r.constraint} ({r.table}): {r.rows} row(s), e.g. id {ids}{more}")
        return "\n".join(lines)


def find_violations(graph, sample: int = 5) -> SchemaViolations:
    """Evaluate every rule the schema would enforce, as SELECTs. A row
    the CHECK would reject is exactly a row where the scoped check body
    is false -- NOT of it, with SQL's NULL-passes semantics matching the
    CHECK's for free. Read-only: no DDL, nothing registered."""
    from sqlalchemy import not_, select

    schema = graph._schema
    node_target, edge_target = graph._schema_targets()
    groups = [
        (node_target, graph.nodes_tbl, graph.node_id_col,
         [rule for nt in schema.node_types
          for rule in _type_rules(node_target, "type", nt.name, nt.properties)]),
        (edge_target, graph.edges_tbl, graph.edge_id_col,
         [rule for kind, properties in _properties_per_kind(schema.edge_types).items()
          for rule in _type_rules(edge_target, "kind", kind, properties)]),
    ]
    unique_groups = [
        (node_target, graph.nodes_tbl, graph.node_id_col, "type",
         [(nt.name, p) for nt in schema.node_types for p in nt.properties if p.unique]),
        (edge_target, graph.edges_tbl, graph.edge_id_col, "kind",
         [(kind, p) for kind, properties in _properties_per_kind(schema.edge_types).items()
          for p in properties if p.unique]),
    ]

    violated = []
    with graph.engine.connect() as connection:
        for target, table, id_col_name, rules in groups:
            id_col = getattr(table.c, id_col_name)
            for name, expression in rules:
                failing = not_(target.scope_check(expression))
                rows = connection.execute(
                    select(func.count()).select_from(table).where(failing)).scalar()
                if not rows:
                    continue
                ids = tuple(row[0] for row in connection.execute(
                    select(id_col).select_from(table).where(failing)
                    .order_by(id_col).limit(sample)).all())
                violated.append(RuleViolations(name, target.label, rows, ids))

        for target, table, id_col_name, discriminator, unique_rules in unique_groups:
            id_col = getattr(table.c, id_col_name)
            for type_name, p in unique_rules:
                # a unique rule's violations are every row in a duplicate
                # group -- the ADD INDEX failure names one pair, this
                # names them all
                name = (f"uq_schema_{_graph_token(target.graph)}_"
                        f"{_slug(type_name)}_{_slug(p.name)}")[:63]
                value = table.c.properties[p.name].astext
                scoped = [graph._scoped(table),
                          table.c.properties[discriminator].astext == type_name,
                          value.isnot(None)]
                duplicated = (select(value).where(*scoped)
                              .group_by(value).having(func.count() > 1))
                failing = select(id_col).where(*scoped, value.in_(duplicated))
                rows = connection.execute(select(func.count()).select_from(
                    failing.subquery())).scalar()
                if not rows:
                    continue
                ids = tuple(row[0] for row in connection.execute(
                    failing.order_by(id_col).limit(sample)).all())
                violated.append(RuleViolations(name, target.label, rows, ids))
    return SchemaViolations(tuple(violated))


# ---------------------------------------------------------------------
# Inference: existing rows -> GraphSchema
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TypeConflict:
    """One property observed with more than one non-null JSON type --
    the 42-versus-"42" situation. The schema carries the honest type
    set; this entry is the pointer to the data worth cleaning."""
    table: str
    type_name: str
    key: str
    json_types: tuple


@dataclass
class InferenceReport:
    """What infer_schema() saw that the schema alone cannot say -- read
    it before adopting: `required` on a three-row type means much less
    than on a three-million-row one."""
    node_counts: dict
    edge_counts: dict
    untyped_nodes: int
    untyped_edges: int
    skipped_endpoint_edges: int
    conflicts: tuple
    sampled: Optional[float] = None

    def __str__(self) -> str:
        lines = []
        if self.sampled is not None:
            lines.append(f"sampled {self.sampled}% of rows -- counts are estimates")
        lines += [
            f"nodes: {sum(self.node_counts.values())} typed across "
            f"{len(self.node_counts)} type(s) {dict(sorted(self.node_counts.items()))}, "
            f"{self.untyped_nodes} untyped (outside the schema)",
            f"edges: {sum(self.edge_counts.values())} with a kind across "
            f"{len(self.edge_counts)} kind(s) {dict(sorted(self.edge_counts.items()))}, "
            f"{self.untyped_edges} kindless, "
            f"{self.skipped_endpoint_edges} skipped (endpoint node's type is not "
            f"one of the above)",
        ]
        for c in self.conflicts:
            lines.append(f"conflict: {c.table}/{c.type_name}.{c.key} observed as "
                         f"{list(c.json_types)}")
        return "\n".join(lines)


def _named(value: Optional[str]) -> bool:
    """A usable type/kind name. NULL means the key is absent (or held a
    JSON null); '' would be a NodeType the model rejects -- neither can
    be invented into a type, so both count as untyped."""
    return value is not None and value != ""


def _source(table, sample_percent: Optional[float]):
    """The table, or a TABLESAMPLE SYSTEM view of it. Sampling trades
    exactness for speed and the caller opted in -- the report carries
    the flag so nobody reads estimated counts as truth."""
    if sample_percent is None:
        return table
    from sqlalchemy import tablesample
    return tablesample(table, sample_percent)


def _observed_counts(connection, graph, source, discriminator: str) -> dict:
    from sqlalchemy import func, select
    name = source.c.properties[discriminator].astext
    rows = connection.execute(
        select(name.label("name"), func.count().label("rows"))
        .where(graph._scoped(source)).group_by(name)
    ).all()
    return {r.name: r.rows for r in rows}


def _observed_keys(connection, graph, source, discriminator: str) -> list:
    """(type name, key, json type, row count) per distinct combination
    -- one lateral jsonb_each scan, Postgres doing all the work."""
    from sqlalchemy import func, select
    kv = func.jsonb_each(source.c.properties).table_valued(
        "key", "value", joins_implicitly=True)
    name = source.c.properties[discriminator].astext
    json_type = func.jsonb_typeof(kv.c.value)
    return connection.execute(
        select(name.label("name"), kv.c.key.label("key"),
               json_type.label("json_type"), func.count().label("rows"))
        .where(graph._scoped(source))
        .group_by(name, kv.c.key, json_type)
    ).all()


def _derive_properties(counts: dict, key_rows: list, discriminator: str,
                       table_label: str, conflicts: list) -> dict:
    """type name -> [Property], with required-ness from presence counts:
    a key on EVERY row of its type is required, anything else optional;
    an observed JSON null makes it nullable. Mixed non-null types become
    a type set and a report entry -- never a silent winner."""
    per_type: dict = {}
    for row in key_rows:
        if not _named(row.name) or row.key == discriminator:
            continue
        per_type.setdefault(row.name, {}).setdefault(row.key, {})[row.json_type] = row.rows
    properties: dict = {}
    for type_name in sorted(k for k in counts if _named(k)):
        props = []
        for key, observed in sorted(per_type.get(type_name, {}).items()):
            json_types = tuple(sorted(observed))
            non_null = [t for t in json_types if t != "null"]
            if len(non_null) > 1:
                conflicts.append(TypeConflict(table_label, type_name, key, json_types))
            props.append(Property(key, json_types,
                                  required=sum(observed.values()) == counts[type_name]))
        properties[type_name] = props
    return properties


def infer_schema(graph, sample_percent: Optional[float] = None) -> tuple:
    """Derive (GraphSchema, InferenceReport) from the rows this graph
    already holds. Read-only, never registers itself as the handle's
    schema -- see the module docstring for why observation and contract
    stay separate. Every query is scoped to this graph.

    sample_percent reads TABLESAMPLE SYSTEM (p) instead of every row:
    counts become estimates, a rare property or edge triple can be
    missed entirely, and `required` is tentative -- a key present on
    every SAMPLED row may be missing elsewhere -- so the report carries
    `sampled` and says estimates. The endpoint-triple join samples the
    edges side only: sampling the node side too would count edges whose
    node fell outside the sample as endpoint-less, manufacturing
    skipped-edge noise the data does not contain."""
    from sqlalchemy import and_, func, select

    if sample_percent is not None and not 0 < sample_percent <= 100:
        raise ValueError(
            f"sample_percent must satisfy 0 < sample_percent <= 100, "
            f"got {sample_percent!r}")

    nt, et = graph.nodes_tbl, graph.edges_tbl
    node_source = _source(nt, sample_percent)
    edge_source = _source(et, sample_percent)
    conflicts: list = []
    with graph.engine.connect() as connection:
        node_counts = _observed_counts(connection, graph, node_source, "type")
        edge_counts = _observed_counts(connection, graph, edge_source, "kind")
        node_props = _derive_properties(node_counts, _observed_keys(
            connection, graph, node_source, "type"), "type", "nodes", conflicts)
        edge_props = _derive_properties(edge_counts, _observed_keys(
            connection, graph, edge_source, "kind"), "kind", "edges", conflicts)

        edges = edge_source
        source, target = nt.alias("src"), nt.alias("dst")
        kind = edges.c.properties["kind"].astext
        node_id = graph.node_id_col
        triples = connection.execute(
            select(kind.label("kind"),
                   source.c.properties["type"].astext.label("source"),
                   target.c.properties["type"].astext.label("target"),
                   func.count().label("rows"))
            .select_from(
                edges.join(source, getattr(edges.c, graph.edge_start_col) == getattr(source.c, node_id))
                     .join(target, getattr(edges.c, graph.edge_end_col) == getattr(target.c, node_id)))
            .where(and_(graph._scoped(edges), graph._scoped(source), graph._scoped(target)))
            .group_by(kind, source.c.properties["type"].astext,
                      target.c.properties["type"].astext)
        ).all()

    node_types = [NodeType(name, properties=props)
                  for name, props in sorted(node_props.items())]
    # The endpoint join above reads the WHOLE nodes table, while
    # node_types comes from the sample -- so under sample_percent an edge
    # can name an endpoint type the node sample never saw. Inference then
    # built EdgeType(source='person') against an empty node-type list and
    # GraphSchema rejected the schema its own inference had just
    # produced, which is a raise from a function whose signature promises
    # a (schema, report) pair. An unobserved type is not a type this run
    # can claim, so the edge type goes the same way an untyped endpoint
    # does -- counted, not silently dropped.
    observed = {t.name for t in node_types}
    edge_types = []
    skipped = 0
    for row in sorted(triples, key=lambda r: (r.kind or "", r.source or "", r.target or "")):
        if not _named(row.kind):
            continue  # already counted as kindless below
        if row.source not in observed or row.target not in observed:
            skipped += row.rows
            continue
        edge_types.append(EdgeType(row.kind, source=row.source, target=row.target,
                                   properties=edge_props.get(row.kind, ())))

    report = InferenceReport(
        node_counts={k: v for k, v in node_counts.items() if _named(k)},
        edge_counts={k: v for k, v in edge_counts.items() if _named(k)},
        untyped_nodes=sum(v for k, v in node_counts.items() if not _named(k)),
        untyped_edges=sum(v for k, v in edge_counts.items() if not _named(k)),
        skipped_endpoint_edges=skipped,
        conflicts=tuple(conflicts),
        sampled=sample_percent,
    )
    return GraphSchema(node_types, edge_types), report


# ---------------------------------------------------------------------
# Persistence: the to_json() document, parsed back
# ---------------------------------------------------------------------

def _properties_from_json(spec, owner: str) -> list:
    """[Property] from a _properties_json() rendering. Structural
    checks raise naming the path; everything else is left to the
    Property constructor, so a corrupted document fails with the same
    complaint hand-written bad input would."""
    if not isinstance(spec, dict) or not isinstance(spec.get("properties", {}), dict):
        raise ValueError(
            f"{owner}: expected a JSON Schema object spec with a 'properties' "
            f"mapping, got {spec!r}")
    required = spec.get("required", [])
    if not isinstance(required, (list, tuple)):
        # 'in' on a string would do substring matching -- a corrupted
        # "required": "email" must not quietly mark nothing (or the
        # wrong thing) required
        raise ValueError(f"{owner}: 'required' must be a list of property names, "
                         f"got {required!r}")
    properties = []
    for name, p in spec.get("properties", {}).items():
        if not isinstance(p, dict) or "type" not in p:
            raise ValueError(
                f"{owner} property {name!r}: expected a spec object with a 'type', "
                f"got {p!r}")
        properties.append(Property(
            name, p["type"],
            required=name in required,
            unique=bool(p.get("unique", False)),
            values=tuple(p.get("enum", ())),
            format=p.get("format"),
            properties=(_properties_from_json(p, owner=f"{owner} property {name!r}")
                        if "properties" in p else ()),
        ))
    return properties


def schema_from_document(document) -> GraphSchema:
    """The exact inverse of GraphSchema.to_json() -- how load_schema()
    turns a stored hopai_schema row back into the canonical dataclasses.

    The stored document is data, not trusted state: every piece routes
    through the same Property/NodeType/EdgeType/GraphSchema constructors
    define_schema() uses, so a hand-corrupted row raises the real
    validation error and never half-loads."""
    if (not isinstance(document, dict)
            or not isinstance(document.get("nodes"), dict)
            or not isinstance(document.get("edges"), list)):
        raise ValueError(
            "not a hopai schema document: expected {'nodes': {...}, 'edges': [...]} "
            "as written by save_schema()")
    node_types = [
        NodeType(name, properties=_properties_from_json(spec, owner=f"node type {name!r}"))
        for name, spec in document["nodes"].items()
    ]
    edge_types = []
    for entry in document["edges"]:
        if not isinstance(entry, dict):
            raise ValueError(f"schema document edge entries must be objects, got {entry!r}")
        edge_types.append(EdgeType(
            entry.get("kind"), source=entry.get("source"), target=entry.get("target"),
            properties=_properties_from_json(
                entry.get("properties", {"type": "object", "properties": {}}),
                owner=f"edge kind {entry.get('kind')!r}"),
        ))
    return GraphSchema(node_types, edge_types)
