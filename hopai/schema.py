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

WHAT ENFORCEMENT DOES NOT COVER: endpoint types. "works_at connects only
person -> company" needs a subquery against nodes, which a CHECK cannot
contain. Declaring it documents intent and feeds every representation,
but the database does not police it (a trigger could -- tracked as a
follow-up, still pure Postgres). Also uncovered: two edge types sharing
a kind with DIFFERENT property schemas. Enforcement is keyed on the kind
alone (endpoints being unenforceable, above), so it refuses that shape
outright rather than enforcing the wrong merge of the two.

The node discriminator is the `type` property and the edge discriminator
is `kind` -- the convention the Cypher front end already uses for
labels. Nothing requires a row to carry them until the schema is
enforced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import types as _pytypes
import typing
from dataclasses import dataclass
from typing import Optional, Union

from sqlalchemy import func, or_, text
from sqlalchemy.dialects import postgresql

from .constraints import JSON_TYPES, _add_check, _literal, _slug, _Target

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
    input form -- or any order -- compares equal."""
    name: str
    json_type: tuple
    required: bool = False

    def __init__(self, name: str, json_type, required: bool = False):
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
            # sqlmodel tolerates pydantic v1, so v1 can genuinely be what
            # is installed -- and v1's create_model builds models with
            # different semantics rather than failing, which is worse.
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


# ---------------------------------------------------------------------
# Class notation -> canonical model
# ---------------------------------------------------------------------

def _is_pydantic_model(obj) -> bool:
    # Duck-typed on purpose: recognizing a pydantic model must not cost
    # hopai a pydantic import -- the caller who passed one already paid
    # it. model_fields is the v2 marker; v1 models (which sqlmodel still
    # tolerates) fall through and are refused as unsupported input.
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
    """One annotation -> (json_type tuple, is_optional). Refuses anything
    outside the documented table -- a datetime silently stored as "we'll
    see" is the approximation this library exists not to make."""
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
    base = _ORIGIN_TYPES.get(origin) if origin is not None else _ANNOTATION_TYPES.get(annotation)
    if base is None:
        raise TypeError(
            f"{owner}.{field}: unsupported annotation {annotation!r}. The mapped annotations "
            f"are str, int, float, bool, dict, list and Optional[...] of those; anything else "
            f"must say what it stores, e.g. Property({field!r}, \"string\") in the primitive "
            f"notation -- guessing a storage type silently is refused on purpose"
        )
    return ((base, "null") if optional else (base,)), optional


def _bag_properties(cls: type, skip: tuple = ()) -> list:
    """A class used as a property bag -> [Property]. A field with no
    default is required; Optional[X] is not required and permits null."""
    properties = []
    for name, annotation, has_default in _class_fields(cls):
        if name in skip:
            continue
        type_names, optional = _map_annotation(cls.__name__, name, annotation)
        properties.append(Property(name, type_names, required=not optional and not has_default))
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


# ---------------------------------------------------------------------
# Representations: shared pieces
# ---------------------------------------------------------------------

def _properties_json(properties: tuple) -> dict:
    spec = {
        "type": "object",
        "properties": {
            p.name: {"type": p.json_type[0] if len(p.json_type) == 1 else list(p.json_type)}
            for p in properties
        },
    }
    required = [p.name for p in properties if p.required]
    if required:
        spec["required"] = required
    return spec


_PYTHON_TYPES = {"string": str, "number": float, "boolean": bool, "object": dict,
                 "array": list, "null": type(None)}


def _pydantic_model(name: str, properties: tuple, create_model):
    fields = {}
    for p in properties:
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


def _type_constraints(target: _Target, discriminator: str, type_name: str,
                      properties: tuple) -> list:
    """(name, ddl) pairs for one node type or edge kind: one presence
    CHECK covering every required property, plus one jsonb_typeof CHECK
    per property. Distinct req_/typ_ name prefixes, so a property
    named 'required' can never collide with the presence check."""
    guard = target.properties_col[discriminator].astext.is_distinct_from(type_name)
    token = _graph_token(target.graph)
    pairs = []
    required = tuple(p.name for p in properties if p.required)
    if required:
        name = f"ck_schema_req_{token}_{_slug(type_name)}"[:63]
        body = _literal(target.scope_check(
            or_(guard, target.properties_col.has_all(postgresql.array(required)))))
        pairs.append((name, _add_check(target, name, body)))
    for p in properties:
        name = f"ck_schema_typ_{token}_{_slug(type_name)}_{_slug(p.name)}"[:63]
        typeof = func.jsonb_typeof(target.properties_col[p.name])
        # jsonb_typeof of a MISSING key is NULL and a CHECK passes on
        # NULL, so this constrains the type of a value that is there --
        # presence is the req_ constraint's job, same split as
        # PropertyType vs Required in constraints.py.
        expression = typeof == p.json_type[0] if len(p.json_type) == 1 else typeof.in_(p.json_type)
        body = _literal(target.scope_check(or_(guard, expression)))
        pairs.append((name, _add_check(target, name, body)))
    return pairs


def compile_node_constraints(schema: GraphSchema, target: _Target) -> list:
    return [pair for nt in schema.node_types
            for pair in _type_constraints(target, "type", nt.name, nt.properties)]


def compile_edge_constraints(schema: GraphSchema, target: _Target) -> list:
    return [pair for kind, properties in _properties_per_kind(schema.edge_types).items()
            for pair in _type_constraints(target, "kind", kind, properties)]
