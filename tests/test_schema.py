"""
Test suite for hopai.schema.

The definition/representation half runs with no database -- schema
declaration never connects, the same guarantee build_query gives -- and
every test in it uses a Graph bound to a DSN nothing listens on, so a
regression that sneaks a connection in fails loudly here.

The enforcement half runs against real PostgreSQL, because a constraint
the server does not enforce is not a constraint: every "rejected" claim
is proven by attempting the write.

The class-notation fixtures live at module level on purpose: this file
uses `from __future__ import annotations`, which turns annotations into
strings, and typing.get_type_hints() resolves them against the module
globals -- classes defined inside a test function would not resolve.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import pydantic
import pytest
from sqlalchemy import text

from hopai import ConstraintViolation, EdgeType, Graph, GraphSchema, NodeType, Property
from hopai.constraints import JSON_TYPES

OFFLINE_DSN = "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline"


@pytest.fixture()
def offgraph():
    """A fresh offline Graph per test -- unlike conftest's session-scoped
    offline_graph, this one may have a schema defined on it without
    leaking into the next test."""
    return Graph(OFFLINE_DSN)


# -- class-notation fixtures (module level, see module docstring) ------

@dataclass
class Person:
    email: str                  # no default -> required
    nickname: str = ""          # defaulted -> not required
    age: Optional[int] = None   # Optional -> not required, permits null


@dataclass
class Company:
    name: str


@dataclass
class WorksAt:
    source: Person
    target: Company
    since: int


@dataclass
class Meeting:               # node class that would read as an edge
    source: str
    target: str


@dataclass
class BadEndpoints:          # edge class whose endpoints are not classes
    source: str
    target: Company


@dataclass
class Event:
    created_at: datetime     # unmapped annotation


@dataclass
class Flag:
    value: Union[int, str]   # a union that is not Optional


@dataclass
class Coupon:
    code: str | None = None  # PEP 604 spelling of Optional


@dataclass
class Basket:
    tags: list[str]                          # parametrized generic
    extra: Optional[dict[str, int]] = None   # parametrized generic inside Optional


@dataclass
class WideFlag:
    value: Union[int, str, None]   # None present, but not Optional-shaped


PydanticPerson = pydantic.create_model(
    "Person", email=(str, ...), nickname=(str, ""), age=(Optional[int], None))
PydanticCompany = pydantic.create_model("Company", name=(str, ...))
PydanticWorksAt = pydantic.create_model(
    "WorksAt", source=(PydanticPerson, ...), target=(PydanticCompany, ...), since=(int, ...))


def primitive_schema(graph):
    """The Person/Company/WorksAt shape spelled in the primitive
    notation -- the equality baseline the class notations must match."""
    return graph.define_schema(
        nodes=[NodeType("person", properties=[Property("email", "string", required=True),
                                              Property("nickname", "string"),
                                              Property("age", ("number", "null"))]),
               NodeType("company", properties=[Property("name", "string", required=True)])],
        edges=[EdgeType("works_at", source="person", target="company",
                        properties=[Property("since", "number", required=True)])],
    )


# ---------------------------------------------------------------------
# Definition: the primitive notation and the canonical model
# ---------------------------------------------------------------------

class TestSchemaDefinition:
    def test_primitives_round_trip_through_schema(self, offgraph):
        """Every type, property, required flag and endpoint must come
        back from .schema exactly as declared -- lossy normalization
        would make the schema lie about itself."""
        declared = primitive_schema(offgraph)
        assert offgraph.schema is declared
        person, company = declared.node_types
        assert person.name == "person" and company.name == "company"
        assert person.properties == (Property("email", "string", required=True),
                                     Property("nickname", "string"),
                                     Property("age", ("number", "null")))
        (works_at,) = declared.edge_types
        assert (works_at.kind, works_at.source, works_at.target) == (
            "works_at", "person", "company")

    def test_json_type_normalizes_to_a_sorted_set(self):
        """Property equality across input forms depends on order and
        duplicates being normalized away -- without it, the same shape
        written twice compares unequal and AC-4 breaks."""
        assert Property("x", ["number", "string", "number"]) == Property("x", ("string", "number"))
        assert Property("x", ("string", "number")).json_type == ("number", "string")

    def test_bad_json_type_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="must be one of") as exc:
            Property("x", "text")
        for json_type in JSON_TYPES:
            assert json_type in str(exc.value)

    def test_duplicate_node_type_rejected(self):
        with pytest.raises(ValueError, match="duplicate node type"):
            GraphSchema(node_types=[NodeType("person"), NodeType("person")])

    def test_edge_identity_is_the_whole_triple(self):
        """The same kind may connect several endpoint pairs -- collapsing
        identity to the kind alone would refuse a legitimate schema --
        but an exact duplicate triple is a declaration error."""
        GraphSchema(
            node_types=[NodeType("person"), NodeType("company")],
            edge_types=[EdgeType("likes", source="person", target="person"),
                        EdgeType("likes", source="person", target="company")],
        )
        with pytest.raises(ValueError, match=r"\(kind, source, target\)"):
            GraphSchema(
                node_types=[NodeType("person")],
                edge_types=[EdgeType("likes", source="person", target="person"),
                            EdgeType("likes", source="person", target="person")],
            )

    def test_edge_referencing_unknown_node_type_lists_the_defined_ones(self):
        with pytest.raises(ValueError, match="not a defined node type") as exc:
            GraphSchema(node_types=[NodeType("person"), NodeType("company")],
                        edge_types=[EdgeType("works_at", source="person", target="companny")])
        assert "'company'" in str(exc.value) and "'person'" in str(exc.value)

    def test_edge_property_named_source_or_target_refused(self):
        """'source'/'target' name the endpoints; letting them double as
        properties would make schema_json ambiguous about which is
        which."""
        with pytest.raises(TypeError, match="cannot be edge properties"):
            EdgeType("works_at", source="person", target="company",
                     properties=[Property("source", "string")])

    def test_duplicate_property_rejected(self):
        with pytest.raises(ValueError, match="duplicate property"):
            NodeType("person", properties=[Property("email", "string"),
                                           Property("email", "number")])

    def test_non_property_entries_are_named(self):
        with pytest.raises(TypeError, match="each property must be a Property"):
            NodeType("person", properties=["email"])
        with pytest.raises(TypeError, match="list of Property"):
            NodeType("person", properties="email")   # a str IS iterable -- of characters
        with pytest.raises(TypeError, match="NodeType|dataclass"):
            Graph(OFFLINE_DSN).define_schema(nodes=[42])

    def test_redefinition_replaces(self, offgraph):
        primitive_schema(offgraph)
        offgraph.define_schema(nodes=[NodeType("robot")])
        assert [nt.name for nt in offgraph.schema.node_types] == ["robot"]

    def test_schema_kwarg_adopts_a_finished_graph_schema(self, offgraph):
        """define_schema(schema=...) is the adoption step of the
        infer -> review -> define -> enforce loop; it must register the
        object as-is, refuse to mix with nodes=/edges=, and name what it
        takes when handed something else."""
        built = GraphSchema(node_types=[NodeType("person")])
        assert offgraph.define_schema(schema=built) is built
        assert offgraph.schema is built
        with pytest.raises(ValueError) as exc:
            offgraph.define_schema(nodes=[NodeType("robot")], schema=built)
        assert str(exc.value).startswith("pass either schema= or nodes=/edges=")
        with pytest.raises(TypeError, match="GraphSchema") as exc:
            offgraph.define_schema(schema={"nodes": []})
        assert "got dict" in str(exc.value)   # the refusal names what it got


# ---------------------------------------------------------------------
# Definition: the class notation
# ---------------------------------------------------------------------

class TestClassNotation:
    def test_dataclass_fields_map_required_optional_nullable(self, offgraph):
        """The three field shapes -- defaultless, defaulted, Optional --
        must land as required / not-required / not-required-plus-null.
        Getting required-ness wrong turns into wrong CHECK constraints
        the moment the schema is enforced."""
        offgraph.define_schema(nodes=[Person, Company], edges=[WorksAt])
        person = offgraph.schema.node_types[0]
        by_name = {p.name: p for p in person.properties}
        assert by_name["email"] == Property("email", "string", required=True)
        assert by_name["nickname"] == Property("nickname", "string", required=False)
        assert by_name["age"] == Property("age", ("number", "null"), required=False)
        (works_at,) = offgraph.schema.edge_types
        assert works_at.properties == (Property("since", "number", required=True),)

    def test_camel_case_becomes_snake_case(self, offgraph):
        offgraph.define_schema(nodes=[Person, Company], edges=[WorksAt])
        assert offgraph.schema.edge_types[0].kind == "works_at"

    def test_the_three_notations_cannot_drift(self, offgraph):
        """Dataclasses, pydantic models and primitives declaring the same
        shape must normalize to EQUAL schemas -- otherwise which notation
        a team picked would silently change what gets enforced."""
        baseline = primitive_schema(Graph(OFFLINE_DSN))
        from_dataclasses = Graph(OFFLINE_DSN).define_schema(
            nodes=[Person, Company], edges=[WorksAt])
        from_pydantic = offgraph.define_schema(
            nodes=[PydanticPerson, PydanticCompany], edges=[PydanticWorksAt])
        assert from_dataclasses == baseline
        assert from_pydantic == baseline

    def test_node_class_with_endpoints_rejected(self, offgraph):
        with pytest.raises(TypeError, match="edge class"):
            offgraph.define_schema(nodes=[Meeting])

    def test_edge_class_without_endpoints_rejected(self, offgraph):
        with pytest.raises(TypeError, match=r"missing \['source', 'target'\]"):
            offgraph.define_schema(nodes=[Person], edges=[Person])

    def test_edge_endpoints_must_be_annotated_with_classes(self, offgraph):
        with pytest.raises(TypeError, match="annotated with a node class"):
            offgraph.define_schema(nodes=[Company], edges=[BadEndpoints])

    def test_unmapped_annotation_names_field_and_rewrite(self, offgraph):
        """A datetime silently stored as string-or-whatever is the
        approximation this library refuses; the error must hand the
        caller the explicit-Property rewrite."""
        with pytest.raises(TypeError) as exc:
            offgraph.define_schema(nodes=[Event])
        assert "created_at" in str(exc.value)
        assert "Property('created_at'" in str(exc.value)
        # ...and the CLASS: two classes can share a field name, and an
        # error that only says the field leaves the caller grepping
        assert "Event" in str(exc.value)

    def test_non_optional_union_refused(self, offgraph):
        with pytest.raises(TypeError, match="union") as exc:
            offgraph.define_schema(nodes=[Flag])
        # the members must be named readably -- "int, str", not reprs or
        # placeholders -- or the suggested type-set rewrite is a puzzle
        assert "int, str" in str(exc.value)

    def test_pep604_optional_matches_typing_optional(self, offgraph):
        """`str | None` and Optional[str] are the same annotation spelled
        two ways; get_origin() returns a DIFFERENT sentinel for each, so
        without the types.UnionType arm the modern spelling would be
        refused as an unmapped annotation."""
        offgraph.define_schema(nodes=[Coupon])
        (coupon,) = offgraph.schema.node_types
        assert coupon.properties == (Property("code", ("string", "null"), required=False),)

    def test_parametrized_generics_map_by_origin(self, offgraph):
        """list[str] and dict[str, int] are not the classes list and dict
        -- they map via get_origin(), bare and inside Optional alike.
        Without the origin arm both would be refused as unmapped."""
        offgraph.define_schema(nodes=[Basket])
        (basket,) = offgraph.schema.node_types
        assert basket.properties == (Property("tags", "array", required=True),
                                     Property("extra", ("object", "null"), required=False))

    def test_union_with_null_but_several_types_is_still_refused(self, offgraph):
        """Union[int, str, None] is not Optional-shaped -- it has no
        single JSON type to attach 'null' to. Accepting it would mean
        guessing; the refusal must hold even though None is present."""
        with pytest.raises(TypeError, match="union"):
            offgraph.define_schema(nodes=[WideFlag])

    def test_class_as_property_bag_in_the_primitive_notation(self, offgraph):
        """NodeType(name, properties=SomeClass) must produce the same
        properties as passing the class bare -- the two spellings of
        'this class is my property schema' may not disagree."""
        bare = Graph(OFFLINE_DSN).define_schema(nodes=[Person])
        bagged = offgraph.define_schema(nodes=[NodeType("person", properties=Person)])
        assert bagged == bare


# ---------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------

class TestSchemaRepresentations:
    def test_schema_json_is_dumps_clean_and_json_schema_shaped(self, offgraph):
        """This dict is what gets pasted into a system prompt; a value
        json.dumps refuses, or invented vocabulary, breaks the one
        consumer the representation exists for."""
        primitive_schema(offgraph)
        spec = offgraph.schema_json
        json.dumps(spec)
        person = spec["nodes"]["person"]
        assert person["type"] == "object"
        assert person["properties"]["email"] == {"type": "string"}
        assert person["properties"]["age"] == {"type": ["null", "number"]}
        assert person["required"] == ["email"]
        (works_at,) = spec["edges"]
        assert works_at["kind"] == "works_at"
        assert works_at["source"] == "person" and works_at["target"] == "company"
        assert works_at["properties"]["required"] == ["since"]

    def test_networkx_is_multi_and_preserves_parallel_edge_types(self, offgraph):
        """Two kinds between the same pair, and one kind with two
        endpoint pairs, are distinct edge types; a plain DiGraph would
        collapse them to one edge each and the meta-graph would lie."""
        import networkx as nx
        offgraph.define_schema(
            nodes=[NodeType("person"), NodeType("company")],
            edges=[EdgeType("knows", source="person", target="person"),
                   EdgeType("likes", source="person", target="person"),
                   EdgeType("likes", source="person", target="company")],
        )
        meta = offgraph.schema_networkx
        assert isinstance(meta, nx.MultiDiGraph)
        assert meta.number_of_nodes() == 2
        assert meta.number_of_edges() == 3
        assert set(meta.get_edge_data("person", "person")) == {"knows", "likes"}

    def test_networkx_missing_names_the_extra(self, offgraph, monkeypatch):
        primitive_schema(offgraph)
        monkeypatch.setitem(sys.modules, "networkx", None)
        with pytest.raises(ImportError, match=r"pip install hopai\[networkx\]"):
            _ = offgraph.schema_networkx

    def test_pydantic_models_actually_validate(self, offgraph):
        """The models must carry the schema's rules, not just its field
        names -- a generated model that accepts anything would be a
        representation in name only."""
        primitive_schema(offgraph)
        models = offgraph.schema_pydantic
        person = models["person"](email="a@x.com")
        assert person.age is None
        assert models["person"](email="a@x.com", age=None).age is None
        # ("number", "null") must admit the number too, not only the null
        assert models["person"](email="a@x.com", age=31).age == 31
        with pytest.raises(pydantic.ValidationError):
            models["person"]()                    # missing required
        with pytest.raises(pydantic.ValidationError):
            models["person"](email=42)            # wrong type
        assert models["works_at"](since=3).since == 3

    def test_pydantic_missing_names_the_extra(self, offgraph, monkeypatch):
        primitive_schema(offgraph)
        monkeypatch.setitem(sys.modules, "pydantic", None)
        with pytest.raises(ImportError, match=r"pip install hopai\[pydantic\]"):
            _ = offgraph.schema_pydantic

    def test_pydantic_model_names_are_camel_case(self, offgraph):
        """The generated class is what shows up in ValidationError text
        and reprs; 'works_at' must come back as WorksAt, not worksat or
        the raw kind."""
        primitive_schema(offgraph)
        models = offgraph.schema_pydantic
        assert models["person"].__name__ == "Person"
        assert models["works_at"].__name__ == "WorksAt"

    def test_pydantic_type_sets_and_null_only_properties(self, offgraph):
        """A type-set property must validate as the UNION of its types
        and a "null"-only property as exactly None -- collapsing either
        to a single arbitrary member would accept what the schema
        refuses, or refuse what it accepts."""
        offgraph.define_schema(nodes=[NodeType("reading", properties=[
            Property("val", ("number", "string"), required=True),
            Property("gone", "null"),
        ])])
        model = offgraph.schema_pydantic["reading"]
        assert model(val=3).val == 3
        assert model(val="high").val == "high"
        with pytest.raises(pydantic.ValidationError):
            model(val=[1])
        assert model(val=1, gone=None).gone is None
        with pytest.raises(pydantic.ValidationError):
            model(val=1, gone="not-null")

    def test_everything_before_define_schema_names_the_fix(self, offgraph):
        """None for .schema is the existence check; every derived
        representation must instead say HOW to get one -- a bare
        AttributeError or None propagating into json.dumps helps
        nobody."""
        assert offgraph.schema is None
        for name, accessor in (("schema_json", lambda: offgraph.schema_json),
                               ("schema_networkx", lambda: offgraph.schema_networkx),
                               ("schema_pydantic", lambda: offgraph.schema_pydantic),
                               ("schema_ddl", offgraph.schema_ddl),
                               ("enforce_schema", offgraph.enforce_schema)):
            with pytest.raises(ValueError, match=r"define_schema") as exc:
                accessor()
            # the message must LEAD with which accessor needed the
            # schema -- "something failed somewhere" is not an error
            # that names the fix
            assert str(exc.value).startswith(name)

    def test_defining_and_reading_needs_no_database(self, offgraph):
        """offgraph's DSN has nothing listening: if any part of
        definition or representation opened a connection, this test
        would error out -- the same guarantee build_query gives."""
        offgraph.define_schema(nodes=[Person, Company], edges=[WorksAt])
        assert offgraph.schema is not None
        assert offgraph.schema_json
        assert offgraph.schema_networkx is not None
        assert offgraph.schema_pydantic
        assert offgraph.schema_ddl()

    def test_in_graph_handle_starts_without_a_schema(self, offgraph):
        """A different graph is allowed a different shape, so a schema
        must never travel implicitly to another graph's handle."""
        primitive_schema(offgraph)
        assert offgraph.in_graph("other").schema is None

    def test_schema_ddl_shows_the_checks_without_running_them(self, offgraph):
        """Node AND edge constraints must both be in the preview -- a
        schema_ddl() that silently forgot the edges table would review
        clean and then enforce something else."""
        primitive_schema(offgraph)
        ddl = offgraph.schema_ddl()
        assert all(statement.startswith("ALTER TABLE") for statement in ddl)
        assert any("IS DISTINCT FROM 'person'" in statement for statement in ddl)
        assert any("?& ARRAY['email']" in statement for statement in ddl)
        assert any("IN ('null', 'number')" in statement for statement in ddl)
        # the pg dialect's subscript rendering: _literal() compiling with
        # the DEFAULT dialect instead (mutant x__literal__mutmut_4) emits
        # `properties -> 'age'` here and nothing else in the DDL differs
        assert any("jsonb_typeof(properties['age'])" in statement for statement in ddl)
        assert any('ALTER TABLE "edges"' in statement
                   and "IS DISTINCT FROM 'works_at'" in statement for statement in ddl)

    def test_conflicting_property_schemas_for_one_kind_refused(self, offgraph):
        """Enforcement and models key on the kind alone (endpoints are
        unenforceable in a CHECK), so two declarations of one kind that
        disagree must refuse rather than enforce a merge neither
        declared."""
        offgraph.define_schema(
            nodes=[NodeType("person"), NodeType("company")],
            edges=[EdgeType("likes", source="person", target="person",
                            properties=[Property("weight", "number")]),
                   EdgeType("likes", source="person", target="company")],
        )
        with pytest.raises(ValueError, match="different property schemas"):
            _ = offgraph.schema_pydantic
        with pytest.raises(ValueError, match="different property schemas"):
            offgraph.schema_ddl()

    def test_node_name_colliding_with_edge_kind_refused_in_pydantic(self, offgraph):
        offgraph.define_schema(
            nodes=[NodeType("likes"), NodeType("person")],
            edges=[EdgeType("likes", source="person", target="person")],
        )
        with pytest.raises(ValueError, match="both a node type and an edge kind"):
            _ = offgraph.schema_pydantic


# ---------------------------------------------------------------------
# Enforcement -- against real PostgreSQL
# ---------------------------------------------------------------------

def schema_checks(graph) -> set:
    """The ck_schema_* constraints currently on the write schema."""
    with graph.engine.connect() as conn:
        return {r[0] for r in conn.execute(text(
            "SELECT conname FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace "
            "WHERE c.contype = 'c' AND n.nspname = 'hopai_write' "
            "AND c.conname LIKE 'ck\\_schema\\_%'"
        )).all()}


class TestSchemaEnforcement:
    def test_missing_required_rejected_on_every_write_path(self, fresh_graph):
        """The point of compiling to CHECKs is that the SERVER rejects
        the row, whichever door it came through -- the Python API and
        the Cypher front end must both surface ConstraintViolation."""
        primitive_schema(fresh_graph)
        fresh_graph.enforce_schema()
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"type": "person", "nickname": "no-email"}])
        assert exc.value.constraint.startswith("ck_schema_req_")
        with pytest.raises(ConstraintViolation):
            fresh_graph.cypher("CREATE (a:person {nickname: 'still-no-email'})")

    def test_wrong_json_type_rejected(self, fresh_graph):
        """PropertyType's reason to exist, per node type: an LLM that
        writes "42" where the schema says number must be stopped at the
        table, not discovered downstream."""
        primitive_schema(fresh_graph)
        fresh_graph.enforce_schema()
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"type": "person", "email": "a@x.com", "age": "42"}])
        assert exc.value.constraint.startswith("ck_schema_typ_")
        # the declared type set is ("number", "null"): both must pass
        fresh_graph.add_nodes([{"type": "person", "email": "b@x.com", "age": 42},
                               {"type": "person", "email": "c@x.com", "age": None}])

    def test_untyped_rows_pass_every_per_type_check(self, fresh_graph):
        """The guard is IS DISTINCT FROM: a row with no discriminator
        belongs to no declared type and no per-type rule may bind it.
        Naive <> would evaluate to NULL and also pass -- but by
        accident; this pins the semantics on purpose."""
        primitive_schema(fresh_graph)
        fresh_graph.enforce_schema()
        assert fresh_graph.add_nodes([{"name": "untyped, and welcome"}]) == 1

    def test_edge_kind_properties_enforced(self, fresh_graph):
        primitive_schema(fresh_graph)
        fresh_graph.enforce_schema()
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "works_at", "since": 5}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "works_at"}])
        # a kind the schema does not declare is not bound by it
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "undeclared"}])

    def test_idempotent_and_reconciles_on_schema_change(self, fresh_graph):
        """Re-running must converge, not accrete: same schema -> same
        constraints and no error; changed schema -> the constraint the
        schema no longer produces is DROPPED, proven by the previously
        rejected write now landing. Constraints from
        define_constraints() are not enforcement's to drop."""
        from hopai import Required
        fresh_graph.define_constraints(nodes=[Required("kept")])
        primitive_schema(fresh_graph)
        first = fresh_graph.enforce_schema()
        assert fresh_graph.enforce_schema() == first
        assert "ck_schema_typ_default_person_age" in schema_checks(fresh_graph)

        fresh_graph.define_schema(
            nodes=[NodeType("person", properties=[Property("email", "string", required=True)]),
                   NodeType("company", properties=[Property("name", "string", required=True)])],
            edges=[EdgeType("works_at", source="person", target="company")],
        )
        fresh_graph.enforce_schema()
        assert "ck_schema_typ_default_person_age" not in schema_checks(fresh_graph)
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com", "age": "free-form again",
                                "kept": 1}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"type": "robot"}])
        assert exc.value.constraint == "ck_required_nodes_kept"

    def test_enforcement_binds_only_its_own_graph(self, fresh_graph):
        """The scope guard in action: graph 'default' declaring a schema
        must not make its rules law for other graphs sharing the
        tables -- the multi-graph invariant, applied to enforcement."""
        primitive_schema(fresh_graph)
        fresh_graph.enforce_schema()
        other = fresh_graph.in_graph("elsewhere")
        assert other.add_nodes([{"type": "person", "nickname": "lawless"}]) == 1


# ---------------------------------------------------------------------
# Inference -- against real PostgreSQL
# ---------------------------------------------------------------------

def as_shape(schema) -> tuple:
    """A schema as order-insensitive dicts. Declaration preserves the
    author's ordering while inference sorts, so round-trip equality is
    about CONTENT: types, properties, required flags, endpoint triples."""
    return (
        {nt.name: {p.name: (p.json_type, p.required) for p in nt.properties}
         for nt in schema.node_types},
        {(et.kind, et.source, et.target): {p.name: (p.json_type, p.required)
                                           for p in et.properties}
         for et in schema.edge_types},
    )


def seed_chaotic(graph) -> None:
    """A graph grown with no declared schema: consistent people, a
    mixed-type key, untyped rows, one kind spanning two endpoint pairs,
    and an edge into an untyped node."""
    graph.add_nodes([
        {"id": 1, "type": "person", "email": "a@x.com", "age": 42},
        {"id": 2, "type": "person", "email": "b@x.com", "age": None, "nickname": "b"},
        {"id": 3, "type": "person", "email": "c@x.com"},
        {"id": 4, "type": "company", "name": "acme", "mixed": 1},
        {"id": 5, "type": "company", "name": "globex", "mixed": "one"},
        {"id": 6, "name": "untyped"},
    ])
    graph.add_edges([
        {"start_id": 1, "end_id": 4, "kind": "works_at", "since": 2019},
        {"start_id": 2, "end_id": 5, "kind": "works_at", "since": 2020},
        {"start_id": 1, "end_id": 2, "kind": "likes"},
        {"start_id": 1, "end_id": 4, "kind": "likes"},
        {"start_id": 1, "end_id": 6, "kind": "dangles"},
        {"start_id": 6, "end_id": 1},
    ])


class TestSchemaInference:
    def test_required_optional_nullable_inferred_from_presence(self, fresh_graph):
        """The declaration-side semantics, read backwards: on every row
        -> required; missing on some -> optional; an observed JSON null
        -> nullable. Getting any of these wrong freezes the wrong CHECK
        the moment the inferred schema is adopted and enforced."""
        seed_chaotic(fresh_graph)
        inferred, _ = fresh_graph.infer_schema()
        person = {p.name: p for nt in inferred.node_types if nt.name == "person"
                  for p in nt.properties}
        assert person["email"] == Property("email", "string", required=True)
        assert person["age"] == Property("age", ("number", "null"), required=False)
        assert person["nickname"] == Property("nickname", "string", required=False)

    def test_mixed_types_infer_a_set_and_are_reported(self, fresh_graph):
        """42 next to "one" must surface BOTH as the honest type set and
        as a report entry -- a silent winner is the approximation this
        library refuses, and an unreported set hides the data worth
        cleaning."""
        seed_chaotic(fresh_graph)
        inferred, report = fresh_graph.infer_schema()
        company = {p.name: p for nt in inferred.node_types if nt.name == "company"
                   for p in nt.properties}
        assert company["mixed"].json_type == ("number", "string")
        (conflict,) = report.conflicts
        assert (conflict.table, conflict.type_name, conflict.key) == ("nodes", "company", "mixed")
        assert conflict.json_types == ("number", "string")

    def test_one_kind_several_endpoint_pairs_becomes_several_edge_types(self, fresh_graph):
        """likes person->person and likes person->company are two
        observations; collapsing them to one triple would invent an
        endpoint pair nobody wrote."""
        seed_chaotic(fresh_graph)
        inferred, _ = fresh_graph.infer_schema()
        likes = {(et.source, et.target) for et in inferred.edge_types if et.kind == "likes"}
        assert likes == {("person", "person"), ("person", "company")}

    def test_untyped_rows_are_counted_not_invented(self, fresh_graph):
        """A row with no discriminator belongs to no type: it must stay
        OUT of the schema (there is nothing true to say about it) and IN
        the report (silence would read as 'covered everything')."""
        seed_chaotic(fresh_graph)
        inferred, report = fresh_graph.infer_schema()
        assert {nt.name for nt in inferred.node_types} == {"company", "person"}
        assert report.untyped_nodes == 1
        assert report.untyped_edges == 1        # the kindless edge
        assert report.skipped_endpoint_edges == 1   # 'dangles' into the untyped node
        assert "dangles" not in {et.kind for et in inferred.edge_types}
        assert report.edge_counts["dangles"] == 1   # ...but the report still shows it

    def test_report_carries_per_type_row_counts(self, fresh_graph):
        """required=True on a 3-row type means almost nothing; the counts
        are what let a human judge that before adopting."""
        seed_chaotic(fresh_graph)
        _, report = fresh_graph.infer_schema()
        assert report.node_counts == {"person": 3, "company": 2}
        assert report.edge_counts == {"works_at": 2, "likes": 2, "dangles": 1}

    def test_infer_define_enforce_locks_the_graph_down(self, fresh_graph):
        """The loop the feature exists for: a chaotically-grown graph
        becomes a server-validated one in three calls, and the next
        violating write is refused by Postgres itself."""
        seed_chaotic(fresh_graph)
        inferred, _ = fresh_graph.infer_schema()
        fresh_graph.define_schema(schema=inferred)
        fresh_graph.enforce_schema()
        from hopai import ConstraintViolation
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"type": "person", "nickname": "no-email"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_edges([{"start_id": 1, "end_id": 4, "kind": "works_at"}])

    def test_round_trip_declared_written_inferred(self, fresh_graph):
        """Rows written in conformance with a declared schema must infer
        back the same shape -- if declaration and inference disagree
        about the same data, one of them is lying. Every declared
        property is exercised at least once, nulls included, since
        inference can only see what the data shows."""
        declared = primitive_schema(fresh_graph)
        fresh_graph.add_nodes([
            {"id": 1, "type": "person", "email": "a@x.com", "age": 42, "nickname": "a"},
            {"id": 2, "type": "person", "email": "b@x.com", "age": None},
            # no age at all -- present-on-every-row would otherwise
            # (correctly!) infer required=True where optional was declared
            {"id": 3, "type": "person", "email": "c@x.com"},
            {"id": 4, "type": "company", "name": "acme"},
        ])
        fresh_graph.add_edges([
            {"start_id": 1, "end_id": 4, "kind": "works_at", "since": 2019},
        ])
        inferred, report = fresh_graph.infer_schema()
        assert as_shape(inferred) == as_shape(declared)
        assert not report.conflicts
        json.dumps(inferred.to_json())

    def test_two_graphs_infer_independently(self, fresh_graph):
        """The _scoped() invariant applied to inference: graph A's scan
        must never see graph B's rows, or one tenant's chaos becomes
        another tenant's schema."""
        seed_chaotic(fresh_graph)
        other = fresh_graph.in_graph("elsewhere")
        other.add_nodes([{"id": 100, "type": "robot", "serial": "r2"}])
        inferred_other, report_other = other.infer_schema()
        assert {nt.name for nt in inferred_other.node_types} == {"robot"}
        assert report_other.node_counts == {"robot": 1}
        inferred_default, _ = fresh_graph.infer_schema()
        assert "robot" not in {nt.name for nt in inferred_default.node_types}

    def test_inference_registers_nothing(self, fresh_graph):
        """Observation is not contract: after infer_schema() the handle
        must still have NO schema -- adoption is define_schema(schema=),
        deliberately a second call."""
        seed_chaotic(fresh_graph)
        fresh_graph.infer_schema()
        assert fresh_graph.schema is None

    def test_unreachable_database_fails_loudly(self, offgraph):
        """Unlike definition, inference NEEDS the database: on a dead
        DSN it must raise the driver's connection error -- not hang, and
        never return a silent empty schema that reads as 'no types'."""
        from sqlalchemy.exc import OperationalError
        with pytest.raises(OperationalError):
            offgraph.infer_schema()
