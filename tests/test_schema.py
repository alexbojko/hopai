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

import enum
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
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
    created_at: complex      # genuinely unmapped annotation


class Status(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class Mixed(enum.Enum):
    A = "a"
    B = 2


@dataclass
class Coordinates:
    lat: float
    lon: float


@dataclass
class Ticket:
    status: Status                    # Enum -> string + allowed values
    opened: datetime                  # -> string + date-time format
    location: Coordinates             # nested dataclass -> object schema
    day: date = date(2026, 1, 1)      # -> string + date format


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


# ---------------------------------------------------------------------
# A property named the same as a real column
# ---------------------------------------------------------------------

class TestColumnCollision:
    """Property('id', ...) would compile a CHECK on properties->>'id',
    a JSONB key that can never hold what the real `id` column already
    does -- add_nodes()/merge_nodes() route a flat row's 'id' there
    directly. define_schema() refuses rather than declaring a rule
    enforce_schema() could never let a correct row satisfy."""

    def test_node_property_named_id_is_refused(self, offgraph):
        with pytest.raises(ValueError, match=r"NodeType\('person'\).*\['id'\]"):
            offgraph.define_schema(
                nodes=[NodeType("person", properties=[Property("id", "number")])])
        assert offgraph.schema is None   # the refusal must not half-adopt it

    def test_edge_property_named_start_id_is_refused(self, offgraph):
        with pytest.raises(ValueError, match=r"EdgeType\('knows'\).*\['start_id'\]"):
            offgraph.define_schema(
                nodes=[NodeType("person")],
                edges=[EdgeType("knows", source="person", target="person",
                                properties=[Property("start_id", "number")])])

    def test_dataclass_field_colliding_with_an_extra_column_is_refused(self):
        """The scenario this exists for: a project's OWN dataclass
        happens to name a field the same as a real column its custom
        table carries. Unlike 'id', 'user_id' is not a SQL convention
        anyone would recognize on sight -- an honest accident, which is
        exactly why it needs a guard rather than relying on the author
        noticing."""
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        nodes = Table("nodes", md, Column("id", BigInteger, primary_key=True),
                      Column("user_id", BigInteger), Column("properties", JSONB))
        g = Graph(OFFLINE_DSN, node_table=nodes, graph_col=None)

        @dataclass
        class Person:
            email: str
            user_id: int

        with pytest.raises(ValueError, match=r"\['user_id'\]"):
            g.define_schema(nodes=[Person])

    def test_a_non_colliding_property_is_unaffected(self, offgraph):
        schema = offgraph.define_schema(
            nodes=[NodeType("person", properties=[Property("email", "string")])])
        assert schema.node_types[0].properties[0].name == "email"

    def test_nested_properties_never_collide(self, offgraph):
        """A nested key compiles to properties->'address'->>'id', never
        properties->>'id' -- no ambiguity with the real column no
        matter what the nested key is named."""
        schema = offgraph.define_schema(nodes=[NodeType("person", properties=[
            Property("address", "object", properties=[Property("id", "string")])])])
        assert schema.node_types[0].properties[0].properties[0].name == "id"

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
                               ("schema_violations", offgraph.schema_violations),
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
        # the return value NAMES the constraints in force, same contract
        # as define_constraints -- a list of Nones satisfies any test
        # that only compares two runs of it (enforce_schema__mutmut_36)
        assert "ck_schema_req_default_person" in first
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


class TestSchemaViolations:
    def seed_dirty(self, graph) -> None:
        graph.add_nodes([
            {"id": 1, "type": "person", "email": "a@x.com", "age": 42},   # conforming
            {"id": 2, "type": "person", "nickname": "no-email"},          # missing required
            {"id": 3, "type": "person", "email": "c@x.com", "age": "42"}, # wrong JSON type
            {"id": 4, "name": "untyped"},                                 # outside the schema
        ])

    def test_reports_each_violated_rule_under_its_enforcement_name(self, fresh_graph):
        """The report is the work list for the enforcement that would
        fail, so each entry must carry the SAME ck_schema_* name the
        CHECK would get, the right count, and the offending ids --
        while conforming and untyped rows stay out of it."""
        self.seed_dirty(fresh_graph)
        primitive_schema(fresh_graph)
        report = fresh_graph.schema_violations()
        assert bool(report)
        by_name = {r.constraint: r for r in report.rules}
        assert by_name["ck_schema_req_default_person"].sample_ids == (2,)
        assert by_name["ck_schema_typ_default_person_age"].sample_ids == (3,)
        assert all(r.rows == 1 and r.table == "nodes" for r in report.rules)
        assert len(report.rules) == 2   # ids 1 and 4 appear nowhere

    def test_edge_rules_are_reported_against_the_edges_table(self, fresh_graph):
        """Every assertion above is about nodes, so the whole edge half
        of the report went unread: `_schema_targets()` could label the
        edge target anything at all and the suite stayed green. The
        label is not decoration -- `rule.table` is how a caller knows
        which table holds the rows to go fix, and the two halves of the
        report are otherwise indistinguishable."""
        fresh_graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"},
                               {"id": 2, "type": "company", "name": "Acme"}])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "works_at", "since": 2019},
                               {"start_id": 1, "end_id": 2, "kind": "works_at"}])  # no `since`
        primitive_schema(fresh_graph)

        (rule,) = fresh_graph.schema_violations().rules
        assert rule.constraint == "ck_schema_req_default_works_at"
        assert rule.table == "edges"
        assert rule.rows == 1

    def test_a_clean_graph_reports_falsy_and_enforcement_succeeds(self, fresh_graph):
        """Falsy-when-clean is the contract that makes
        `if graph.schema_violations():` read correctly -- and clean must
        mean enforce_schema() actually goes through."""
        fresh_graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"}])
        primitive_schema(fresh_graph)
        report = fresh_graph.schema_violations()
        assert not report
        assert "enforce_schema() would succeed" in str(report)
        assert fresh_graph.enforce_schema()

    def test_dry_run_and_enforcement_cannot_disagree(self, fresh_graph):
        """The point of sharing _type_rules: after cleaning exactly the
        rows the report named, enforcement must succeed -- a dry-run
        that checks different rules than the DDL enforces would be worse
        than none."""
        from sqlalchemy.exc import IntegrityError
        self.seed_dirty(fresh_graph)
        primitive_schema(fresh_graph)
        with pytest.raises(IntegrityError):
            fresh_graph.enforce_schema()   # dirty: the driver refuses ADD CONSTRAINT
        for rule in fresh_graph.schema_violations().rules:
            with fresh_graph.engine.begin() as conn:
                conn.execute(text(f"DELETE FROM edges WHERE start_id IN "
                                  f"({','.join(map(str, rule.sample_ids))}) OR end_id IN "
                                  f"({','.join(map(str, rule.sample_ids))})"))
                conn.execute(text(f"DELETE FROM nodes WHERE id IN "
                                  f"({','.join(map(str, rule.sample_ids))})"))
        assert not fresh_graph.schema_violations()
        assert fresh_graph.enforce_schema()

    def test_sampling_caps_ids_but_not_the_count(self, fresh_graph):
        fresh_graph.add_nodes([{"id": i, "type": "person", "nickname": f"n{i}"}
                               for i in range(1, 8)])
        primitive_schema(fresh_graph)
        (rule,) = fresh_graph.schema_violations(sample=3).rules
        assert rule.rows == 7
        assert rule.sample_ids == (1, 2, 3)
        assert ", ..." in str(fresh_graph.schema_violations(sample=3))

    def test_violations_are_read_only_and_per_graph(self, fresh_graph):
        """No DDL may exist after a dry-run, and another graph's dirt
        must not appear in this graph's report -- the _scoped invariant,
        applied to reading violations."""
        self.seed_dirty(fresh_graph)
        other = fresh_graph.in_graph("elsewhere")
        other.add_nodes([{"id": 100, "type": "person", "nickname": "dirty-elsewhere"}])
        primitive_schema(fresh_graph)
        report = fresh_graph.schema_violations()
        assert all(100 not in r.sample_ids for r in report.rules)
        assert schema_checks(fresh_graph) == set()   # read-only: nothing created
        primitive_schema(other)
        (rule,) = other.schema_violations().rules
        assert rule.sample_ids == (100,)


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


class TestEndpointEnforcement:
    def declare(self, graph) -> None:
        graph.add_nodes([
            {"id": 1, "type": "person", "email": "a@x.com"},
            {"id": 2, "type": "company", "name": "acme"},
            {"id": 3, "type": "robot"},
            {"id": 4, "name": "untyped"},
        ])
        graph.define_schema(
            nodes=[NodeType("person"), NodeType("company"), NodeType("robot")],
            edges=[EdgeType("works_at", source="person", target="company"),
                   EdgeType("likes", source="person", target="person"),
                   EdgeType("likes", source="person", target="company")],
        )

    @staticmethod
    def trigger_count(graph) -> int:
        with graph.engine.connect() as conn:
            return conn.execute(text(
                "SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'hopai_write' AND t.tgname LIKE 'ck\\_schema\\_end%'"
            )).scalar()

    def test_wrong_endpoint_rejected_on_every_write_path(self, fresh_graph):
        """The whole point of a trigger over a Python check: the SERVER
        rejects the edge whichever door it came through, and the error
        names the kind, the observed types, and the declared triples --
        an error that names the fix."""
        self.declare(fresh_graph)
        fresh_graph.enforce_schema(endpoints=True)
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "works_at"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_edges([{"start_id": 3, "end_id": 2, "kind": "works_at"}])
        assert exc.value.constraint == "ck_schema_end_default"
        assert "works_at connects robot -> company" in str(exc.value)
        assert "works_at: person -> company" in str(exc.value)
        with pytest.raises(ConstraintViolation):
            fresh_graph.cypher(
                "MATCH (a {type: 'robot'}), (b {name: 'acme'}) CREATE (a)-[:works_at]->(b)")

    def test_a_kind_with_several_pairs_accepts_each_and_only_each(self, fresh_graph):
        """Edge identity is the (kind, source, target) triple, and the
        trigger must honor every declared triple of a kind -- while an
        UNdeclared pair of a declared kind stays rejected."""
        self.declare(fresh_graph)
        fresh_graph.enforce_schema(endpoints=True)
        fresh_graph.add_edges([{"start_id": 1, "end_id": 1, "kind": "likes"},
                               {"start_id": 1, "end_id": 2, "kind": "likes"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_edges([{"start_id": 3, "end_id": 1, "kind": "likes"}])

    def test_undeclared_and_kindless_pass_untyped_endpoint_fails(self, fresh_graph):
        """Only declared kinds are policed -- consistent with the
        per-type CHECKs. But a DECLARED kind into an untyped node cannot
        satisfy any triple, and the error must say (untyped), not
        pretend a type."""
        self.declare(fresh_graph)
        fresh_graph.enforce_schema(endpoints=True)
        fresh_graph.add_edges([{"start_id": 3, "end_id": 1, "kind": "undeclared"}])
        fresh_graph.add_edges([{"start_id": 4, "end_id": 1}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_edges([{"start_id": 1, "end_id": 4, "kind": "works_at"}])
        assert "person -> (untyped)" in str(exc.value)

    def test_idempotent_scoped_and_reconciled_away(self, fresh_graph):
        """Re-running converges; another graph's edges are never
        validated; and enforce_schema() WITHOUT the flag removes the
        trigger and its function -- a stale trigger would keep policing
        a rule the caller just opted out of."""
        self.declare(fresh_graph)
        first = fresh_graph.enforce_schema(endpoints=True)
        assert fresh_graph.enforce_schema(endpoints=True) == first
        assert "ck_schema_end_default" in first
        assert self.trigger_count(fresh_graph) == 1

        other = fresh_graph.in_graph("elsewhere")
        other.add_nodes([{"id": 100, "type": "robot"}, {"id": 101, "type": "company"}])
        assert other.add_edges([{"start_id": 100, "end_id": 101, "kind": "works_at"}]) == 1

        fresh_graph.enforce_schema()   # endpoints=False -> converge to unpoliced
        assert self.trigger_count(fresh_graph) == 0
        with fresh_graph.engine.connect() as conn:
            functions = conn.execute(text(
                "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE n.nspname = 'hopai_write' AND p.proname LIKE 'ck\\_schema\\_endf%'"
            )).scalar()
        assert functions == 0
        assert fresh_graph.add_edges(
            [{"start_id": 3, "end_id": 2, "kind": "works_at"}]) == 1   # unpoliced again

    def test_default_stays_off_and_ddl_previews_without_executing(self, fresh_graph):
        """endpoints=True is an opt-in with a per-row write cost, so the
        default must create NO trigger -- and schema_ddl(endpoints=True)
        must show the exact SQL while executing none of it."""
        self.declare(fresh_graph)
        fresh_graph.enforce_schema()
        assert self.trigger_count(fresh_graph) == 0
        ddl = fresh_graph.schema_ddl(endpoints=True)
        assert any("CREATE CONSTRAINT TRIGGER" in statement for statement in ddl)
        assert any("CREATE OR REPLACE FUNCTION" in statement for statement in ddl)
        assert self.trigger_count(fresh_graph) == 0   # preview executed nothing


class TestRicherMappings:
    def test_class_notation_maps_enum_datetime_and_nested(self, offgraph):
        """The three former refusals, now mapped: a single-value-type
        Enum carries its allowed values, datetime/date become string
        with a JSON Schema format, and a nested dataclass becomes an
        object with a nested property schema -- each losslessly in the
        canonical form."""
        offgraph.define_schema(nodes=[Ticket])
        (ticket,) = offgraph.schema.node_types
        by_name = {p.name: p for p in ticket.properties}
        assert by_name["status"] == Property("status", "string", required=True,
                                             values=("open", "closed"))
        assert by_name["opened"] == Property("opened", "string", required=True,
                                             format="date-time")
        assert by_name["day"] == Property("day", "string", format="date")
        location = by_name["location"]
        assert location.json_type == ("object",)
        assert location.properties == (Property("lat", "number", required=True),
                                       Property("lon", "number", required=True))

    def test_primitive_spellings_equal_the_class_output(self, offgraph):
        """values=/format=/properties= spelled explicitly must normalize
        identically to the class notation -- the cannot-drift rule,
        extended to the new fields."""
        explicit = Graph(OFFLINE_DSN).define_schema(nodes=[NodeType("ticket", properties=[
            Property("status", "string", required=True, values=("open", "closed")),
            Property("opened", "string", required=True, format="date-time"),
            Property("location", "object", required=True, properties=[
                Property("lat", "number", required=True),
                Property("lon", "number", required=True)]),
            Property("day", "string", format="date"),
        ])])
        assert offgraph.define_schema(nodes=[Ticket]) == explicit

    def test_mixed_value_type_enum_still_refused(self, offgraph):
        @dataclass
        class Bad:
            state: Mixed
        Bad.__annotations__["state"] = Mixed   # resolvable without module globals
        with pytest.raises(TypeError, match="mixes value types"):
            offgraph.define_schema(nodes=[Bad])

    def test_schema_json_renders_enum_format_unique_and_nesting(self, offgraph):
        offgraph.define_schema(nodes=[NodeType("ticket", properties=[
            Property("status", "string", values=("open", "closed")),
            Property("opened", "string", format="date-time"),
            Property("code", "string", unique=True),
            Property("location", "object", properties=[Property("lat", "number",
                                                                required=True)]),
        ])])
        spec = offgraph.schema_json["nodes"]["ticket"]["properties"]
        json.dumps(spec)
        assert spec["status"] == {"type": "string", "enum": ["open", "closed"]}
        assert spec["opened"] == {"type": "string", "format": "date-time"}
        assert spec["code"] == {"type": "string", "unique": True}
        assert spec["location"]["type"] == "object"
        assert spec["location"]["properties"]["lat"] == {"type": "number"}
        assert spec["location"]["required"] == ["lat"]

    def test_schema_json_keeps_a_nested_propertys_own_type_set(self, offgraph):
        """A nullable nested object is ("null", "object"), and its JSON
        rendering must say so -- collapsing the set to the bare "object"
        _properties_json assumes would misdescribe a null as a
        violation."""
        offgraph.define_schema(nodes=[NodeType("ticket", properties=[
            Property("location", ("null", "object"), properties=[
                Property("lat", "number")])])])
        spec = offgraph.schema_json["nodes"]["ticket"]["properties"]["location"]
        assert spec["type"] == ["null", "object"]
        assert spec["properties"]["lat"] == {"type": "number"}

    def test_pydantic_models_validate_the_new_shapes(self, offgraph):
        """Literal rejects a non-member, datetime parses ISO strings,
        and the nested model validates its own fields -- real validation,
        not annotation decoration."""
        offgraph.define_schema(nodes=[Ticket])
        model = offgraph.schema_pydantic["ticket"]
        ok = model(status="open", opened="2026-08-15T09:00:00",
                   location={"lat": 1.5, "lon": 2.5}, day="2026-08-15")
        assert ok.opened.year == 2026 and ok.location.lat == 1.5
        # format "date" is its own branch, distinct from "date-time":
        # a date field parses to datetime.date, and a full timestamp is
        # NOT a date
        assert ok.day == date(2026, 8, 15)
        with pytest.raises(pydantic.ValidationError):
            model(status="open", opened="2026-08-15T09:00:00",
                  location={"lat": 1.5, "lon": 2.5}, day="2026-08-15T09:00:00")
        with pytest.raises(pydantic.ValidationError):
            model(status="reopened", opened="2026-08-15T09:00:00",
                  location={"lat": 1.5, "lon": 2.5})
        with pytest.raises(pydantic.ValidationError):
            model(status="open", opened="2026-08-15T09:00:00",
                  location={"lat": "north", "lon": 2.5})

    def test_inference_stays_at_json_type_granularity(self, fresh_graph):
        """Observation is not declaration: rows that happen to hold enum
        members or ISO strings infer plain string properties -- no
        values, no format, no uniqueness."""
        fresh_graph.add_nodes([{"id": 1, "type": "ticket", "status": "open",
                                "opened": "2026-08-15T09:00:00"}])
        inferred, _ = fresh_graph.infer_schema()
        (ticket,) = inferred.node_types
        for p in ticket.properties:
            assert p.values == () and p.format is None and p.unique is False


class TestTier2Enforcement:
    def declare(self, graph) -> None:
        graph.define_schema(nodes=[
            NodeType("person", properties=[Property("email", "string", unique=True),
                                           Property("status", "string",
                                                    values=("active", "gone"))]),
            NodeType("robot"),
        ])

    def test_unique_is_per_type(self, fresh_graph):
        """unique=True compiles to the PARTIAL index: two people cannot
        share the email, a robot may reuse it, and rows missing the
        property repeat freely (SQL NULL semantics, same as Unique)."""
        self.declare(fresh_graph)
        applied = fresh_graph.enforce_schema()
        assert "uq_schema_default_person_email" in applied
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"},
                               {"type": "robot", "email": "a@x.com"},
                               {"type": "person"}, {"type": "person"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"}])
        assert exc.value.constraint == "uq_schema_default_person_email"

    def test_values_are_enforced(self, fresh_graph):
        self.declare(fresh_graph)
        fresh_graph.enforce_schema()
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com", "status": "active"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"type": "person", "email": "b@x.com",
                                    "status": "resting"}])
        assert exc.value.constraint.startswith("ck_schema_val_")

    def test_unique_reconciles_away(self, fresh_graph):
        """Dropping the flag and re-enforcing must drop the index --
        a stale unique index would keep refusing writes the schema now
        allows."""
        self.declare(fresh_graph)
        fresh_graph.enforce_schema()
        fresh_graph.define_schema(nodes=[NodeType("person", properties=[
            Property("email", "string")]), NodeType("robot")])
        fresh_graph.enforce_schema()
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"},
                               {"type": "person", "email": "a@x.com"}])   # now legal

    def test_schema_violations_names_every_duplicate(self, fresh_graph):
        """The dry-run reports the WHOLE duplicate group under the
        index's name -- the ADD INDEX failure would name one pair."""
        fresh_graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"},
                               {"id": 2, "type": "person", "email": "a@x.com"},
                               {"id": 3, "type": "person", "email": "b@x.com"},
                               {"id": 4, "type": "robot", "email": "a@x.com"}])
        self.declare(fresh_graph)
        (rule,) = fresh_graph.schema_violations().rules
        assert rule.constraint == "uq_schema_default_person_email"
        assert rule.rows == 2 and rule.sample_ids == (1, 2)

    def test_nested_schemas_check_the_top_level_type_only(self, fresh_graph):
        """The documented boundary: a wrong INNER value passes
        enforcement (representations validate it, CHECKs do not), while
        a non-object at the top level still fails jsonb_typeof."""
        fresh_graph.define_schema(nodes=[NodeType("ticket", properties=[
            Property("location", "object", properties=[Property("lat", "number",
                                                                required=True)])])])
        fresh_graph.enforce_schema()
        fresh_graph.add_nodes([{"type": "ticket", "location": {"lat": "not-a-number"}}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"type": "ticket", "location": 5}])


# ---------------------------------------------------------------------
# Inference sampling -- TABLESAMPLE, against real PostgreSQL
# ---------------------------------------------------------------------

class TestInferenceSampling:
    def test_full_sample_equals_the_exact_scan(self, fresh_graph):
        """TABLESAMPLE SYSTEM (100) reads every page, so sampling at
        100 must reproduce the exact pipeline's output -- proving the
        sampled path changes WHERE rows come from, never what is done
        with them (a phantom skipped-endpoint edge here would mean the
        node side got sampled too)."""
        seed_chaotic(fresh_graph)
        exact_schema, exact = fresh_graph.infer_schema()
        sampled_schema, sampled = fresh_graph.infer_schema(sample_percent=100)
        assert as_shape(sampled_schema) == as_shape(exact_schema)
        assert sampled.node_counts == exact.node_counts
        assert sampled.edge_counts == exact.edge_counts
        assert sampled.untyped_nodes == exact.untyped_nodes
        assert sampled.untyped_edges == exact.untyped_edges
        assert sampled.skipped_endpoint_edges == exact.skipped_endpoint_edges

    def test_report_says_estimates_only_when_sampling(self, fresh_graph):
        """The epistemics flag: an exact run must NOT carry it (or every
        report would cry wolf), a sampled run must lead with it --
        estimated counts read as truth are how a tentative `required`
        gets enforced."""
        seed_chaotic(fresh_graph)
        _, exact = fresh_graph.infer_schema()
        assert exact.sampled is None
        assert "estimates" not in str(exact)
        _, sampled = fresh_graph.infer_schema(sample_percent=100)
        assert sampled.sampled == 100
        assert str(sampled).startswith(
            "sampled 100% of rows -- counts are estimates\n")

    def test_partial_sample_still_returns_a_valid_schema(self, fresh_graph):
        """SYSTEM sampling is page-level, so on a tiny table a 5% run
        may see all rows or none -- the contract is only that the flag
        propagates and the result is a well-formed (schema, report)
        pair, whatever the sample contained."""
        seed_chaotic(fresh_graph)
        schema, report = fresh_graph.infer_schema(sample_percent=5)
        assert isinstance(schema, GraphSchema)
        assert report.sampled == 5

    def test_a_sample_that_saw_no_nodes_drops_the_edge_types_naming_them(
            self, fresh_graph, monkeypatch):
        """The MIXED sample -- the case the test above could not survive.

        Nodes and edges are sampled independently, so the edge sample can
        return rows while the node sample returns none; endpoints come
        from a join against the FULL nodes table, so inference built
        EdgeType('likes', source='person') against an empty node-type
        list and GraphSchema raised out of a function whose signature
        promises a (schema, report) pair.

        Forced rather than waited for: as a TABLESAMPLE race it fired on
        roughly one CI job in three and one local run in forty, which is
        exactly often enough to be mistaken for infrastructure."""
        from sqlalchemy import false, select

        import hopai.schema as schema_module

        real_source = schema_module._source

        def node_sample_sees_nothing(table, sample_percent):
            if table is fresh_graph.nodes_tbl:
                return select(table).where(false()).subquery()
            return real_source(table, 100)

        monkeypatch.setattr(schema_module, "_source", node_sample_sees_nothing)
        seed_chaotic(fresh_graph)
        schema, report = fresh_graph.infer_schema(sample_percent=5)

        assert list(schema.node_types) == []
        # Not "no edges observed" -- they were seen and then disqualified,
        # which is what the report has to keep saying.
        assert list(schema.edge_types) == []
        assert report.edge_counts == {"works_at": 2, "likes": 2, "dangles": 1}
        assert report.skipped_endpoint_edges == 5
        assert "not one of the above" in str(report)
        # The point of dropping rather than raising: what comes back is
        # still something define_schema() accepts. An inferred schema
        # its own library refuses is not a usable answer.
        fresh_graph.define_schema(schema=schema)

    def test_out_of_range_percent_is_refused_offline(self, offgraph):
        """Validation names the range and runs BEFORE anything connects
        -- TABLESAMPLE with a bad percentage would otherwise surface as
        a server error naming no parameter."""
        for bad in (0, -5, 100.5):
            with pytest.raises(ValueError, match=r"0 < sample_percent <= 100"):
                offgraph.infer_schema(sample_percent=bad)


# ---------------------------------------------------------------------
# Persistence -- save_schema()/load_schema(), against real PostgreSQL
# ---------------------------------------------------------------------

class TestSchemaPersistence:
    def test_save_then_load_on_a_second_handle(self, fresh_graph):
        """The feature's point: a second process loads the contract
        instead of re-declaring it -- equal schema, ADOPTED on the
        handle, and enforce_schema() works from the loaded copy."""
        declared = primitive_schema(fresh_graph)
        fresh_graph.save_schema()
        other = Graph(fresh_graph.engine)
        assert other.schema is None
        loaded = other.load_schema()
        assert loaded == declared
        assert other.schema == declared      # adopted, not just returned
        other.enforce_schema()
        with pytest.raises(ConstraintViolation):
            other.add_nodes([{"id": 1, "type": "person", "nickname": "no-email"}])

    def test_round_trip_is_exact_including_every_flag(self, fresh_graph):
        """Lossless means every flag: unique, allowed values, formats,
        nested object schemas and multi-type sets must all survive the
        database round trip -- a dropped flag would quietly weaken the
        enforce_schema() another process runs from the loaded copy."""
        declared = fresh_graph.define_schema(
            nodes=[NodeType("ticket", properties=[
                Property("code", "string", required=True, unique=True),
                Property("status", "string", values=("open", "closed")),
                Property("opened", "string", format="date-time"),
                Property("age", ("null", "number")),
                Property("location", ("null", "object"), properties=[
                    Property("lat", "number", required=True),
                    Property("lon", "number", required=True)])])],
            edges=[EdgeType("blocks", source="ticket", target="ticket",
                            properties=[Property("hard", "boolean", required=True)])],
        )
        fresh_graph.save_schema()
        assert Graph(fresh_graph.engine).load_schema() == declared

    def test_graphs_persist_independently(self, fresh_graph):
        """One row per graph_id: graph A's save must never become graph
        B's contract -- the _scoped() discipline applied to metadata."""
        primitive_schema(fresh_graph)
        fresh_graph.save_schema()
        elsewhere = fresh_graph.in_graph("elsewhere")
        elsewhere.define_schema(nodes=[NodeType("robot")])
        elsewhere.save_schema()
        loaded = Graph(fresh_graph.engine).load_schema()
        assert {nt.name for nt in loaded.node_types} == {"person", "company"}
        loaded_elsewhere = Graph(fresh_graph.engine).in_graph("elsewhere").load_schema()
        assert {nt.name for nt in loaded_elsewhere.node_types} == {"robot"}

    def test_load_refuses_a_schema_that_collides_with_this_table(self, fresh_graph):
        """save_schema()/load_schema() can cross tables -- the schema
        was a valid contract on the table that declared it, but this
        SECOND handle's table carries a real user_id column the same
        schema names as a JSONB property. load_schema() must catch
        that rather than adopt a contract enforce_schema() could never
        let a correct row satisfy."""
        fresh_graph.define_schema(
            nodes=[NodeType("person", properties=[Property("user_id", "number")])])
        fresh_graph.save_schema()

        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB
        md = MetaData()
        nodes = Table("nodes", md, Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text), Column("user_id", BigInteger),
                      Column("properties", JSONB))
        custom = Graph(fresh_graph.engine, node_table=nodes)
        with pytest.raises(ValueError, match=r"\['user_id'\]"):
            custom.load_schema()
        assert custom.schema is None   # the refusal must not half-adopt it

    def test_load_without_save_names_both_fixes(self, fresh_graph):
        """No table yet (nobody ever persisted) and the error must name
        save_schema() AND define_schema -- the two ways out."""
        with pytest.raises(ValueError) as exc:
            fresh_graph.load_schema()
        assert "no saved schema for graph 'default'" in str(exc.value)
        assert "save_schema()" in str(exc.value)
        assert "define_schema(" in str(exc.value)
        assert fresh_graph.schema is None

    def test_load_missing_row_when_the_table_exists(self, fresh_graph):
        """Same error when SOME graph saved but this one did not --
        another graph's contract must never be handed over as a
        fallback."""
        primitive_schema(fresh_graph)
        fresh_graph.save_schema()
        with pytest.raises(ValueError, match="no saved schema for graph 'elsewhere'"):
            fresh_graph.in_graph("elsewhere").load_schema()

    def test_resave_replaces_the_row(self, fresh_graph):
        """Upsert proven: one row per graph after a re-save, carrying
        the NEW schema and a newer saved_at -- an INSERT-only
        implementation would either fail or leave a stale contract for
        load_schema() to pick nondeterministically."""
        primitive_schema(fresh_graph)
        fresh_graph.save_schema()
        with fresh_graph.engine.connect() as conn:
            first = conn.execute(text('SELECT saved_at FROM "hopai_schema"')).scalar()
        fresh_graph.define_schema(nodes=[NodeType("only")])
        fresh_graph.save_schema()
        loaded = Graph(fresh_graph.engine).load_schema()
        assert {nt.name for nt in loaded.node_types} == {"only"}
        with fresh_graph.engine.connect() as conn:
            rows = conn.execute(text('SELECT saved_at FROM "hopai_schema"')).all()
        assert len(rows) == 1
        assert rows[0][0] > first

    def test_corrupted_document_fails_loudly_and_adopts_nothing(self, fresh_graph):
        """The trust boundary: the stored row is data, so a
        hand-corrupted document must raise the real validation error and
        leave the handle schema-less -- a half-loaded schema would
        enforce a contract nobody wrote."""
        primitive_schema(fresh_graph)
        fresh_graph.save_schema()
        corruptions = [
            ('\'{"nodes": {"person": {"type": "object", '
             '"properties": {"email": "corrupted"}}}, "edges": []}\'',
             r"expected a spec object with a 'type'"),
            ("'[1, 2]'", "not a hopai schema document"),
            ('\'{"nodes": {}, "edges": [{"kind": "works_at", '
             '"source": "ghost", "target": "ghost"}]}\'',
             "not a defined node type"),
        ]
        for document, complaint in corruptions:
            with fresh_graph.engine.begin() as conn:
                conn.execute(text(
                    f'UPDATE "hopai_schema" SET document = CAST({document} AS json)'))
            other = Graph(fresh_graph.engine)
            with pytest.raises(ValueError, match=complaint):
                other.load_schema()
            assert other.schema is None

    def test_never_persisting_never_creates_the_table(self, fresh_graph):
        """hopai_schema is metadata for those who opt in: declare,
        enforce and write all you like -- the table must not exist until
        the first save_schema()."""
        primitive_schema(fresh_graph)
        fresh_graph.enforce_schema()
        fresh_graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"}])
        with fresh_graph.engine.connect() as conn:
            assert conn.execute(
                text("SELECT to_regclass('hopai_schema')")).scalar() is None

    def test_save_without_schema_refuses_offline(self, offgraph):
        """The accessor contract: no declared schema means the refusal
        names define_schema -- and it happens before any connection, so
        the dead DSN proves nothing was touched."""
        with pytest.raises(ValueError, match=r"save_schema\(\) needs a schema"):
            offgraph.save_schema()


# ---------------------------------------------------------------------
# Mermaid -- no database, like the other representations
# ---------------------------------------------------------------------

class TestSchemaMermaid:
    def test_flowchart_with_property_bags_and_labeled_arrows(self, offgraph):
        """The canonical shape: one node per type labeled with the
        bounded property bag (* required, same markers as
        tool_summary), one arrow per (kind, source, target) triple."""
        primitive_schema(offgraph)
        lines = offgraph.schema_mermaid.split("\n")
        assert lines[0] == "flowchart LR"
        assert '    person["person (email*, nickname, age)"]' in lines
        assert '    company["company (name*)"]' in lines
        assert "    person -- works_at --> company" in lines

    def test_parallel_endpoint_pairs_stay_parallel_arrows(self, offgraph):
        """One kind across two endpoint pairs is two arrows -- the same
        no-silent-collapse rule to_networkx() enforces with
        MultiDiGraph."""
        offgraph.define_schema(
            nodes=[NodeType("person"), NodeType("company")],
            edges=[EdgeType("likes", source="person", target="person"),
                   EdgeType("likes", source="person", target="company")])
        diagram = offgraph.schema_mermaid
        assert "person -- likes --> person" in diagram
        assert "person -- likes --> company" in diagram

    def test_names_needing_sanitization_keep_identity_in_the_label(self, offgraph):
        """works-at and works_at slug to the same Mermaid id: the ids
        must stay distinct (or the diagram merges two types) while each
        label keeps the real name; a kind with a space needs the quoted
        edge-label form to stay parseable."""
        offgraph.define_schema(
            nodes=[NodeType("works-at"), NodeType("works_at")],
            edges=[EdgeType("has space", source="works-at", target="works_at")])
        diagram = offgraph.schema_mermaid
        assert 'works_at["works-at"]' in diagram
        assert 'works_at_1["works_at"]' in diagram
        assert '    works_at -- "has space" --> works_at_1' in diagram

    def test_quotes_in_names_cannot_break_the_label(self, offgraph):
        """'\"' is the one character that terminates a Mermaid label
        early; unescaped it would truncate the diagram at the first
        quoted name."""
        offgraph.define_schema(nodes=[NodeType('a"b')])
        diagram = offgraph.schema_mermaid
        assert '["a#quot;b"]' in diagram
        assert 'a"b' not in diagram

    def test_wide_types_cap_with_n_more(self, offgraph):
        """Bounded like tool_summary: a 15-property type shows 12 and
        says +3 more -- a diagram node the width of the screen is not a
        picture anymore."""
        offgraph.define_schema(nodes=[NodeType("wide", properties=[
            Property(f"p{i:02d}", "string") for i in range(15)])])
        diagram = offgraph.schema_mermaid
        assert "+3 more" in diagram
        assert "p11" in diagram and "p12" not in diagram

    def test_undefined_schema_raises_naming_the_fix(self, offgraph):
        """The accessor contract shared by every representation."""
        with pytest.raises(ValueError, match=r"schema_mermaid needs a schema"):
            _ = offgraph.schema_mermaid
