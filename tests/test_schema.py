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

    def test_non_optional_union_refused(self, offgraph):
        with pytest.raises(TypeError, match="union"):
            offgraph.define_schema(nodes=[Flag])

    def test_pep604_optional_matches_typing_optional(self, offgraph):
        """`str | None` and Optional[str] are the same annotation spelled
        two ways; get_origin() returns a DIFFERENT sentinel for each, so
        without the types.UnionType arm the modern spelling would be
        refused as an unmapped annotation."""
        offgraph.define_schema(nodes=[Coupon])
        (coupon,) = offgraph.schema.node_types
        assert coupon.properties == (Property("code", ("string", "null"), required=False),)

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
            # the message must say WHICH accessor needed the schema --
            # "something failed somewhere" is not an error that names
            # the fix
            assert name in str(exc.value)

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
