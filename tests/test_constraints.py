"""
Test suite for hopai.constraints.

These run against a real PostgreSQL because a constraint that is not
enforced by the server is not a constraint. Compiling the right DDL is
necessary and not sufficient -- every test here that claims something is
rejected proves it by trying the write.

The partial-unique tests are the ones to read first: "unique among nodes
of this type" is the capability Neo4j sells with an enterprise licence
and Postgres gives away as a partial index.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from hopai import (
    GT, OR, Check, Col, ConstraintViolation, Index, PropertyType, Required, Unique,
)
from hopai.constraints import compile_constraint, _Target


def indexes(graph) -> dict:
    with graph.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'hopai_write'"
        )).all()
    return dict(rows)


def checks(graph) -> set:
    with graph.engine.connect() as conn:
        return {r[0] for r in conn.execute(text(
            "SELECT conname FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace "
            "WHERE c.contype = 'c' AND n.nspname = 'hopai_write'"
        )).all()}


# ---------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------

class TestUnique:
    def test_rejects_a_duplicate(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.add_nodes([{"email": "a@x.com"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"email": "a@x.com"}])
        assert exc.value.constraint == "uq_nodes_email"

    def test_allows_distinct_values(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        assert fresh_graph.add_nodes([{"email": "a@x.com"}, {"email": "b@x.com"}]) == 2

    def test_does_not_constrain_rows_missing_the_property(self, fresh_graph):
        """A missing property is NULL to the index, and SQL uniqueness
        lets NULLs repeat. So Unique means 'no two share one', not
        'everyone has one' -- the same semantics as Neo4j's uniqueness
        constraint. Pair it with Required to get both."""
        fresh_graph.define_constraints(nodes=[Unique("email")])
        assert fresh_graph.add_nodes([{"name": "no email"}, {"name": "also none"}]) == 2

    def test_composite_constrains_the_combination_only(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("tenant", "slug")])
        fresh_graph.add_nodes([{"tenant": "a", "slug": "x"}, {"tenant": "b", "slug": "x"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"tenant": "a", "slug": "x"}])

    def test_partial_unique_scopes_the_rule(self, fresh_graph):
        """Unique among people, unconstrained for everything else. This
        is a partial index, and it is the constraint Neo4j has no
        equivalent for at any price."""
        fresh_graph.define_constraints(nodes=[Unique("email", where={"type": "person"})])
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"},
                               {"type": "robot", "email": "a@x.com"},
                               {"type": "robot", "email": "a@x.com"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"}])

    def test_partial_unique_with_a_compound_filter(self, fresh_graph):
        fresh_graph.define_constraints(
            nodes=[Unique("email", where=OR({"type": "person"}, {"type": "admin"}),
                          name="uq_human_email")])
        fresh_graph.add_nodes([{"type": "person", "email": "a@x.com"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"type": "admin", "email": "a@x.com"}])
        fresh_graph.add_nodes([{"type": "robot", "email": "a@x.com"}])

    def test_unique_over_real_columns_and_properties_together(self, fresh_graph):
        """One edge of a given kind between a given pair -- the most
        common edge constraint there is, and it needs both real columns
        and a JSONB property in one index."""
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        fresh_graph.define_constraints(
            edges=[Unique(Col("start_id"), Col("end_id"), "kind")])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"},
                               {"start_id": 1, "end_id": 2, "kind": "likes"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])

    def test_unique_needs_at_least_one_key(self):
        with pytest.raises(ValueError, match="at least one key"):
            Unique()

    def test_unknown_column_is_named(self, fresh_graph):
        with pytest.raises(ValueError, match="no column 'nope'"):
            fresh_graph.constraint_ddl(edges=[Unique(Col("nope"))])

    @pytest.mark.parametrize("target,other", [("nodes", "edges"), ("edges", "nodes")])
    def test_unknown_column_says_which_table(self, fresh_graph, target, other):
        """`Col('nope')` on the wrong table is a typo the caller fixes
        by looking at ONE table, so the message has to name which --
        the columns it goes on to list differ between the two. Nothing
        pinned that name, and a mutant relabelling the edges target
        survived the suite."""
        with pytest.raises(ValueError) as raised:
            fresh_graph.constraint_ddl(**{target: [Unique(Col("nope"))]})
        message = str(raised.value)
        assert message.startswith(f"{target} has no column 'nope'")
        assert other not in message


# ---------------------------------------------------------------------
# Presence and type
# ---------------------------------------------------------------------

class TestRequired:
    def test_rejects_a_missing_property(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Required("type")])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"name": "no type"}])

    def test_accepts_any_value_including_false_and_null(self, fresh_graph):
        """Presence, not truthiness: `false` and JSON null are values."""
        fresh_graph.define_constraints(nodes=[Required("flag")])
        assert fresh_graph.add_nodes([{"flag": False}, {"flag": None}, {"flag": 0}]) == 3

    def test_several_keys_at_once(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Required("type", "name")])
        fresh_graph.add_nodes([{"type": "a", "name": "b"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"type": "a"}])

    def test_rejects_col_keys(self):
        with pytest.raises(TypeError, match="already NOT NULL"):
            Required(Col("start_id"))


class TestPropertyType:
    def test_rejects_the_wrong_json_type(self, fresh_graph):
        """The failure this exists for: a model writing "42" instead of
        42, which breaks every numeric comparison silently and later."""
        fresh_graph.define_constraints(nodes=[PropertyType("age", "number")])
        fresh_graph.add_nodes([{"age": 42}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"age": "42"}])

    def test_passes_when_the_property_is_absent(self, fresh_graph):
        """jsonb_typeof(NULL) is NULL and a CHECK passes on NULL, so this
        constrains the type of a value that is there, not its presence."""
        fresh_graph.define_constraints(nodes=[PropertyType("age", "number")])
        assert fresh_graph.add_nodes([{"name": "ageless"}]) == 1

    @pytest.mark.parametrize("json_type,good,bad", [
        ("string", "x", 1),
        ("number", 1, "x"),
        ("boolean", True, "true"),
        ("array", [1, 2], {"a": 1}),
        ("object", {"a": 1}, [1, 2]),
    ])
    def test_each_json_type(self, fresh_graph, json_type, good, bad):
        fresh_graph.define_constraints(nodes=[PropertyType("v", json_type)])
        fresh_graph.add_nodes([{"v": good}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"v": bad}])

    def test_unknown_type_is_rejected_at_declaration(self):
        with pytest.raises(ValueError, match="json_type must be"):
            PropertyType("age", "integer")


class TestCheck:
    def test_arbitrary_filter_as_a_check(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Check(GT("age", 0), name="ck_age_positive")])
        fresh_graph.add_nodes([{"age": 1}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"age": -1}])
        assert exc.value.constraint == "ck_age_positive"

    def test_compound_filter(self, fresh_graph):
        fresh_graph.define_constraints(
            nodes=[Check(OR({"type": "person"}, {"type": "company"}), name="ck_known_type")])
        fresh_graph.add_nodes([{"type": "person"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_nodes([{"type": "alien"}])

    def test_check_requires_a_name(self):
        with pytest.raises(ValueError, match="explicit name"):
            compile_constraint(Check(GT("age", 0)), _Target(None, "nodes"))


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------

class TestLifecycle:
    def test_plain_index_is_created(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Index("type")])
        definition = indexes(fresh_graph)["ix_nodes_type"]
        assert "UNIQUE" not in definition and "type" in definition

    def test_defining_twice_is_a_no_op(self, fresh_graph):
        """Idempotent on purpose: this belongs in a start-up path next to
        create_schema(), and a second run must not explode."""
        declarations = {"nodes": [Unique("email"), Required("type"),
                                  PropertyType("age", "number"),
                                  Check(GT("age", 0), name="ck_age")]}
        first = fresh_graph.define_constraints(**declarations)
        second = fresh_graph.define_constraints(**declarations)
        assert first == second
        assert len(checks(fresh_graph)) == len({n for n in second if n.startswith("ck")})

    def test_drop_constraints_is_the_inverse(self, fresh_graph):
        declarations = {"nodes": [Unique("email"), Required("type")]}
        fresh_graph.define_constraints(**declarations)
        dropped = fresh_graph.drop_constraints(**declarations)
        # the return value names what was dropped, same contract as
        # define_constraints -- a list of Nones would satisfy any test
        # that only counts it
        assert dropped == ["uq_nodes_email", "ck_required_nodes_type"]
        assert "uq_nodes_email" not in indexes(fresh_graph)
        assert fresh_graph.add_nodes([{"email": "a@x.com"}, {"email": "a@x.com"}]) == 2

    def test_dropping_what_was_never_created_is_fine(self, fresh_graph):
        fresh_graph.drop_constraints(nodes=[Unique("nothing_here")])

    def test_constraint_ddl_does_not_execute(self, fresh_graph):
        ddl = fresh_graph.constraint_ddl(nodes=[Unique("email")])
        assert ddl == ['CREATE UNIQUE INDEX IF NOT EXISTS "uq_nodes_email" '
                       'ON "nodes" (graph_id, (properties ->> \'email\'))'], ddl
        assert "uq_nodes_email" not in indexes(fresh_graph)

    def test_create_schema_is_idempotent(self, fresh_graph):
        fresh_graph.create_schema()
        fresh_graph.create_schema()
        assert "ix_edges_graph_start_id" in indexes(fresh_graph)

    def test_create_schema_makes_the_traversal_indexes(self, fresh_graph):
        """Without these every hop is a sequential scan. They are part of
        the schema, not a tuning step the caller has to remember."""
        made = indexes(fresh_graph)
        assert {"ix_edges_graph_start_id", "ix_edges_graph_end_id",
                "ix_nodes_graph", "ix_nodes_properties", "ix_edges_properties"} <= set(made)
        # graph_id must LEAD, or a second graph makes the index useless
        assert made["ix_edges_graph_start_id"].endswith("(graph_id, start_id)")
        assert "gin" in made["ix_nodes_properties"].lower()

    def test_drop_schema(self, fresh_graph):
        fresh_graph.drop_schema()
        assert indexes(fresh_graph) == {}


class TestNaming:
    def test_names_are_deterministic(self, offline_graph):
        assert (offline_graph.constraint_ddl(nodes=[Unique("a", "b")])
                == offline_graph.constraint_ddl(nodes=[Unique("a", "b")]))

    def test_explicit_name_wins(self, offline_graph):
        assert "my_name" in offline_graph.constraint_ddl(nodes=[Unique("a", name="my_name")])[0]

    def test_generated_names_fit_postgres_identifier_limit(self, offline_graph):
        ddl = offline_graph.constraint_ddl(nodes=[Unique(*(f"property_{i}" for i in range(20)))])[0]
        name = ddl.split('"')[1]
        assert len(name) <= 63

    def test_unknown_constraint_object_is_rejected(self, offline_graph):
        with pytest.raises(TypeError, match="expected one of"):
            offline_graph.constraint_ddl(nodes=["Unique('email')"])


class TestViolationErrors:
    def test_message_names_the_constraint_and_the_cause_is_kept(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.add_nodes([{"email": "a@x.com"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"email": "a@x.com"}])
        assert "uq_nodes_email" in str(exc.value)
        assert exc.value.detail and "a@x.com" in exc.value.detail
        assert exc.value.__cause__ is not None  # the driver error is not thrown away
