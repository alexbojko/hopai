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
# A bare string colliding with a real column
# ---------------------------------------------------------------------

class TestColumnCollision:
    """A bare string that names a real column can only be a mistake --
    that column is written and read by name already, never through
    `properties`. Unique/Index/Required/PropertyType all refuse it
    rather than silently compiling a rule that can never see the value
    it is testing for; Col(...) is the documented, still-working escape
    hatch when the real column genuinely is what's meant."""

    def test_unique_refuses_it(self, fresh_graph):
        with pytest.raises(TypeError, match="'start_id' is a real column"):
            fresh_graph.constraint_ddl(edges=[Unique("start_id")])

    def test_index_refuses_it(self, fresh_graph):
        with pytest.raises(TypeError, match="'end_id' is a real column"):
            fresh_graph.constraint_ddl(edges=[Index("end_id")])

    def test_required_refuses_it(self, fresh_graph):
        with pytest.raises(TypeError, match="'id' is a real column"):
            fresh_graph.constraint_ddl(nodes=[Required("id")])

    def test_property_type_refuses_it(self, fresh_graph):
        with pytest.raises(TypeError, match="'id' is a real column"):
            fresh_graph.constraint_ddl(nodes=[PropertyType("id", "number")])

    def test_the_message_names_the_fix(self, fresh_graph):
        with pytest.raises(TypeError, match=r"Col\('start_id'\)"):
            fresh_graph.constraint_ddl(edges=[Unique("start_id")])

    def test_col_is_the_working_escape_hatch(self, fresh_graph):
        """The point of the refusal: the real column stays reachable,
        it just has to say so -- Unique(Col("start_id"), "kind") is
        exactly what TestMerge.test_merging_edges already relies on."""
        ddl = fresh_graph.constraint_ddl(edges=[Unique(Col("start_id"), "kind")])
        assert ddl

    def test_a_non_colliding_property_is_unaffected(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Required("email"), Unique("email")])
        assert fresh_graph.add_nodes([{"email": "a@x.com"}]) == 1


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

    def test_drop_constraints_reaches_the_edges_too(self, fresh_graph):
        """Every drop_constraints test passed `nodes=` only, so the
        edges half of `_targets(nodes, edges)` could be dropped
        entirely -- `drop_constraints(edges=[...])` silently doing
        nothing, returning [], and leaving the constraint enforcing.

        Asserted by the write, not by the return value: a name in the
        list proves the DDL was composed, not that it ran."""
        fresh_graph.add_nodes([{"id": 1}, {"id": 2}])
        declarations = {"edges": [Unique(Col("start_id"), Col("end_id"), "kind")]}
        fresh_graph.define_constraints(**declarations)
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
        with pytest.raises(ConstraintViolation):
            fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])

        assert fresh_graph.drop_constraints(**declarations) == ["uq_edges_start_id_end_id_kind"]
        assert fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}]) == 1

    def test_dropping_what_was_never_created_is_fine(self, fresh_graph):
        fresh_graph.drop_constraints(nodes=[Unique("nothing_here")])

    def test_constraint_ddl_does_not_execute(self, fresh_graph):
        ddl = fresh_graph.constraint_ddl(nodes=[Unique("email")])
        table = f"{fresh_graph.nodes_tbl.schema}.nodes" if fresh_graph.nodes_tbl.schema else "nodes"
        assert ddl == ['CREATE UNIQUE INDEX IF NOT EXISTS "uq_nodes_email" '
                       f'ON {table} (graph_id, (properties ->> \'email\'))'], ddl
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

    @pytest.mark.parametrize("target", ["nodes", "edges"])
    def test_a_renamed_graph_column_reaches_both_targets(self, target):
        """Table and column names are configurable, and every constraint
        is scoped by the graph column -- so a graph discriminator named
        anything but `graph_id` has to reach the DDL, or the constraint
        is scoped by a column that does not exist.

        Parametrized because `_targets()` builds the node and edge sides
        on two separate lines: dropping graph_col from the EDGES line
        alone left the whole suite green, which is the same
        node-works/edge-twin-untested shape as every other edge defect
        this feature has had."""
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        from hopai import Graph

        metadata = MetaData()
        nodes = Table("nodes", metadata,
                      Column("id", BigInteger, primary_key=True),
                      Column("tenant", Text),
                      Column("properties", JSONB))
        edges = Table("edges", metadata,
                      Column("id", BigInteger, primary_key=True),
                      Column("tenant", Text),
                      Column("start_id", BigInteger),
                      Column("end_id", BigInteger),
                      Column("properties", JSONB))
        graph = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                      node_table=nodes, edge_table=edges, graph_col="tenant")
        ddl = graph.constraint_ddl(**{target: [Unique("email")]})[0]
        assert "tenant" in ddl
        assert "graph_id" not in ddl


class TestSQLAlchemyMetadata:
    """define_constraints() attaches real sqlalchemy.Index/CheckConstraint
    objects to graph.nodes_tbl/edges_tbl -- not just hand-built DDL kept
    off to the side -- so a tool reading that metadata (Alembic's
    --autogenerate, chiefly) sees hopai's declared shape natively."""

    def test_unique_attaches_a_real_index(self, fresh_graph):
        from sqlalchemy import Index as SAIndex

        fresh_graph.define_constraints(nodes=[Unique("email")])
        match = [ix for ix in fresh_graph.nodes_tbl.indexes if ix.name == "uq_nodes_email"]
        assert len(match) == 1
        assert isinstance(match[0], SAIndex)
        assert match[0].unique is True

    def test_required_attaches_a_real_check_constraint(self, fresh_graph):
        from sqlalchemy import CheckConstraint as SACheckConstraint

        fresh_graph.define_constraints(nodes=[Required("type")])
        match = [c for c in fresh_graph.nodes_tbl.constraints
                 if getattr(c, "name", None) == "ck_required_nodes_type"]
        assert len(match) == 1
        assert isinstance(match[0], SACheckConstraint)

    def test_defining_twice_does_not_duplicate_metadata(self, fresh_graph):
        """SQLAlchemy does not dedupe two same-named Index/CheckConstraint
        objects attached to one table -- attach the same declaration twice
        without replacing and Alembic would see two competing definitions
        for one name."""
        declarations = {"nodes": [Unique("email"), Required("type")]}
        fresh_graph.define_constraints(**declarations)
        fresh_graph.define_constraints(**declarations)
        assert sum(1 for ix in fresh_graph.nodes_tbl.indexes
                   if ix.name == "uq_nodes_email") == 1
        assert sum(1 for c in fresh_graph.nodes_tbl.constraints
                   if getattr(c, "name", None) == "ck_required_nodes_type") == 1

    def test_previewing_still_attaches(self, fresh_graph):
        """constraint_ddl() never executes anything, but it still has to
        attach: a project calling it to preview a migration should see
        the same metadata a caller of define_constraints() would."""
        fresh_graph.constraint_ddl(nodes=[Unique("email")])
        assert any(ix.name == "uq_nodes_email" for ix in fresh_graph.nodes_tbl.indexes)

    def test_drop_constraints_detaches(self, fresh_graph):
        declarations = {"nodes": [Unique("email"), Required("type")]}
        fresh_graph.define_constraints(**declarations)
        fresh_graph.drop_constraints(**declarations)
        assert not any(ix.name == "uq_nodes_email" for ix in fresh_graph.nodes_tbl.indexes)
        assert not any(getattr(c, "name", None) == "ck_required_nodes_type"
                       for c in fresh_graph.nodes_tbl.constraints)


class TestViolationErrors:
    def test_message_names_the_constraint_and_the_cause_is_kept(self, fresh_graph):
        fresh_graph.define_constraints(nodes=[Unique("email")])
        fresh_graph.add_nodes([{"email": "a@x.com"}])
        with pytest.raises(ConstraintViolation) as exc:
            fresh_graph.add_nodes([{"email": "a@x.com"}])
        assert "uq_nodes_email" in str(exc.value)
        assert exc.value.detail and "a@x.com" in exc.value.detail
        assert exc.value.__cause__ is not None  # the driver error is not thrown away
