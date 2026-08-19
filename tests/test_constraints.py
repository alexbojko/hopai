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


class TestDeclaredEdgeType:
    """Graph.define_edge_type() -- issue #80's guaranteed btree index on
    (graph_id, properties ->> 'kind'), the narrow index a `kind`
    equality can use instead of falling back to the whole-properties
    GIN index every other property filter shares."""

    def test_declares_a_functional_btree_on_graph_id_and_kind(self, fresh_graph):
        name = fresh_graph.define_edge_type()
        assert name == "ix_edges_kind"
        definition = indexes(fresh_graph)["ix_edges_kind"]
        assert "UNIQUE" not in definition
        assert "graph_id" in definition
        assert "(properties ->> 'kind'::text)" in definition

    def test_it_is_the_exact_index_define_constraints_would_make(self, fresh_graph):
        """No second naming scheme: Index("kind") through
        define_constraints() directly produces the identical index name
        and DDL, so a caller who already knows constraints.py gets no
        surprise from reaching for the dedicated call instead."""
        table = f"{fresh_graph.edges_tbl.schema}.edges" if fresh_graph.edges_tbl.schema else "edges"
        assert fresh_graph.constraint_ddl(edges=[Index("kind")]) == [
            f'CREATE INDEX IF NOT EXISTS "ix_edges_kind" ON {table} '
            "(graph_id, (properties ->> 'kind'))"
        ]

    def test_flips_the_declared_flag(self, fresh_graph):
        assert fresh_graph.edge_type_declared is False
        fresh_graph.define_edge_type()
        assert fresh_graph.edge_type_declared is True

    def test_declaring_twice_is_a_no_op(self, fresh_graph):
        """Idempotent like define_constraints() -- safe next to
        create_schema() in a start-up path."""
        first = fresh_graph.define_edge_type()
        second = fresh_graph.define_edge_type()
        assert first == second == "ix_edges_kind"
        assert len([n for n in indexes(fresh_graph) if n == "ix_edges_kind"]) == 1

    def test_declared_flag_is_per_handle_not_per_database(self, fresh_graph):
        """An IN-MEMORY flag, not a database read -- the same
        existence-check contract `.schema`/`.vectors` already keep. A
        second handle over the SAME already-declared database still
        answers False until IT calls define_edge_type()."""
        from hopai import Graph
        fresh_graph.define_edge_type()
        second_handle = Graph(fresh_graph.engine)
        assert second_handle.edge_type_declared is False

    def test_works_on_a_caller_supplied_edge_table(self, write_engine):
        """No new column, so nothing for a custom edge_table= to be
        missing -- the same generality Index()/Unique() already have
        over an arbitrary table, see constraints.py's module docstring."""
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        from hopai import Graph

        with write_engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS hopai_custom_edge_type CASCADE"))
            conn.execute(text("CREATE SCHEMA hopai_custom_edge_type"))
        meta = MetaData(schema="hopai_custom_edge_type")
        nodes = Table("nodes", meta, Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text, nullable=False, server_default="default"),
                      Column("properties", JSONB, nullable=False, server_default="{}"))
        edges = Table("edges", meta, Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text, nullable=False, server_default="default"),
                      Column("start_id", BigInteger, nullable=False),
                      Column("end_id", BigInteger, nullable=False),
                      Column("properties", JSONB, nullable=False, server_default="{}"))
        meta.create_all(write_engine)
        graph = Graph(write_engine, node_table=nodes, edge_table=edges)
        name = graph.define_edge_type()
        with write_engine.connect() as conn:
            found = conn.execute(text(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'hopai_custom_edge_type' "
                "AND indexname = :name"), {"name": name}).first()
        assert found is not None


class TestViaStoredInShorthand:
    """via=<name> (Hop.via as a bare string) end to end: same results as
    via={"kind": <name>}, and -- once define_edge_type() has run --
    equal to it in the sense that both compile through the exact same
    fast path (filters.resolve_via(), pinned in tests/test_query_shape.py's
    TestResolveVia and TestQueryStructure)."""

    def test_matches_the_same_edges_as_the_dict_form(self, fresh_graph):
        from hopai import Hop, Start
        fresh_graph.add_nodes([{"id": 1, "type": "leaf"}, {"id": 2, "type": "hub"},
                               {"id": 3, "type": "hub"}])
        fresh_graph.add_edges([
            {"start_id": 1, "end_id": 2, "kind": "knows"},
            {"start_id": 1, "end_id": 3, "kind": "wrong_kind"},
        ])
        by_dict = fresh_graph.traverse(Start(where={"type": "leaf"}), Hop(via={"kind": "knows"}))
        by_shorthand = fresh_graph.traverse(Start(where={"type": "leaf"}), Hop(via="knows"))
        assert {n["id"] for n in by_dict.nodes} == {n["id"] for n in by_shorthand.nodes} == {"1", "2"}
        assert ({(e["start_id"], e["end_id"]) for e in by_dict.edges}
                == {(e["start_id"], e["end_id"]) for e in by_shorthand.edges})

    def test_matches_the_same_edges_once_edge_type_is_declared(self, fresh_graph):
        """The point of define_edge_type(): the ordinary dict form and
        the shorthand agree not just on RESULTS (the test above, true
        regardless) but on which SQL shape reaches the database."""
        from hopai import Hop, Start
        fresh_graph.define_edge_type()
        fresh_graph.add_nodes([{"id": 1, "type": "leaf"}, {"id": 2, "type": "hub"}])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
        by_dict = fresh_graph.traverse(Start(where={"type": "leaf"}), Hop(via={"kind": "knows"}))
        by_shorthand = fresh_graph.traverse(Start(where={"type": "leaf"}), Hop(via="knows"))
        assert {n["id"] for n in by_dict.nodes} == {n["id"] for n in by_shorthand.nodes} == {"1", "2"}

    def test_json_api_accepts_the_bare_string_shorthand(self, fresh_graph):
        from hopai import traverse_json
        fresh_graph.add_nodes([{"id": 1, "type": "leaf"}, {"id": 2, "type": "hub"}])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
        result = traverse_json(fresh_graph, {
            "start": {"where": {"type": "leaf"}},
            "hops": [{"via": "knows"}],
        })
        assert {n["id"] for n in result["nodes"]} == {"1", "2"}


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

    def test_a_declaration_cannot_replace_the_composite_foreign_key(self):
        """The attach machinery matched a same-named object by NAME
        alone, and `table.constraints` holds the table's own primary key
        and foreign keys too -- so a constraint named after one of them
        REPLACED it, on the module-level Edge every default Graph()
        shares. `edges` then got created without the composite FK that
        makes a cross-graph edge impossible.

        Uses the real hopai.models.Edge deliberately: the shared
        singleton is what made this global, and models.py names these
        two foreign keys in cleartext for anyone to collide with. It is
        reachable through constraint_ddl(), which runs nothing."""
        from sqlalchemy.schema import CreateTable

        from hopai.models import Edge

        with pytest.raises(ValueError, match="hopai did not declare"):
            compile_constraint(
                Check(GT("weight", 0), name="fk_edges_start_same_graph"),
                _Target(Edge, "edges", None, "graph_id"))

        kinds = {c.name: type(c).__name__ for c in Edge.constraints if c.name}
        assert kinds["fk_edges_start_same_graph"] == "ForeignKeyConstraint"
        assert "fk_edges_start_same_graph" in str(CreateTable(Edge).compile())

    def test_a_unique_cannot_shadow_an_index_backed_constraint(self):
        """A PRIMARY KEY / UNIQUE constraint owns a backing index of its
        own name, so it shares one namespace with the indexes Unique()
        emits. Without the refusal, Unique(name="uq_nodes_id_graph")
        compiled a CREATE UNIQUE INDEX IF NOT EXISTS that Postgres
        skipped as already-present -- the declared rule silently never
        enforced, which is the worst answer this library can give."""
        from hopai.models import Node

        with pytest.raises(ValueError, match="hopai did not declare"):
            compile_constraint(Unique("email", name="uq_nodes_id_graph"),
                               _Target(Node, "nodes", None, "graph_id"))
        assert not any(i.name == "uq_nodes_id_graph" for i in Node.indexes)

    def test_a_check_may_reuse_a_name_no_index_owns(self, fresh_graph):
        """The mirror of the two above: a CHECK owns no index, so it is
        NOT in the index namespace and a check named after one of this
        graph's own indexes has to keep working -- a refusal widened to
        every container would break it."""
        fresh_graph.define_constraints(nodes=[Index("type", name="shared_name")])
        assert fresh_graph.define_constraints(
            nodes=[Check(GT("age", 0), name="shared_name")]) == ["shared_name"]

    def test_suspending_restores_on_an_exception(self):
        """create_schema() hides hopai's declarations while it issues
        CREATE TABLE, so they are not baked into it. If the CREATE
        raises -- an unreachable database, a permissions error -- the
        `finally` has to put them back, or the process is left with a
        Graph whose declared constraints have silently vanished from its
        own metadata. Coverage cannot see this: the try/finally is one
        statement either way."""
        from hopai.constraints import _attach_index, suspended_declarations
        from hopai.models import Node

        _attach_index(Node, "ix_suspend_probe", [Node.c.graph_id])
        try:
            with suspended_declarations(Node):
                assert not any(i.name == "ix_suspend_probe" for i in Node.indexes)
                raise RuntimeError("whatever CREATE TABLE would have raised")
        except RuntimeError:
            pass
        try:
            assert any(i.name == "ix_suspend_probe" for i in Node.indexes)
        finally:
            # Node is a module-level singleton shared by every default
            # Graph(); leaving a probe on it would follow the whole session.
            Node.indexes.discard(
                next(i for i in Node.indexes if i.name == "ix_suspend_probe"))

    def test_drop_constraints_leaves_a_foreign_object_alone(self, fresh_graph):
        """detach_constraint() is ownership-aware on the way out too --
        dropping must not evict a table's own constraint that happens to
        share the name."""
        from hopai.constraints import detach_constraint
        from hopai.models import Edge

        detach_constraint(Edge, "check", "fk_edges_start_same_graph")
        assert any(c.name == "fk_edges_start_same_graph" for c in Edge.constraints)

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
