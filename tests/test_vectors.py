"""
Vector search: declarations, the migration, exact-cosine SQL, traversal
integration, and live end-to-end behavior.

The offline classes follow test_query_shape.py's pattern -- query
building never connects, so everything up to the compiled SQL runs with
no database. The *Live classes need Postgres like the rest of the suite
and run on the write schema, since migration is DDL.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from hopai import (
    AGGREGATE_TOOL_SCHEMA, Count, Graph, Hop, INGEST_TOOL_SCHEMA, Near, Start,
    TRAVERSE_TOOL_SCHEMA, Vector, parse_near, spec_to_traversal, vector_search_json,
)
from hopai.constraints import ConstraintViolation


def norm(statement, literal_binds: bool = False) -> str:
    kwargs = {"compile_kwargs": {"literal_binds": True}} if literal_binds else {}
    return " ".join(str(statement.compile(dialect=postgresql.dialect(), **kwargs)).split())


def offline() -> Graph:
    return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")


@pytest.fixture()
def vg() -> Graph:
    """An offline graph with vector fields declared on both targets."""
    g = offline()
    g.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)],
                     edges=[Vector("rel", 3)])
    return g


# ---------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------

class TestVectorDeclaration:
    def test_define_returns_and_exposes_the_registry(self, vg):
        assert set(vg.vectors) == {"nodes", "edges"}
        assert vg.vectors["nodes"]["summary"].dimensions == 3
        assert vg.vectors["edges"]["rel"].column_name == "vec_rel"

    def test_vectors_is_none_before_define(self):
        """None is the existence check, exactly like Graph.schema."""
        assert offline().vectors is None

    def test_redefine_replaces_not_merges(self, vg):
        """Same contract as define_schema(): calling again replaces.
        Merging would make the registry depend on call history."""
        vg.define_vectors(nodes=[Vector("other", 2)])
        assert set(vg.vectors["nodes"]) == {"other"}
        assert vg.vectors["edges"] == {}

    def test_in_graph_starts_without_vectors(self, vg):
        """A different graph may use different fields and DIFFERENT
        dimensions for the same field, so the registry must not travel
        implicitly -- the same rule as schemas."""
        assert vg.in_graph("other").vectors is None

    def test_columns_become_visible_to_sqlalchemy(self, vg):
        """Queries reference vec_<name> columns, so define must attach
        them to the table metadata -- without this every near= query
        fails at build with a KeyError instead of compiling."""
        assert "vec_summary" in vg.nodes_tbl.c
        assert "vec_rel" in vg.edges_tbl.c
        # Nullable, load-bearingly: create_schema() emits this metadata,
        # and a NOT NULL vector column would reject every node written
        # before its embedding exists -- which is all of them. A mutant
        # that flipped this survived the suite in silence.
        assert vg.nodes_tbl.c.vec_summary.nullable is True

    @pytest.mark.parametrize("bad", ["Vec", "9x", "", "a-b", "a b", "vec.sum", None, 42])
    def test_field_names_are_held_to_identifier_rules(self, bad):
        """The name becomes a real column; failing here beats failing
        later as mangled DDL."""
        with pytest.raises(ValueError, match=r"must match \[a-z\]"):
            Vector(bad, 3)

    def test_overlong_name_names_the_identifier_limit(self):
        with pytest.raises(ValueError, match="63-character identifier limit"):
            Vector("x" * 60, 3)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3", True, None])
    def test_dimensions_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="dimensions must be a positive integer"):
            Vector("summary", bad)

    def test_duplicate_field_is_rejected(self):
        with pytest.raises(ValueError, match="duplicate vector field 'summary'"):
            offline().define_vectors(nodes=[Vector("summary", 3), Vector("summary", 4)])

    def test_non_vector_entries_are_rejected(self):
        with pytest.raises(TypeError, match=r"takes Vector\(name, dimensions\) entries"):
            offline().define_vectors(nodes=["summary"])


# ---------------------------------------------------------------------
# The migration's DDL
# ---------------------------------------------------------------------

class TestVectorDDL:
    def test_column_storage_and_check_per_field(self, vg):
        ddl = vg.vector_ddl()
        assert 'ALTER TABLE "nodes" ADD COLUMN IF NOT EXISTS "vec_summary" real[]' in ddl
        # EXTERNAL skips TOAST compression: float noise does not
        # compress, and decompressing every row would tax every scan.
        assert 'ALTER TABLE "nodes" ALTER COLUMN "vec_summary" SET STORAGE EXTERNAL' in ddl
        checks = [s for s in ddl if "ck_vec_dims_default_nodes_summary" in s]
        assert len(checks) == 1
        assert "array_ndims(vec_summary) = 1" in checks[0]
        assert "array_length(vec_summary, 1) = 3" in checks[0]

    def test_check_is_graph_scoped(self, vg):
        """A CHECK binds the whole table; without the guard one graph's
        3-dim rule would reject another graph's 1536-dim vectors."""
        check = [s for s in vg.vector_ddl() if "ADD CONSTRAINT" in s][0]
        assert "graph_id != 'default'" in check
        assert "vec_summary IS NULL" in check

    def test_non_default_graph_gets_its_own_constraint_name(self, vg):
        """Two graphs declaring the same field need two constraints --
        same rule as scope_name() everywhere else. The graph token sits
        right after the prefix, mirroring schema.py's ck_schema_*."""
        other = vg.in_graph("team_a")
        other.define_vectors(nodes=[Vector("summary", 5)])
        checks = [s for s in other.vector_ddl() if "ADD CONSTRAINT" in s]
        default = [s for s in vg.vector_ddl() if "ADD CONSTRAINT" in s][0]
        assert "ck_vec_dims_default_nodes_summary" in default
        assert "ck_vec_dims_default_nodes_summary" not in checks[0]
        assert "graph_id != 'team_a'" in checks[0]
        assert "array_length(vec_summary, 1) = 5" in checks[0]

    @pytest.mark.parametrize("graph_a,field_a,graph_b,field_b", [
        # Long enough that the old name filled the 63-char budget and
        # the graph suffix was truncated away entirely -- so BOTH graphs
        # got one constraint. The second migrate_vectors() then found
        # the first graph's constraint, matched the dimension regex, and
        # reported success having added nothing: that graph accepted any
        # dimensionality, and drop_vectors() in one graph removed the
        # other's check. Vector() cannot catch this -- it validates the
        # column budget, which knows nothing of the table or graph name.
        ("tenant_one", "s" * 46, "tenant_two", "s" * 46),
        # No truncation needed: _slug'd names carry '_', so 'v_b' in
        # graph 'a' and 'v' in graph 'b_a' used to produce one string.
        ("a", "v_b", "b_a", "v"),
    ])
    def test_field_and_graph_can_never_share_a_constraint_name(
            self, vg, graph_a, field_a, graph_b, field_b):
        import re

        def name_for(graph, field):
            handle = vg.in_graph(graph)
            handle.define_vectors(nodes=[Vector(field, 3)])
            ddl = [s for s in handle.vector_ddl() if "ADD CONSTRAINT" in s][0]
            return re.search(r'ADD CONSTRAINT "([^"]+)"', ddl).group(1)

        first, second = name_for(graph_a, field_a), name_for(graph_b, field_b)
        assert first != second
        assert len(first) <= 63 and len(second) <= 63
        # Deterministic, or a re-run of migrate_vectors() would add a
        # second constraint instead of being a no-op.
        assert name_for(graph_a, field_a) == first

    def test_custom_tables_without_discriminator_have_no_guard(self):
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        v = Table("vertex", md, Column("vid", BigInteger, primary_key=True),
                  Column("properties", JSONB))
        e = Table("link", md, Column("lid", BigInteger, primary_key=True),
                  Column("src", BigInteger), Column("dst", BigInteger),
                  Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=v, edge_table=e, node_id_col="vid", edge_id_col="lid",
                  edge_start_col="src", edge_end_col="dst", graph_col=None)
        g.define_vectors(nodes=[Vector("summary", 3)])
        check = [s for s in g.vector_ddl() if "ADD CONSTRAINT" in s][0]
        assert "graph_id" not in check
        assert '"vertex"' in check

    def test_migrate_without_define_names_the_fix_before_connecting(self):
        with pytest.raises(ValueError, match=r"call define_vectors\(...\) first"):
            offline().migrate_vectors()

    def test_no_declaration_means_no_ddl(self):
        assert offline().vector_ddl() == []


# ---------------------------------------------------------------------
# Near validation
# ---------------------------------------------------------------------

class TestNearValidation:
    @pytest.mark.parametrize("bad", ["x", 42, [], (), {}, None])
    def test_vector_must_be_a_non_empty_sequence(self, bad):
        with pytest.raises(TypeError, match="non-empty list of numbers"):
            Near("summary", bad)

    @pytest.mark.parametrize("bad", [[1.0, float("nan")], [float("inf"), 0.0],
                                     [1.0, "2"], [True, False]])
    def test_non_finite_and_non_numeric_elements_are_refused(self, bad):
        """One NaN makes every similarity involving the vector NaN, and
        NaN comparisons rank as garbage without ever raising -- the
        silent-wrong-answer failure this library refuses everywhere."""
        with pytest.raises(ValueError, match="finite number"):
            Near("summary", bad)

    def test_zero_query_vector_is_refused(self):
        with pytest.raises(ValueError, match="all zeros"):
            Near("summary", [0.0, 0.0, 0.0])

    @pytest.mark.parametrize("bad", [None, 42, b"summary", ""])
    def test_field_must_be_a_name(self, bad):
        with pytest.raises(TypeError, match="field must be a vector field name"):
            Near(bad, [1.0])

    def test_search_with_no_near_at_all_is_rejected(self, vg):
        """The caller name leads the message so the reader knows which
        call to fix -- matched exactly, or a mutant can mangle it."""
        with pytest.raises(ValueError, match=r"vector_search\(\): near=\[\] is empty"):
            vg.build_vector_search_query()

    def test_numpy_style_arrays_are_accepted_via_tolist(self):
        """Embeddings usually arrive as numpy float32 arrays; refusing
        them over the container type would be pedantry."""
        class FakeArray:
            def tolist(self):
                return [1.0, 2.0]
        assert Near("summary", FakeArray()).vector == (1.0, 2.0)

    @pytest.mark.parametrize("bad", [0, 0.0, float("nan"), "2", True])
    def test_weight_must_be_non_zero_and_finite(self, bad):
        with pytest.raises(ValueError, match="non-zero finite"):
            Near("summary", [1.0], weight=bad)

    @pytest.mark.parametrize("bad", [1.5, -2, "0.5", True])
    def test_min_similarity_must_be_a_cosine_bound(self, bad):
        with pytest.raises(ValueError, match="between -1 and 1"):
            Near("summary", [1.0], min_similarity=bad)

    def test_missing_mode_is_validated(self):
        with pytest.raises(ValueError, match="missing must be one of"):
            Near("summary", [1.0], missing="drop")

    def test_repr_shows_dims_not_the_vector(self):
        """A 1536-float repr in a traceback helps nobody; the non-default
        knobs are what distinguish two Near specs."""
        assert repr(Near("s", [1, 2, 3])) == "Near('s', 3 dims)"
        assert (repr(Near("s", [1.0], weight=0.5, min_similarity=0.8, missing="zero"))
                == "Near('s', 1 dims, weight=0.5, min_similarity=0.8, missing='zero')")

    def test_near_inside_where_is_refused_by_name(self, vg):
        """Near is an ordering, not a boolean; falling through to the
        generic 'filter must be' message would leave the caller hunting
        for what a Near IS allowed to be."""
        with pytest.raises(TypeError, match="near= on Start/Hop"):
            vg.build_query(Start(where=Near("summary", [1.0, 0.0, 0.0])), [])

    @pytest.mark.parametrize("factory", [
        lambda: Start(k=5),
        lambda: Hop(k=5),
    ])
    def test_k_without_near_is_rejected(self, factory):
        """k is 'keep the top k by similarity'; without a Near there is
        nothing to rank by -- and the message must also head off the
        obvious misreading of k as the hop count."""
        with pytest.raises(ValueError, match="k is not the hop count"):
            factory()

    @pytest.mark.parametrize("bad", [0, -1, "5", True, 2.5])
    def test_k_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            Start(near=Near("summary", [1.0]), k=bad)

    def test_empty_near_list_is_rejected(self):
        with pytest.raises(ValueError, match=r"near=\[\] is empty"):
            Start(near=[])

    def test_near_without_k_or_threshold_changes_nothing_and_is_refused(self, vg):
        """Ranking with no limit and no bound keeps every row -- letting
        it pass would let callers believe it filtered something."""
        with pytest.raises(ValueError, match="changes nothing"):
            vg.build_query(Start(near=Near("summary", [1.0, 0.0, 0.0])), [])

    def test_undefined_field_names_the_defined_ones(self, vg):
        with pytest.raises(ValueError, match=r"no vector field 'body'.*\['summary', 'title'\]"):
            vg.build_vector_search_query(Near("body", [1.0, 0.0, 0.0]))

    def test_no_registry_at_all_names_define_vectors(self):
        """The message leads with the CALLER, so someone holding a
        traceback knows which call to fix -- pinned, because a mutant
        that passed None in place of the caller name still matched a
        test looking only for 'define_vectors'."""
        with pytest.raises(ValueError, match=r"^vector_search\(\) needs vector fields"):
            offline().build_vector_search_query(Near("summary", [1.0]))

    def test_undefined_field_from_a_hop_names_the_hop(self, vg):
        """The same caller-naming contract on the traversal side: a
        chain of five hops must say WHICH one named a field that does
        not exist."""
        with pytest.raises(ValueError, match=r"^hop 1 \(ranked\): no vector field 'body'"):
            vg.build_query(Start(), [Hop(), Hop(near=Near("body", [1.0]), k=2, label="ranked")])

    def test_dimension_mismatch_names_both_sizes(self, vg):
        with pytest.raises(ValueError, match="has 2 dimensions, the field is defined with 3"):
            vg.build_vector_search_query(Near("summary", [1.0, 2.0]))

    def test_non_near_entries_are_rejected(self, vg):
        with pytest.raises(TypeError, match=r"takes Near\(field, vector\) specs"):
            vg.build_query(Start(near=[{"field": "summary"}], k=3), [])

    def test_invalid_target_is_rejected(self, vg):
        with pytest.raises(ValueError, match="target must be one of"):
            vg.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0]), target="rows")

    @pytest.mark.parametrize("bad", [0, -1, True, "10"])
    def test_search_k_must_be_a_positive_integer(self, vg, bad):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            vg.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0]), k=bad)


# ---------------------------------------------------------------------
# The search query's shape
# ---------------------------------------------------------------------

class TestSearchQueryShape:
    @staticmethod
    def sql(vg, *near, literal_binds=False, **kw) -> str:
        return norm(vg.build_vector_search_query(*near, **kw), literal_binds)

    def test_exact_cosine_arithmetic_is_present(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 2.0, 3.0]))
        for piece in ("unnest", "sum", "sqrt", "nullif"):
            assert piece in sql

    def test_sums_accumulate_in_float8(self, vg):
        """sum(real) accumulates in float4; over a long embedding that
        drifts. The DOUBLE PRECISION casts before the products are the
        fix, and losing them changes answers silently."""
        assert "AS DOUBLE PRECISION" in self.sql(vg, Near("summary", [1.0, 2.0, 3.0]))

    def test_query_norm_is_a_python_constant(self, vg):
        """The query vector never changes mid-query, so its norm is
        computed once in Python -- exactly one sqrt (the stored side)
        and one unnest may appear, or a third of the per-row work is
        being recomputed for a constant."""
        sql = self.sql(vg, Near("summary", [1.0, 2.0, 3.0]))
        assert sql.count("unnest") == 1
        assert sql.count("sqrt") == 1

    def test_search_is_graph_scoped(self, vg):
        """Every read goes through _scoped() -- a vector search is a new
        query path and needs its own proof."""
        assert "graph_id = 'default'" in self.sql(vg, Near("summary", [1.0, 2.0, 3.0]),
                                                  literal_binds=True)

    def test_query_vector_values_are_bound_not_inlined(self, vg):
        """Query vectors are caller data; like every filter value they
        must reach the server as parameters, not SQL text."""
        sql = self.sql(vg, Near("summary", [1.5, 2.5, 3.5]))
        assert "1.5" not in sql
        assert "%(param" in sql

    def test_where_filter_and_null_guard_prefilter_candidates(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 2.0, 3.0]), where={"type": "person"})
        assert "properties @> CAST" in sql
        assert "vec_summary IS NOT NULL" in sql

    def test_ranked_most_similar_first_with_deterministic_ties(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 2.0, 3.0]))
        assert "ORDER BY similarity DESC" in sql
        # The tiebreak is the RAW id: as text, '10' would sort before '9'.
        assert "._id LIMIT" in sql

    def test_ids_are_cast_to_text_like_every_other_result(self, vg):
        assert "AS VARCHAR" in self.sql(vg, Near("summary", [1.0, 2.0, 3.0]))

    def test_edges_target_searches_edges_and_reports_endpoints(self, vg):
        sql = self.sql(vg, Near("rel", [1.0, 2.0, 3.0]), target="edges")
        assert "FROM edges" in sql
        assert "vec_rel" in sql
        assert "start_id" in sql and "end_id" in sql

    def test_single_statement_one_round_trip(self, vg):
        assert ";" not in self.sql(vg, Near("summary", [1.0, 2.0, 3.0]))

    def test_default_k_is_ten_and_the_two_entry_points_agree(self, vg):
        """build_vector_search_query() is documented as *the statement
        vector_search() runs*, so a drifted default would make the
        preview a different query than the one executed -- silently, in
        exactly the call someone reaches for to check what will run.
        Mutants on either default survived the suite."""
        import inspect

        sql = self.sql(vg, Near("summary", [1.0, 2.0, 3.0]), literal_binds=True)
        assert "LIMIT 10" in sql
        defaults = {
            name: inspect.signature(getattr(Graph, name)).parameters["k"].default
            for name in ("vector_search", "build_vector_search_query")
        }
        assert defaults == {"vector_search": 10, "build_vector_search_query": 10}

    def test_multivector_weights_land_in_the_combined_score(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0], weight=0.7),
                       Near("title", [0.0, 1.0, 0.0], weight=0.3), literal_binds=True)
        assert sql.count("unnest") == 2
        assert "* 0.7" in sql and "* 0.3" in sql

    def test_exclude_mode_guards_every_field(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0]), Near("title", [0.0, 1.0, 0.0]))
        assert "vec_summary IS NOT NULL" in sql
        assert "vec_title IS NOT NULL" in sql
        assert "coalesce" not in sql

    def test_zero_mode_coalesces_instead_of_excluding(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0]),
                       Near("title", [0.0, 1.0, 0.0], missing="zero"))
        assert "coalesce" in sql
        # summary still excludes; title no longer guards candidacy.
        assert "vec_summary IS NOT NULL" in sql

    def test_all_zero_mode_still_requires_some_vector(self, vg):
        """A row with NO vectors would otherwise score a meaningless 0
        and pad the results."""
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0], missing="zero"),
                       Near("title", [0.0, 1.0, 0.0], missing="zero"))
        assert "vec_summary IS NOT NULL OR" in sql

    def test_min_similarity_filters_that_field(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0], min_similarity=0.5),
                       literal_binds=True)
        assert ">= 0.5" in sql

    def test_custom_table_and_column_names_are_used(self):
        """Same contract as the traversal's custom-schema test. The
        default tables name BOTH id columns `id`, so picking the edge
        id column for a node search is invisible there -- only distinct
        names can catch it, and a mutant doing exactly that survived."""
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        v = Table("vertex", md, Column("vid", BigInteger, primary_key=True),
                  Column("properties", JSONB))
        e = Table("link", md, Column("lid", BigInteger, primary_key=True),
                  Column("src", BigInteger), Column("dst", BigInteger),
                  Column("properties", JSONB))
        g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/x",
                  node_table=v, edge_table=e, node_id_col="vid", edge_id_col="lid",
                  edge_start_col="src", edge_end_col="dst", graph_col=None)
        g.define_vectors(nodes=[Vector("summary", 3)], edges=[Vector("rel", 3)])

        nodes_sql = norm(g.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0])))
        assert "vertex.vid" in nodes_sql and "lid" not in nodes_sql
        assert "graph_id" not in nodes_sql
        edges_sql = norm(g.build_vector_search_query(Near("rel", [1.0, 0.0, 0.0]),
                                                     target="edges"))
        for name in ("link.lid", "link.src", "link.dst"):
            assert name in edges_sql
        assert "vid" not in edges_sql


# ---------------------------------------------------------------------
# Traversal integration
# ---------------------------------------------------------------------

class TestTraversalNearShape:
    def test_seed_cte_is_ranked_and_limited(self, vg):
        sql = norm(vg.build_query(Start(near=Near("summary", [1.0, 0.0, 0.0]), k=4),
                                  [Hop(via={"kind": "x"})]))
        assert "seed AS" in sql
        assert "unnest" in sql and "ORDER BY" in sql and "LIMIT" in sql
        # Downstream is untouched: the walk still hangs off `seed`.
        assert "walk_0" in sql and "match_0" in sql

    def test_seed_near_keeps_every_table_access_scoped(self, vg):
        """The count existing tests pin for the near-less query (seed,
        walk base, recursive term, match join, edge hydration) must
        hold with a ranked seed too -- similarity must not open a
        cross-graph window."""
        sql = norm(vg.build_query(Start(near=Near("summary", [1.0, 0.0, 0.0]), k=4),
                                  [Hop()]), literal_binds=True)
        assert sql.count("graph_id = 'default'") == 5

    @pytest.mark.parametrize("start,hops", [
        (Start(near=Near("summary", [1.0, 0.0, 0.0]), k=3), [Hop()]),
        (Start(), [Hop(near=Near("summary", [1.0, 0.0, 0.0]), k=3)]),
    ])
    def test_ranked_ctes_break_ties_on_the_id(self, vg, start, hops):
        """When more nodes tie on similarity than k keeps, WHICH ones
        survive must not be arbitrary -- two runs of the same traversal
        would otherwise return different subgraphs. The search side has
        the same tiebreak; a mutant dropping it here survived, because
        every behavioral test used distinct similarities."""
        import re

        sql = norm(vg.build_query(start, hops))
        assert re.search(r"ORDER BY [^ ]+ DESC, \w+\.node_id", sql), sql

    def test_hop_near_ranks_after_deduplication(self, vg):
        """Many walks can reach one node; its similarity is one number.
        The reached_i subquery dedupes BEFORE the per-row subquery runs,
        or the most-fanned-in node pays its similarity many times."""
        sql = norm(vg.build_query(Start(), [Hop(near=Near("summary", [1.0, 0.0, 0.0]),
                                               k=2, hops=(1, 3))]))
        assert "reached_0" in sql
        assert "match_0" in sql and "LIMIT" in sql

    def test_threshold_only_near_filters_without_limiting(self, vg):
        sql = norm(vg.build_query(
            Start(), [Hop(near=Near("summary", [1.0, 0.0, 0.0], min_similarity=0.5))]),
            literal_binds=True)
        assert ">= 0.5" in sql
        assert "LIMIT" not in sql

    def test_defining_vectors_changes_no_near_less_query(self, vg):
        """Declaring fields must be free: the same traversal must emit
        byte-identical SQL whether or not vectors are defined, or every
        existing query plan is up for renegotiation."""
        plain = offline()
        start, hops = Start(where={"type": "person"}), [Hop(hops=(1, 3)), Hop(optional=True)]
        assert norm(plain.build_query(start, hops)) == norm(vg.build_query(start, hops))

    def test_aggregates_run_over_the_ranked_set(self, vg):
        """near + k then aggregate means 'summarize the k most similar
        (and what they reach)' -- and the aggregation path shares
        _walk_matches, so it must accept near the same way."""
        sql = norm(vg.build_aggregate_query(
            Start(near=Near("summary", [1.0, 0.0, 0.0]), k=7), [Hop()], {"n": Count()}))
        assert "LIMIT" in sql
        for cte in ("hop_edges", "all_edges", "edge_rows"):
            assert cte not in sql

    def test_optional_and_near_compose_on_the_last_hop(self, vg):
        sql = norm(vg.build_query(
            Start(), [Hop(), Hop(near=Near("summary", [1.0, 0.0, 0.0]), k=3, optional=True)]))
        assert "match_0.node_id" in sql

    def test_hop_near_error_names_the_hop_like_optional_does(self, vg):
        with pytest.raises(ValueError, match=r"hop 1 \(ranked\): near= without k="):
            vg.build_query(Start(), [Hop(), Hop(near=Near("summary", [1.0, 0.0, 0.0]),
                                               label="ranked")])

    def test_traversal_near_keeps_the_presence_guards(self, vg):
        """The guards are not just an optimization: in all-zero missing
        mode a vectorless node's score coalesces to 0, which is NOT
        NULL -- so without the presence guard it would quietly join the
        ranked set. A mutant that dropped the guards survived every
        behavioral test because exclude mode also filters via NULL
        propagation; the guard has to be pinned in the SQL itself."""
        exclude = norm(vg.build_query(
            Start(near=Near("summary", [1.0, 0.0, 0.0]), k=3), [Hop()]))
        assert "vec_summary IS NOT NULL" in exclude
        zero = norm(vg.build_query(
            Start(near=[Near("summary", [1.0, 0.0, 0.0], missing="zero"),
                        Near("title", [0.0, 1.0, 0.0], missing="zero")], k=3), [Hop()]))
        assert "vec_summary IS NOT NULL OR" in zero


# ---------------------------------------------------------------------
# The JSON front end
# ---------------------------------------------------------------------

class TestNearJsonPythonEquivalence:
    """The JSON near form compiles through the same validate/build path
    as the Python form; comparing full SQL, values included, proves it."""

    def test_same_traversal_sql(self, vg):
        spec_start, spec_hops = spec_to_traversal({
            "start": {"near": {"field": "summary", "vector": [1.0, 0.0, 0.0]}, "k": 4},
            "hops": [{"near": [{"field": "summary", "vector": [0.0, 1.0, 0.0],
                                "weight": 0.5, "min_similarity": 0.2, "missing": "zero"}],
                      "k": 2}],
        })
        python = vg.build_query(
            Start(near=Near("summary", [1.0, 0.0, 0.0]), k=4),
            [Hop(near=Near("summary", [0.0, 1.0, 0.0], weight=0.5, min_similarity=0.2,
                           missing="zero"), k=2)])
        assert norm(vg.build_query(spec_start, spec_hops), literal_binds=True) \
            == norm(python, literal_binds=True)

    @pytest.mark.parametrize("bad,message", [
        ({"field": "s"}, r"needs \['vector'\]"),
        ({"vector": [1.0]}, r"needs \['field'\]"),
        ({"field": "s", "vector": [1.0], "top_k": 3}, "unknown"),
        # Anchored: an unanchored "empty list" kept matching after a
        # mutant mangled the message's edges.
        ([], '^"near" is an empty list'),
        # "got str", not just "must be an object": the message names the
        # actual offending type -- same standard as parse_aggregate,
        # where a mutant hardcoding NoneType survived a looser match.
        ("s", "must be an object or a list of objects -- got str"),
    ])
    def test_malformed_near_specs_are_rejected(self, bad, message):
        with pytest.raises((ValueError, TypeError), match=message):
            parse_near(bad)

    def test_vector_search_json_requires_near(self, vg):
        with pytest.raises(ValueError, match='"near" key'):
            vector_search_json(vg, {"k": 5})


class TestToolSchemasStayVectorFree:
    def test_no_schema_advertises_an_embedding_parameter(self):
        """DELIBERATE asymmetry with the parsers, pinned: a tool-calling
        model asked to fill a "vector" invents plausible floats, and an
        invented embedding finds confidently wrong neighbors. If this
        fails, someone widened a schema -- vectors.py explains why not."""
        for schema in (TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA, INGEST_TOOL_SCHEMA):
            dumped = json.dumps(schema)
            assert '"near"' not in dumped
            assert "embedding" not in dumped
            assert "similarity" not in dumped


# ---------------------------------------------------------------------
# Live: migration
# ---------------------------------------------------------------------

def _migrated(fresh_graph) -> Graph:
    fresh_graph.define_vectors(nodes=[Vector("docvec", 3), Vector("titlevec", 3)],
                               edges=[Vector("relvec", 3)])
    fresh_graph.migrate_vectors()
    return fresh_graph


class TestVectorMigrationLive:
    def test_migrate_is_idempotent(self, fresh_graph):
        g = _migrated(fresh_graph)
        first = g.migrate_vectors()
        assert first == g.migrate_vectors()
        assert "nodes.vec_docvec" in first and "edges.vec_relvec" in first

    def test_storage_is_external_and_constraint_exists(self, fresh_graph):
        g = _migrated(fresh_graph)
        with g.engine.connect() as conn:
            storage = conn.execute(text(
                "SELECT attstorage FROM pg_attribute "
                "WHERE attrelid = CAST('nodes' AS regclass) AND attname = 'vec_docvec'"
            )).scalar()
            assert storage == "e"
            names = {row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('nodes' AS regclass) AND contype = 'c'"
            ))}
        assert "ck_vec_dims_default_nodes_docvec" in names

    def test_wrong_dimensions_are_rejected_by_the_server_too(self, fresh_graph):
        """The CHECK is what protects writes that bypass this library
        entirely -- the same reason enforce_schema() compiles to
        constraints instead of Python validation."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        with pytest.raises(Exception, match="ck_vec_dims_default_nodes_docvec"), \
                g.engine.begin() as conn:
            conn.execute(text(
                "UPDATE nodes SET vec_docvec = '{1,2}' WHERE id = 1"))

    def test_changing_dimensions_is_refused_with_the_fix_named(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.define_vectors(nodes=[Vector("docvec", 4)])
        with pytest.raises(ValueError, match=r"different dimensions.*drop_vectors"):
            g.migrate_vectors()

    def test_conflicting_column_type_is_refused(self, fresh_graph):
        """On its own tables rather than the shared Node metadata: the
        define_vectors() here would otherwise attach vec_clash to the
        global Table, and the NEXT in-process run of this test would
        then create `nodes` with the column and die on the raw ALTER.
        mutmut's baseline runs the suite twice in one process, so that
        leak read as `0/N mutants checked` on every vector PR."""
        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        v = Table("clash_nodes", md, Column("id", BigInteger, primary_key=True),
                  Column("properties", JSONB))
        e = Table("clash_edges", md, Column("id", BigInteger, primary_key=True),
                  Column("start_id", BigInteger), Column("end_id", BigInteger),
                  Column("properties", JSONB))
        with fresh_graph.engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE clash_nodes (id bigint primary key, "
                "properties jsonb, vec_clash text)"))
        g = Graph(fresh_graph.engine, node_table=v, edge_table=e, graph_col=None)
        g.define_vectors(nodes=[Vector("clash", 3)])
        with pytest.raises(ValueError, match="not a float array"):
            g.migrate_vectors()

    def test_existing_violating_rows_fail_the_migration_by_name(self, fresh_graph):
        """ADD CONSTRAINT validates existing rows; pre-vector data of
        the wrong shape must surface as a named fix, not a driver error."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        with g.engine.begin() as conn:
            conn.execute(text(
                'ALTER TABLE nodes DROP CONSTRAINT "ck_vec_dims_default_nodes_docvec"'))
            conn.execute(text("UPDATE nodes SET vec_docvec = '{1,2}' WHERE id = 1"))
        with pytest.raises(ConstraintViolation, match="drop_vectors"):
            g.migrate_vectors()

    def test_two_graphs_may_give_one_field_different_dimensions(self, fresh_graph):
        """The reason vectors are real[] plus a scoped CHECK instead of
        a typed vector(d) column: per-graph dimensionality on shared
        tables. Each graph's rule binds only its own rows -- so BOTH
        constraints must exist under distinct names, and each must
        reject the other's size. Asserting only that 'some constraint
        fired' would pass while the two graphs shared one check."""
        a = _migrated(fresh_graph)
        b = fresh_graph.in_graph("other")
        b.define_vectors(nodes=[Vector("docvec", 2)])
        b.migrate_vectors()
        a.add_nodes([{"id": 1, "t": "a"}])
        b.add_nodes([{"id": 2, "t": "b"}])
        assert a.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 2.0, 3.0]}]) == 1
        assert b.set_vectors(nodes=[{"id": 2, "docvec": [1.0, 2.0]}]) == 1

        with fresh_graph.engine.connect() as conn:
            names = sorted(row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('nodes' AS regclass) AND contype = 'c' "
                "AND conname LIKE 'ck\\_vec\\_dims\\_%docvec' ESCAPE '\\'")))
        assert len(names) == 2, f"one constraint is serving both graphs: {names}"

        # Each graph rejects the OTHER graph's dimensionality.
        for row_id, bad in ((2, "{1,2,3}"), (1, "{1,2}")):
            with pytest.raises(Exception, match="ck_vec_dims_"), \
                    fresh_graph.engine.begin() as conn:
                conn.execute(text(
                    f"UPDATE nodes SET vec_docvec = '{bad}' WHERE id = {row_id}"))


# ---------------------------------------------------------------------
# Live: writing and reading vectors
# ---------------------------------------------------------------------

class TestSetGetVectorsLive:
    def test_roundtrip_within_float4_precision(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        assert g.set_vectors(nodes=[{"id": 1, "docvec": [0.1, 0.2, 0.3]}]) == 1
        stored = g.get_vectors(nodes=[1])["nodes"]["1"]
        assert stored["docvec"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
        assert stored["titlevec"] is None

    def test_string_ids_from_results_are_accepted_back(self, fresh_graph):
        """Traversal and search hand back string ids; making the caller
        int() them before set_vectors would be a first-use papercut."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 7, "type": "doc"}])
        assert g.set_vectors(nodes=[{"id": "7", "docvec": [1.0, 0.0, 0.0]}]) == 1
        assert "7" in g.get_vectors(nodes=["7"])["nodes"]

    def test_none_clears_a_vector(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        g.set_vectors(nodes=[{"id": 1, "docvec": None}])
        assert g.get_vectors(nodes=[1])["nodes"]["1"]["docvec"] is None

    def test_edge_vectors_roundtrip(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "t": "a"}, {"id": 2, "t": "b"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
        with g.engine.connect() as conn:
            edge_id = conn.execute(text("SELECT id FROM edges")).scalar()
        assert g.set_vectors(edges=[{"id": edge_id, "relvec": [0.5, 0.5, 0.0]}]) == 1
        stored = g.get_vectors(edges=[edge_id])["edges"][str(edge_id)]
        assert stored["relvec"] == pytest.approx([0.5, 0.5, 0.0], abs=1e-6)

    def test_unknown_field_names_the_defined_ones(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match=r"no vector field 'body'.*\['docvec', 'titlevec'\]"):
            g.set_vectors(nodes=[{"id": 1, "body": [1.0, 0.0, 0.0]}])

    def test_wrong_dimensions_fail_in_python_first(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="has 2 dimensions, the field is defined with 3"):
            g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 2.0]}])

    def test_missing_row_fails_the_whole_call(self, fresh_graph):
        """One transaction, like every write: a partial vector write
        would leave a retry believing rows were never written."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        with pytest.raises(ValueError, match="no node with id 99.*nothing from this call"):
            g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                                 {"id": 99, "docvec": [0.0, 1.0, 0.0]}])
        assert g.get_vectors(nodes=[1])["nodes"]["1"]["docvec"] is None

    def test_rows_in_another_graph_do_not_count_as_existing(self, fresh_graph):
        """set_vectors is scoped like every write -- updating another
        graph's row by id collision is the one bug multi-graph must
        never produce."""
        g = _migrated(fresh_graph)
        other = g.in_graph("other")
        other.define_vectors(nodes=[Vector("docvec", 3)])
        other.add_nodes([{"id": 1, "t": "other"}])
        with pytest.raises(ValueError, match="no node with id 1 in graph 'default'"):
            g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])

    def test_duplicate_id_in_one_call_is_rejected(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="appears twice"):
            g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                                 {"id": 1, "titlevec": [1.0, 0.0, 0.0]}])

    def test_row_without_fields_or_without_id_is_rejected(self, fresh_graph):
        g = _migrated(fresh_graph)
        with pytest.raises(ValueError, match="names no vector fields"):
            g.set_vectors(nodes=[{"id": 1}])
        with pytest.raises(ValueError, match="dict with an 'id'"):
            g.set_vectors(nodes=[{"docvec": [1.0, 0.0, 0.0]}])

    def test_get_vectors_narrows_by_fields_and_validates_them(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        only = g.get_vectors(nodes=[1], fields=["docvec"])["nodes"]["1"]
        assert set(only) == {"docvec"}
        with pytest.raises(ValueError, match="no vector field 'nope'"):
            g.get_vectors(nodes=[1], fields=["nope"])

    def test_absent_ids_are_simply_absent(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        assert g.get_vectors(nodes=[1, 42])["nodes"].keys() == {"1"}

    def test_non_numeric_string_ids_pass_through_for_custom_tables(self):
        """The str->int coercion serves the default BIGINT ids; a
        custom table with text ids must not have them mangled."""
        from hopai.vectors import _coerce_id
        assert _coerce_id("42") == 42
        assert _coerce_id("doc-a") == "doc-a"
        assert _coerce_id(7) == 7


# ---------------------------------------------------------------------
# Live: search
# ---------------------------------------------------------------------

def _corpus(fresh_graph) -> Graph:
    """Vectors chosen so every cosine is hand-checkable: unit-ish 3-d
    vectors against query (1, 0, 0) give 1.0, 0.8, 0.6, 0.0, -1.0."""
    g = _migrated(fresh_graph)
    g.add_nodes([
        {"id": 1, "type": "doc", "name": "exact"},
        {"id": 2, "type": "doc", "name": "close"},
        {"id": 3, "type": "doc", "name": "closer"},
        {"id": 4, "type": "doc", "name": "orthogonal"},
        {"id": 5, "type": "doc", "name": "opposite"},
        {"id": 6, "type": "doc", "name": "no-vector"},
        {"id": 7, "type": "memo", "name": "wrong-type"},
    ])
    g.set_vectors(nodes=[
        {"id": 1, "docvec": [1.0, 0.0, 0.0]},
        {"id": 2, "docvec": [0.6, 0.8, 0.0]},
        {"id": 3, "docvec": [0.8, 0.6, 0.0]},
        {"id": 4, "docvec": [0.0, 1.0, 0.0]},
        {"id": 5, "docvec": [-1.0, 0.0, 0.0]},
        {"id": 7, "docvec": [1.0, 0.0, 0.0]},
    ])
    return g


QUERY = [1.0, 0.0, 0.0]


class TestVectorSearchLive:
    def test_ranking_and_scores_match_hand_computed_cosine(self, fresh_graph):
        g = _corpus(fresh_graph)
        hits = g.vector_search(Near("docvec", QUERY), k=10, where={"type": "doc"})
        assert [h["id"] for h in hits] == ["1", "3", "2", "4", "5"]
        assert [h["similarity"] for h in hits] == pytest.approx(
            [1.0, 0.8, 0.6, 0.0, -1.0], abs=1e-6)
        assert hits[0]["properties"]["name"] == "exact"

    def test_k_truncates_and_where_prefilters(self, fresh_graph):
        g = _corpus(fresh_graph)
        assert len(g.vector_search(Near("docvec", QUERY), k=2, where={"type": "doc"})) == 2
        ids = {h["id"] for h in g.vector_search(Near("docvec", QUERY), k=10)}
        assert "7" in ids            # no filter: the memo competes too
        assert "6" not in ids        # no vector, never a candidate

    def test_min_similarity_is_a_floor(self, fresh_graph):
        g = _corpus(fresh_graph)
        hits = g.vector_search(Near("docvec", QUERY, min_similarity=0.7),
                               k=10, where={"type": "doc"})
        assert [h["id"] for h in hits] == ["1", "3"]

    def test_ties_break_by_id_deterministically(self, fresh_graph):
        g = _corpus(fresh_graph)
        hits = g.vector_search(Near("docvec", QUERY, min_similarity=0.99), k=10)
        assert [h["id"] for h in hits] == ["1", "7"]

    def test_wrong_length_stored_vector_never_scores_a_prefix_cosine(self, fresh_graph):
        """unnest(a, b) pads the shorter array with NULLs and sum()
        skips them, so a stored vector of the wrong size would score a
        confident cosine over the prefix the two happen to share --
        1.0 for a 4-of-8 match. Reachable whenever declared dimensions
        and stored rows disagree, which redefining a field without
        re-migrating does. It must read as missing, not as similar."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "short"}, {"id": 2, "n": "right"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.6, 0.8, 0.0]}])
        # Redefine to 6 dims WITHOUT re-migrating: node 1's stored 3-dim
        # vector is now a perfect match on the shared prefix.
        g.define_vectors(nodes=[Vector("docvec", 6)])
        hits = g.vector_search(Near("docvec", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), k=10)
        assert hits == []

    def test_zero_mode_excludes_a_row_whose_only_vector_has_no_direction(self, fresh_graph):
        """With every field in missing="zero" nothing NULLs the combined
        score, so an all-zero vector -- which the module defines as
        missing -- would rank at 0, ABOVE a row whose real vector points
        the other way. Presence of a column is not presence of a
        direction."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "directionless"}, {"id": 2, "n": "opposite"},
                     {"id": 3, "n": "aligned"}])
        g.set_vectors(nodes=[
            {"id": 1, "docvec": [0.0, 0.0, 0.0]},
            {"id": 2, "titlevec": [-1.0, 0.0, 0.0]},
            {"id": 3, "docvec": [1.0, 0.0, 0.0], "titlevec": [1.0, 0.0, 0.0]},
        ])
        hits = g.vector_search(Near("docvec", QUERY, missing="zero"),
                               Near("titlevec", QUERY, missing="zero"), k=10)
        assert [h["id"] for h in hits] == ["3", "2"]

    def test_threshold_only_and_k_agree_on_membership(self, fresh_graph):
        """The same Near specs must match the same rows whether or not
        k is set. `combined IS NOT NULL` used to live inside the k
        branch, so a threshold-only traversal kept a row whose
        exclude-mode vector had no direction -- one the identical search
        with k correctly rejected."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "directionless-doc"}, {"id": 2, "n": "good"}])
        g.set_vectors(nodes=[
            {"id": 1, "docvec": [0.0, 0.0, 0.0], "titlevec": [1.0, 0.0, 0.0]},
            {"id": 2, "docvec": [1.0, 0.0, 0.0], "titlevec": [1.0, 0.0, 0.0]},
        ])
        nears = [Near("docvec", QUERY), Near("titlevec", QUERY, min_similarity=0.5)]
        with_k = {h["id"] for h in g.vector_search(*nears, k=10)}
        result = g.traverse(Start(near=nears), Hop(optional=True))
        threshold_only = {n["id"] for n in result.nodes}
        assert with_k == {"2"}
        assert threshold_only == with_k

    def test_stored_zero_vector_ranks_as_missing(self, fresh_graph):
        """A zero vector has no direction; NULL-ing its similarity and
        excluding it beats a division error mid-statement."""
        g = _corpus(fresh_graph)
        g.set_vectors(nodes=[{"id": 6, "docvec": [0.0, 0.0, 0.0]}])
        ids = {h["id"] for h in g.vector_search(Near("docvec", QUERY), k=10)}
        assert "6" not in ids

    def test_multivector_combines_weighted_fields(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "both"}, {"id": 2, "n": "swapped"}])
        g.set_vectors(nodes=[
            {"id": 1, "docvec": [1.0, 0.0, 0.0], "titlevec": [0.0, 1.0, 0.0]},
            {"id": 2, "docvec": [0.0, 1.0, 0.0], "titlevec": [1.0, 0.0, 0.0]},
        ])
        hits = g.vector_search(Near("docvec", QUERY, weight=0.7),
                               Near("titlevec", QUERY, weight=0.3), k=10)
        assert [h["id"] for h in hits] == ["1", "2"]
        assert [h["similarity"] for h in hits] == pytest.approx([0.7, 0.3], abs=1e-6)

    def test_missing_exclude_versus_zero(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "full"}, {"id": 2, "n": "doc-only"}])
        g.set_vectors(nodes=[
            {"id": 1, "docvec": [0.6, 0.8, 0.0], "titlevec": [1.0, 0.0, 0.0]},
            {"id": 2, "docvec": [1.0, 0.0, 0.0]},
        ])
        strict = g.vector_search(Near("docvec", QUERY, weight=0.5),
                                 Near("titlevec", QUERY, weight=0.5), k=10)
        assert [h["id"] for h in strict] == ["1"]

        lenient = g.vector_search(Near("docvec", QUERY, weight=0.5),
                                  Near("titlevec", QUERY, weight=0.5, missing="zero"), k=10)
        assert [h["id"] for h in lenient] == ["1", "2"]
        # full: 0.5*0.6 + 0.5*1.0 = 0.8; doc-only: 0.5*1.0 + 0 = 0.5
        assert [h["similarity"] for h in lenient] == pytest.approx([0.8, 0.5], abs=1e-6)

    def test_edges_search_reports_endpoints(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "t": "a"}, {"id": 2, "t": "b"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "close"},
                     {"start_id": 2, "end_id": 1, "kind": "far"}])
        with g.engine.connect() as conn:
            close_id = conn.execute(text(
                "SELECT id FROM edges WHERE properties->>'kind' = 'close'")).scalar()
            far_id = conn.execute(text(
                "SELECT id FROM edges WHERE properties->>'kind' = 'far'")).scalar()
        g.set_vectors(edges=[{"id": close_id, "relvec": [1.0, 0.0, 0.0]},
                             {"id": far_id, "relvec": [0.0, 1.0, 0.0]}])
        hits = g.vector_search(Near("relvec", QUERY), target="edges", k=1)
        assert hits[0]["id"] == str(close_id)
        assert (hits[0]["start_id"], hits[0]["end_id"]) == ("1", "2")
        assert hits[0]["properties"]["kind"] == "close"

    def test_graphs_do_not_see_each_others_vectors(self, fresh_graph):
        g = _corpus(fresh_graph)
        other = g.in_graph("other")
        other.define_vectors(nodes=[Vector("docvec", 3)])
        other.add_nodes([{"id": 100, "type": "doc"}])
        other.set_vectors(nodes=[{"id": 100, "docvec": [1.0, 0.0, 0.0]}])
        assert [h["id"] for h in other.vector_search(Near("docvec", QUERY), k=10)] == ["100"]
        assert "100" not in {h["id"] for h in g.vector_search(Near("docvec", QUERY), k=10)}

    def test_vector_search_json_end_to_end(self, fresh_graph):
        g = _corpus(fresh_graph)
        result = vector_search_json(g, {
            "near": {"field": "docvec", "vector": QUERY, "min_similarity": 0.7},
            "k": 2, "where": {"type": "doc"},
        })
        assert json.loads(json.dumps(result)) == result
        assert [h["id"] for h in result["results"]] == ["1", "3"]

    def test_vector_search_json_k_actually_truncates(self, fresh_graph):
        """`k` in the spec has to REACH the search. Pairing it with a
        min_similarity that already limits the result hides a misread
        key behind the threshold's answer -- a mutant that looked up
        "K" survived exactly that way."""
        g = _corpus(fresh_graph)
        spec = {"near": {"field": "docvec", "vector": QUERY}, "where": {"type": "doc"}}
        assert len(vector_search_json(g, {**spec, "k": 1})["results"]) == 1
        assert len(vector_search_json(g, {**spec, "k": 3})["results"]) == 3
        # ...and the documented default when the key is absent.
        assert len(vector_search_json(g, spec)["results"]) == 5


# ---------------------------------------------------------------------
# Live: similarity inside a traversal
# ---------------------------------------------------------------------

class TestTraversalNearLive:
    def test_similar_seeds_walk_and_dead_ends_still_prune(self, fresh_graph):
        """Start(near=, k=) seeds the walk with the top-k similar --
        and a similar seed with no matching edges must STILL vanish
        from the result, because reported nodes derive from edges
        found, never from the seed set."""
        g = _migrated(fresh_graph)
        g.add_nodes([
            {"id": 1, "name": "similar-connected"},
            {"id": 2, "name": "similar-dead-end"},
            {"id": 3, "name": "dissimilar"},
            {"id": 4, "name": "target"},
        ])
        g.set_vectors(nodes=[
            {"id": 1, "docvec": [1.0, 0.0, 0.0]},
            {"id": 2, "docvec": [0.8, 0.6, 0.0]},
            {"id": 3, "docvec": [0.0, 1.0, 0.0]},
        ])
        g.add_edges([{"start_id": 1, "end_id": 4, "kind": "knows"},
                     {"start_id": 3, "end_id": 4, "kind": "knows"}])
        result = g.traverse(Start(near=Near("docvec", QUERY), k=2),
                            Hop(via={"kind": "knows"}))
        assert {n["id"] for n in result.nodes} == {"1", "4"}
        assert len(result.edges) == 1

    def test_hop_near_keeps_only_the_most_similar_reached(self, fresh_graph):
        """A semantic beam: of everything the hop reaches, only the k
        most similar continue -- and only their edges are reported."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "name": "seed", "seed": True},
                     {"id": 2, "name": "close"}, {"id": 3, "name": "far"}])
        g.set_vectors(nodes=[{"id": 2, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 3, "docvec": [0.0, 1.0, 0.0]}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"},
                     {"start_id": 1, "end_id": 3, "kind": "knows"}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via={"kind": "knows"}, near=Near("docvec", QUERY), k=1))
        assert {n["id"] for n in result.nodes} == {"1", "2"}
        assert len(result.edges) == 1

    def test_aggregate_over_the_k_most_similar(self, fresh_graph):
        g = _corpus(fresh_graph)
        counted = g.aggregate(Start(near=Near("docvec", QUERY), k=3, where={"type": "doc"}),
                              aggregates={"n": Count()})
        assert counted == {"n": 3}


# ---------------------------------------------------------------------
# Live: dropping
# ---------------------------------------------------------------------

class TestDropVectorsLive:
    def test_drop_nulls_this_graph_and_removes_its_constraint(self, fresh_graph):
        g = _corpus(fresh_graph)
        dropped = g.drop_vectors(nodes=["docvec"])
        assert dropped == ["docvec"]
        with g.engine.connect() as conn:
            remaining = conn.execute(text(
                "SELECT count(*) FROM nodes WHERE vec_docvec IS NOT NULL")).scalar()
            names = {row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('nodes' AS regclass) AND contype = 'c'"))}
        assert remaining == 0
        assert "ck_vec_dims_default_nodes_docvec" not in names
        assert "docvec" not in g.vectors["nodes"]

    def test_drop_leaves_other_graphs_vectors_alone(self, fresh_graph):
        """The column is shared; dropping a field is a per-graph act.
        Nulling another graph's rows would be the cross-graph write
        this design exists to prevent."""
        g = _migrated(fresh_graph)
        other = g.in_graph("other")
        other.define_vectors(nodes=[Vector("docvec", 3)])
        other.migrate_vectors()
        other.add_nodes([{"id": 10, "t": "o"}])
        other.set_vectors(nodes=[{"id": 10, "docvec": [1.0, 0.0, 0.0]}])
        g.drop_vectors(nodes=["docvec"])
        kept = other.get_vectors(nodes=[10])["nodes"]["10"]["docvec"]
        assert kept == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_drop_of_a_never_migrated_field_is_ignored(self, fresh_graph):
        assert fresh_graph.drop_vectors(nodes=["ghost"]) == ["ghost"]

    def test_drop_works_on_a_handle_that_never_declared_the_field(self, fresh_graph):
        """drop_vectors() takes bare names and probes the catalog, not
        the registry -- so a teardown script, or any fresh in_graph()
        handle, is a supported caller. It used to fail to compile
        ("Unconsumed column names") whenever define_vectors() had not
        attached the column to the table metadata in this process."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "t": "doc"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])

        # Its own Table objects, so the vec_* column is genuinely absent
        # from the metadata -- the state a fresh process starts in,
        # which the shared module-level tables would otherwise hide.
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        n = Table("nodes", md, Column("id", BigInteger, primary_key=True),
                  Column("graph_id", Text), Column("properties", JSONB))
        e = Table("edges", md, Column("id", BigInteger, primary_key=True),
                  Column("graph_id", Text), Column("start_id", BigInteger),
                  Column("end_id", BigInteger), Column("properties", JSONB))
        undeclared = Graph(fresh_graph.engine, node_table=n, edge_table=e)
        assert "vec_docvec" not in undeclared.nodes_tbl.c
        assert undeclared.drop_vectors(nodes=["docvec"]) == ["docvec"]
        with fresh_graph.engine.connect() as conn:
            assert conn.execute(text(
                "SELECT count(*) FROM nodes WHERE vec_docvec IS NOT NULL")).scalar() == 0


# ---------------------------------------------------------------------
# Live: the seam with schema inference
# ---------------------------------------------------------------------

class TestVectorsAreInvisibleToSchemaInference:
    def test_inferred_schema_never_reports_a_vector_field(self, fresh_graph):
        """`vec_*` are real columns, not properties -- so a schema
        inferred from the same rows must describe the properties only.
        The two features landed independently; this pins the seam, since
        an inference that ever reached for columns instead of JSONB keys
        would start emitting 1536-dimension noise into every schema an
        agent is shown."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc", "title": "a"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        schema, report = g.infer_schema()
        # `type` names the inferred node type rather than repeating as
        # one of its properties, so `title` is the whole property set --
        # and no vec_* column joins it.
        properties = {p.name for nt in schema.node_types for p in nt.properties}
        assert properties == {"title"}
        assert [nt.name for nt in schema.node_types] == ["doc"]
        assert report.node_counts == {"doc": 1}
