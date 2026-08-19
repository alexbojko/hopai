"""
Vector search: declarations, the migration, exact-cosine SQL, traversal
integration, and live end-to-end behavior.

The offline classes follow test_query_shape.py's pattern -- query
building never connects, so everything up to the compiled SQL runs with
no database. The *Live classes need Postgres like the rest of the suite
and run on the write schema, since migration is DDL.
"""

from __future__ import annotations

import ast
import json
import pathlib
import random
import re

import pytest
from sqlalchemy import event, func, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from hopai import (
    AGGREGATE_TOOL_SCHEMA, Count, Graph, Hop, INGEST_TOOL_SCHEMA, Near, Start, aggregate_json,
    traverse_json,
    TRAVERSE_TOOL_SCHEMA, VECTOR_SEARCH_TOOL_SCHEMA, Vector, parse_near,
    spec_to_traversal, vector_search_json,
)
from hopai import Boost, Embedder
from hopai import pgvector as pg
from hopai.vectors import (
    build_search_many_query, build_search_query, parse_boost, pgvector_exit_ddl,
)
from hopai.constraints import ConstraintViolation


#: Helpers that take their CALLER's name as an argument, mapped to the
#: position that argument sits in. Several refusals are shared by many
#: entry points, and the label is the only thing telling a caller which
#: one they reached -- so each call site needs its own assertion, and
#: four separate mutation rounds each surfaced one more that had none.
#: Reading the sites out of the source is what stops the fifth.
_LABEL_ARG = {"_check_k": 1, "validate_nears": 4, "validate_boosts": 1,
              "_field": 3, "_defined": 2, "_check_keys": 2, "refuse_vectors": 1,
              "_embedder": 2, "_resolve_query_texts": 3}


#: mutmut copies every function once per mutant into the tree it runs
#: from, so the source this reads THERE also contains the mutated labels
#: -- "XXvector_search()XX", "VECTOR_SEARCH()". Counting those would fail
#: this test under every mutant, which fails the BASELINE, which mutmut
#: reports as "0 checked": a broken harness that reads like a clean
#: sweep. The `_orig` copy carries the real labels, so skipping the
#: numbered ones is both correct and sufficient.
_MUTANT_COPY = re.compile(r"__mutmut_\d+$")


def _calls(node):
    """Every Call below `node`, not descending into mutmut's copies."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and _MUTANT_COPY.search(child.name):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _calls(child)


def declared_caller_labels() -> set:
    """Every literal caller label passed to one of those helpers."""
    from hopai import json_api, vectors

    found = set()
    for module in (vectors, json_api):
        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        for node in _calls(tree):
            if not isinstance(node.func, ast.Name):
                continue
            index = _LABEL_ARG.get(node.func.id)
            if index is None or len(node.args) <= index:
                continue
            argument = node.args[index]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.add(argument.value)
    return found


def norm(statement, literal_binds: bool = False) -> str:
    kwargs = {"compile_kwargs": {"literal_binds": True}} if literal_binds else {}
    return " ".join(str(statement.compile(dialect=postgresql.dialect(), **kwargs)).split())


def offline() -> Graph:
    return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")


def offline_pgvector() -> Graph:
    """offline()'s twin on vector_backend='pgvector'. The backend is a
    constructor argument and nothing about it connects either, so the
    whole compile-time surface is testable with no extension and no
    database -- which is also what proves the choice is inert until a
    query actually ranks by similarity."""
    return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                 vector_backend="pgvector")


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

    def test_no_migrate_kwarg_is_byte_identical_to_today(self, monkeypatch):
        """The default (issue #57): no `migrate=` at all, and `migrate=False`
        explicitly, must both be exactly today's behavior -- no DDL, the
        registry returned -- for every existing caller that never heard of
        the kwarg."""
        called = []
        monkeypatch.setattr(Graph, "migrate_vectors", lambda self: called.append(True))

        g = offline()
        result = g.define_vectors(nodes=[Vector("summary", 3)])
        assert result == g.vectors
        assert isinstance(result, dict) and set(result) == {"nodes", "edges"}

        g2 = offline()
        result2 = g2.define_vectors(nodes=[Vector("summary", 3)], migrate=False)
        assert result2 == g2.vectors

        assert called == []  # migrate_vectors() never ran in either call

    def test_migrate_true_creates_the_column_and_returns_migrate_vectors_result(
            self, fresh_graph):
        """The one-call form (issue #57): define_vectors(migrate=True) must
        have the same effect on the database as the two-call
        define_vectors()+migrate_vectors() sequence, and must hand back
        migrate_vectors()'s own return value -- not the registry -- so the
        DDL result stays visible instead of happening invisibly."""
        result = fresh_graph.define_vectors(nodes=[Vector("docvec", 3)], migrate=True)

        assert result == ["nodes.vec_docvec"]  # migrate_vectors()'s own return, verbatim
        assert result != fresh_graph.vectors  # NOT the registry

        with fresh_graph.engine.connect() as conn:
            udt = conn.execute(text(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'nodes' AND column_name = 'vec_docvec'"
            )).scalar()
            names = {row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('nodes' AS regclass) AND contype = 'c'"))}
        assert udt == "_float4"
        assert "ck_vec_dims_default_nodes_docvec" in names

    def test_migrate_true_matches_calling_migrate_vectors_separately(self, fresh_graph):
        """Equivalence check against the existing two-call path: same
        column, same constraint, on an equally fresh graph."""
        one_call = fresh_graph.define_vectors(nodes=[Vector("docvec", 3)], migrate=True)

        two_call_graph = fresh_graph.in_graph("two_call")
        two_call_graph.define_vectors(nodes=[Vector("docvec", 3)])
        two_call = two_call_graph.migrate_vectors()

        assert one_call == two_call == ["nodes.vec_docvec"]


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

    def test_create_schema_emits_a_nullable_real_array_column(self, vg):
        """define_vectors() attaches the vec_* columns to the shared
        Table metadata, so create_schema() emits them -- and a NOT NULL
        there would make the tables unusable outright. Vectors are
        written only by set_vectors(), which UPDATEs rows that already
        exist, so a row has to be insertable without one first.

        The type is asserted against a FRESH table rather than the
        fixture's, because Table metadata is shared between handles:
        _attach() only appends when the column is absent, so whether
        this graph's call is the one that set the type depends on which
        test ran first. Against a table nobody has touched it does not."""
        ddl = str(CreateTable(vg.nodes_tbl).compile(dialect=postgresql.dialect()))
        assert re.search(r"vec_summary REAL\[\](?!\s+NOT NULL)", ddl), ddl

        from sqlalchemy import BigInteger, Column, MetaData, Table
        from sqlalchemy.dialects.postgresql import ARRAY, REAL
        from hopai.vectors import _attach

        fresh = Table("t", MetaData(), Column("id", BigInteger))
        column = _attach(fresh, "vec_x")
        assert isinstance(column.type, ARRAY) and isinstance(column.type.item_type, REAL)
        assert column.nullable is True

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
        assert "vertex" in check

    def test_migrate_without_define_names_the_fix_before_connecting(self):
        with pytest.raises(ValueError, match=r"call define_vectors\(...\) first"):
            offline().migrate_vectors()

    def test_no_declaration_raises_like_its_siblings(self):
        """An empty list from a DDL previewer reads as "nothing to
        migrate", not "you forgot to declare" -- and schema_ddl(), the
        contract this cites, raises."""
        with pytest.raises(ValueError, match=r"call define_vectors\(...\) first"):
            offline().vector_ddl()


# ---------------------------------------------------------------------
# Near validation
# ---------------------------------------------------------------------

class TestNearValidation:
    # None and "x" are absent on purpose: None means "no query was
    # given" (the neither-of-them refusal) and a str is TEXT to embed,
    # not a malformed vector -- both live in TestNearText.
    @pytest.mark.parametrize("bad", [42, [], (), {}])
    def test_vector_must_be_a_non_empty_sequence(self, bad):
        with pytest.raises(TypeError, match="non-empty list of numbers"):
            Near("summary", bad)

    @pytest.mark.parametrize("bad,name", [(42, "int"), ({}, "dict")])
    def test_the_refusal_names_the_type_actually_passed(self, bad, name):
        """`got {type(vector).__name__}` is the whole diagnostic -- every
        case above matches only the shared sentence, so the type name
        could have been hardcoded to any one of them and stayed green.
        A caller who passed a dict being told "got NoneType" goes looking
        for the wrong bug."""
        with pytest.raises(TypeError, match=rf"got {name}$"):
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

    @pytest.mark.parametrize("where", [
        {"summary": Near("summary", [1.0, 0.0, 0.0])},
        {"summary": [Near("summary", [1.0, 0.0, 0.0])]},
    ])
    def test_near_as_a_property_value_is_refused_by_name_too(self, vg, where):
        """The value position is the LIKELIER mistake, because GT/BETWEEN
        live there: `where={"age": GT(30)}` is what the DSL teaches, so
        `where={"summary": Near(...)}` is the shape a reader reaches for.
        Guarded only at the filter position, it reached json.dumps and
        surfaced as "Object of type Near is not JSON serializable" --
        naming nothing the caller can act on."""
        with pytest.raises(TypeError, match="near= on Start/Hop"):
            vg.build_query(Start(where=where), [])

    def test_the_refusal_names_which_key_held_the_near(self, vg):
        """One offending key out of five is what the caller needs; the
        bare sentence sends them re-reading the whole filter."""
        with pytest.raises(TypeError, match=r"in where=\{'summary': \.\.\.\}"):
            vg.build_query(Start(where={"type": "doc",
                                        "summary": Near("summary", [1.0, 0.0, 0.0])}), [])

    @pytest.mark.parametrize("factory", [
        lambda: Start(keep=5),
        lambda: Hop(keep=5),
    ])
    def test_k_without_near_is_rejected(self, factory):
        """`keep` is 'keep the top N by similarity'; without a Near
        there is nothing to rank by."""
        with pytest.raises(ValueError, match="without near= orders nothing"):
            factory()

    @pytest.mark.parametrize("bad", [0, -1, "5", True, 2.5])
    def test_k_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="keep must be a positive integer"):
            Start(near=Near("summary", [1.0]), keep=bad)

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
            vg.build_query(Start(), [Hop(), Hop(near=Near("body", [1.0]), keep=2, label="ranked")])

    def test_dimension_mismatch_names_both_sizes(self, vg):
        with pytest.raises(ValueError, match="has 2 dimensions, the field is defined with 3"):
            vg.build_vector_search_query(Near("summary", [1.0, 2.0]))

    def test_non_near_entries_are_rejected(self, vg):
        with pytest.raises(TypeError, match=r"takes Near\(field, vector\) specs"):
            vg.build_query(Start(near=[{"field": "summary"}], keep=3), [])

    def test_invalid_target_is_rejected(self, vg):
        with pytest.raises(ValueError, match="target must be one of"):
            vg.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0]), target="rows")

    @pytest.mark.parametrize("bad", [0, -1, True, "10"])
    def test_search_k_must_be_a_positive_integer(self, vg, bad):
        with pytest.raises(ValueError, match="k must be a positive integer or None"):
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
        # The LABELS, not the bare names: "start_id" is also the edges
        # column, so `edges.start_id AS _start` satisfies the substring
        # whether or not the endpoints are actually selected.
        assert "AS start_id" in sql and "AS end_id" in sql

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

    def test_per_field_report_columns_read_the_lateral_not_a_new_evaluation(self, vg):
        """The per-field `similarities` report (issue #54) has to expose
        each LATERAL's raw value outward -- but CLAUDE.md is emphatic
        that similarity is never a correlated scalar subquery, so
        reading it again must not add a second `unnest`/`sqrt` per
        field: it is one more column projected off the SAME LATERAL
        join, not a second computation."""
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0]),
                       Near("title", [0.0, 1.0, 0.0]))
        assert sql.count("unnest") == 2       # one per field, not one per field per use
        assert sql.count("sqrt") == 2
        assert "sim_report_0" in sql and "sim_report_1" in sql

    def test_boost_report_column_reaches_the_outer_select(self, vg):
        sql = self.sql(vg, Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.5))
        # boost_0 is computed inside the inner subquery either way; the
        # OUTER select is what search() reads to build `boosts`, so its
        # presence there specifically is the thing worth pinning.
        outer, _, _ = sql.partition(" FROM (SELECT")
        assert "boost_0" in outer

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
        sql = norm(vg.build_query(Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=4),
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
        sql = norm(vg.build_query(Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=4),
                                  [Hop()]), literal_binds=True)
        assert sql.count("graph_id = 'default'") == 5

    @pytest.mark.parametrize("start,hops", [
        (Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=3), [Hop()]),
        (Start(), [Hop(near=Near("summary", [1.0, 0.0, 0.0]), keep=3)]),
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
                                               keep=2, hops=(1, 3))]))
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
            Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=7), [Hop()], {"n": Count()}))
        assert "LIMIT" in sql
        for cte in ("hop_edges", "all_edges", "edge_rows"):
            assert cte not in sql

    def test_optional_and_near_compose_on_the_last_hop(self, vg):
        sql = norm(vg.build_query(
            Start(), [Hop(), Hop(near=Near("summary", [1.0, 0.0, 0.0]), keep=3, optional=True)]))
        assert "match_0.node_id" in sql

    def test_hop_near_error_names_the_hop_like_optional_does(self, vg):
        with pytest.raises(ValueError, match=r"hop 1 \(ranked\): near= without keep="):
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
            Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=3), [Hop()]))
        assert "vec_summary IS NOT NULL" in exclude
        zero = norm(vg.build_query(
            Start(near=[Near("summary", [1.0, 0.0, 0.0], missing="zero"),
                        Near("title", [0.0, 1.0, 0.0], missing="zero")], keep=3), [Hop()]))
        assert "vec_summary IS NOT NULL OR" in zero


# ---------------------------------------------------------------------
# The JSON front end
# ---------------------------------------------------------------------

class TestNearJsonPythonEquivalence:
    """The JSON near form compiles through the same validate/build path
    as the Python form; comparing full SQL, values included, proves it."""

    def test_same_traversal_sql(self, vg):
        spec_start, spec_hops = spec_to_traversal({
            "start": {"near": {"field": "summary", "vector": [1.0, 0.0, 0.0]}, "keep": 4},
            "hops": [{"near": [{"field": "summary", "vector": [0.0, 1.0, 0.0],
                                "weight": 0.5, "min_similarity": 0.2},
                               {"field": "title", "vector": [1.0, 0.0, 0.0],
                                "missing": "zero"}],
                      "keep": 2}],
        })
        python = vg.build_query(
            Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=4),
            [Hop(near=[Near("summary", [0.0, 1.0, 0.0], weight=0.5, min_similarity=0.2),
                       Near("title", [1.0, 0.0, 0.0], missing="zero")], keep=2)])
        assert norm(vg.build_query(spec_start, spec_hops), literal_binds=True) \
            == norm(python, literal_binds=True)

    @pytest.mark.parametrize("bad,message", [
        ({"field": "s"}, r'needs "text" to embed OR "vector".*not neither'),
        ({"field": "s", "text": "a", "vector": [1.0]},
         r'needs "text" to embed OR "vector".*not both'),
        ({"vector": [1.0]}, r'^a near spec needs "field" -- '),
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

    def test_the_text_form_carries_every_optional_key_across(self, vg):
        """The JSON half of the same resolution the Python half does --
        and the keys travel through parse_near, Near, and _with_vector
        before reaching SQL, so any of the three could drop one."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder()),
                                 Vector("title", 3, embed=counting_embedder())])
        start, hops = spec_to_traversal({
            "start": {"near": [{"field": "summary", "text": "apple", "weight": 0.25,
                                "min_similarity": 0.75, "missing": "zero"},
                               {"field": "title", "text": "banana"}],
                      "keep": 4},
        })
        sql = norm(vg.build_query(start, hops), literal_binds=True)
        assert "0.25" in sql and "0.75" in sql and "coalesce" in sql.lower()
        assert "97" in sql and "98" in sql       # both texts reached the embedder

    def test_a_json_text_query_and_the_python_one_compile_alike(self, vg):
        """The JSON front end holds no query logic -- text= is parsed
        into exactly the Near the Python caller would have written."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder())])
        start, hops = spec_to_traversal(
            {"start": {"near": {"field": "summary", "text": "apple"}, "keep": 4}})
        python = vg.build_query(Start(near=Near("summary", text="apple"), keep=4), [])
        assert norm(vg.build_query(start, hops), literal_binds=True) \
            == norm(python, literal_binds=True)


class TestToolSchemasStayVectorFree:
    @staticmethod
    def _parameter_names(schema: dict) -> set:
        found = set()

        def walk(node):
            if isinstance(node, dict):
                for name, child in (node.get("properties") or {}).items():
                    found.add(name)
                    walk(child)
                # $defs and the combinators too: a near spec lives in
                # $defs and is reached by $ref, so a walker that stopped
                # at properties/items would have seen neither `text` nor
                # a `vector` someone put back beside it.
                for key in ("items", "additionalProperties", "anyOf", "oneOf", "allOf"):
                    walk(node.get(key))
                for child in (node.get("$defs") or {}).values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(schema.get("parameters", schema))
        return found

    @pytest.mark.parametrize("schema", [TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA,
                                        VECTOR_SEARCH_TOOL_SCHEMA, INGEST_TOOL_SCHEMA])
    def test_no_schema_advertises_a_vector_parameter(self, schema):
        """The DELIBERATE asymmetry with the parsers, pinned, and now
        down to exactly one key: a tool-calling model asked to fill a
        "vector" invents plausible floats, and an invented embedding
        finds confidently wrong neighbors. `text` is the way in -- the
        field embeds it with the application's own client -- so it is
        advertised and this must not creep back to cover it.

        Asserted on PARAMETER NAMES, not by grepping the serialized
        schema: the descriptions must stay free to talk about vectors
        and embeddings in prose."""
        assert not self._parameter_names(schema) & {"vector", "embedding"}

    @pytest.mark.parametrize("schema", [TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA,
                                        VECTOR_SEARCH_TOOL_SCHEMA])
    def test_the_text_half_is_advertised(self, schema):
        """The other side of the same rule. A schema that omitted `text`
        along with `vector` would leave a model no way to search by
        meaning at all -- which is what it did before, and why it was
        told to say it could not."""
        names = self._parameter_names(schema)
        assert "text" in names and "near" in names

    def test_per_graph_tool_schemas_stay_vector_free_too(self):
        """tool_schemas() summarizes THIS graph's declared schema into
        each description, so it is the second place a vector field
        could reach a model -- and it is generated, not hand-written,
        which is exactly how a leak would arrive. Vector FIELD names
        are not graph-schema properties: a model given "summary" as a
        property on traverse_graph/aggregate_graph/ingest_graph/
        mutate_graph would filter on it and get nothing.

        Since issue #51 a field name IS legitimate in exactly one
        place: the appended vector search tool's own `field` enum,
        which exists so a model picks among real fields instead of
        guessing one (the whole point of narrowing it from a bare
        string) -- so this checks the enum names them correctly and
        checks the OTHER four tools stay exactly as vector-free as
        before. The raw column name and the literal word "embedding"
        stay forbidden everywhere, enum included: `vec_*` is a storage
        detail no schema anywhere should name, and "embedding" would
        invite a model to send floats under a different word than
        "vector" -- the same invariant test_vectors.py pins for the
        static schemas, just no longer blind to a name appearing
        on purpose."""
        from hopai import NodeType, Property

        g = offline()
        g.define_schema(nodes=[NodeType("doc", properties=[Property("title", "string")])])
        g.define_vectors(nodes=[Vector("summary", 1536)], edges=[Vector("rel", 8)])
        tools = g.tool_schemas()
        search = next(t for t in tools if t["name"] == "search_graph_by_meaning")
        assert search["parameters"]["$defs"]["near"]["properties"]["field"]["enum"] \
            == ["rel", "summary"]

        others = json.dumps([t for t in tools if t is not search])
        assert "title" in others                      # declared properties DO appear
        for leaked in ("summary", "rel"):
            assert leaked not in others, leaked
        for leaked in ("vec_summary", "vec_rel", "embedding"):
            assert leaked not in json.dumps(tools), leaked

    #: The three that advertise a near spec. INGEST_TOOL_SCHEMA writes
    #: rows and has no similarity surface at all, which is why it is in
    #: the vector-free check above and not in these two.
    NEAR_SCHEMAS = [TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA, VECTOR_SEARCH_TOOL_SCHEMA]

    @staticmethod
    def _nodes(node):
        """Every dict in the schema, wherever it sits."""
        if isinstance(node, dict):
            yield node
            for child in node.values():
                yield from TestToolSchemasStayVectorFree._nodes(child)
        elif isinstance(node, list):
            for child in node:
                yield from TestToolSchemasStayVectorFree._nodes(child)

    @pytest.mark.parametrize("schema", NEAR_SCHEMAS)
    def test_every_ref_resolves(self, schema):
        """A $ref naming no $def is a parameter that describes nothing.
        Providers validate these before the model ever sees them, and
        nothing here would have noticed -- the schema stays a dict
        either way, and json.dumps() is just as happy."""
        defs = schema["parameters"].get("$defs", {})
        refs = [node["$ref"] for node in self._nodes(schema) if "$ref" in node]
        assert refs, "no $ref at all -- did the near spec stop being shared?"
        for ref in refs:
            assert ref.startswith("#/$defs/"), ref
            assert ref.split("/")[-1] in defs, f"{ref} resolves to nothing"

    @pytest.mark.parametrize("schema", NEAR_SCHEMAS)
    def test_a_near_parameter_takes_one_spec_or_a_list(self, schema):
        """parse_near() accepts both forms, so the schema has to offer
        both -- under `anyOf`, the key a validator actually knows. A
        misspelled combinator leaves the parameter unconstrained, which
        is the one failure mode a "does it serialize" test cannot see.

        _near_schema() is CALLED here as well as read off the schema.
        The schemas are module-level constants built once at import, so
        a test that only reads them cannot exercise the builder under a
        harness that imports the module once and varies behavior per
        test -- which is how the mutation runner works, and how three
        mutants in this exact function came back "survived" while
        failing this assertion on a fresh interpreter."""
        from hopai.json_api import _near_schema
        built = _near_schema("anything")
        assert set(built) == {"description", "anyOf"}
        assert built["anyOf"] == [{"$ref": "#/$defs/near"},
                                  {"type": "array", "items": {"$ref": "#/$defs/near"}}]

        found = 0
        for node in self._nodes(schema):
            for name in ("near", "via_near"):
                spec = (node.get("properties") or {}).get(name)
                if spec is None:
                    continue
                found += 1
                assert spec["anyOf"] == built["anyOf"]     # same shared shape
        assert found, "no near parameter advertised at all"

    def test_descriptions_point_the_model_at_the_right_tool(self):
        """A model holding only the schema has to be told which half of
        the surface answers which question -- it used to be told that
        searching by meaning was impossible, which is no longer true."""
        for schema in (TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA):
            assert "EXACT property matches" in schema["description"]
            assert "near" in schema["description"]
        assert "not an exact property match" in VECTOR_SEARCH_TOOL_SCHEMA["description"]


class TestJsonFrontEndRefusesInventedVectors:
    def test_traverse_json_refuses_similarity_keys_by_default(self, vg):
        """The invariant was advertised and unenforced: this is the call
        an agent integration wires up, and it happily built a subgraph
        from invented floats."""
        spec = {"start": {"where": {"type": "doc"},
                          "near": {"field": "summary", "vector": [0.1, 0.2, 0.9]},
                          "keep": 2}}
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            traverse_json(vg, spec)

    def test_hop_level_keys_are_refused_too(self, vg):
        spec = {"start": {"where": {"type": "doc"}},
                "hops": [{"via_near": {"field": "rel", "vector": [1.0, 0.0, 0.0]},
                          "via_keep": 1}]}
        with pytest.raises(ValueError, match="via_near=.*cannot come from a tool call"):
            traverse_json(vg, spec)

    def test_a_vector_inside_a_list_of_near_specs_is_found(self, vg):
        """near= takes a list, and checking only the object form would
        leave the list the way through."""
        spec = {"start": {"near": [{"field": "title", "text": "a"},
                                   {"field": "summary", "vector": [1.0, 0.0, 0.0]}],
                          "keep": 2}}
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            traverse_json(vg, spec)

    def test_the_refusal_shows_the_rewrite_for_the_field_asked_for(self, vg):
        """The message's whole job is to be copy-pasteable, so it names
        the caller's own field -- and falls back to a placeholder rather
        than an invented field name when the spec gave none, which reads
        as "you asked for summary" to someone who did not."""
        with pytest.raises(ValueError, match=r'\{"field": \'title\', "text": "\.\.\."\}'):
            traverse_json(vg, {"start": {"near": {"field": "title", "vector": [1.0]}}})
        with pytest.raises(ValueError, match=r'\{"field": \'<your field>\''):
            traverse_json(vg, {"start": {"near": {"vector": [1.0]}}})

    def test_text_is_not_refused(self, fresh_graph):
        """The whole point of the narrowing: a model CAN say what it is
        looking for, because the field embeds it with the application's
        own client. Refusing this left semantic search unreachable from
        a tool call at all."""
        g = _corpus(fresh_graph)
        g.define_vectors(nodes=[Vector("docvec", 3, embed=lambda t: [QUERY for _ in t])])
        result = traverse_json(g, {
            "start": {"where": {"type": "doc"},
                      "near": {"field": "docvec", "text": "graph databases"}, "keep": 2},
        })
        assert {n["id"] for n in result["nodes"]} == {"1", "3"}

    def test_keep_and_boost_alone_are_not_refused(self, vg):
        """They hold an integer and a property name -- nothing a model
        could invent an embedding into. Refusing them was collateral
        damage from the coarser rule, and it took `keep` with it, so a
        text query had no way to say how many to keep."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=lambda t: [[1.0, 0.0, 0.0]])])
        spec = {"start": {"near": {"field": "summary", "text": "a"}, "keep": 2,
                          "boost": {"property": "rank", "weight": 0.5}}}
        spec_to_traversal(spec)                        # parses
        from hopai.json_api import refuse_vectors
        refuse_vectors(spec, "traverse_json()")       # and is not refused

    def test_application_code_opts_in(self, fresh_graph):
        """A caller holding a REAL embedding says so, and it runs."""
        g = _corpus(fresh_graph)
        result = traverse_json(g, {
            "start": {"where": {"type": "doc"},
                      "near": {"field": "docvec", "vector": QUERY}, "keep": 2},
            "hops": [{"optional": True}],
        }, allow_vectors=True)
        assert {n["id"] for n in result["nodes"]} == {"1", "3"}

    def test_aggregate_json_refuses_the_same_keys(self, vg):
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            aggregate_json(vg, {"start": {"near": {"field": "summary",
                                                   "vector": [1.0, 0.0, 0.0]}, "keep": 1},
                                "aggregates": {"n": {"fn": "count"}}})


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

    def test_two_graphs_may_give_one_EDGE_field_different_dimensions(self, fresh_graph):
        """The same rule on the other table, which had no test at all.

        `edges` is not the node path with a different string: it has its
        own _target_for(), its own id column, and its own pass through
        migrate_vectors(). Every edge-side defect this feature has had
        was of exactly this shape -- a node path that worked and an edge
        twin nobody exercised -- so the per-graph rule is asserted here
        against the edges table directly rather than assumed by
        symmetry."""
        a = _migrated(fresh_graph)                      # edges relvec = 3
        b = fresh_graph.in_graph("other")
        b.define_vectors(edges=[Vector("relvec", 2)])
        b.migrate_vectors()

        for graph, ids in ((a, (1, 2)), (b, (3, 4))):
            graph.add_nodes([{"id": ids[0], "t": "x"}, {"id": ids[1], "t": "y"}])
            graph.add_edges([{"start_id": ids[0], "end_id": ids[1], "kind": "k"}])
        with fresh_graph.engine.connect() as conn:
            edges = dict(conn.execute(text("SELECT graph_id, id FROM edges")).all())

        assert a.set_vectors(edges=[{"id": edges[a.graph], "relvec": [1.0, 2.0, 3.0]}]) == 1
        assert b.set_vectors(edges=[{"id": edges[b.graph], "relvec": [1.0, 2.0]}]) == 1

        with fresh_graph.engine.connect() as conn:
            names = sorted(row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('edges' AS regclass) AND contype = 'c' "
                "AND conname LIKE 'ck\\_vec\\_dims\\_%relvec' ESCAPE '\\'")))
        assert len(names) == 2, f"one constraint is serving both graphs: {names}"

        for edge_id, bad in ((edges[b.graph], "{1,2,3}"), (edges[a.graph], "{1,2}")):
            with pytest.raises(Exception, match="ck_vec_dims_"), \
                    fresh_graph.engine.begin() as conn:
                conn.execute(text(
                    f"UPDATE edges SET vec_relvec = '{bad}' WHERE id = {edge_id}"))


# ---------------------------------------------------------------------
# Live: recovering a lost declaration (#53)
# ---------------------------------------------------------------------

class TestSearchRefusalsNameTheRightFixLive:
    """The symptom issue #53 was filed about: a raw UndefinedColumn
    where every other refusal here names the fix. Two DIFFERENT causes
    look identical to the registry (both read as "not usable yet"), so
    both must still read as two DIFFERENT refusals to a caller."""

    def test_undeclared_field_still_refuses_before_touching_the_database(self, fresh_graph):
        """The registry check in _defined() already catches this --
        this pins that it keeps doing so, and that the message is the
        ordinary "define_vectors() first" one, not the new
        migrate_vectors() one below."""
        with pytest.raises(ValueError,
                           match=r"needs vector fields and none are defined") as excinfo:
            fresh_graph.vector_search(Near("nosuchfield", QUERY))
        assert "migrate_vectors" not in str(excinfo.value)

    def test_declared_but_never_migrated_names_migrate_vectors(self, fresh_graph):
        """define_vectors() ran, migrate_vectors() never did -- the ONE
        case validate_nears() cannot catch from the registry alone,
        since the registry says "declared" either way. Without the fix
        this raises psycopg2.errors.UndefinedColumn instead.

        A field name used nowhere else in this file: create_schema()
        emits every vec_* column EVER attached to the shared
        Node/Edge metadata in this process (see define_vectors()'s
        docstring), so reusing "docvec" here would find the column
        already created by an earlier test and never reach the bug
        this pins."""
        fresh_graph.define_vectors(nodes=[Vector("unmigrated_solo", 3)])
        with pytest.raises(ValueError, match=r"declared for nodes in this graph but never "
                                             r"migrated.*migrate_vectors\(\)") as excinfo:
            fresh_graph.vector_search(Near("unmigrated_solo", QUERY))
        # The two refusals must stay TEXTUALLY distinguishable -- a
        # caller who already called define_vectors() must not be told
        # to call it again.
        assert "define_vectors" not in str(excinfo.value)

    def test_declared_but_never_migrated_names_migrate_vectors_for_search_many_too(
            self, fresh_graph):
        fresh_graph.define_vectors(nodes=[Vector("unmigrated_many", 3)])
        with pytest.raises(ValueError, match=r"declared for nodes in this graph but never "
                                             r"migrated.*migrate_vectors\(\)"):
            fresh_graph.vector_search_many([Near("unmigrated_many", QUERY)])

    def test_a_field_that_never_existed_is_not_reported_as_unmigrated(self, fresh_graph):
        """Multivector search over one migrated and one never-migrated
        field: the refusal must name only the field actually missing
        its column, not every field in the query."""
        g = _migrated(fresh_graph)          # docvec + titlevec migrated
        g.define_vectors(nodes=[Vector("docvec", 3), Vector("titlevec", 3),
                                Vector("unmigrated_mixed", 3)])
        with pytest.raises(ValueError, match=r"\['unmigrated_mixed'\].*never migrated"):
            g.vector_search(Near("docvec", QUERY), Near("titlevec", QUERY),
                            Near("unmigrated_mixed", QUERY))

    def test_after_migrating_the_same_query_succeeds(self, fresh_graph):
        """The refusal path (rollback + two catalog probes on the SAME
        connection the failed query used) must not leave that
        connection, or this graph's declaration, unusable afterward."""
        fresh_graph.define_vectors(nodes=[Vector("unmigrated_recovers", 3)])
        with pytest.raises(ValueError, match="migrate_vectors"):
            fresh_graph.vector_search(Near("unmigrated_recovers", QUERY))
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1, "t": "doc"}])
        fresh_graph.set_vectors(nodes=[{"id": 1, "unmigrated_recovers": QUERY}])
        assert [h["id"] for h in
                fresh_graph.vector_search(Near("unmigrated_recovers", QUERY))] == ["1"]


class TestRaiseIfUnmigratedLeavesUnrelatedErrorsAlone:
    """_raise_if_unmigrated() has two "this was not about a vec_* field
    after all, let the original error through" branches -- a different
    SQLSTATE, and a SQLSTATE match where every named column turns out
    to exist anyway. Neither is reachable through vector_search()
    itself (both mean "the UndefinedColumn was about something else"),
    so they are pinned directly: called exactly the way the except
    block above calls them, asserting they return instead of raising."""

    class _FakeOrig:
        def __init__(self, pgcode):
            self.pgcode = pgcode

    class _FakeExc(Exception):
        def __init__(self, pgcode):
            super().__init__()
            self.orig = TestRaiseIfUnmigratedLeavesUnrelatedErrorsAlone._FakeOrig(pgcode)

    def test_a_different_sqlstate_is_left_alone(self, fresh_graph):
        from hopai.vectors import _raise_if_unmigrated
        fresh_graph.define_vectors(nodes=[Vector("irrelevant_field", 3)])
        with fresh_graph.engine.connect() as conn:
            # No exception: undefined_table (42P01), not undefined_column,
            # so this diagnosis has nothing to say about it.
            _raise_if_unmigrated(fresh_graph, "nodes", ["irrelevant_field"], conn,
                                 self._FakeExc("42P01"), "vector_search()")

    def test_a_column_that_actually_exists_is_left_alone(self, fresh_graph):
        from hopai.vectors import _raise_if_unmigrated
        g = _migrated(fresh_graph)          # docvec's column genuinely exists
        with fresh_graph.engine.connect() as conn:
            _raise_if_unmigrated(g, "nodes", ["docvec"], conn,
                                 self._FakeExc("42703"), "vector_search()")


class TestLoadVectorsLive:
    def test_a_second_handle_recovers_and_can_search(self, fresh_graph):
        """The scenario issue #53 opens with: a fresh process/handle
        that never called define_vectors()."""
        _corpus(fresh_graph)
        second = Graph(fresh_graph.engine)
        assert second.vectors is None
        recovered = second.load_vectors()
        assert recovered["nodes"]["docvec"].dimensions == 3
        assert recovered["nodes"]["titlevec"].dimensions == 3
        assert recovered["edges"]["relvec"].dimensions == 3   # migrated by _corpus() too
        # Populates the handle itself, not just the return value.
        assert second.vectors["nodes"]["docvec"].dimensions == 3
        hits = second.vector_search(Near("docvec", QUERY), k=10, where={"type": "doc"})
        assert [h["id"] for h in hits] == ["1", "3", "2", "4", "5"]

    def test_embed_and_source_come_back_as_defaults_not_guesses(self, fresh_graph):
        """embed= is an application's own HTTP client and a non-default
        source= is a choice of property -- neither is stored in SQL, so
        recovering them would mean inventing one. Documented as "not
        recovering the policy"; asserted here as the two defaults
        Vector() itself would pick."""
        g = fresh_graph
        g.define_vectors(nodes=[Vector("summary", 3, source="abstract",
                                       embed=lambda t: [QUERY for _ in t])])
        g.migrate_vectors()
        second = Graph(fresh_graph.engine)
        recovered = second.load_vectors()
        field = recovered["nodes"]["summary"]
        assert field.embed is None
        assert field.source == "summary"          # not "abstract"

    def test_a_field_migrated_by_a_different_graph_is_not_recovered(self, fresh_graph):
        """vec_* columns are shared storage; the dimension CHECK is
        scoped per graph. A column present only because ANOTHER graph
        migrated it must not be guessed at for this one -- there is no
        dimension to read for a constraint that does not exist here."""
        g = _migrated(fresh_graph)
        other = fresh_graph.in_graph("elsewhere")
        other.define_vectors(nodes=[Vector("onlythere", 4)])
        other.migrate_vectors()
        recovered = g.load_vectors()
        assert "onlythere" not in recovered["nodes"]
        assert set(recovered["nodes"]) == {"docvec", "titlevec"}

    def test_two_graphs_recover_their_own_dimensions(self, fresh_graph):
        """The per-graph CHECK is what makes this safe: two graphs
        sharing one vec_docvec column, at different dimensions, must
        each recover THEIR OWN size rather than either one's."""
        _migrated(fresh_graph)                            # docvec = 3
        b = fresh_graph.in_graph("other")
        b.define_vectors(nodes=[Vector("docvec", 2)])
        b.migrate_vectors()

        a2 = Graph(fresh_graph.engine)
        assert a2.load_vectors()["nodes"]["docvec"].dimensions == 3
        b2 = Graph(fresh_graph.engine, graph="other")
        assert b2.load_vectors()["nodes"]["docvec"].dimensions == 2

    def test_columns_this_library_did_not_name_are_ignored(self, fresh_graph):
        """A vec_*-prefixed column load_vectors() cannot have created
        (an invalid field-name suffix) must not blow up the whole
        call -- it is someone else's column, not a field this library
        forgot."""
        _migrated(fresh_graph)
        with fresh_graph.engine.begin() as conn:
            conn.execute(text("ALTER TABLE nodes ADD COLUMN vec_9invalid real[]"))
        second = Graph(fresh_graph.engine)
        recovered = second.load_vectors()
        assert "9invalid" not in recovered["nodes"]
        assert set(recovered["nodes"]) == {"docvec", "titlevec"}


class TestInGraphLazyVectorsLive:
    """in_graph() no longer starts PERMANENTLY blank on vectors -- see
    its docstring. Offline behavior (test_in_graph_starts_without_vectors
    in TestVectorDeclaration) is unchanged: `.vectors` itself never
    touches the database, on ANY handle including one from in_graph(),
    so reading it before ever connecting still answers None for a lazy
    handle even when the database could prove otherwise -- only
    vector_search()/vector_search_many()/load_vectors() actually
    connect and trigger the recovery. .vectors's own docstring says so
    explicitly (issue #53's review flagged the silent-until-asked gap;
    the fix is stating it, not making a property connect)."""

    def test_in_graph_handle_searches_a_field_migrated_by_the_originating_handle(
            self, fresh_graph):
        g = _corpus(fresh_graph)
        lazy = Graph(fresh_graph.engine).in_graph(g.graph)
        assert lazy.vectors is None                # no implicit trigger yet
        hits = lazy.vector_search(Near("docvec", QUERY), k=10, where={"type": "doc"})
        assert [h["id"] for h in hits] == ["1", "3", "2", "4", "5"]
        assert lazy.vectors is not None             # the search WAS the trigger

    def test_in_graph_handle_search_many_also_carries_vectors_lazily(self, fresh_graph):
        g = _corpus(fresh_graph)
        lazy = Graph(fresh_graph.engine).in_graph(g.graph)
        results = lazy.vector_search_many([Near("docvec", QUERY)], k=10,
                                          where={"type": "doc"})
        assert [h["id"] for h in results[0]] == ["1", "3", "2", "4", "5"]

    def test_in_graph_to_a_genuinely_different_graph_still_refuses_by_name(self, fresh_graph):
        """Lazy is not magic: a graph this in_graph() handle points at
        that was NEVER migrated still refuses normally, exactly like a
        plain fresh Graph() would."""
        _migrated(fresh_graph)                      # migrates "default" only
        lazy = fresh_graph.in_graph("nobody_migrated_this_one")
        with pytest.raises(ValueError, match="needs vector fields and none are defined"):
            lazy.vector_search(Near("docvec", QUERY))

    def test_a_plain_fresh_graph_stays_immediate_not_lazy(self, fresh_graph):
        """The lazy fallback is in_graph()'s alone. A plain Graph()
        that never called define_vectors() or load_vectors() must keep
        refusing immediately -- auto-recovering for every unset
        registry would make "you forgot define_vectors()" quietly mean
        something else."""
        _corpus(fresh_graph)
        plain = Graph(fresh_graph.engine)
        assert plain._vectors_lazy is False
        with pytest.raises(ValueError, match="needs vector fields and none are defined"):
            plain.vector_search(Near("docvec", QUERY))


# ---------------------------------------------------------------------
# Live: writing and reading vectors
# ---------------------------------------------------------------------

class TestSetGetVectorsLive:
    def test_roundtrip_within_float4_precision(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        assert g.set_vectors(nodes=[{"id": 1, "docvec": [0.1, 0.2, 0.3]}]) == 1
        stored = g.get_vectors(node_ids=[1])["nodes"]["1"]
        assert stored["docvec"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
        assert stored["titlevec"] is None

    def test_one_call_writing_both_targets_counts_both(self, fresh_graph):
        """`nodes=` and `edges=` in ONE call is a supported shape -- the
        signature offers it and the whole call is one transaction -- but
        every test wrote a single target, so the count accumulated
        across chunks was never observed. `total = len(chunk)` in place
        of `+=` returns the LAST group's size and nothing objected: the
        rows land correctly, the number reported is just wrong."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}, {"id": 2, "type": "doc"},
                     {"id": 3, "type": "doc"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "k"},
                     {"start_id": 2, "end_id": 3, "kind": "k"}])
        with g.engine.connect() as conn:
            edge_ids = [row[0] for row in conn.execute(text("SELECT id FROM edges ORDER BY id"))]

        written = g.set_vectors(
            nodes=[{"id": i, "docvec": [1.0, 0.0, 0.0]} for i in (1, 2, 3)],
            edges=[{"id": e, "relvec": [0.0, 1.0, 0.0]} for e in edge_ids])
        assert written == 5                      # 3 nodes + 2 edges, not 2
        # ...and both groups really landed, so the count is not right by
        # accident while one target was skipped.
        assert len(g.get_vectors(node_ids=[1, 2, 3])["nodes"]) == 3
        assert len(g.get_vectors(edge_ids=edge_ids)["edges"]) == 2

    def test_string_ids_from_results_are_accepted_back(self, fresh_graph):
        """Traversal and search hand back string ids; making the caller
        int() them before set_vectors would be a first-use papercut."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 7, "type": "doc"}])
        assert g.set_vectors(nodes=[{"id": "7", "docvec": [1.0, 0.0, 0.0]}]) == 1
        assert "7" in g.get_vectors(node_ids=["7"])["nodes"]

    def test_none_clears_a_vector(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        g.set_vectors(nodes=[{"id": 1, "docvec": None}])
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] is None

    def test_edge_vectors_roundtrip(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "t": "a"}, {"id": 2, "t": "b"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
        with g.engine.connect() as conn:
            edge_id = conn.execute(text("SELECT id FROM edges")).scalar()
        assert g.set_vectors(edges=[{"id": edge_id, "relvec": [0.5, 0.5, 0.0]}]) == 1
        stored = g.get_vectors(edge_ids=[edge_id])["edges"][str(edge_id)]
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
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] is None

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
        only = g.get_vectors(node_ids=[1], node_fields=["docvec"])["nodes"]["1"]
        assert set(only) == {"docvec"}
        with pytest.raises(ValueError, match="no vector field 'nope'"):
            g.get_vectors(node_ids=[1], node_fields=["nope"])

    def test_absent_ids_are_simply_absent(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        assert g.get_vectors(node_ids=[1, 42])["nodes"].keys() == {"1"}

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


def needs_jq():
    """The jq binding is an optional extra (`hopai[rerankers]`), and
    `document_from` is evaluated with it -- so the reranking tests skip
    without it, exactly as tests/test_rerankers.py does."""
    return pytest.importorskip("jq")


def _searchable(fresh_graph) -> Graph:
    """_corpus(), with docvec declaring an embedder.

    Reranking needs the query as TEXT, and the text becomes a vector
    through the FIELD's own embed= -- so the dense stage and the rerank
    stage see the same query. Every text here embeds to QUERY, which
    keeps the similarity ordering the hand-checked one above."""
    g = _corpus(fresh_graph)
    g.define_vectors(
        nodes=[Vector("docvec", 3, embed=lambda texts: [list(QUERY) for _ in texts]),
               Vector("titlevec", 3)],
        edges=[Vector("relvec", 3)])
    return g


def _by_name(order: list, calls=None):
    """A reranker client whose score is `order`'s reverse index of the
    document -- so the ranking a test expects is written in the test,
    not left to a provider. `calls` collects (query, documents) per
    call, which is how "one call per step" is asserted."""
    def client(query, documents):
        if calls is not None:
            calls.append((query, list(documents)))
        return [float(len(order) - order.index(d)) if d in order else -1.0
                for d in documents]
    return client


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
        # similarities is keyed by FIELD NAME, not by sim_0/sim_1 -- and
        # each field's OWN cosine, not its weighted contribution to the
        # combined score (issue #54).
        assert hits[0]["similarities"] == pytest.approx({"docvec": 1.0, "titlevec": 0.0}, abs=1e-6)
        assert hits[1]["similarities"] == pytest.approx({"docvec": 0.0, "titlevec": 1.0}, abs=1e-6)
        # No boost= was passed -- the key is still present, empty, so a
        # caller never has to check whether one was given before reading it.
        assert hits[0]["boosts"] == {}

    def test_single_near_search_still_returns_a_similarities_dict(self, fresh_graph):
        """Shape stability (issue #54): a caller writing
        `hit["similarities"][field]` should never have to branch on how
        many Near clauses a search used."""
        g = _corpus(fresh_graph)
        hits = g.vector_search(Near("docvec", QUERY), k=1, where={"type": "doc"})
        assert hits[0]["similarities"] == pytest.approx({"docvec": 1.0}, abs=1e-6)
        assert hits[0]["similarities"]["docvec"] == hits[0]["similarity"]

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
        # The COMBINED score above treats doc-only's missing titlevec as
        # 0 (missing="zero"'s whole point). The PER-FIELD report must
        # keep saying "missing" honestly -- None, not 0.0 -- or a field
        # with no vector would look identical to one that matched
        # poorly, which is exactly the ambiguity issue #54 exists to
        # remove.
        by_id = {h["id"]: h for h in lenient}
        assert by_id["2"]["similarities"]["titlevec"] is None
        assert by_id["2"]["similarities"]["docvec"] == pytest.approx(1.0, abs=1e-6)
        assert by_id["1"]["similarities"] == pytest.approx(
            {"docvec": 0.6, "titlevec": 1.0}, abs=1e-6)

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
        }, allow_vectors=True)
        assert json.loads(json.dumps(result)) == result
        assert [h["id"] for h in result["results"]] == ["1", "3"]
        # json_api.py is a thin front end with no query logic of its
        # own (CLAUDE.md) -- it calls graph.vector_search() directly, so
        # similarities/boosts (issue #54) should already be here with no
        # extra plumbing. A None similarity (had min_similarity excluded
        # a missing field here) has to survive the JSON round trip too,
        # which the assertion above already covers structurally.
        assert result["results"][0]["similarities"] == {"docvec": pytest.approx(1.0, abs=1e-6)}
        assert result["results"][0]["boosts"] == {}

    def test_vector_search_json_reports_a_missing_field_as_json_null(self, fresh_graph):
        """None -> JSON null -> None has to survive round-tripping
        through an actual HTTP-shaped call, not just Python equality --
        the earlier test's corpus never exercises missing="zero"."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "doc-only"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        result = vector_search_json(g, {
            "near": [{"field": "docvec", "vector": QUERY},
                    {"field": "titlevec", "vector": QUERY, "missing": "zero"}],
            "k": 1,
        }, allow_vectors=True)
        dumped = json.loads(json.dumps(result))
        assert dumped == result
        assert dumped["results"][0]["similarities"]["titlevec"] is None
        assert dumped["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_vector_search_json_k_actually_truncates(self, fresh_graph):
        """`k` in the spec has to REACH the search. Pairing it with a
        min_similarity that already limits the result hides a misread
        key behind the threshold's answer -- a mutant that looked up
        "K" survived exactly that way."""
        g = _corpus(fresh_graph)
        spec = {"near": {"field": "docvec", "vector": QUERY}, "where": {"type": "doc"}}
        run = lambda s: vector_search_json(g, s, allow_vectors=True)  # noqa: E731
        assert len(run({**spec, "k": 1})["results"]) == 1
        assert len(run({**spec, "k": 3})["results"]) == 3
        # ...and the documented default when the key is absent.
        assert len(run(spec)["results"]) == 5

    def test_a_model_can_search_by_meaning_with_no_floats_at_all(self, fresh_graph):
        """VECTOR_SEARCH_TOOL_SCHEMA's whole promise, end to end: the
        spec a tool-calling model can actually produce -- a field name
        and words -- runs and ranks. Nothing here opts into vectors."""
        g = _corpus(fresh_graph)
        g.define_vectors(nodes=[Vector("docvec", 3, embed=lambda t: [QUERY for _ in t])])
        result = vector_search_json(g, {
            "near": {"field": "docvec", "text": "how do nodes agree?"},
            "k": 2, "where": {"type": "doc"},
        })
        assert json.loads(json.dumps(result)) == result
        assert [h["id"] for h in result["results"]] == ["1", "3"]

    def test_a_field_that_cannot_embed_says_so_through_json_too(self, fresh_graph):
        """A model that sends text to a field with no embedder gets the
        declaration to change, not an empty result set."""
        g = _corpus(fresh_graph)
        with pytest.raises(ValueError, match="declares no embedder"):
            vector_search_json(g, {"near": {"field": "docvec", "text": "anything"}})

    # -- reranking: the third stage ------------------------------------

    def test_rerank_reorders_and_reports_both_scores(self, fresh_graph):
        """The whole point, and the shape of the answer. Without this,
        `rerank=` could be accepted and quietly ignored: the hits would
        still come back, in similarity order, with nothing to see."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        calls = []
        # "opposite" is the WORST cosine (-1.0) and the best document --
        # only a real rerank can put it first.
        rerank = Rerank(_by_name(["opposite", "exact"], calls),
                        document_from=".properties.name", candidates=5)
        hits = g.vector_search(Near("docvec", text="how do nodes agree?"), k=2,
                               where={"type": "doc"}, rerank=rerank)
        assert [h["id"] for h in hits] == ["5", "1"]
        assert [h["rerank_score"] for h in hits] == [2.0, 1.0]
        # `similarity` KEEPS the retrieval stage's number rather than
        # being overwritten, so a caller can see what the reranker moved.
        assert [h["similarity"] for h in hits] == pytest.approx([-1.0, 1.0], abs=1e-6)
        # One provider call, with the caller's own text as the query.
        assert len(calls) == 1 and calls[0][0] == "how do nodes agree?"

    def test_candidates_bounds_the_input_and_k_bounds_the_output(self, fresh_graph):
        """Two DIFFERENT bounds that never overlap -- LanceDB's `top_n`
        beside `.limit()` is the trap this avoids. `candidates` decides
        how many rows the SQL fetches (and how many documents are
        billed); `k` decides how many come back."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        calls = []
        rerank = Rerank(_by_name([], calls), document_from=".properties.name", candidates=3)
        hits = g.vector_search(Near("docvec", text="q"), k=2, where={"type": "doc"},
                               rerank=rerank)
        assert len(calls[0][1]) == 3          # candidates reached the reranker
        assert len(hits) == 2                 # k truncated afterwards

    def test_no_rerank_leaves_the_hit_shape_exactly_as_it_was(self, fresh_graph):
        """Additive only: without rerank= there must be no rerank_score
        key and no reordering, or every existing caller's result shape
        changed the day this landed."""
        g = _corpus(fresh_graph)
        hits = g.vector_search(Near("docvec", QUERY), k=3, where={"type": "doc"})
        assert all("rerank_score" not in hit for hit in hits)
        assert [h["id"] for h in hits] == ["1", "3", "2"]

    def test_candidates_below_k_is_refused_by_the_call_too(self, fresh_graph):
        """hop.py holds this rule for Start/Hop; vector_search() reaches
        the SAME validator rather than restating it, so the two can
        never disagree about whether reranking 5 to keep 10 is legal."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        with pytest.raises(ValueError, match="reranks fewer candidates than k=10 keeps"):
            g.vector_search(Near("docvec", text="q"), k=10,
                            rerank=Rerank(_by_name([]), document_from=".properties.name",
                                          candidates=5))

    def test_a_raw_vector_query_with_rerank_is_refused_before_any_sql(self, fresh_graph):
        """A reranker reads the query. A list of floats is not something
        it can read, and no implementation makes it one -- so this
        refuses rather than growing a second way to supply the query,
        which could disagree with the first silently."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        with pytest.raises(ValueError, match="rerank= needs the query as TEXT"):
            g.vector_search(Near("docvec", QUERY), k=2,
                            rerank=Rerank(_by_name([]), document_from=".properties.name"))

    def test_two_different_texts_have_no_single_query_to_score_with(self, fresh_graph):
        """A multivector query is ONE question asked of several fields.
        Two texts give the reranker no single query, and picking the
        first would score every document against a question the caller
        never asked -- a plausible, confidently wrong ranking."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        g.define_vectors(nodes=[
            Vector("docvec", 3, embed=lambda t: [list(QUERY) for _ in t]),
            Vector("titlevec", 3, embed=lambda t: [list(QUERY) for _ in t])])
        with pytest.raises(ValueError, match="carry 2 different texts"):
            g.vector_search(Near("docvec", text="raft"), Near("titlevec", text="paxos"), k=2,
                            rerank=Rerank(_by_name([]), document_from=".properties.name"))

    def test_the_provider_is_called_after_the_round_trip_closes(self, fresh_graph):
        """The rule set_vectors() already keeps, on the read side: an
        HTTP round trip inside an open transaction holds a snapshot for
        its whole duration. Asserted by making the client itself LOOK --
        a connection still checked out here means the provider call is
        sitting on top of one."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        held = [0]
        timeline = []

        @event.listens_for(g.engine, "checkout")
        def out(*args):                                   # noqa: ARG001
            held[0] += 1

        @event.listens_for(g.engine, "checkin")
        def back(*args):                                  # noqa: ARG001
            held[0] -= 1

        def client(query, documents):
            timeline.append(held[0])
            return [0.0] * len(documents)

        try:
            g.vector_search(Near("docvec", text="q"), k=2,
                            rerank=Rerank(client, document_from=".properties.name",
                                          candidates=3))
        finally:
            event.remove(g.engine, "checkout", out)
            event.remove(g.engine, "checkin", back)
        assert timeline == [0], "the provider was called with a connection still open"

    def test_an_empty_result_never_reaches_the_provider(self, fresh_graph):
        """No documents, no call: an empty rerank request is billed by
        some providers and refused by others, and neither is an answer."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        calls = []
        g.vector_search(Near("docvec", text="q", min_similarity=0.999), k=2,
                        where={"type": "nothing-matches-this"},
                        rerank=Rerank(_by_name([], calls), document_from=".properties.name"))
        assert calls == []

    def test_rerank_with_no_near_at_all_is_refused(self, fresh_graph):
        """hop.py's "nothing to reorder" rule sees `near=None`, and
        vector_search() spells the same absence `near=[]` -- which is
        not None, so it needs saying here too. A reranker REORDERS a
        candidate list; it cannot produce one."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        with pytest.raises(ValueError, match=r"rerank= reorders the candidates near= ranks"):
            g.vector_search(k=2, rerank=Rerank(_by_name([]),
                                               document_from=".properties.name"))


# ---------------------------------------------------------------------
# Live: similarity inside a traversal
# ---------------------------------------------------------------------

class TestTraversalNearLive:
    def test_similar_seeds_walk_and_dead_ends_still_prune(self, fresh_graph):
        """Start(near=, keep=) seeds the walk with the top-k similar --
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
        result = g.traverse(Start(near=Near("docvec", QUERY), keep=2),
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
                            Hop(via={"kind": "knows"}, near=Near("docvec", QUERY), keep=1))
        assert {n["id"] for n in result.nodes} == {"1", "2"}
        assert len(result.edges) == 1

    def test_aggregate_over_the_k_most_similar(self, fresh_graph):
        g = _corpus(fresh_graph)
        counted = g.aggregate(Start(near=Near("docvec", QUERY), keep=3, where={"type": "doc"}),
                              aggregates={"n": Count()})
        assert counted == {"n": 3}

    # -- step-wise reranking -------------------------------------------
    #
    # A traversal compiles to ONE recursive CTE and a reranker is a
    # network call in the middle of the walk, so a reranked traversal
    # probes, reranks, then RE-RUNS THE ORDINARY TRAVERSAL with each
    # step's survivors pinned. Everything below is about that final
    # re-run being an ordinary traversal: if it ever became a stitching
    # of partial results, fan-in, multi-hop edge reconstruction and
    # dead-end pruning would each have to be re-derived by hand, which
    # is exactly what the invariants in tests/test_hopai.py forbid.

    def _fan_in_graph(self, fresh_graph) -> Graph:
        """Two parents feeding one intermediate, a second reachable node
        for the reranker to choose between, and a dead end whose only
        edge is the wrong kind.

            p1, p2 -> shared -> tail
            p1     -> other
            p1     -> deadend   (kind: wrong)
        """
        g = _searchable(fresh_graph)
        g.add_nodes([{"id": 10, "type": "seed", "name": "p1"},
                     {"id": 11, "type": "seed", "name": "p2"},
                     {"id": 12, "name": "shared"},
                     {"id": 13, "name": "other"},
                     {"id": 14, "name": "deadend"},
                     {"id": 15, "name": "tail"}])
        g.set_vectors(nodes=[{"id": 10, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 11, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 12, "docvec": [0.6, 0.8, 0.0]},
                             {"id": 13, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 14, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 15, "docvec": [1.0, 0.0, 0.0]}])
        g.add_edges([{"start_id": 10, "end_id": 12, "kind": "cites"},
                     {"start_id": 11, "end_id": 12, "kind": "cites"},
                     {"start_id": 10, "end_id": 13, "kind": "cites"},
                     {"start_id": 10, "end_id": 14, "kind": "wrong"},
                     {"start_id": 12, "end_id": 15, "kind": "cites"}])
        return g

    def test_a_hop_rerank_prunes_and_fan_in_is_still_preserved(self, fresh_graph):
        """The invariant test_fan_in_both_parents_preserved protects,
        under a rerank: `shared` is the WORST cosine and the best
        document, so only a real rerank keeps it -- and once it is kept,
        BOTH parents' edges must still be reported. A node survives or
        is dropped as a UNIT, which is what makes that true."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        result = g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="fan in"), keep=1,
                rerank=Rerank(_by_name(["shared"]), document_from=".properties.name",
                              candidates=5)))
        assert {n["id"] for n in result.nodes} == {"10", "11", "12"}
        # Both in-edges, from both parents -- the fan-in the per-hop path
        # tracking exists to keep.
        assert {(e["start_id"], e["end_id"]) for e in result.edges} \
            == {("10", "12"), ("11", "12")}

    def test_a_reranked_hop_still_prunes_dead_ends_and_wrong_edges(self, fresh_graph):
        """Reported nodes derive from the EDGES found, never from a
        surviving id list. `deadend` is reachable only by a `wrong` edge,
        so however highly the reranker scores it, the hop never reaches
        it and it never appears."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        result = g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=2,
                rerank=Rerank(_by_name(["deadend", "other", "shared"]),
                              document_from=".properties.name", candidates=5)))
        assert "14" not in {n["id"] for n in result.nodes}

    def test_multi_hop_edge_reconstruction_survives_a_rerank(self, fresh_graph):
        """A hop spanning several real edges must report ALL of them,
        not one fabricated edge between the endpoints. The final
        traversal is an ordinary one, so `hop_edges_i` still unnests
        `local_edges` -- this is what would notice if pinning ever
        started reporting edges from the pinned id list instead."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        result = g.traverse(
            Start(where={"name": "p1"}),
            Hop(via={"kind": "cites"}, hops=(2, 2), near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(_by_name(["tail"]), document_from=".properties.name",
                              candidates=5)))
        # p1 -> shared -> tail is two real edges, both reported.
        assert {(e["start_id"], e["end_id"]) for e in result.edges} \
            == {("10", "12"), ("12", "15")}

    def test_a_start_rerank_prunes_the_seed_before_any_hop_walks(self, fresh_graph):
        """The seed set is a ranked set too, so `rerank=` belongs there
        as well -- and pruning it changes what the whole chain reaches,
        which is the point rather than a side effect."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        result = g.traverse(
            Start(where={"type": "seed"}, near=Near("docvec", text="q"), keep=1,
                  rerank=Rerank(_by_name(["p2"]), document_from=".properties.name",
                                candidates=5)),
            Hop(via={"kind": "cites"}))
        # p2's only edge goes to `shared`; p1 never seeded, so `other`
        # is unreachable.
        assert {n["id"] for n in result.nodes} == {"11", "12"}

    def test_a_hop_document_may_read_the_paths_that_reached_it(self, fresh_graph):
        """The capability a flat reranker structurally cannot offer: at
        a hop a candidate is a node PLUS how it was reached. `.paths` is
        every route, canonically ordered, with the candidate itself
        dropped -- so `.[-1]` is the immediate parent ("cited by") and
        not the node restating its own title."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        seen = []

        def client(query, documents):
            seen.extend(documents)
            return [0.0] * len(documents)

        g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(client, candidates=5,
                              document_from='.properties.name + " <- " '
                                            '+ (.paths | map(.[-1].properties.name) '
                                            '| join(","))')))
        assert sorted(seen) == ["other <- p1", "shared <- p1,p2"]

    def test_paths_are_hydrated_only_when_the_filter_reads_them(self, fresh_graph):
        """Path context costs a wider probe. A filter reading only the
        node's own properties must not pay for it -- and must not be
        handed a `paths` key it never asked about, which would change
        what a `.` filter or a `keys` produces."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)

        class Watching(Rerank):
            """A Rerank that reports the SHAPE of what it was handed."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.shapes = []

            def build_documents(self, candidates, **kwargs):
                self.shapes.extend(sorted(candidate) for candidate in candidates)
                return super().build_documents(candidates, **kwargs)

        def run(document_from):
            rerank = Watching(_by_name([]), document_from=document_from, candidates=5)
            g.traverse(Start(where={"type": "seed"}),
                       Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                           rerank=rerank))
            return rerank.shapes

        plain = run(".properties.name")
        assert plain and all(shape == ["id", "properties"] for shape in plain)
        reading = run('.paths | map(.[-1].properties.name) | join(",")')
        assert reading and all(shape == ["id", "paths", "properties"] for shape in reading)

    def test_per_path_scores_a_node_by_its_BEST_route(self, fresh_graph):
        """per_path=True is one document per (node, path) and the node's
        score is the MAX over its paths -- not the sum, not the mean --
        so one strong route is enough to keep it. That is the same "any
        valid parent counts" semantics the fan-in invariant protects; a
        mean would let a node's weak second parent vote it out."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        seen = []

        def client(query, documents):
            seen.extend(documents)
            # `shared` via p1 is terrible, via p2 is the best document
            # here. Under max it survives; under mean or sum of these
            # numbers `other` would win instead.
            return [{"shared <- p1": -10.0, "shared <- p2": 5.0}.get(d, 1.0)
                    for d in documents]

        result = g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(client, candidates=5, per_path=True,
                              document_from='.properties.name + " <- " '
                                            '+ (.paths | map(.[-1].properties.name) '
                                            '| join(","))')))
        # One document per (node, path): `shared` was scored twice.
        assert sorted(seen) == ["other <- p1", "shared <- p1", "shared <- p2"]
        assert {n["id"] for n in result.nodes} == {"10", "11", "12"}

    def test_max_paths_caps_what_one_document_may_quote(self, fresh_graph):
        """A high fan-in node can be reached hundreds of ways, and a
        document quoting all of them blows the provider's token limit --
        as a hard error if you are lucky, as a silent server-side
        truncation if you are not. The cap is visible and deterministic,
        which is why the paths are canonically ordered first."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        seen = []

        def client(query, documents):
            seen.extend(documents)
            return [0.0] * len(documents)

        g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(client, candidates=5, max_paths=1,
                              document_from='.properties.name + " <- " '
                                            '+ (.paths | map(.[-1].properties.name) '
                                            '| join(","))')))
        assert sorted(seen) == ["other <- p1", "shared <- p1"]

    def test_reading_paths_at_a_start_is_refused_before_any_sql(self, fresh_graph):
        """A seed has no provenance -- nothing reached it -- so `.paths`
        there would be `null` and every document would quietly change
        shape. "You have to know that .paths is hop-only" is the kind of
        gap this library treats as a defect, so it refuses and names the
        rewrite."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        with pytest.raises(ValueError, match=r"reads \.paths, but a seed has no provenance"):
            g.traverse(Start(where={"type": "seed"}, near=Near("docvec", text="q"), keep=1,
                             rerank=Rerank(_by_name([]), candidates=5,
                                           document_from='.paths | map(.[-1].properties.name) '
                                                         '| join(",")')),
                       Hop(via={"kind": "cites"}))

    def test_per_path_at_a_start_is_refused(self, fresh_graph):
        """per_path=True means one provider call per (node, path), and a
        SEED has no paths -- nothing reached it. So at a Start the flag
        is not merely cheap, it IS the default under another name: the
        caller asked for per-path scoring and quietly got per-node
        scoring, which is "a constraint the options discard is not no
        constraint". It refuses beside the `.paths`-at-a-Start refusal
        and says where the mode belongs. Without this the flag was
        accepted and dropped, with nothing said and a bill either way."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        calls = []

        def client(query, documents):
            calls.append(documents)
            return [0.0] * len(documents)

        with pytest.raises(ValueError, match="per_path=True.*a seed has no provenance"):
            g.traverse(
                Start(where={"type": "seed"}, near=Near("docvec", text="q"), keep=1,
                      rerank=Rerank(client, document_from=".properties.name", candidates=5,
                                    per_path=True)),
                Hop(via={"kind": "cites"}))
        assert not calls, "refused before a document was built, let alone billed"

    def test_a_filter_the_grammar_rejects_is_refused_as_a_filter(self, fresh_graph):
        """A document_from outside the safe subset must be refused with
        `document_from` as the owner and the offending construct named
        -- NOT with the ".paths at a Start" message, which would blame
        something the caller never wrote. Deciding "does it read paths"
        by parsing means an unparseable filter has no answer, and the
        one it gets must be the one that does not mislead."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        with pytest.raises(Exception, match="document_from") as raised:
            g.traverse(Start(where={"type": "seed"}, near=Near("docvec", text="q"), keep=1,
                             rerank=Rerank(_by_name([]), candidates=5,
                                           document_from="def f: .properties.name; f")),
                       Hop(via={"kind": "cites"}))
        assert "paths" not in str(raised.value)

    def test_a_step_whose_probe_finds_nothing_never_calls_the_provider(self, fresh_graph):
        """An empty candidate set pins nothing and costs no provider
        call -- and the traversal that follows is simply empty, rather
        than falling back to the unpruned walk, which would be a
        different answer with no signal."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        calls = []
        result = g.traverse(
            Start(where={"type": "nothing-matches-this"}, near=Near("docvec", text="q"),
                  keep=1, rerank=Rerank(_by_name([], calls), document_from=".properties.name",
                                        candidates=5)),
            Hop(via={"kind": "cites"}))
        assert calls == [] and result.nodes == [] and result.edges == []

    def test_rerank_scores_never_reach_the_traversal_result(self, fresh_graph):
        """A traversal returns a SUBGRAPH, not a ranking -- Start's and
        Hop's docstrings both say so, and similarity scores have never
        survived into it. Reranking gets no exception: `result.nodes`
        keeps its exact shape, and vector_search() is where scores
        live."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        result = g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(_by_name(["shared"]), document_from=".properties.name",
                              candidates=5)))
        assert all(sorted(node) == ["id", "properties"] for node in result.nodes)

    def test_a_reranked_traversal_costs_one_plus_two_per_reranked_step(self, fresh_graph):
        """Rule 6: "probably fine" is not a measurement. A reranked
        traversal is a probe and a hydration per reranked step, then the
        ordinary traversal -- whose own hydration SELECTs are the two
        it has always issued. Pinned here so a change that doubles the
        round trips is visible in a diff rather than in a latency
        graph."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        statements = []

        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):       # noqa: ARG001
            statements.append(statement)

        def rerank(order):
            return Rerank(_by_name(order), document_from=".properties.name", candidates=5)

        try:
            plain = g.traverse(Start(where={"type": "seed"}), Hop(via={"kind": "cites"}))
            baseline = len(statements)
            statements.clear()
            g.traverse(Start(where={"type": "seed"}, near=Near("docvec", text="q"), keep=2,
                             rerank=rerank(["p1", "p2"])),
                       Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                           rerank=rerank(["shared"])))
            reranked = len(statements)
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        assert plain.nodes                       # the baseline really ran
        assert baseline == 3                     # the walk plus its two hydrations
        assert reranked == baseline + 2 * 2      # + (probe, hydration) per reranked step

    def test_aggregates_run_over_the_reranked_survivors(self, fresh_graph):
        """aggregate() shares _walk_matches(), so pinning reaches it for
        free -- and that is the right answer rather than an accident:
        aggregates run over the LAST step's matched nodes, and after a
        rerank those nodes ARE the survivors, exactly as they are after
        a `keep=`."""
        needs_jq()
        from hopai import Rerank

        g = self._fan_in_graph(fresh_graph)
        counted = g.aggregate(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(_by_name(["shared"]), document_from=".properties.name",
                              candidates=5)),
            aggregates={"n": Count()})
        assert counted == {"n": 1}

    def test_a_traversal_with_no_rerank_never_probes(self, fresh_graph):
        """The negative half: reranking is opt-in per step, so a chain
        with none must not pay a probe, a hydration or a provider call
        -- and must reach build_query() with pins=None."""
        g = self._fan_in_graph(fresh_graph)
        statements = []

        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):       # noqa: ARG001
            statements.append(statement)
        try:
            g.traverse(Start(where={"type": "seed"}),
                       Hop(via={"kind": "cites"}, near=Near("docvec", QUERY), keep=1))
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        assert len(statements) == 3


class TestProbeHydrationIsProjected:
    """Issue #77: a reranked step's probe fetches only the `properties`
    keys `document_from` can read, instead of the whole JSONB column --
    the same insight #76 applied to what jq is handed, one layer
    earlier, at the SQL that hydrates the candidates
    (`Graph._rerank_properties()`).

    THE INSTRUMENT is what `_rerank_properties()` actually returns, not
    the compiled SQL text: the projected query binds each property key
    as a parameter (`nodes.properties[%(properties_1)s]`, not a literal
    `'title'`), so grepping the statement string would only prove a
    bind placeholder exists, not which key it carries or what came
    back. A node carrying a 100KB `junk` property beside the one the
    filter reads makes the two cases impossible to confuse either way:
    present in the fetched dict is the whole-column fallback, absent is
    the projection -- the same "byte count, not a threshold" instrument
    `TestDocumentBuildingStaysOffTheLoop` uses one layer down, adapted
    to what crosses the wire rather than what crosses into jq."""

    JUNK = "x" * 100_000

    def _graph(self, fresh_graph) -> Graph:
        # ids past _corpus()'s 1-7, the same convention
        # TestTraversalNearLive._fan_in_graph() above uses.
        g = _searchable(fresh_graph)
        g.add_nodes([
            {"id": 20, "type": "seed", "name": "p1"},
            {"id": 21, "name": "shared", "title": "Raft", "meta": {"title": "nested"},
             "junk": self.JUNK},
        ])
        g.set_vectors(nodes=[{"id": 20, "docvec": QUERY}, {"id": 21, "docvec": QUERY}])
        g.add_edges([{"start_id": 20, "end_id": 21, "kind": "cites"}])
        return g

    def _fetched(self, fresh_graph, monkeypatch, document_from):
        """One reranked traversal, with `Graph._rerank_properties()`
        spied on rather than replaced -- the real query still runs, so
        this proves what Postgres actually sent back, not what the
        projection function alone would compute."""
        needs_jq()
        from hopai import Rerank
        import hopai.core as core_module

        g = self._graph(fresh_graph)
        calls = []
        original = core_module.Graph._rerank_properties

        def spy(self, session, node_id_col, node_ids, keys=None):
            result = original(self, session, node_id_col, node_ids, keys=keys)
            calls.append((keys, result))
            return result

        monkeypatch.setattr(core_module.Graph, "_rerank_properties", spy)
        g.traverse(
            Start(where={"type": "seed"}),
            Hop(via={"kind": "cites"}, near=Near("docvec", text="q"), keep=1,
                rerank=Rerank(_by_name(["shared"]), document_from=document_from, candidates=5)))
        assert calls, "the probe never hydrated -- nothing was measured"
        keys, fetched = calls[-1]
        return keys, fetched["21"]

    def test_a_single_level_filter_prunes_the_fetch(self, fresh_graph, monkeypatch):
        """The common case, and the one the issue measured: reading a
        title never sees the row's other properties, because Postgres
        never sent them -- `properties -> 'title'` rather than
        `properties`."""
        keys, properties = self._fetched(fresh_graph, monkeypatch, ".properties.title")
        assert keys == frozenset({"title"})
        assert properties == {"title": "Raft"}

    def test_a_filter_reading_no_properties_fetches_none(self, fresh_graph, monkeypatch):
        """.id never touches the properties column -- the empty
        projection still runs one query (a candidate's own id still has
        to be confirmed against the graph), just for zero keys and zero
        bytes of `properties` back."""
        keys, properties = self._fetched(fresh_graph, monkeypatch, ".id")
        assert keys == frozenset()
        assert properties == {}

    def test_a_dotted_path_past_one_level_falls_back_to_the_whole_column(
            self, fresh_graph, monkeypatch):
        """`_pruned()`'s non-object-intermediate and ambiguous-dotted-
        key rulings are made per ROW, at evaluation time -- this SQL is
        compiled once, before any row exists, so it cannot make them.
        `.properties.meta.title` -- one level past what
        `_rerank_property_projection()` will prune -- keeps today's
        whole-column fetch rather than guess, unchanged behaviour."""
        keys, properties = self._fetched(fresh_graph, monkeypatch, ".properties.meta.title")
        assert keys is None
        assert properties.get("junk") == self.JUNK

    def test_a_filter_reading_the_whole_row_falls_back_too(self, fresh_graph, monkeypatch):
        """A bare `.` is `_projection_paths()`'s own _WHOLE_ROW answer
        (reused here rather than restated), and the reason is the same
        one that answer already carries: there is no reported set to
        trust."""
        keys, properties = self._fetched(fresh_graph, monkeypatch, ". | tostring")
        assert keys is None
        assert properties.get("junk") == self.JUNK


# ---------------------------------------------------------------------
# Live: dropping
# ---------------------------------------------------------------------

class TestDropVectorsLive:
    def test_edge_fields_reach_the_drop_too(self, fresh_graph):
        """Every other case here passes only node_fields, so blanking
        the edge_fields argument on the way through Graph.drop_vectors()
        survived the whole class -- drop_vectors(edges=...) would have
        silently dropped nothing and reported success, which is the same
        shape as the drop_constraints(edges=[...]) gap this branch
        already fixed."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}, {"id": 2}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"}])
        with g.engine.connect() as conn:
            edge_id = conn.execute(text("SELECT id FROM edges")).scalar()
        g.set_vectors(edges=[{"id": edge_id, "relvec": [1.0, 0.0, 0.0]}])

        # `for entry in names or ()` -- a blanked edge_fields drops
        # NOTHING and still returns, so both halves are asserted: the
        # report, and the values it claims to have cleared.
        assert g.drop_vectors(edge_fields=["relvec"]) == ["edges.vec_relvec"]
        with g.engine.connect() as conn:
            remaining = conn.execute(text(
                "SELECT count(*) FROM edges WHERE vec_relvec IS NOT NULL")).scalar()
        assert remaining == 0

    def test_drop_nulls_this_graph_and_removes_its_constraint(self, fresh_graph):
        g = _corpus(fresh_graph)
        dropped = g.drop_vectors(node_fields=["docvec"])
        assert dropped == ["nodes.vec_docvec"]
        with g.engine.connect() as conn:
            remaining = conn.execute(text(
                "SELECT count(*) FROM nodes WHERE vec_docvec IS NOT NULL")).scalar()
            names = {row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('nodes' AS regclass) AND contype = 'c'"))}
        assert remaining == 0
        assert "ck_vec_dims_default_nodes_docvec" not in names
        # The DECLARATION survives: drop_constraints() does not mutate
        # declarations either, and migrate_vectors()'s dimension-change
        # recipe ("drop, then migrate") only works if it does not.
        assert "docvec" in g.vectors["nodes"]

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
        g.drop_vectors(node_fields=["docvec"])
        kept = other.get_vectors(node_ids=[10])["nodes"]["10"]["docvec"]
        assert kept == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)

    def test_drop_of_a_never_migrated_field_is_ignored(self, fresh_graph):
        assert fresh_graph.drop_vectors(node_fields=["ghost"]) == ["nodes.vec_ghost"]

    def test_drop_of_an_undeclared_field_alongside_declared_ones(self, fresh_graph):
        """Dropping a name this handle never declared is a no-op, and
        that includes the case where the handle DOES declare others --
        the registry cleanup must not assume the field is in it. The
        no-registry test above cannot catch this: it never reaches the
        cleanup at all."""
        g = _migrated(fresh_graph)
        assert set(g.vectors["nodes"]) == {"docvec", "titlevec"}
        assert g.drop_vectors(node_fields=["ghost"]) == ["nodes.vec_ghost"]
        assert set(g.vectors["nodes"]) == {"docvec", "titlevec"}

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
        assert undeclared.drop_vectors(node_fields=["docvec"]) == ["nodes.vec_docvec"]
        with fresh_graph.engine.connect() as conn:
            assert conn.execute(text(
                "SELECT count(*) FROM nodes WHERE vec_docvec IS NOT NULL")).scalar() == 0


# ---------------------------------------------------------------------
# Live: the seam with schema inference
# ---------------------------------------------------------------------

class TestVectorsAreInvisibleToSchemaInference:
    # sample_percent=None reads every row; a percentage takes a different
    # path to the same jsonb_each. Inference is the ONE feature that reads
    # the database to decide what a schema contains, so every path through
    # it is a place a vec_* column could start appearing.
    @pytest.mark.parametrize("sample_percent", [None, 100.0])
    def test_inferred_schema_never_reports_a_vector_field(self, fresh_graph, sample_percent):
        """`vec_*` are real columns, not properties -- so a schema
        inferred from the same rows must describe the properties only.
        The two features landed independently; this pins the seam, since
        an inference that ever reached for columns instead of JSONB keys
        would start emitting 1536-dimension noise into every schema an
        agent is shown."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "type": "doc", "title": "a"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        schema, report = g.infer_schema(sample_percent=sample_percent)
        # `type` names the inferred node type rather than repeating as
        # one of its properties, so `title` is the whole property set --
        # and no vec_* column joins it.
        properties = {p.name for nt in schema.node_types for p in nt.properties}
        assert properties == {"title"}
        assert [nt.name for nt in schema.node_types] == ["doc"]
        assert report.node_counts == {"doc": 1}


# ---------------------------------------------------------------------
# Batch search: many queries, one round trip
# ---------------------------------------------------------------------

class TestSearchManyShape:
    def test_queries_travel_as_one_values_list(self, vg):
        """The whole point is one statement: the queries become a
        VALUES list joined LATERAL-ly to a per-query top-k. Two copies
        of that list would mean the planner scoring every row against
        every query -- not slower, wrong."""
        sql = norm(vg.build_vector_search_many_query(
            [Near("summary", [1.0, 0.0, 0.0]), Near("summary", [0.0, 1.0, 0.0])], k=3))
        assert sql.count("AS queries") == 1
        assert "JOIN LATERAL" in sql and "LIMIT" in sql
        assert ";" not in sql

    def test_query_vectors_are_bound_not_inlined(self, vg):
        sql = norm(vg.build_vector_search_many_query([Near("summary", [1.5, 2.5, 3.5])], k=1))
        assert "1.5" not in sql

    def test_batch_is_graph_scoped(self, vg):
        sql = norm(vg.build_vector_search_many_query([Near("summary", [1.0, 0.0, 0.0])], k=1),
                   literal_binds=True)
        assert "graph_id = 'default'" in sql

    def test_edges_target_reports_endpoints_here_too(self, vg):
        """The single-query path pins this; the batch path did not, so
        `_result_columns(inner, target)` could be handed anything and
        every batch edge hit would come back without the endpoints that
        make it usable.

        Asserted on the OUTPUT labels, not on "start_id" anywhere in the
        statement: that is also the literal edges column name, so the
        bare substring is already there via `edges.start_id AS _start`
        and matches whether the endpoints are selected or not."""
        sql = norm(vg.build_vector_search_many_query(
            [Near("rel", [1.0, 0.0, 0.0])], target="edges", k=1))
        assert "FROM edges" in sql and "vec_rel" in sql
        assert "AS start_id" in sql and "AS end_id" in sql
        assert "hits.start_id" in sql and "hits.end_id" in sql

    def test_report_columns_reach_the_hits_lateral(self, vg):
        """The batch path builds its per-query select separately from
        the single-query path's -- `similarities`/`boosts` (issue #54)
        need their own SQL-shape proof that sim_report_i/boost_j reach
        the `hits` LATERAL the outer query reads, not just the inner
        subquery search_many() never surfaces on its own."""
        sql = norm(vg.build_vector_search_many_query(
            [[Near("summary", [1.0, 0.0, 0.0]), Near("title", [0.0, 1.0, 0.0])]],
            k=1, boost=Boost("score", 0.5)))
        assert "hits.sim_report_0" in sql and "hits.sim_report_1" in sql
        assert "hits.boost_0" in sql

    def test_rows_without_a_vector_are_filtered_before_the_cosine(self, vg):
        """The single-query path has this guard pinned; the batch path
        did not, and dropping it changes no ANSWER -- the outer
        `similarity IS NOT NULL` removes those rows anyway. It changes
        the COST: without it the unnest+sum LATERAL runs once per query
        for every row that cannot score. A behaviour test structurally
        cannot see that, which is why this is a shape test."""
        sql = norm(vg.build_vector_search_many_query([Near("summary", [1.0, 0.0, 0.0])], k=1))
        pre_filter, _, post_filter = sql.partition(") AS anon_1")
        assert "vec_summary IS NOT NULL" in pre_filter
        assert "similarity IS NOT NULL" in post_filter or "sim_0 IS NOT NULL" in post_filter

    def test_mismatched_query_shapes_are_refused(self, vg):
        """One statement can only express one shape. Ranking the second
        query with the first one's weights would answer a question
        nobody asked."""
        # Anchored on the whole opening clause: "must share a shape"
        # alone survives the message losing the emphasis that explains
        # WHY -- one statement, not one per query.
        with pytest.raises(ValueError,
                           match=r"^vector_search_many\(\) ranks every query with ONE "
                                 r"statement, so the queries must share a shape"):
            vg.build_vector_search_many_query(
                [Near("summary", [1.0, 0.0, 0.0]),
                 Near("summary", [0.0, 1.0, 0.0], weight=0.5)], k=2)
        with pytest.raises(ValueError, match="must share a shape"):
            vg.build_vector_search_many_query(
                [Near("summary", [1.0, 0.0, 0.0]), Near("title", [0.0, 1.0, 0.0])], k=2)

    @pytest.mark.parametrize("bad", [[], (), "near", None, 42])
    def test_queries_must_be_a_non_empty_list(self, vg, bad):
        with pytest.raises(TypeError, match="must be a non-empty list"):
            vg.build_vector_search_many_query(bad, k=2)

    def test_multivector_queries_are_allowed_per_entry(self, vg):
        sql = norm(vg.build_vector_search_many_query(
            [[Near("summary", [1.0, 0.0, 0.0]), Near("title", [0.0, 1.0, 0.0])],
             [Near("summary", [0.0, 0.0, 1.0]), Near("title", [1.0, 0.0, 0.0])]], k=2))
        assert sql.count("unnest") == 2      # one per field, not one per query


class TestSearchManyLive:
    def test_each_query_gets_its_own_ranking(self, fresh_graph):
        g = _corpus(fresh_graph)
        batch = g.vector_search_many(
            [Near("docvec", QUERY), Near("docvec", [0.0, 1.0, 0.0])], k=2,
            where={"type": "doc"})
        assert [[h["id"] for h in one] for one in batch] == [["1", "3"], ["4", "2"]]

    def test_batch_equals_running_the_searches_one_by_one(self, fresh_graph):
        """The contract that makes the optimization safe to take."""
        g = _corpus(fresh_graph)
        queries = [QUERY, [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        one_by_one = [[h["id"] for h in g.vector_search(Near("docvec", q), k=3)]
                      for q in queries]
        batched = [[h["id"] for h in one]
                   for one in g.vector_search_many([Near("docvec", q) for q in queries], k=3)]
        assert batched == one_by_one

    def test_a_query_matching_nothing_keeps_its_slot(self, fresh_graph):
        """Index i must always answer query i, or the caller has to
        guess which of their questions came back empty."""
        g = _corpus(fresh_graph)
        batch = g.vector_search_many(
            [Near("docvec", QUERY, min_similarity=0.99),
             Near("docvec", [0.0, 0.0, 1.0], min_similarity=0.99)], k=5)
        assert [len(one) for one in batch] == [2, 0]

    def test_hits_carry_their_properties(self, fresh_graph):
        """Every batch assertion read `id` or `similarity`, so the
        properties column could be dropped from the per-query select and
        nothing objected -- every caller would get bare ids back from a
        search whose single-query twin returns whole rows."""
        g = _corpus(fresh_graph)
        (one,) = g.vector_search_many([Near("docvec", QUERY)], k=1)
        assert one[0]["properties"]["name"] == "exact"

    def test_hits_carry_similarities_too(self, fresh_graph):
        """The batch path builds its per-query SELECT separately from
        vector_search()'s -- nothing shares column lists automatically,
        so `similarities`/`boosts` need their own proof here (issue
        #54), matching the single-query path's."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "both"}])
        g.set_vectors(nodes=[
            {"id": 1, "docvec": [1.0, 0.0, 0.0], "titlevec": [0.0, 1.0, 0.0]}])
        (one,) = g.vector_search_many(
            [[Near("docvec", QUERY, weight=0.5), Near("titlevec", QUERY, weight=0.5)]], k=1)
        assert one[0]["similarities"] == pytest.approx(
            {"docvec": 1.0, "titlevec": 0.0}, abs=1e-6)
        assert one[0]["boosts"] == {}

    def test_edge_hits_carry_their_endpoints(self, fresh_graph):
        """The single-query path has had this since it was written; the
        batch path was only ever exercised against nodes, so nothing
        noticed if an edge hit came back as a bare id."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "t": "a"}, {"id": 2, "t": "b"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "close"},
                     {"start_id": 2, "end_id": 1, "kind": "far"}])
        with g.engine.connect() as conn:
            ids = dict(conn.execute(text(
                "SELECT properties->>'kind', id FROM edges")).all())
        g.set_vectors(edges=[{"id": ids["close"], "relvec": [1.0, 0.0, 0.0]},
                             {"id": ids["far"], "relvec": [0.0, 1.0, 0.0]}])
        batch = g.vector_search_many(
            [Near("relvec", QUERY), Near("relvec", [0.0, 1.0, 0.0])], target="edges", k=1)
        assert [(h["start_id"], h["end_id"]) for one in batch for h in one] \
            == [("1", "2"), ("2", "1")]

    def test_one_round_trip(self, fresh_graph):
        """Two queries, one statement -- the reason this exists rather
        than a Python loop."""
        g = _corpus(fresh_graph)
        statements = []
        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):    # noqa: ARG001
            statements.append(statement)
        try:
            g.vector_search_many([Near("docvec", QUERY), Near("docvec", [0.0, 1.0, 0.0])], k=2)
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        assert len(statements) == 1

    def test_boost_reaches_the_batched_call_too(self, fresh_graph):
        """vector_search_many() does nothing but forward its arguments to
        search_many() -- drop the boost= forwarding and every batched
        call would quietly stop boosting while the single-query
        vector_search() kept working, unnoticed unless something calls
        it with a boost that actually reorders."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "closest", "score": 0.0},
                     {"id": 2, "n": "runner-up", "score": 0.9}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.8, 0.6, 0.0]}])
        (plain,) = g.vector_search_many([Near("docvec", QUERY)], k=10)
        (boosted,) = g.vector_search_many([Near("docvec", QUERY)], k=10,
                                          boost=Boost("score", 1.0))
        assert [h["id"] for h in plain] == ["1", "2"]
        assert [h["id"] for h in boosted] == ["2", "1"]

    def test_rerank_is_per_call_and_costs_one_provider_call_per_query(self, fresh_graph):
        """`rerank=` sits where `boost=` sits -- per CALL, not per query
        -- because `candidates` decides how many rows the ONE statement
        fetches and this call takes ONE k. But a score IS the (query,
        document) relationship, so one provider call cannot serve two
        queries: N queries mean N calls, each with its own text."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        calls = []
        rerank = Rerank(_by_name(["opposite", "exact"], calls),
                        document_from=".properties.name", candidates=5)
        batch = g.vector_search_many(
            [Near("docvec", text="first question"), Near("docvec", text="second question")],
            k=2, where={"type": "doc"}, rerank=rerank)
        assert [[h["id"] for h in one] for one in batch] == [["5", "1"], ["5", "1"]]
        assert [query for query, _ in calls] == ["first question", "second question"]
        # candidates, not k, decided how wide each query's list was.
        assert [len(documents) for _, documents in calls] == [5, 5]

    def test_the_batch_still_costs_one_round_trip_when_it_reranks(self, fresh_graph):
        """Reranking must not turn the one statement back into N. The
        provider calls are N by necessity; the SQL is not."""
        needs_jq()
        from hopai import Rerank

        g = _searchable(fresh_graph)
        statements = []

        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):       # noqa: ARG001
            statements.append(statement)
        try:
            g.vector_search_many([Near("docvec", text="a"), Near("docvec", text="b")], k=1,
                                 rerank=Rerank(_by_name([]),
                                               document_from=".properties.name",
                                               candidates=3))
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        assert len(statements) == 1


# ---------------------------------------------------------------------
# Hybrid ranking
# ---------------------------------------------------------------------

class TestBoost:
    def test_boost_lands_in_the_score_not_the_filter(self, vg):
        """A boost reorders; it must never change which rows qualify,
        or a ranking knob becomes a silent filter."""
        sql = norm(vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.5)), literal_binds=True)
        assert "boost_0" in sql and "* 0.5" in sql
        assert "jsonb_typeof" in sql          # non-numeric reads as absent
        assert "coalesce" in sql              # absent contributes `missing`, never NULL

    def test_scale_normalized_is_the_default_and_emits_a_window_function(self, vg):
        """#55: the default rescales the boost with a min-max window
        function over the select's own candidate rows -- OVER () with no
        PARTITION BY, so it is the whole result set at that select level,
        not the whole table."""
        sql = norm(vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.2)), literal_binds=True)
        assert "OVER ()" in sql
        assert "min(coalesce" in sql and "max(coalesce" in sql
        assert "nullif" in sql

    def test_scale_raw_emits_no_window_function(self, vg):
        """The escape hatch's whole point: with scale="raw" the compiled
        SQL has no OVER() at all -- the coefficient multiplies the
        coalesced property exactly as #55 found it."""
        sql = norm(vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.2, scale="raw")),
            literal_binds=True)
        assert "OVER ()" not in sql
        assert "boost_0" in sql and "* 0.2" in sql

    def test_scale_must_be_normalized_or_raw(self):
        with pytest.raises(ValueError, match=r"scale must be one of"):
            Boost("score", scale="clamped")

    def test_scale_reaches_ranked_ids_too(self, vg):
        """Boost's normalization is not vector_search()-only -- Start/Hop
        near= builds its seed/match set through ranked_ids(), which calls
        the same _boost_columns()."""
        sql = norm(vg.build_query(
            Start(near=Near("summary", [1.0, 0.0, 0.0]), keep=5,
                 boost=Boost("score", 0.2)), []), literal_binds=True)
        assert "OVER ()" in sql

    def test_callable_boost_is_passed_the_properties_column(self, vg):
        sql = norm(vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]),
            boost=Boost(lambda p: func.length(p["title"].astext), 0.1)))
        assert "length" in sql

    @pytest.mark.parametrize("bad", [0, float("nan"), "1", True, None])
    def test_boost_weight_must_be_a_non_zero_finite_number(self, bad):
        with pytest.raises(ValueError, match="finite number|non-zero"):
            Boost("score", bad)

    def test_the_finite_number_message_names_which_argument(self):
        """`weight` and `default` share one validation loop, so the only
        thing separating "you passed a bad weight" from "you passed a
        bad default" is the label -- which was mutable with the suite
        green. It matters here more than usual: `default` is the
        argument that got renamed from `missing`, so a caller reading
        this message is often checking exactly that."""
        with pytest.raises(ValueError, match=r"^Boost weight must be a finite number"):
            Boost("score", float("inf"))
        with pytest.raises(ValueError, match=r"^Boost default must be a finite number"):
            Boost("score", 1.0, default=float("nan"))

    def test_the_zero_weight_message_names_the_type(self):
        with pytest.raises(ValueError, match=r"^Boost weight must be non-zero"):
            Boost("score", 0)

    def test_the_default_weight_is_one_in_both_spellings(self):
        """Every test passed `weight` explicitly, so the default could
        be any number. It is spelled TWICE -- once on Boost, once in
        parse_boost's `spec.get("weight", 1.0)` -- and two copies of a
        default that nothing compares is how the JSON form and the
        Python form come to mean different things."""
        assert Boost("score").weight == 1.0
        assert Boost("score").default == 0.0
        assert parse_boost({"property": "score"}).weight == Boost("score").weight

    @pytest.mark.parametrize("bad", [None, 42, b"x", ""])
    def test_boost_key_must_be_a_property_or_callable(self, bad):
        with pytest.raises(TypeError, match="property name or a callable"):
            Boost(bad)

    def test_boost_without_near_is_refused(self):
        """Nothing is ranked without near=, so the boost would be a
        silent no-op -- which reads exactly like a working feature."""
        with pytest.raises(ValueError, match="an edge beam"):
            Start(boost=Boost("score", 1.0))
        with pytest.raises(ValueError, match="an edge beam"):
            Hop(boost=Boost("score", 1.0))

    def test_reprs_are_exact(self):
        assert repr(Boost("score", 0.5)) == "Boost('score', weight=0.5)"
        assert repr(Boost("score", 1.0, default=-1.0)) \
            == "Boost('score', weight=1.0, default=-1.0)"
        # scale="normalized" is the default -- it stays out of repr like
        # any other unstated default, so a caller who never touched the
        # knob does not see an implementation detail in every repr.
        assert repr(Boost("score", 0.5, scale="normalized")) == "Boost('score', weight=0.5)"
        assert repr(Boost("score", 0.5, scale="raw")) == "Boost('score', weight=0.5, scale='raw')"

    def test_json_form_matches_the_python_form(self, vg):
        python = vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.5, default=-1.0))
        parsed = parse_boost({"property": "score", "weight": 0.5, "default": -1.0})
        assert norm(vg.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0]), boost=parsed),
                    literal_binds=True) == norm(python, literal_binds=True)

    def test_json_form_carries_scale_too(self, vg):
        """scale="raw" from JSON must reach the same query shape as the
        Python form -- otherwise a caller behind json_api.py has no way
        to reach the escape hatch #55 added."""
        python = vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.5, scale="raw"))
        parsed = parse_boost({"property": "score", "weight": 0.5, "scale": "raw"})
        assert parsed.scale == "raw"
        assert norm(vg.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0]), boost=parsed),
                    literal_binds=True) == norm(python, literal_binds=True)

    @pytest.mark.parametrize("bad,message", [
        ({}, 'needs "property"'),
        ({"property": "s", "w": 1}, "unknown"),
        ([], "empty list"),
        ("score", "must be an object"),
    ])
    def test_malformed_json_boosts_are_rejected(self, bad, message):
        with pytest.raises((ValueError, TypeError), match=message):
            parse_boost(bad)


class TestBoostLive:
    def test_boost_reorders_without_changing_membership(self, fresh_graph):
        """scale="normalized" is the default (#55): the boost is rescaled
        into similarity's own range by a min-max window over the
        candidates (0.0/0.9/default-0.0 here -> 0/1/0), so node 2's
        contribution is weight * 1.0, not weight * 0.9 -- 1.0 rather than
        the raw property."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "closest", "score": 0.0},
                     {"id": 2, "n": "runner-up", "score": 0.9},
                     {"id": 3, "n": "no-score"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.8, 0.6, 0.0]},
                             {"id": 3, "docvec": [0.6, 0.8, 0.0]}])
        plain = [h["id"] for h in g.vector_search(Near("docvec", QUERY), k=10)]
        boosted = g.vector_search(Near("docvec", QUERY), boost=Boost("score", 1.0), k=10)
        assert plain == ["1", "2", "3"]
        assert [h["id"] for h in boosted] == ["2", "1", "3"]
        # Same rows, different order -- and the missing property is 0,
        # not a dropped row.
        assert {h["id"] for h in boosted} == set(plain)
        assert boosted[0]["similarity"] == pytest.approx(0.8 + 1.0, abs=1e-6)

    def test_scale_raw_reproduces_the_old_unbounded_behavior(self, fresh_graph):
        """The regression test that scale="raw" is a real, working escape
        hatch and not just accepted and ignored: identical setup to
        test_boost_reorders_without_changing_membership, but the
        combined score is exactly what it was BEFORE #55 -- the raw
        property multiplied straight in, unbounded, node 2's runner-up
        cosine buried under a boost of 0.9 rather than the normalized 1.0
        above."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "closest", "score": 0.0},
                     {"id": 2, "n": "runner-up", "score": 0.9},
                     {"id": 3, "n": "no-score"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.8, 0.6, 0.0]},
                             {"id": 3, "docvec": [0.6, 0.8, 0.0]}])
        boosted = g.vector_search(Near("docvec", QUERY), boost=Boost("score", 1.0, scale="raw"),
                                  k=10)
        assert [h["id"] for h in boosted] == ["2", "1", "3"]
        assert boosted[0]["similarity"] == pytest.approx(0.8 + 0.9, abs=1e-6)
        # boosts is keyed by the boosted PROPERTY, so a caller can see
        # the boost's own contribution and not just the combined total
        # (issue #54) -- node 2's boost alone (0.9) is what actually
        # pushed it past node 1 despite a lower raw similarity.
        by_id = {h["id"]: h for h in boosted}
        assert by_id["2"]["boosts"] == pytest.approx({"score": 0.9}, abs=1e-6)
        assert by_id["3"]["boosts"] == pytest.approx({"score": 0.0}, abs=1e-6)

    def test_callable_boost_falls_back_to_its_slot_name(self, fresh_graph):
        """A callable Boost has no property name to key the report by --
        it gets a stable positional fallback instead of vanishing from
        `boosts` or crashing the formatter. `boosts` reports the boost's
        OWN value (what `_boost_columns()` already computes, reused
        outward per the task's minimal-change design), not
        weight-multiplied -- the same choice `similarities` makes for
        min_similarity/weight: the raw signal, not its contribution."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "title": "abc"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        hits = g.vector_search(
            Near("docvec", QUERY),
            boost=Boost(lambda p: func.length(p["title"].astext), 0.1, scale="raw"), k=10)
        assert hits[0]["boosts"] == pytest.approx({"boost_0": 3.0}, abs=1e-6)
        assert hits[0]["similarity"] == pytest.approx(1.0 + 3.0 * 0.1, abs=1e-6)

    def test_two_boosts_on_the_same_property_sum_in_the_report(self, fresh_graph):
        """Two Boosts naming the same property is unusual but legal --
        the COMBINED score already adds both terms weighted, so the
        per-property report sums their RAW values too (0.4 + 0.4)
        rather than the second silently overwriting the first, which
        would under-report the very thing this feature exists to
        surface. Weights of 0.5 and 2.0 (not equal, and not summing to
        the boost count) keep this from passing by the coincidence a
        matched pair of weights could hide."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "score": 0.4}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        hits = g.vector_search(
            Near("docvec", QUERY),
            boost=[Boost("score", 0.5, scale="raw"), Boost("score", 2.0, scale="raw")], k=10)
        assert hits[0]["boosts"] == pytest.approx({"score": 0.4 + 0.4}, abs=1e-6)
        assert hits[0]["similarity"] == pytest.approx(1.0 + 0.4 * 0.5 + 0.4 * 2.0, abs=1e-6)

    def test_a_thousand_fold_property_no_longer_dominates_a_perfect_match(self, fresh_graph):
        """The bug report itself: Boost("importance", 0.2) on a view count
        in the thousands used to multiply an unbounded quantity by 0.2 and
        add it to a cosine that never exceeds 1 -- so a perfect match
        (similarity 1.0) could still lose to a mediocre match with a big
        enough property. Under the new default it cannot: the property is
        rescaled into [-1, 1] before `weight` is applied, so 20% weight
        really is 20%, capped at a 0.2 contribution."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "perfect-match", "importance": 0.0},
                     {"id": 2, "n": "irrelevant-match", "importance": 4200.0}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.0, 1.0, 0.0]}])  # orthogonal: sim 0.0
        hits = g.vector_search(Near("docvec", QUERY), boost=Boost("importance", 0.2), k=10)
        assert [h["id"] for h in hits] == ["1", "2"]
        # The perfect match's contribution is bounded: it starts at 1.0
        # and the whole boost term can add at most `weight` (0.2), so it
        # can never be beaten by a property alone, however large.
        assert hits[0]["similarity"] <= 1.0 + 0.2 + 1e-6
        # Node 2 is exactly the pre-#55 winner: raw would have added
        # 0.2 * 4200 = 840 to a similarity that never exceeds 1.
        raw_hits = g.vector_search(Near("docvec", QUERY),
                                   boost=Boost("importance", 0.2, scale="raw"), k=10)
        assert [h["id"] for h in raw_hits] == ["2", "1"]

    def test_non_numeric_property_contributes_missing_not_an_error(self, fresh_graph):
        """One row carrying "high" where a number was expected must not
        abort the statement -- the judgement aggregates.py already makes."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "score": "high"}, {"id": 2, "score": 0.5}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [1.0, 0.0, 0.0]}])
        hits = g.vector_search(Near("docvec", QUERY), boost=Boost("score", 1.0, default=0.0), k=10)
        assert [h["id"] for h in hits] == ["2", "1"]
        assert hits[1]["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_boost_applies_to_a_ranked_traversal_seed(self, fresh_graph):
        """Boost's normalization is not vector_search()-only: Start(near=)
        builds its ranked seed through the same ranked_ids()/
        _boost_columns() path (#55)."""
        g = _migrated(fresh_graph)
        # weight=2.0: at weight=1.0 the normalized boost (0 vs. the
        # candidate-set max, 1.0) exactly cancels node 1's similarity
        # lead (1.0 vs 0.0), a real tie this test must not depend on
        # Postgres breaking one particular way.
        g.add_nodes([{"id": 1, "score": 0.0}, {"id": 2, "score": 5.0}, {"id": 3, "n": "target"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.0, 1.0, 0.0]}])
        g.add_edges([{"start_id": 1, "end_id": 3, "kind": "k"},
                     {"start_id": 2, "end_id": 3, "kind": "k"}])
        result = g.traverse(Start(near=Near("docvec", QUERY), keep=1,
                                  boost=Boost("score", 2.0)), Hop(via={"kind": "k"}))
        # Node 2 loses on similarity (0.0 vs node 1's 1.0) and wins on the
        # boost (normalized to 1.0, weight 2.0 -> +2.0 vs node 1's +0.0).
        assert {n["id"] for n in result.nodes} == {"2", "3"}

    def test_zero_spread_boost_is_a_no_op_and_keeps_combined_not_null(self, fresh_graph):
        """Every candidate sharing one boost value means no spread to
        normalize against -- nullif(0, 0) is NULL, and left uncoalesced
        that NULL would propagate through `total = term + term` in
        _combined() and null out rows whose SIMILARITY was real, breaking
        `combined IS NOT NULL`'s meaning for every row a constant boost
        touches. It must instead contribute nothing, identically to no
        boost at all -- including when a boost is the only ranking signal
        near= min_similarity=None would otherwise refuse (min_similarity
        keeps this call legal without k)."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "score": 3.0}, {"id": 2, "score": 3.0}, {"id": 3, "score": 3.0}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.8, 0.6, 0.0]},
                             {"id": 3, "docvec": [0.0, 1.0, 0.0]}])
        plain = g.vector_search(Near("docvec", QUERY, min_similarity=-1.0), k=10)
        boosted = g.vector_search(Near("docvec", QUERY, min_similarity=-1.0),
                                  boost=Boost("score", 1.0), k=10)
        assert [(h["id"], h["similarity"]) for h in boosted] \
            == [(h["id"], h["similarity"]) for h in plain]
        # combined IS NOT NULL still means "some similarity had a
        # direction" -- all three rows, including the orthogonal one
        # (similarity 0.0, a real direction, not missing), still score.
        assert {h["id"] for h in boosted} == {"1", "2", "3"}


# ---------------------------------------------------------------------
# Edge similarity inside a hop
# ---------------------------------------------------------------------

class TestEdgeBeamShape:
    def test_beam_is_a_lateral_in_both_walk_terms(self, vg):
        sql = norm(vg.build_query(Start(), [Hop(via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                via_keep=2, hops=(1, 3))]))
        assert "beam_0" in sql and "beam_rec_0" in sql
        assert sql.count("JOIN LATERAL") >= 2

    def test_cycle_guard_sits_inside_the_beam(self, vg):
        """A top-k beam that spent slots on edges leading back into the
        path would follow fewer than via_k usable edges, and how many
        fewer is invisible from outside the query."""
        sql = norm(vg.build_query(Start(), [Hop(via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                via_keep=2, hops=(1, 3))]))
        beam = sql[sql.index("UNION ALL"):sql.index("AS beam_rec_0")]
        assert "local_path" in beam, beam
        # Before the LIMIT, which is what "inside the beam" means: the
        # guard has to shrink the candidate set the top-k picks FROM.
        assert beam.index("local_path") < beam.index("LIMIT")

    def test_via_still_filters_the_edges_the_beam_ranks(self, vg):
        """`via=` and `via_near=` compose: rank the most similar edges
        AMONG the ones via allows. Nothing combined them, so `via` could
        be dropped on the way into the beam -- at the call site or
        inside it -- and the walk would follow the nearest edge of ANY
        kind. That is a silently wrong answer, not an error.

        Counted per walk term, like the null guards: the beam is built
        for the anchor and again for the recursive reference, and `via`
        surviving on only one of them is the same bug at half depth."""
        sql = norm(vg.build_query(Start(), [Hop(via={"kind": "cites"}, hops=(1, 2),
                                                via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                via_keep=3)]), literal_binds=True)
        assert sql.count('{"kind": "cites"}') == 2

    def test_via_stored_in_shorthand_composes_with_via_near_too(self, vg):
        """edge_beam() resolves `via` through resolve_via(), same as the
        plain (non-ranked) walk terms in core.py -- so the STORED_IN
        shorthand filters what the beam ranks, not just what an
        ordinary via= dict filters."""
        sql = norm(vg.build_query(Start(), [Hop(via="cites", hops=(1, 2),
                                                via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                via_keep=3)]), literal_binds=True)
        assert sql.count("properties ->> 'kind') = 'cites'") == 2
        assert "@>" not in sql

    def test_via_dict_form_upgrades_in_the_beam_too_once_declared(self, vg):
        """edge_beam() passes graph._edge_type_declared into resolve_via()
        as a THIRD argument, separate from `via` itself -- the same
        upgrade TestViaStoredInShorthand pins for the plain (non-ranked)
        walk terms in core.py, but nothing exercised it for the beam
        before now. Dropping that argument silently falls back to
        resolve_via()'s edge_type_declared=False default: every result
        stays correct (both SQL shapes match the same rows), so only a
        shape assertion -- not a results comparison -- can tell the
        beam actually reached for the index issue #80 added, instead of
        always compiling the ordinary `via={"kind": ...}` dict form to
        the old whole-properties containment test regardless of what
        the graph has declared."""
        vg._edge_type_declared = True
        sql = norm(vg.build_query(Start(), [Hop(via={"kind": "cites"}, hops=(1, 2),
                                                via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                via_keep=3)]), literal_binds=True)
        assert sql.count("properties ->> 'kind') = 'cites'") == 2
        assert "@>" not in sql

    def test_edges_without_a_vector_are_filtered_before_the_cosine(self, vg):
        """The same shape the batch path needed, for the same reason.
        Dropping these guards changes no ANSWER -- the beam's own
        `combined IS NOT NULL` removes those edges anyway -- but it
        changes the COST: the unnest+sum LATERAL then runs, once per
        anchor row of a recursive walk, over every edge that cannot
        score. edge_beam()'s docstring promises "one set of guards"
        shared with the node path; this is what holds it to that."""
        sql = norm(vg.build_query(Start(), [Hop(via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                via_keep=2)]))
        # Once per walk term: the beam is built for the anchor and again
        # for the recursive reference, and a guard on only one of them
        # would leave the recursive half -- the expensive half -- naked.
        assert sql.count("vec_rel IS NOT NULL") == 2
        anchor = sql[sql.index("via_base_0"):sql.index("UNION ALL")]
        assert anchor.index("vec_rel IS NOT NULL") < anchor.index("sim_0 IS NOT NULL")

    def test_threshold_only_beam_has_no_limit(self, vg):
        sql = norm(vg.build_query(
            Start(), [Hop(via_near=Near("rel", [1.0, 0.0, 0.0], min_similarity=0.5))]),
            literal_binds=True)
        assert ">= 0.5" in sql
        assert "LIMIT" not in sql

    def test_via_near_is_validated_against_EDGE_fields(self, vg):
        """`summary` is a node field; naming it in via_near is the
        mistake most worth catching by name."""
        with pytest.raises(ValueError, match=r"no vector field 'summary'.*edges"):
            vg.build_query(Start(), [Hop(via_near=Near("summary", [1.0, 0.0, 0.0]), via_keep=1)])

    def test_via_k_without_via_near_is_rejected(self):
        with pytest.raises(ValueError, match="without via_near= orders nothing"):
            Hop(via_keep=3)

    def test_via_near_error_names_the_hop(self, vg):
        with pytest.raises(ValueError, match=r"hop 1 \(ranked\) via_near"):
            vg.build_query(Start(), [Hop(), Hop(via_near=Near("rel", [1.0, 0.0, 0.0]),
                                                label="ranked")])

    def test_node_and_edge_ranking_compose_on_one_hop(self, vg):
        sql = norm(vg.build_query(Start(), [Hop(via_near=Near("rel", [1.0, 0.0, 0.0]), via_keep=2,
                                                near=Near("summary", [1.0, 0.0, 0.0]), keep=3)]))
        assert "beam_0" in sql and "reached_0" in sql and "match_0" in sql


class TestEdgeBeamLive:
    @staticmethod
    def _edges(g):
        with g.engine.connect() as conn:
            return {row[0]: row[1] for row in conn.execute(text(
                "SELECT properties->>'kind', id FROM edges"))}

    def _fixture(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "n": "src", "seed": True}, {"id": 2, "n": "near"},
                     {"id": 3, "n": "far"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "aligned"},
                     {"start_id": 1, "end_id": 3, "kind": "orthogonal"}])
        ids = self._edges(g)
        g.set_vectors(edges=[{"id": ids["aligned"], "relvec": [1.0, 0.0, 0.0]},
                             {"id": ids["orthogonal"], "relvec": [0.0, 1.0, 0.0]}])
        return g

    def test_via_wins_over_similarity(self, fresh_graph):
        """The beam ranks the edges `via` allows -- it does not rank all
        of them and hope `via` agrees. Proven by making the DISALLOWED
        edge the more similar one: if `via` is dropped anywhere on the
        way into the beam, the walk follows `orthogonal` to node 3 and
        the result is confidently wrong rather than empty."""
        g = self._fixture(fresh_graph)
        ids = self._edges(g)
        # aligned points AWAY from the query; orthogonal points at it.
        g.set_vectors(edges=[{"id": ids["aligned"], "relvec": [0.0, 1.0, 0.0]},
                             {"id": ids["orthogonal"], "relvec": [1.0, 0.0, 0.0]}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via={"kind": "aligned"},
                                via_near=Near("relvec", QUERY), via_keep=1))
        assert {n["id"] for n in result.nodes} == {"1", "2"}      # NOT node 3
        assert [e["properties"]["kind"] for e in result.edges] == ["aligned"]

    def test_via_k_follows_only_the_most_similar_edge(self, fresh_graph):
        g = self._fixture(fresh_graph)
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY), via_keep=1))
        assert {n["id"] for n in result.nodes} == {"1", "2"}
        assert len(result.edges) == 1
        assert result.edges[0]["properties"]["kind"] == "aligned"

    def test_the_beam_ranks_by_similarity_not_by_edge_id(self, fresh_graph):
        """Every other case in this class inserts the most similar edge
        FIRST, so it also carries the lowest id -- and `ORDER BY
        similarity DESC, edge_id` and a bare `ORDER BY edge_id` then
        pick the same winner. Dropping the similarity term survived the
        entire class for exactly that reason, which is the whole feature
        silently becoming "follow the oldest edge".

        Here the similar edge is inserted LAST, so the two orderings
        disagree and only one of them is right. edge_id stays in the
        ORDER BY as the tie-break that keeps equal scores deterministic;
        what must not go is the term in front of it."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "seed": True}, {"id": 2}, {"id": 3}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "older_but_wrong"},
                     {"start_id": 1, "end_id": 3, "kind": "newer_but_right"}])
        ids = self._edges(g)
        assert ids["older_but_wrong"] < ids["newer_but_right"]
        g.set_vectors(edges=[{"id": ids["older_but_wrong"], "relvec": [0.0, 1.0, 0.0]},
                             {"id": ids["newer_but_right"], "relvec": [1.0, 0.0, 0.0]}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY), via_keep=1))
        assert [e["properties"]["kind"] for e in result.edges] == ["newer_but_right"]
        assert {n["id"] for n in result.nodes} == {"1", "3"}

    def test_threshold_filters_which_edges_are_worth_walking(self, fresh_graph):
        g = self._fixture(fresh_graph)
        kept = g.traverse(Start(where={"seed": True}),
                          Hop(via_near=Near("relvec", QUERY, min_similarity=0.5)))
        assert {n["id"] for n in kept.nodes} == {"1", "2"}
        both = g.traverse(Start(where={"seed": True}),
                          Hop(via_near=Near("relvec", QUERY, min_similarity=-1.0)))
        assert {n["id"] for n in both.nodes} == {"1", "2", "3"}

    def test_beam_is_per_source_node_not_a_global_truncation(self, fresh_graph):
        """via_k=1 from EACH node, so a second source keeps its own best
        edge -- a global top-1 would starve it entirely."""
        g = self._fixture(fresh_graph)
        g.add_nodes([{"id": 4, "n": "src2", "seed": True}, {"id": 5, "n": "target2"}])
        g.add_edges([{"start_id": 4, "end_id": 5, "kind": "second"}])
        ids = self._edges(g)
        g.set_vectors(edges=[{"id": ids["second"], "relvec": [0.9, 0.1, 0.0]}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY), via_keep=1))
        assert {n["id"] for n in result.nodes} == {"1", "2", "4", "5"}
        assert len(result.edges) == 2

    def test_edges_without_vectors_are_not_followed(self, fresh_graph):
        g = self._fixture(fresh_graph)
        g.add_nodes([{"id": 9, "n": "unreachable"}])
        g.add_edges([{"start_id": 1, "end_id": 9, "kind": "vectorless"}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY, min_similarity=-1.0)))
        assert "9" not in {n["id"] for n in result.nodes}

    def test_multi_hop_beam_reports_every_edge_it_walked(self, fresh_graph):
        """The invariant a restructured walk could break: a hop spanning
        several edges reports all of them, not one fabricated edge."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": i, "n": str(i), "seed": i == 1} for i in (1, 2, 3)])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "a"},
                     {"start_id": 2, "end_id": 3, "kind": "b"}])
        ids = self._edges(g)
        g.set_vectors(edges=[{"id": ids["a"], "relvec": [1.0, 0.0, 0.0]},
                             {"id": ids["b"], "relvec": [1.0, 0.0, 0.0]}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY), via_keep=2, hops=(2, 2)))
        assert {n["id"] for n in result.nodes} == {"1", "2", "3"}
        assert len(result.edges) == 2

    def test_cycle_guard_runs_inside_the_beam_not_after_it(self, fresh_graph):
        """The back edge is the MOST similar one, so a beam that ranks
        before excluding it spends its only slot walking home and never
        reaches node 3. Filtering after the beam cannot recover that:
        the slot is already gone, and how many were wasted is invisible
        from the outside."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": i, "n": str(i), "seed": i == 1} for i in (1, 2, 3)])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "out"},
                     {"start_id": 2, "end_id": 1, "kind": "back"},
                     {"start_id": 2, "end_id": 3, "kind": "onward"}])
        ids = self._edges(g)
        g.set_vectors(edges=[{"id": ids["out"], "relvec": [1.0, 0.0, 0.0]},
                             {"id": ids["back"], "relvec": [1.0, 0.0, 0.0]},
                             {"id": ids["onward"], "relvec": [0.8, 0.6, 0.0]}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY), via_keep=1,
                                hops=(1, 2)))
        assert {n["id"] for n in result.nodes} == {"1", "2", "3"}
        assert sorted(e["properties"]["kind"] for e in result.edges) == ["onward", "out"]

    def test_beam_does_not_leak_across_graphs(self, fresh_graph):
        g = self._fixture(fresh_graph)
        other = g.in_graph("other")
        other.define_vectors(edges=[Vector("relvec", 3)])
        other.add_nodes([{"id": 100, "seed": True}, {"id": 101}])
        other.add_edges([{"start_id": 100, "end_id": 101, "kind": "elsewhere"}])
        result = g.traverse(Start(where={"seed": True}),
                            Hop(via_near=Near("relvec", QUERY, min_similarity=-1.0)))
        assert "101" not in {n["id"] for n in result.nodes}

    def test_beam_scopes_by_graph_where_ids_actually_collide(self, fresh_graph):
        """The default schema makes node ids globally unique and ties
        both endpoints to the edge's own graph, so no foreign edge can
        even join this graph's seeds -- which is why the test above
        proves nothing about the discriminator. On CALLER-SUPPLIED
        tables keyed by (id, graph_id) the ids do collide, and an
        unscoped beam spends its only slot on the foreign edge because
        that one is more similar. The walk then carries an edge id this
        graph does not own, the scoped edge-report step drops it, and
        the traversal returns NOTHING -- silence standing in for the
        neighbor that was there all along."""
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        from conftest import _retry_ddl_race

        engine = fresh_graph.engine
        meta = MetaData(schema="hopai_beam_scope")
        nodes = Table("nodes", meta,
                      Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text, primary_key=True),
                      Column("properties", JSONB, nullable=False))
        edges = Table("edges", meta,
                      Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text, primary_key=True),
                      Column("start_id", BigInteger, nullable=False),
                      Column("end_id", BigInteger, nullable=False),
                      Column("properties", JSONB, nullable=False))
        def _setup():
            with engine.begin() as conn:
                conn.execute(text("DROP SCHEMA IF EXISTS hopai_beam_scope CASCADE"))
                conn.execute(text("CREATE SCHEMA hopai_beam_scope"))
            meta.create_all(engine)

        _retry_ddl_race(_setup)
        with engine.begin() as conn:
            conn.execute(nodes.insert(), [
                {"id": 1, "graph_id": "g1", "properties": {"seed": True}},
                {"id": 2, "graph_id": "g1", "properties": {"m": "mine"}},
                {"id": 3, "graph_id": "g1", "properties": {"m": "ours"}},
                {"id": 1, "graph_id": "g2", "properties": {"seed": True}},
                {"id": 3, "graph_id": "g2", "properties": {"m": "theirs"}},
            ])
            conn.execute(edges.insert(), [
                {"id": 1, "graph_id": "g1", "start_id": 1, "end_id": 2,
                 "properties": {"kind": "mine"}},
                # A DIFFERENT edge id on purpose: reuse g1's and the
                # scoped edge-report step resolves the stolen id back to
                # g1's own edge, hiding the leak behind a right answer.
                {"id": 99, "graph_id": "g2", "start_id": 1, "end_id": 3,
                 "properties": {"kind": "theirs"}},
            ])

        g1 = Graph(engine, graph="g1", node_table=nodes, edge_table=edges)
        g1.define_vectors(edges=[Vector("relvec", 3)])
        g1.migrate_vectors()
        # The foreign edge is the MORE similar one, so an unscoped beam
        # prefers it over this graph's own.
        g1.set_vectors(edges=[{"id": 1, "relvec": [0.8, 0.6, 0.0]}])
        g2 = g1.in_graph("g2")
        g2.define_vectors(edges=[Vector("relvec", 3)])
        g2.set_vectors(edges=[{"id": 99, "relvec": [1.0, 0.0, 0.0]}])

        result = g1.traverse(Start(where={"seed": True}),
                             Hop(via_near=Near("relvec", QUERY), via_keep=1))
        assert {n["id"] for n in result.nodes} == {"1", "2"}
        assert [e["properties"]["kind"] for e in result.edges] == ["mine"]


# ---------------------------------------------------------------------
# Re-embedding and the pgvector exit
# ---------------------------------------------------------------------

class TestStaleVectorsLive:
    def test_reports_missing_and_wrong_dimension_rows(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}, {"id": 2}, {"id": 3}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]}])
        stale = g.stale_vectors(node_fields=["docvec"])["nodes"]["docvec"]
        assert stale == {"missing": ["2", "3"], "wrong_dimensions": []}

        # Redefine wider without re-migrating: node 1's stored vector no
        # longer fits, which is exactly the window this call closes.
        g.define_vectors(nodes=[Vector("docvec", 6)])
        stale = g.stale_vectors(node_fields=["docvec"])["nodes"]["docvec"]
        assert stale == {"missing": ["2", "3"], "wrong_dimensions": ["1"]}

    def test_ids_feed_straight_back_into_set_vectors(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}, {"id": 2}])
        for node_id in g.stale_vectors(node_fields=["docvec"])["nodes"]["docvec"]["missing"]:
            g.set_vectors(nodes=[{"id": node_id, "docvec": [1.0, 0.0, 0.0]}])
        assert g.stale_vectors(node_fields=["docvec"])["nodes"]["docvec"]["missing"] == []

    def test_defaults_to_every_declared_field_and_is_graph_scoped(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1}])
        other = g.in_graph("other")
        other.define_vectors(nodes=[Vector("docvec", 3)])
        other.add_nodes([{"id": 50}])
        stale = g.stale_vectors()
        assert set(stale["nodes"]) == {"docvec", "titlevec"}
        assert stale["nodes"]["docvec"]["missing"] == ["1"]      # not 50

    def test_limit_caps_each_field(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": i} for i in range(1, 6)])
        assert len(g.stale_vectors(node_fields=["docvec"], limit=2)["nodes"]["docvec"]["missing"]) == 2

    def test_unknown_field_names_the_defined_ones(self, fresh_graph):
        g = _migrated(fresh_graph)
        with pytest.raises(ValueError, match="no vector field 'ghost'"):
            g.stale_vectors(node_fields=["ghost"])

    def test_after_walks_past_the_ids_already_seen(self, fresh_graph):
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": i} for i in range(1, 6)])
        assert g.stale_vectors(node_fields=["docvec"], limit=2,
                               after=None)["nodes"]["docvec"]["missing"] == ["1", "2"]
        assert g.stale_vectors(node_fields=["docvec"], limit=2,
                               after="2")["nodes"]["docvec"]["missing"] == ["3", "4"]
        assert g.stale_vectors(node_fields=["docvec"],
                               after="5")["nodes"]["docvec"]["missing"] == []

    def test_the_cursor_compares_ids_the_way_it_orders_them(self, fresh_graph):
        """Ids come back as strings but the walk is ordered by the raw
        column, where 9 < 10. A cursor comparing the TEXT would put
        '10' before '9', so paging past '9' would skip every id from 10
        up -- rows silently never returned, which is how a backfill
        finishes while leaving work behind."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": i} for i in (8, 9, 10, 11)])
        assert g.stale_vectors(node_fields=["docvec"],
                               after="9")["nodes"]["docvec"]["missing"] == ["10", "11"]


def _embedding_graph(fresh_graph, log=None, source=None, dimensions=3):
    """A migrated graph whose `docvec` field knows how to embed itself."""
    fresh_graph.define_vectors(
        nodes=[Vector("docvec", dimensions, source=source,
                      embed=counting_embedder(dimensions, log)),
               Vector("titlevec", 3)],
        edges=[Vector("relvec", 3, embed=counting_embedder(3, log))])
    fresh_graph.migrate_vectors()
    return fresh_graph


class TestSetVectorsFromTextLive:
    def test_a_string_is_embedded_and_stored(self, fresh_graph):
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}])
        assert g.set_vectors(nodes=[{"id": 1, "docvec": "apple"}]) == 1
        # ord('a') == 97: the stored vector is the embedder's answer, so
        # this covers the whole path rather than "something was written".
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] == [97.0, 0.0, 0.0]

    def test_text_and_vectors_mix_in_one_call(self, fresh_graph):
        """Nothing forces a caller to pick one form for a whole batch,
        and a half-migrated codebase will have both."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}, {"id": 2}])
        g.set_vectors(nodes=[{"id": 1, "docvec": "apple"},
                             {"id": 2, "docvec": [1.0, 2.0, 3.0]}])
        stored = g.get_vectors(node_ids=[1, 2])["nodes"]
        assert stored["1"]["docvec"] == [97.0, 0.0, 0.0]
        assert stored["2"]["docvec"] == [1.0, 2.0, 3.0]

    def test_edges_take_text_too(self, fresh_graph):
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}, {"id": 2}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "cites"}])
        with g.engine.connect() as connection:
            edge_id = connection.execute(text("SELECT id FROM edges")).scalar()
        g.set_vectors(edges=[{"id": edge_id, "relvec": "banana"}])
        assert g.get_vectors(edge_ids=[edge_id])["edges"][str(edge_id)]["relvec"] \
            == [98.0, 0.0, 0.0]

    def test_every_row_is_embedded_in_one_provider_call(self, fresh_graph):
        """Per row would be 500 HTTP round trips for 500 rows. The
        Embedder chunks to the provider's cap on top of this; what
        matters here is that set_vectors() does not defeat it."""
        log = []
        g = _embedding_graph(fresh_graph, log=log)
        g.add_nodes([{"id": i} for i in range(1, 6)])
        g.set_vectors(nodes=[{"id": i, "docvec": f"text-{i}"} for i in range(1, 6)])
        assert log == [[f"text-{i}" for i in range(1, 6)]]

    def test_a_field_with_no_embedder_refuses_text(self, fresh_graph):
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match=r"set_vectors\(\): field 'titlevec' on nodes "
                                             r"was given text, but declares no embedder"):
            g.set_vectors(nodes=[{"id": 1, "titlevec": "apple"}])

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_blank_text_is_refused_rather_than_embedded(self, fresh_graph, blank):
        """Whitespace has no meaning to embed; every provider either
        errors or returns a vector for nothing. Refusing here names
        None as the way to actually clear the field."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="the text to embed is empty"):
            g.set_vectors(nodes=[{"id": 1, "docvec": blank}])

    def test_none_still_clears_the_vector(self, fresh_graph):
        """The one thing blank text must not become: a field with an
        embedder still has to be clearable."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}])
        g.set_vectors(nodes=[{"id": 1, "docvec": "apple"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": None}])
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] is None

    def test_a_provider_failure_writes_nothing(self, fresh_graph):
        """The reason embedding happens before the transaction opens:
        a provider that dies halfway must not leave half a batch
        written, and a retry must not collide with rows that landed."""
        def dies(texts):
            raise RuntimeError("provider is down")

        g = _embedding_graph(fresh_graph)
        g.define_vectors(nodes=[Vector("docvec", 3, embed=dies)])
        g.add_nodes([{"id": 1}, {"id": 2}])
        with pytest.raises(Exception, match="provider is down"):
            g.set_vectors(nodes=[{"id": 1, "docvec": "a"}, {"id": 2, "docvec": "b"}])
        assert g.get_vectors(node_ids=[1, 2])["nodes"]["1"]["docvec"] is None

    def test_an_embedder_of_the_wrong_width_writes_nothing(self, fresh_graph):
        g = _embedding_graph(fresh_graph)
        g.define_vectors(nodes=[Vector("docvec", 3, embed=lambda texts: [[1.0, 2.0]])])
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="returned 2 dimensions, the field is defined "
                                             "with 3 -- nothing was written"):
            g.set_vectors(nodes=[{"id": 1, "docvec": "apple"}])
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] is None

    def test_the_provider_is_called_before_the_transaction_opens(self, fresh_graph):
        """Stated as an invariant in set_vectors()' docstring and
        untestable by inspection: an HTTP call inside an open
        transaction holds row locks for a network round trip."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}])
        seen_texts, opened = [], []

        def watching(texts):
            seen_texts.extend(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

        def on_begin(conn):
            opened.append(len(seen_texts))

        g.define_vectors(nodes=[Vector("docvec", 3, embed=watching)])
        event.listen(g.engine, "begin", on_begin)
        try:
            g.set_vectors(nodes=[{"id": 1, "docvec": "apple"}])
        finally:
            event.remove(g.engine, "begin", on_begin)
        # Every transaction this call opened already had the embedding
        # in hand; a zero here is a provider call holding row locks.
        assert opened and all(count == 1 for count in opened)

    def test_every_row_across_both_targets_is_validated_before_any_embed_call(
            self, fresh_graph):
        """set_vectors(nodes=[...], edges=[...]) validates EVERY row on
        BOTH targets before embedding either -- a call refused for the
        edges half (here: a duplicate id) must not have already spent a
        provider round trip embedding the nodes half. plan_vector_writes()
        is split into a pure grouping pass (_group_vector_rows(), which
        raises on this) and a separate embed-then-finalize pass per
        target precisely so this holds -- issue #74's async review
        found the two had drifted apart when it checked the async twin
        against this same invariant."""
        log = []
        g = _embedding_graph(fresh_graph, log=log)
        g.add_nodes([{"id": 1}])
        with pytest.raises(ValueError, match="edges id 9 appears twice"):
            g.set_vectors(
                nodes=[{"id": 1, "docvec": "apple"}],
                edges=[{"id": 9, "relvec": "x"}, {"id": 9, "relvec": "y"}])
        assert log == []


class TestEmbedStaleLive:
    def test_it_reads_the_property_of_the_fields_own_name(self, fresh_graph):
        """The naming gap, closed: Vector("docvec") embeds the "docvec"
        property. Before this the field name named a column and nothing
        else, and which property fed it was simply unanswerable."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple"}])
        assert g.embed_stale()["nodes"]["docvec"] == {"embedded": ["1"], "skipped": []}
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] == [97.0, 0.0, 0.0]

    def test_source_points_it_at_another_property(self, fresh_graph):
        g = _embedding_graph(fresh_graph, source="abstract")
        g.add_nodes([{"id": 1, "abstract": "banana", "docvec": "apple"}])
        g.embed_stale()
        # 'b' for the abstract, not 'a' for the same-named property.
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] == [98.0, 0.0, 0.0]

    def test_rows_with_nothing_to_embed_are_reported_not_raised(self, fresh_graph):
        """A node with no abstract legitimately has no abstract vector.
        Silence would leave the caller re-running a backfill that can
        never finish, and an exception would stop the other 999 rows.

        Row 4 is written directly rather than through add_nodes(): a
        non-string value at a declared vector field's name is refused
        at ingestion (#50), so this represents a row that arrived some
        other way -- raw SQL, a migration, data from before the field
        was declared -- which embed_stale() still has to survive."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple"}, {"id": 2},
                     {"id": 3, "docvec": "  "}])
        with g.engine.begin() as conn:
            conn.execute(text('INSERT INTO nodes (id, properties) VALUES (4, \'{"docvec": 5}\')'))
        result = g.embed_stale(node_fields=["docvec"])["nodes"]["docvec"]
        assert result == {"embedded": ["1"], "skipped": ["2", "3", "4"]}

    def test_it_refills_the_graph_after_a_dimension_change(self, fresh_graph):
        """Changing a model means changing a width, and the whole
        re-embed used to be the caller's loop to write. The source text
        is already in the properties, so it is three calls now."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple"}])
        g.embed_stale()
        g.define_vectors(nodes=[Vector("docvec", 4, embed=counting_embedder(4))])
        g.drop_vectors(node_fields=["docvec"])       # refuses to reinterpret
        g.migrate_vectors()
        assert g.embed_stale()["nodes"]["docvec"]["embedded"] == ["1"]
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] == [97.0, 0.0, 0.0, 0.0]

    def test_wrong_dimension_rows_are_in_the_work_set(self, fresh_graph):
        """stale_vectors() reports two categories and this fills in
        both. Taking only `missing` would leave every row a widened
        declaration outgrew sitting there, reported forever.

        The state is built with raw SQL because the API refuses to
        create it: set_vectors() checks the width on the way in."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple"}])
        g.drop_vectors(node_fields=["docvec"])       # take the CHECK away
        with g.engine.begin() as connection:
            connection.execute(text(
                "UPDATE nodes SET vec_docvec = ARRAY[1.0, 2.0]::real[] WHERE id = 1"))
        stale = g.stale_vectors(node_fields=["docvec"])["nodes"]["docvec"]
        assert stale == {"missing": [], "wrong_dimensions": ["1"]}
        assert g.embed_stale()["nodes"]["docvec"]["embedded"] == ["1"]
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["docvec"] == [97.0, 0.0, 0.0]

    def test_a_second_run_finds_nothing_left(self, fresh_graph):
        """The loop has to terminate: a call that re-embeds rows it
        already filled would never come back empty, and a caller
        looping until it does would loop forever."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple"}])
        g.embed_stale()
        assert g.embed_stale()["nodes"]["docvec"] == {"embedded": [], "skipped": []}

    def test_fields_without_an_embedder_are_left_alone(self, fresh_graph):
        """titlevec declares no embed=, so this call has nothing to say
        about it -- and must not report a zero that reads as "done"."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple", "titlevec": "apple"}])
        assert set(g.embed_stale()["nodes"]) == {"docvec"}
        assert g.get_vectors(node_ids=[1])["nodes"]["1"]["titlevec"] is None

    def test_naming_a_field_that_cannot_embed_itself_raises(self, fresh_graph):
        """Skipped from the default sweep, refused when asked for by
        name: a caller who typed the field meant something by it."""
        g = _embedding_graph(fresh_graph)
        with pytest.raises(ValueError, match=r"embed_stale\(\): field 'titlevec' on nodes "
                                             r"was given text, but declares no embedder"):
            g.embed_stale(node_fields=["titlevec"])

    def test_a_graph_where_nothing_can_embed_says_so(self, fresh_graph):
        """An empty result here would read as "nothing was stale",
        which is the opposite of what happened."""
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "docvec": "apple"}])
        with pytest.raises(ValueError, match="no vector field on this graph declares an "
                                             "embedder"):
            g.embed_stale()

    def test_limit_caps_the_rows_per_field(self, fresh_graph):
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": i, "docvec": f"text-{i}"} for i in range(1, 6)])
        assert len(g.embed_stale(limit=2)["nodes"]["docvec"]["embedded"]) == 2

    def test_it_reaches_work_behind_rows_that_can_never_be_filled(self, fresh_graph):
        """The bug a plain LIMIT window has, and the reason paging is a
        keyset cursor: a row with no source text is stale FOREVER, so
        the same leading rows fill the window on every pass and the
        work behind them is never reached. It reported success and
        embedded nothing -- silent, permanent incompleteness."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": 1}, {"id": 2}, {"id": 3},          # nothing to embed
                     {"id": 4, "docvec": "apple"}, {"id": 5, "docvec": "banana"}])
        result = g.embed_stale(node_fields=["docvec"], batch=2)["nodes"]["docvec"]
        assert result == {"embedded": ["4", "5"], "skipped": ["1", "2", "3"]}
        assert g.get_vectors(node_ids=[4])["nodes"]["4"]["docvec"] == [97.0, 0.0, 0.0]

    def test_a_page_smaller_than_the_field_still_finishes_it(self, fresh_graph):
        """batch= bounds memory and transaction size, not how much gets
        done: one call still walks the whole field."""
        g = _embedding_graph(fresh_graph)
        g.add_nodes([{"id": i, "docvec": f"text-{i}"} for i in range(1, 8)])
        result = g.embed_stale(node_fields=["docvec"], batch=2)["nodes"]["docvec"]
        assert result["embedded"] == [str(i) for i in range(1, 8)]
        assert g.embed_stale(node_fields=["docvec"], batch=2)["nodes"]["docvec"]["embedded"] == []

    def test_each_page_is_its_own_provider_call_and_transaction(self, fresh_graph):
        """What bounds the memory: 500k rows must not become one embed
        call holding 500k vectors, nor one transaction holding every
        row lock until the last provider round trip returns."""
        log, opened = [], []
        g = _embedding_graph(fresh_graph, log=log)
        g.add_nodes([{"id": i, "docvec": f"text-{i}"} for i in range(1, 7)])

        def on_begin(conn):
            opened.append(len(log))

        event.listen(g.engine, "begin", on_begin)
        try:
            g.embed_stale(node_fields=["docvec"], batch=2)
        finally:
            event.remove(g.engine, "begin", on_begin)
        assert log == [["text-1", "text-2"], ["text-3", "text-4"], ["text-5", "text-6"]]
        # Read as: one transaction opened before any embedding (the
        # first stale query), then more after each page. Embedding
        # everything first and writing once would put every `begin` at
        # the same count instead of walking 0, 1, 2, 3.
        assert sorted(set(opened)) == [0, 1, 2, 3]

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
    def test_a_meaningless_batch_is_refused(self, fresh_graph, bad):
        """batch=0 would spin forever asking for zero rows."""
        g = _embedding_graph(fresh_graph)
        with pytest.raises(ValueError, match="batch must be a positive integer"):
            g.embed_stale(node_fields=["docvec"], batch=bad)

    def test_it_stays_inside_its_own_graph(self, fresh_graph):
        """Every read and write goes through _scoped(); the property
        read this call adds is a new place to forget it."""
        g = _embedding_graph(fresh_graph)
        other = g.in_graph("other")
        other.define_vectors(nodes=[Vector("docvec", 3, embed=counting_embedder())])
        g.add_nodes([{"id": 1, "docvec": "apple"}])
        other.add_nodes([{"id": 2, "docvec": "banana"}])
        assert g.embed_stale()["nodes"]["docvec"]["embedded"] == ["1"]
        assert other.get_vectors(node_ids=[2])["nodes"]["2"]["docvec"] is None

    def test_undeclared_graphs_say_to_declare_first(self, fresh_graph):
        with pytest.raises(ValueError, match=r"call define_vectors\(...\) first"):
            fresh_graph.embed_stale()


class TestPgvectorDdl:
    def test_emits_extension_conversion_and_index(self, vg):
        ddl = vg.pgvector_exit_ddl()
        assert ddl[0] == "CREATE EXTENSION IF NOT EXISTS vector"
        joined = "\n".join(ddl)
        assert 'TYPE vector(3) USING "vec_summary"::vector(3)' in joined
        # The dimension CHECK must go: vector(d) enforces it in the type,
        # and leaving both would reject nothing while confusing everyone.
        assert 'DROP CONSTRAINT IF EXISTS "ck_vec_dims_default_nodes_summary"' in joined
        assert "USING hnsw" in joined and "vector_cosine_ops" in joined
        assert "vec_rel" in joined          # edges too

    def test_cosine_operator_class_matches_the_metric_this_library_computes(self, vg):
        """An exported index ranking by L2 would answer a different
        question than the code it replaces."""
        assert all("vector_cosine_ops" in s for s in vg.pgvector_exit_ddl() if "CREATE INDEX" in s)

    def test_index_none_emits_conversion_only(self, vg):
        assert not any("CREATE INDEX" in s for s in vg.pgvector_exit_ddl(index=None))

    @pytest.mark.parametrize("bad", ["hnsw2", "gist", "", 3])
    def test_unknown_index_method_is_refused(self, vg, bad):
        with pytest.raises(ValueError, match="index must be None or one of"):
            vg.pgvector_exit_ddl(index=bad)

    def test_needs_a_declaration(self):
        with pytest.raises(ValueError, match=r"call define_vectors\(...\) first"):
            offline().pgvector_exit_ddl()

    def test_generates_without_importing_pgvector(self, vg):
        """The whole point: hopai never depends on the extension, even
        to describe the migration off itself."""
        import sys
        assert "pgvector" not in sys.modules
        vg.pgvector_exit_ddl()
        assert "pgvector" not in sys.modules


# ---------------------------------------------------------------------
# The optional pgvector BACKEND -- vector_backend="pgvector"
#
# Offline first (nothing here connects), then the live classes. Every
# offline test that declares a field goes through `metadata_guard`: a
# vec_* column is attached to the process-global Node/Edge metadata,
# and under this backend it carries pgvector's `vector` TYPE -- which
# _attach() now refuses to share with an exact-backend handle. A leak
# would therefore not be the harmless extra column it used to be; it
# would fail every later exact test that declares the same name.
# ---------------------------------------------------------------------

@pytest.fixture()
def metadata_guard():
    """Restore the shared vec_* attachment set this test was given --
    fresh_graph()'s teardown, for the offline tests that have no
    fixture of their own."""
    from conftest import _restore_vector_columns, _vector_columns
    before = _vector_columns()
    yield
    _restore_vector_columns(before)


@pytest.fixture()
def pgvg(metadata_guard) -> Graph:      # noqa: ARG001 -- the guard is the point
    """`vg`'s pgvector twin: an offline graph on the pgvector backend
    with fields declared on both targets. Field names that no
    exact-backend test uses, so a leak is visible as a failure here
    rather than as a confusing one three modules away."""
    g = Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
              vector_backend="pgvector")
    g.define_vectors(nodes=[Vector("pgsummary", 3), Vector("pgtitle", 3)],
                     edges=[Vector("pgrel", 3)])
    return g


class TestVectorBackendSelection:
    """Which storage a Graph uses is chosen once, at construction, and
    every handle derived from it must agree -- the column is shared, so
    two handles disagreeing means one of them writing the wrong type."""

    def test_the_default_backend_is_exact(self):
        """No caller asked for an extension, so no caller gets one."""
        assert offline().vector_backend == "exact"

    def test_pgvector_is_selected_by_name(self):
        assert offline_pgvector().vector_backend == "pgvector"

    @pytest.mark.parametrize("bad", ["pg_vector", "hnsw", "PGVECTOR", "", None, 1])
    def test_an_unknown_backend_names_both_valid_values(self, bad):
        """A typo must not fall back to a default: 'exact' silently
        would be an approximate answer the caller asked to avoid, and
        the other way round is a missing extension surfacing as an
        undefined-operator error from inside a compiled query."""
        with pytest.raises(ValueError) as exc:
            Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                  vector_backend=bad)
        assert "exact" in str(exc.value) and "pgvector" in str(exc.value)

    def test_in_graph_carries_the_backend_to_the_child_handle(self):
        """in_graph() returns a handle on the SAME tables. A child left
        on the default would read a vector(d) column as real[] and
        compile SQL the column cannot answer."""
        child = offline_pgvector().in_graph("other")
        assert child.vector_backend == "pgvector"

    def test_in_graph_carries_the_cached_version_without_connecting(self):
        """The version is a property of the database both handles point
        at, and in_graph() promises not to connect -- so it is carried,
        never re-probed."""
        parent = offline_pgvector()
        parent._pgvector_version = (0, 8, 0)
        assert parent.in_graph("other")._pgvector_version == (0, 8, 0)

    def test_a_new_handle_starts_with_no_cached_version_so_the_check_runs(self):
        """_ensure_pgvector_ready() probes only while the cache reads
        `is None`. Any other empty-ish starting value ("" say) is
        falsy but not None, so the version check would be SKIPPED for
        the handle's whole life and a server below the floor would go
        unrefused -- the silently-lossy filtered search the floor
        exists to prevent. A surviving mutant on this line said no test
        objected."""
        assert offline_pgvector()._pgvector_version is None
        assert offline().vector_backend == "exact"

    def test_the_first_search_probes_the_version_and_caches_it(self, pgvector_graph):
        """The other half of the line above: the probe must actually
        run, once, and then stop running."""
        pgvector_graph.define_vectors(nodes=[Vector("probever", 3)])
        pgvector_graph.migrate_vectors()
        # migrate_vectors() has already probed, so clear it: what this
        # asserts is that the SEARCH path fills an empty cache too, and
        # a handle that only ever reads never skips the check.
        pgvector_graph._pgvector_version = None
        pgvector_graph.vector_search(Near("probever", [1.0, 0.0, 0.0]), k=1)
        assert pgvector_graph._pgvector_version >= (0, 8, 0)


class TestBackendChoiceIsInvisibleToVectorFreeQueries:
    """The premise of an OPTIONAL backend: a query with no similarity in
    it must compile the same way under both, or every existing plan is
    up for renegotiation the day someone opts in."""

    def test_the_pgvector_backend_changes_no_near_less_query(self, pgvg):
        """The exact backend's twin of this
        (test_defining_vectors_changes_no_near_less_query) is the
        invariant CLAUDE.md names; declaring pgvector fields must be
        just as free."""
        start, hops = Start(where={"type": "person"}), [Hop(hops=(1, 3)), Hop(optional=True)]
        assert norm(offline().build_query(start, hops)) == norm(pgvg.build_query(start, hops))

    def test_the_two_backends_compile_the_same_traversal(
            self, pgvg, metadata_guard):                  # noqa: ARG002 -- the guard is the point
        """Not the same as the test above: here BOTH handles have vector
        fields declared, so the vec_* columns exist in both metadata
        sets and only the backend differs. The backend may touch
        vector-ranking SQL and nothing else."""
        exact = offline()
        exact.define_vectors(nodes=[Vector("exsummary", 3)], edges=[Vector("exrel", 3)])
        start, hops = Start(where={"type": "person"}), [Hop(via={"kind": "knows"}), Hop()]
        assert norm(exact.build_query(start, hops)) == norm(pgvg.build_query(start, hops))

    def test_an_aggregate_with_no_near_is_unchanged_too(self, pgvg):
        """_prepare_vector_scan() gates on the CHAIN carrying a near=,
        not on the backend -- a walk with none must not pay a round trip
        for a setting it never reads, and must not compile differently
        either."""
        query = (Start(where={"type": "person"}), [Hop()], {"n": Count()})
        assert norm(offline().build_aggregate_query(*query)) == \
            norm(pgvg.build_aggregate_query(*query))


class TestPgvectorSearchShape:
    """What the emitted SQL must look like for the index to serve it.
    Every assertion here is the difference between an index scan and a
    sequential one, which is the whole reason this backend exists."""

    def test_the_order_by_holds_the_bare_distance_operator(self, pgvg):
        """An HNSW index is on the OPERATOR. Wrapping `<=>` in `1 - x`
        (or in a CASE mapping NaN to NULL) returns the same rows from a
        sequential scan -- so distance is what the inner query ranks by
        and similarity is computed outside it."""
        sql = norm(build_search_query(pgvg, Near("pgsummary", [1.0, 0.0, 0.0]), k=5))
        assert "ORDER BY nodes.vec_pgsummary <=> CAST(" in sql
        assert "unnest" not in sql              # no exact-backend scoring survived

    def test_the_filters_sit_inside_the_ranked_subquery(self, pgvg):
        """`where=` applied after LIMIT k would mean 'k candidates, some
        of which match' -- which is what a filtered search must never
        become. The guards and the limit share one SELECT."""
        sql = norm(build_search_query(
            pgvg, Near("pgsummary", [1.0, 0.0, 0.0], min_similarity=0.5),
            k=5, where={"type": "doc"}))
        inner = sql[sql.index("FROM (SELECT"):sql.index(") AS anon_1")]
        assert "properties -> " in inner or "properties @>" in inner
        assert "IS NOT NULL" in inner and "<= " in inner
        assert "LIMIT" in inner

    def test_the_query_vector_travels_as_a_parameter(self, pgvg):
        """A 1536-float literal inlined into every statement defeats
        Postgres's plan cache and bloats the logs for nothing."""
        sql = norm(build_search_query(pgvg, Near("pgsummary", [1.0, 0.0, 0.0]), k=5))
        assert "CAST(%(param_2)s AS vector(3))" in sql
        assert "[1.0,0.0,0.0]" not in sql

    def test_a_near_seed_ranks_with_the_operator_too(self, pgvg):
        """The traversal seed is the same shape as a search, deliberately
        -- otherwise Start(near=) would be the one place the extension
        was paid for and not used."""
        sql = norm(pgvg.build_query(
            Start(near=Near("pgsummary", [1.0, 0.0, 0.0]), keep=3), [Hop()]))
        seed = sql[sql.index("seed AS ("):sql.index("walk_0")]
        assert "ORDER BY nodes.vec_pgsummary <=> CAST(" in seed
        assert "unnest" not in seed and "LATERAL" not in seed

    def test_via_near_is_still_a_per_anchor_lateral(self, pgvg):
        """'The k most similar edges' only means something relative to
        where you are standing, so the beam stays per anchor under this
        backend as well -- it just ranks with `<=>`."""
        sql = norm(pgvg.build_query(
            Start(), [Hop(via_near=Near("pgrel", [1.0, 0.0, 0.0]), via_keep=2)]))
        assert "LATERAL" in sql and "<=>" in sql

    def test_search_many_gives_every_query_its_own_ordered_scan(self, pgvg):
        """One statement, one LATERAL invocation per query row -- which
        is what makes each query its own index scan instead of one
        shared sequential one."""
        sql = norm(build_search_many_query(
            pgvg, [Near("pgsummary", [1.0, 0.0, 0.0]), Near("pgsummary", [0.0, 1.0, 0.0])], k=2))
        assert "JOIN LATERAL" in sql
        assert "CAST(queries.v0 AS vector(3))" in sql
        assert sql.count("ORDER BY nodes.vec_pgsummary <=>") == 1


class TestPgvectorRefusesRankingsAnIndexCannotServe:
    """Refusals, not scoped-out features: an HNSW index answers exactly
    one ORDER BY, so anything else would pay for the extension, get the
    exact backend's cost, and quietly have become approximate. Each one
    is asserted on its MESSAGE -- naming the way forward is the point,
    and a bare ValueError names nothing."""

    def test_multivector_names_both_fields_and_the_way_out(self, pgvg):
        with pytest.raises(ValueError) as exc:
            build_search_query(pgvg, [Near("pgsummary", [1.0, 0.0, 0.0]),
                                      Near("pgtitle", [1.0, 0.0, 0.0])], k=3)
        message = str(exc.value)
        assert "'pgsummary', 'pgtitle'" in message
        assert "ONE field per search" in message
        assert "without vector_backend='pgvector'" in message

    def test_boost_is_refused_by_name(self, pgvg):
        with pytest.raises(ValueError) as exc:
            build_search_query(pgvg, Near("pgsummary", [1.0, 0.0, 0.0]), k=3,
                               boost=Boost("priority"))
        message = str(exc.value)
        assert "does not support boost=" in message
        assert "apply the boost to the returned hits yourself" in message

    def test_a_negative_weight_is_refused_because_hnsw_has_one_direction(self, pgvg):
        """A negative weight asks for the LEAST similar rows first.
        There is no 'farthest' scan to fall back on, so serving it would
        silently be a sequential scan returning the opposite of what the
        index is for."""
        with pytest.raises(ValueError) as exc:
            build_search_query(pgvg, Near("pgsummary", [1.0, 0.0, 0.0], weight=-1.0), k=3)
        assert "needs a positive Near weight" in str(exc.value)
        assert "'pgsummary' has weight=-1.0" in str(exc.value)

    def test_the_traversal_seed_refuses_under_its_own_caller_name(self, pgvg):
        """Every refusal in this library names the call the caller made.
        Reaching it through Start(near=) must say `Start:`, not
        `vector_search():` -- the two are different arguments in
        different places."""
        with pytest.raises(ValueError, match=r"^Start: vector_backend='pgvector' ranks ONE"):
            pgvg.build_query(Start(near=[Near("pgsummary", [1.0, 0.0, 0.0]),
                                         Near("pgtitle", [1.0, 0.0, 0.0])], keep=2), [Hop()])

    def test_a_hop_refuses_under_the_hop_s_own_label(self, pgvg):
        with pytest.raises(ValueError, match=r"^hop 0 \(rels\) via_near: "):
            pgvg.build_query(Start(), [Hop(label="rels", via_keep=1,
                                           via_near=Near("pgrel", [1.0, 0.0, 0.0], weight=-2.0))])

    def test_a_hop_near_refuses_under_the_hop_s_own_label(self, pgvg):
        with pytest.raises(ValueError, match=r"^hop 0 \(docs\): vector_backend"):
            pgvg.build_query(Start(), [Hop(label="docs", keep=2, boost=Boost("priority"),
                                           near=Near("pgsummary", [1.0, 0.0, 0.0]))])

    def test_search_many_refuses_a_mixed_batch_before_the_shape_check(self, pgvg):
        """Two fields in ONE query is the multivector refusal; two
        different fields ACROSS queries is search_many's own shape rule.
        Both must still fire under this backend."""
        with pytest.raises(ValueError, match="ONE field per search"):
            build_search_many_query(pgvg, [[Near("pgsummary", [1.0, 0.0, 0.0]),
                                            Near("pgtitle", [1.0, 0.0, 0.0])]], k=2)
        with pytest.raises(ValueError, match="must share a shape"):
            build_search_many_query(pgvg, [Near("pgsummary", [1.0, 0.0, 0.0]),
                                           Near("pgtitle", [0.0, 1.0, 0.0])], k=2)


class TestBackendsMayNotShareOneColumn:
    """The SQLAlchemy Table metadata is process-global, so two handles
    on the same tables share every vec_* Column. Until this backend
    existed an extra attachment changed no behavior; now the TYPE drives
    how a value binds and how a result is read back, so a disagreement
    means one handle writing float arrays into a vector column."""

    def test_a_pgvector_handle_refuses_a_column_an_exact_one_declared(
            self, metadata_guard):                        # noqa: ARG002 -- the guard is the point
        exact = offline()
        exact.define_vectors(nodes=[Vector("clashfield", 3)])
        with pytest.raises(ValueError) as exc:
            offline_pgvector().define_vectors(nodes=[Vector("clashfield", 3)])
        assert "already declared as real[]" in str(exc.value)
        assert "must pass the same vector_backend=" in str(exc.value)

    def test_an_exact_handle_refuses_a_column_a_pgvector_one_declared(
            self, metadata_guard):                        # noqa: ARG002 -- the guard is the point
        """The mirror image, and it is not symmetry for its own sake:
        the refusal reads the EXISTING type and the WANTED one, so a
        mutant swapping the two would only fail one direction."""
        offline_pgvector().define_vectors(edges=[Vector("clashedge", 3)])
        with pytest.raises(ValueError) as exc:
            offline().define_vectors(edges=[Vector("clashedge", 3)])
        assert "already declared as vector" in str(exc.value)
        assert "needs it as real[]" in str(exc.value)

    def test_two_handles_on_the_same_backend_share_it_happily(
            self, metadata_guard):                        # noqa: ARG002 -- the guard is the point
        """The check is a type CONFLICT, not a re-declaration ban --
        in_graph() and a second handle both re-attach the same column
        every time they are built."""
        offline_pgvector().define_vectors(nodes=[Vector("sharedfield", 3)])
        second = offline_pgvector()
        second.define_vectors(nodes=[Vector("sharedfield", 3)])
        assert second.vectors["nodes"]["sharedfield"].dimensions == 3


class TestPgvectorUnits:
    """hopai.pgvector's own helpers, with no Graph and no database --
    the layer that turns a Python list into SQL text and a catalog
    string back into a version tuple."""

    @pytest.mark.parametrize("raw,expected", [
        ("0.8.0", (0, 8, 0)),
        # A packaged build ('0.8.0-1') and a source build ('0.8.0devel')
        # must compare as 0.8.0 rather than failing to parse and refusing
        # a server that is actually fine.
        ("0.8.0-1", (0, 8, 0)),
        ("0.8.0devel", (0, 8, 0)),
        ("0.7.4", (0, 7, 4)),
        ("1.10.2", (1, 10, 2)),
    ])
    def test_parse_version_reads_what_the_catalog_reports(self, raw, expected):
        assert pg.parse_version(raw) == expected

    def test_the_floor_is_a_comparison_not_a_string_match(self):
        """0.7.4 is refused and 0.8.0 accepted, and '0.10.0' must sort
        ABOVE '0.8.0' -- which a string comparison gets backwards."""
        assert pg.parse_version("0.7.4") < pg.MINIMUM_PGVECTOR
        assert pg.parse_version("0.8.0") >= pg.MINIMUM_PGVECTOR
        assert pg.parse_version("0.10.0") >= pg.MINIMUM_PGVECTOR

    def test_an_unparseable_version_refuses_rather_than_passing(self):
        """Refuse, don't approximate: a version this cannot read must
        not read as 'new enough'."""
        assert pg.parse_version("unknown") < pg.MINIMUM_PGVECTOR

    def test_literal_vector_is_pgvector_s_own_text_form(self):
        assert pg.literal_vector([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"

    def test_literal_vector_spells_every_number_as_a_float(self):
        """repr() would render an int as `1` and a numpy scalar as
        `np.float32(1.0)` -- embedding a Python repr into SQL. float()
        first keeps one spelling."""
        assert pg.literal_vector([1, 0, -2]) == "[1.0,0.0,-2.0]"

    def test_index_name_is_shared_by_every_graph_on_the_column(self):
        """Not graph-scoped, and not for brevity: the COLUMN is shared,
        so a second index over it under another graph's name would
        double every other graph's write cost to buy nothing."""
        assert pg.index_name("nodes", "vec_summary") == "ix_nodes_vec_summary_hnsw"

    def test_index_name_is_truncated_to_the_identifier_limit(self):
        """Postgres truncates a >63-character identifier silently, which
        would make CREATE INDEX and DROP INDEX name different things."""
        name = pg.index_name("nodes", "vec_" + "x" * 80)
        assert len(name) == 63
        assert name.startswith("ix_nodes_vec_")

    def test_the_type_renders_sized_and_unsized(self):
        """`vector(d)` is what DDL needs; the unsized `vector` is enough
        to compile a reference to a column drop_vectors()/load_vectors()
        reach without a declaration."""
        assert pg.Vector(1536).get_col_spec() == "vector(1536)"
        assert pg.Vector().get_col_spec() == "vector"

    @pytest.mark.parametrize("value,expected", [
        ([0.5, 0.25], "[0.5,0.25]"),
        (None, None),
        ("[0.5,0.25]", "[0.5,0.25]"),          # already text: passed through
    ])
    def test_the_bind_processor_sends_pgvector_text(self, value, expected):
        """A list binds as a Postgres ARRAY otherwise, and the server
        refuses that against a `vector` parameter."""
        assert pg.Vector(2).bind_processor(postgresql.dialect())(value) == expected

    @pytest.mark.parametrize("value,expected", [
        ("[1,0,-0.5]", [1.0, 0.0, -0.5]),
        ("[]", []),
        (None, None),
        ([1.0, 0.0], [1.0, 0.0]),              # already parsed: passed through
    ])
    def test_the_result_processor_returns_floats_not_text(self, value, expected):
        """get_vectors() promises [floats] | None under BOTH backends --
        a caller doing arithmetic on the result must not discover that
        one of them hands back the string '[1,0,0]'."""
        assert pg.Vector(3).result_processor(postgresql.dialect(), None)(value) == expected

    def test_a_round_trip_through_both_processors_is_the_identity(self):
        dialect = postgresql.dialect()
        vector = [0.125, -0.5, 1.0]
        wire = pg.Vector(3).bind_processor(dialect)(vector)
        assert pg.Vector(3).result_processor(dialect, None)(wire) == vector

    def test_distance_renders_the_operator_the_index_is_built_on(self, pgvg):
        column = pgvg.nodes_tbl.c["vec_pgsummary"]
        assert "<=>" in str(pg.distance(column, [1.0, 0.0, 0.0], 3))

    def test_similarity_is_one_minus_the_distance_and_nothing_else(self, pgvg):
        """`similarity` has to mean the same number under either
        backend, so this is the whole conversion -- and it lives OUTSIDE
        the ranked subquery, because an HNSW index is on the operator
        and `1 - (a <=> b)` is not an expression it can answer."""
        distance = pg.distance(pgvg.nodes_tbl.c["vec_pgsummary"], [1.0, 0.0, 0.0], 3)
        assert norm(pg.similarity(distance), literal_binds=True) == \
            "1.0 - (nodes.vec_pgsummary <=> CAST('[1.0,0.0,0.0]' AS vector(3)))"

    def test_has_direction_detects_nan_with_a_comparison_not_with_d_ne_d(self, pgvg):
        """A stored all-zero vector has no direction and pgvector's
        cosine distance returns NaN for it. Postgres considers NaN equal
        to itself and GREATER than every real number (unlike IEEE), so
        `d <> d` does NOT detect it and `d <> 'NaN'` does -- getting
        that backwards ranks a directionless vector as the WORST match
        rather than as no match, which is a different answer and not a
        rounding difference."""
        distance = pg.distance(pgvg.nodes_tbl.c["vec_pgsummary"], [1.0, 0.0, 0.0], 3)
        rendered = norm(pg.has_direction(distance), literal_binds=True)
        assert rendered.endswith("!= nan")
        assert "<=>" in rendered

    def test_the_ddl_helpers_name_the_column_the_index_and_the_operator_class(self):
        """Both statements are IF NOT EXISTS, which is what makes
        migrate_vectors() safe to call on every start-up."""
        assert pg.column_ddl("public.nodes", "vec_x", 3) == (
            'ALTER TABLE public.nodes ADD COLUMN IF NOT EXISTS "vec_x" vector(3)')
        assert pg.index_ddl("public.nodes", "nodes", "vec_x") == (
            'CREATE INDEX IF NOT EXISTS "ix_nodes_vec_x_hnsw" ON public.nodes '
            'USING hnsw ("vec_x" vector_cosine_ops)')

    def test_cosine_is_the_metric_on_both_sides_of_the_backend_choice(self):
        """`similarity` must mean the same number under either backend,
        so an L2 operator class here would answer a different question
        than the code it replaces."""
        assert pg.HNSW_OPS == "vector_cosine_ops"
        assert pg.COSINE_DISTANCE == "<=>"

    def test_the_scan_setting_is_strict_order(self):
        """`relaxed_order` lets pgvector return rows slightly out of
        distance order to fill k faster -- a caller reading `similarity`
        down a result list would see it go back UP, which the exact
        backend cannot produce and so neither may this one."""
        assert pg.ITERATIVE_SCAN == "strict_order"


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConnection:
    """Enough of a Connection for ensure_available(): the two catalog
    reads it makes, answered from constructor arguments. A real server
    below the floor is not something a test can arrange, and the three
    refusals are the whole point of the function."""

    def __init__(self, installed=None, available=None, fail_create=False,
                 on_path=True, schema="public"):
        self.installed, self.available, self.fail_create = installed, available, fail_create
        self.on_path, self.schema = on_path, schema
        self.executed, self.rolled_back = [], False

    def execute(self, statement, *args):        # noqa: ARG002
        sql = str(statement)
        self.executed.append(sql)
        if "CREATE EXTENSION" in sql:
            if self.fail_create:
                raise RuntimeError("permission denied to create extension \"vector\"")
            self.installed = self.available
            return _FakeResult(None)
        if "pg_available_extensions" in sql:
            return _FakeResult(self.available)
        if "current_schemas" in sql:
            return _FakeResult(self.on_path)
        if "extnamespace" in sql:
            return _FakeResult(self.schema)
        return _FakeResult(self.installed)

    def rollback(self):
        self.rolled_back = True


class TestPgvectorAvailabilityIsNamedNotGuessed:
    """Three different causes, three different fixes -- a single
    'pgvector unavailable' would send every reader to the wrong one."""

    def test_a_server_without_the_extension_files_says_install_it(self):
        with pytest.raises(RuntimeError) as exc:
            pg.ensure_available(_FakeConnection(installed=None, available=None))
        message = str(exc.value)
        assert "does not have it available" in message
        assert "install it" in message
        assert "CREATE EXTENSION" not in message

    def test_an_uncreated_extension_says_create_extension(self):
        """Available but not created is a one-line fix, and naming the
        upgrade instead would send a superuser hunting for packages."""
        with pytest.raises(RuntimeError) as exc:
            pg.ensure_available(_FakeConnection(installed=None, available="0.8.0"))
        message = str(exc.value)
        assert "CREATE EXTENSION vector" in message
        assert "not a superuser" in message

    def test_a_version_below_the_floor_says_what_it_gets_wrong(self):
        """The floor is measured, not preferred: below 0.8 a selective
        filter returns far fewer rows than match it. The message has to
        say so, or the reader downgrades again to save a migration."""
        with pytest.raises(RuntimeError) as exc:
            pg.ensure_available(_FakeConnection(installed="0.7.4"))
        message = str(exc.value)
        assert ">= 0.8.0" in message and "0.7.4" in message
        assert "fewer rows than match" in message.lower()
        assert "hnsw.iterative_scan" in message

    def test_a_good_server_returns_the_parsed_version(self):
        assert pg.ensure_available(_FakeConnection(installed="0.8.0-1")) == (0, 8, 0)

    def test_an_extension_off_the_search_path_is_named_not_left_to_the_driver(self):
        """Everything this backend emits names `vector` and `<=>`
        UNQUALIFIED, so an extension in a schema the search_path does
        not reach fails as `type "vector" does not exist` -- one line
        after this same function confirmed it IS installed, which is
        the most confusing shape an error can take. Routine whenever an
        application pins search_path to its own schema."""
        with pytest.raises(RuntimeError) as exc:
            pg.ensure_available(
                _FakeConnection(installed="0.8.0", on_path=False, schema="extensions"))
        message = str(exc.value)
        assert "'extensions'" in message
        assert "search_path" in message
        assert "does not exist" in message

    def test_a_two_component_version_is_not_read_as_older_than_the_floor(self):
        """Tuple comparison is length-sensitive in the direction that
        refuses a GOOD server: (0, 8) < (0, 8, 0), so '0.8' would be
        rejected as below a floor it actually meets. pgvector reports
        three components today, which is exactly why this would go
        unnoticed until something else reported two."""
        assert pg.parse_version("0.8") == (0, 8, 0)
        assert pg.parse_version("0.8") >= pg.MINIMUM_PGVECTOR
        assert pg.ensure_available(_FakeConnection(installed="0.8")) == (0, 8, 0)

    def test_creating_the_extension_is_skipped_when_it_already_exists(self):
        conn = _FakeConnection(installed="0.8.0", available="0.8.0")
        pg.ensure_available_or_create(conn)
        assert not any("CREATE EXTENSION" in sql for sql in conn.executed)

    def test_a_role_that_may_not_create_it_is_left_to_ensure_available(self):
        """Failing the migration on the permission error would hide the
        real, fixable situation behind it -- so the failure is swallowed
        here and ensure_available() names the fix next."""
        conn = _FakeConnection(installed=None, available="0.8.0", fail_create=True)
        pg.ensure_available_or_create(conn)
        assert conn.rolled_back
        with pytest.raises(RuntimeError, match="CREATE EXTENSION vector"):
            pg.ensure_available(conn)


# ---------------------------------------------------------------------
# Live: the pgvector backend against a server that has the extension
# ---------------------------------------------------------------------

PG_QUERY = [1.0, 0.0, 0.0]


def _pg_migrated(pgvector_graph) -> Graph:
    pgvector_graph.define_vectors(nodes=[Vector("pgdoc", 3), Vector("pgnote", 3)],
                                  edges=[Vector("pgedge", 3)])
    pgvector_graph.migrate_vectors()
    return pgvector_graph


def _pg_corpus(pgvector_graph) -> Graph:
    """_corpus()'s pgvector twin: the same hand-checkable cosines
    against (1, 0, 0) -- 1.0, 0.8, 0.6, 0.0, -1.0 -- plus a row with no
    vector, a row whose vector has no direction, and a row of the wrong
    `type` for the filter tests."""
    g = _pg_migrated(pgvector_graph)
    g.add_nodes([
        {"id": 1, "type": "doc", "name": "exact"},
        {"id": 2, "type": "doc", "name": "close"},
        {"id": 3, "type": "doc", "name": "closer"},
        {"id": 4, "type": "doc", "name": "orthogonal"},
        {"id": 5, "type": "doc", "name": "opposite"},
        {"id": 6, "type": "doc", "name": "no-vector"},
        {"id": 7, "type": "memo", "name": "wrong-type"},
        {"id": 8, "type": "doc", "name": "directionless"},
    ])
    g.set_vectors(nodes=[
        {"id": 1, "pgdoc": [1.0, 0.0, 0.0]},
        {"id": 2, "pgdoc": [0.6, 0.8, 0.0]},
        {"id": 3, "pgdoc": [0.8, 0.6, 0.0]},
        {"id": 4, "pgdoc": [0.0, 1.0, 0.0]},
        {"id": 5, "pgdoc": [-1.0, 0.0, 0.0]},
        {"id": 7, "pgdoc": [1.0, 0.0, 0.0]},
        {"id": 8, "pgdoc": [0.0, 0.0, 0.0]},
    ])
    return g


def _column_type_of(graph, table: str, column: str) -> str:
    with graph.engine.connect() as conn:
        return conn.execute(text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = CAST(:table AS regclass) AND attname = :column"
        ), {"table": table, "column": column}).scalar()


class TestPgvectorMigrationLive:
    def test_the_column_is_a_typed_vector_of_the_declared_size(self, pgvector_graph):
        """real[] plus a CHECK is the exact backend's shape. Here the
        size lives in the TYPE, which is what makes the index possible
        and what the drift refusals below are about."""
        _pg_migrated(pgvector_graph)
        assert _column_type_of(pgvector_graph, "nodes", "vec_pgdoc") == "vector(3)"
        assert _column_type_of(pgvector_graph, "edges", "vec_pgedge") == "vector(3)"

    def test_an_hnsw_cosine_index_is_created_for_every_field(self, pgvector_graph):
        """The index IS the feature. Without it this backend is the
        exact backend's cost plus an extension to install."""
        _pg_migrated(pgvector_graph)
        with pgvector_graph.engine.connect() as conn:
            definitions = dict(conn.execute(text(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema()"
            )).all())
        for index in ("ix_nodes_vec_pgdoc_hnsw", "ix_edges_vec_pgedge_hnsw"):
            assert index in definitions, sorted(definitions)
            assert "USING hnsw" in definitions[index]
            assert "vector_cosine_ops" in definitions[index]

    def test_no_dimension_check_constraint_is_created(self, pgvector_graph):
        """The one real asymmetry between the backends: vector(d) fixes
        d in the type, so the CHECK the exact backend adds would reject
        nothing and confuse everyone. Its absence is also what the
        different-dimension refusal below exists to make up for."""
        _pg_migrated(pgvector_graph)
        with pgvector_graph.engine.connect() as conn:
            checks = {row[0] for row in conn.execute(text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = CAST('nodes' AS regclass) AND contype = 'c'"))}
        assert not any(name.startswith("ck_vec_dims_") for name in checks), checks

    def test_migrate_is_idempotent(self, pgvector_graph):
        """Every DDL statement is IF NOT EXISTS, so start-up may call
        this every time -- and a second CREATE INDEX must not build a
        second index over the same column."""
        g = _pg_migrated(pgvector_graph)
        first = g.migrate_vectors()
        assert first == g.migrate_vectors()
        assert "nodes.vec_pgdoc" in first and "edges.vec_pgedge" in first
        with g.engine.connect() as conn:
            count = conn.execute(text(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = current_schema() "
                "AND indexname LIKE '%_hnsw'")).scalar()
        assert count == 3

    def test_migrating_over_an_exact_backend_column_names_pgvector_exit_ddl(
            self, pgvector_graph):
        """A real[] column is not converted behind the caller's back:
        rewriting a populated column is a long, locking ALTER, and
        pgvector_exit_ddl() is the call that prints it."""
        with pgvector_graph.engine.begin() as conn:
            conn.execute(text("ALTER TABLE nodes ADD COLUMN vec_legacy real[]"))
        pgvector_graph.define_vectors(nodes=[Vector("legacy", 3)])
        with pytest.raises(ValueError) as exc:
            pgvector_graph.migrate_vectors()
        assert "'_float4'" in str(exc.value)
        assert "pgvector_exit_ddl()" in str(exc.value)

    def test_an_exact_graph_over_a_vector_column_names_the_backend_to_pass(
            self, pgvector_graph):
        """The other direction, and the one a caller hits by forgetting
        `vector_backend='pgvector'` on a second handle. Its own Table
        metadata, because the shared one would refuse at
        define_vectors() and never reach the migration."""
        _pg_migrated(pgvector_graph)
        from sqlalchemy import BigInteger, Column, MetaData, Table, Text
        from sqlalchemy.dialects.postgresql import JSONB

        md = MetaData()
        nodes = Table("nodes", md, Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text), Column("properties", JSONB))
        edges = Table("edges", md, Column("id", BigInteger, primary_key=True),
                      Column("graph_id", Text), Column("start_id", BigInteger),
                      Column("end_id", BigInteger), Column("properties", JSONB))
        exact = Graph(pgvector_graph.engine, node_table=nodes, edge_table=edges)
        exact.define_vectors(nodes=[Vector("pgdoc", 3)])
        with pytest.raises(ValueError) as exc:
            exact.migrate_vectors()
        assert "is a pgvector column" in str(exc.value)
        assert "vector_backend='pgvector'" in str(exc.value)

    def test_a_second_graph_may_not_declare_a_different_dimension(self, pgvector_graph):
        """The capability this backend GIVES UP, pinned so nobody
        rediscovers it as a driver error: under the exact backend two
        graphs may size one field differently, because the rule is a
        per-graph CHECK. Here the size is the column's type, which every
        graph in these tables shares."""
        _pg_migrated(pgvector_graph)
        other = pgvector_graph.in_graph("other")
        other.define_vectors(nodes=[Vector("pgdoc", 4)])
        with pytest.raises(ValueError) as exc:
            other.migrate_vectors()
        assert "already exists as vector(3)" in str(exc.value)
        assert "declares 4" in str(exc.value)
        assert "give this graph a differently named field" in str(exc.value)

    def test_a_second_graph_at_the_same_dimension_shares_the_column(self, pgvector_graph):
        """The refusal above is about the SIZE, not about a second graph
        -- sharing the column is the normal case and must stay free."""
        a = _pg_migrated(pgvector_graph)
        b = pgvector_graph.in_graph("other")
        b.define_vectors(nodes=[Vector("pgdoc", 3)])
        assert b.migrate_vectors() == ["nodes.vec_pgdoc"]
        a.add_nodes([{"id": 1, "t": "a"}])
        b.add_nodes([{"id": 2, "t": "b"}])
        assert a.set_vectors(nodes=[{"id": 1, "pgdoc": [1.0, 0.0, 0.0]}]) == 1
        assert b.set_vectors(nodes=[{"id": 2, "pgdoc": [0.0, 1.0, 0.0]}]) == 1
        assert [h["id"] for h in a.vector_search(Near("pgdoc", PG_QUERY), k=10)] == ["1"]

    def test_the_server_rejects_a_wrong_sized_vector_without_a_check(self, pgvector_graph):
        """What replaces the dimension CHECK: writes that bypass this
        library entirely must still be rejected, which is the reason the
        exact backend compiles a constraint at all."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        with pytest.raises(Exception, match="expected 3 dimensions"), \
                g.engine.begin() as conn:
            conn.execute(text("UPDATE nodes SET vec_pgdoc = '[1,2]' WHERE id = 1"))

    def test_load_vectors_recovers_the_size_from_the_column_type(self, pgvector_graph):
        """There is no CHECK to read here, so the recovery reads
        atttypmod instead -- issue #53's fix has to work under this
        backend too, or a fresh handle gets a raw UndefinedColumn."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        g.set_vectors(nodes=[{"id": 1, "pgdoc": PG_QUERY}])
        fresh = g.in_graph(g.graph)
        recovered = fresh.load_vectors()
        assert recovered["nodes"]["pgdoc"].dimensions == 3
        assert recovered["edges"]["pgedge"].dimensions == 3
        assert [h["id"] for h in fresh.vector_search(Near("pgdoc", PG_QUERY), k=5)] == ["1"]

    def test_drop_vectors_nulls_this_graph_s_values_and_keeps_the_column(
            self, pgvector_graph):
        """The column is shared by every graph in the table, so dropping
        it is a deliberate manual ALTER -- under this backend as much as
        under the other one."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        g.set_vectors(nodes=[{"id": 1, "pgdoc": PG_QUERY}])
        assert g.drop_vectors(node_fields=["pgdoc"]) == ["nodes.vec_pgdoc"]
        assert g.get_vectors(node_ids=[1]) == {"nodes": {"1": {"pgdoc": None, "pgnote": None}},
                                               "edges": {}}
        assert _column_type_of(g, "nodes", "vec_pgdoc") == "vector(3)"


class TestPgvectorWritesAndReadsLive:
    def test_set_vectors_writes_into_the_vector_column(self, pgvector_graph):
        """The rows still travel as float arrays -- binding 1536 floats
        as a vector literal would stringify every row for nothing -- and
        Postgres's real[] -> vector cast does the rest on assignment."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        assert g.set_vectors(nodes=[{"id": 1, "pgdoc": [1.0, 0.0, 0.5]}]) == 1
        with g.engine.connect() as conn:
            assert conn.execute(text("SELECT vec_pgdoc FROM nodes WHERE id = 1")).scalar() \
                == "[1,0,0.5]"

    def test_get_vectors_returns_lists_of_floats_not_pgvector_text(self, pgvector_graph):
        """The line above is what the raw column looks like: the string
        '[1,0,0.5]'. get_vectors() promises `[floats] | None` under both
        backends, so a caller doing arithmetic on the result must never
        get that string back."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        g.set_vectors(nodes=[{"id": 1, "pgdoc": [1.0, 0.0, 0.5]}])
        stored = g.get_vectors(node_ids=[1], node_fields=["pgdoc"])["nodes"]["1"]["pgdoc"]
        assert stored == [1.0, 0.0, 0.5]
        assert all(isinstance(value, float) for value in stored)

    def test_an_unwritten_field_reads_back_as_none(self, pgvector_graph):
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "type": "doc"}])
        assert g.get_vectors(node_ids=[1])["nodes"]["1"] == {"pgdoc": None, "pgnote": None}

    def test_edge_vectors_round_trip_too(self, pgvector_graph):
        """Every edge-side defect this feature has had was a node path
        that worked and an edge twin nobody exercised."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": 1, "t": "x"}, {"id": 2, "t": "y"}])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "k"}])
        with g.engine.connect() as conn:
            edge_id = conn.execute(text("SELECT id FROM edges")).scalar()
        assert g.set_vectors(edges=[{"id": edge_id, "pgedge": [0.0, 1.0, 0.0]}]) == 1
        assert g.get_vectors(edge_ids=[edge_id])["edges"][str(edge_id)]["pgedge"] == \
            [0.0, 1.0, 0.0]


class TestPgvectorSearchLive:
    def test_ranking_and_scores_match_hand_computed_cosine(self, pgvector_graph):
        """Same numbers as the exact backend's twin of this test:
        similarity is `1 - (a <=> b)`, so a caller comparing the two
        backends sees one scale rather than two conventions."""
        g = _pg_corpus(pgvector_graph)
        hits = g.vector_search(Near("pgdoc", PG_QUERY), k=10, where={"type": "doc"})
        assert [h["id"] for h in hits] == ["1", "3", "2", "4", "5"]
        assert [h["similarity"] for h in hits] == pytest.approx(
            [1.0, 0.8, 0.6, 0.0, -1.0], abs=1e-6)
        assert hits[0]["properties"]["name"] == "exact"

    def test_similarities_are_keyed_by_field_name_and_boosts_are_empty(self, pgvector_graph):
        """`hit["similarities"][field]` must never need to branch on
        which backend answered -- sim_0 is a SQL label, not an API."""
        g = _pg_corpus(pgvector_graph)
        hit = g.vector_search(Near("pgdoc", PG_QUERY), k=1)[0]
        assert hit["similarities"] == {"pgdoc": pytest.approx(1.0, abs=1e-6)}
        assert hit["boosts"] == {}

    def test_a_null_vector_and_a_zero_vector_are_both_missing(self, pgvector_graph):
        """The zero vector is the one that bites: it has no direction,
        pgvector's cosine distance returns NaN for it, and Postgres
        sorts NaN ABOVE every real number rather than filtering it out.
        Reported as missing -- a zero vector matches nothing rather than
        matching badly -- so it must not appear at all, and certainly
        not ahead of the row pointing the other way."""
        g = _pg_corpus(pgvector_graph)
        ids = [h["id"] for h in g.vector_search(Near("pgdoc", PG_QUERY), k=10)]
        assert "6" not in ids                  # NULL column
        assert "8" not in ids                  # all zeros -> NaN distance
        assert ids[-1] == "5"                  # the opposite vector is still the worst

    def test_k_truncates_and_where_prefilters(self, pgvector_graph):
        g = _pg_corpus(pgvector_graph)
        assert len(g.vector_search(Near("pgdoc", PG_QUERY), k=2, where={"type": "doc"})) == 2
        assert [h["id"] for h in g.vector_search(Near("pgdoc", PG_QUERY), k=10,
                                                 where={"type": "memo"})] == ["7"]

    def test_a_selective_filter_returns_every_row_that_matches(self, pgvector_graph):
        """THE reason the backend refuses pgvector below 0.8. An HNSW
        scan there visits a fixed candidate window and applies where= to
        what comes back, so a filter matching 2 rows out of hundreds
        returns ONE and reports success -- a row the caller asked for,
        missing, with nothing to indicate it. 0.8's
        hnsw.iterative_scan=strict_order resumes the scan until k rows
        survive the filter, and this test is what says the backend
        actually turns it on."""
        g = _pg_migrated(pgvector_graph)
        rng = random.Random(20240819)
        rows, vectors = [], []
        for node_id in range(1, 601):
            needle = node_id in (137, 491)
            rows.append({"id": node_id, "type": "needle" if needle else "hay"})
            # Random directions, so the two needles are nowhere near the
            # front of an unfiltered scan -- which is what makes the
            # window matter.
            vectors.append({"id": node_id,
                            "pgdoc": [rng.random(), rng.random(), rng.random()]})
        g.add_nodes(rows)
        g.set_vectors(nodes=vectors)
        hits = g.vector_search(Near("pgdoc", PG_QUERY), k=10, where={"type": "needle"})
        assert {h["id"] for h in hits} == {"137", "491"}

    def test_only_a_short_filtered_result_pays_for_a_second_query(self, pgvector_graph):
        """The completion pass is what makes the test above true, and it
        has to stay conditional to be worth having: re-asking exactly on
        EVERY search would hand back the speed this backend exists for,
        and re-asking on none of them is the silent under-return. So a
        filtered result short of k is completed, and a full one -- or an
        unfiltered one, which has nothing to be short of that the index
        would not already have found -- runs once."""
        g = _pg_migrated(pgvector_graph)
        rng = random.Random(20240819)
        g.add_nodes([{"id": i, "type": "needle" if i in (137, 491) else "hay"}
                     for i in range(1, 601)])
        g.set_vectors(nodes=[{"id": i, "pgdoc": [rng.random(), rng.random(), rng.random()]}
                             for i in range(1, 601)])
        ranked = []

        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):       # noqa: ARG001
            if "vec_pgdoc <=>" in statement:
                ranked.append(statement)

        try:
            g.vector_search(Near("pgdoc", PG_QUERY), k=10, where={"type": "needle"})
            short_and_filtered = len(ranked)
            ranked.clear()
            g.vector_search(Near("pgdoc", PG_QUERY), k=10)
            unfiltered = len(ranked)
            ranked.clear()
            g.vector_search(Near("pgdoc", PG_QUERY), k=10, where={"type": "hay"})
            filtered_but_full = len(ranked)
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        assert (short_and_filtered, unfiltered, filtered_but_full) == (2, 1, 1)

    def test_min_similarity_is_applied_before_the_limit(self, pgvector_graph):
        """A floor applied after LIMIT k would silently return fewer
        rows than match, exactly like the window above -- so it travels
        as a DISTANCE bound in the same inner WHERE."""
        g = _pg_corpus(pgvector_graph)
        assert [h["id"] for h in g.vector_search(
            Near("pgdoc", PG_QUERY, min_similarity=0.7), k=10, where={"type": "doc"})] == \
            ["1", "3"]
        # ...and with k SMALLER than the number that clears the floor,
        # the floor is still what decides membership.
        assert [h["id"] for h in g.vector_search(
            Near("pgdoc", PG_QUERY, min_similarity=0.7), k=1, where={"type": "doc"})] == ["1"]

    def test_ties_break_by_id_deterministically(self, pgvector_graph):
        """Two rows at the same distance must come back in the same
        order every time, or a paged caller sees a row twice.

        Filtered and short of k, so this is the exactly-completed
        answer -- which is where the id tiebreak lives. The indexed path
        orders by distance ALONE on purpose (a second sort key is
        exactly what an HNSW index cannot supply), so a tie at a FULL
        k's boundary is the planner's to break, and that is a documented
        backend difference rather than something to assert here."""
        g = _pg_corpus(pgvector_graph)
        assert [h["id"] for h in g.vector_search(
            Near("pgdoc", PG_QUERY, min_similarity=0.99), k=10)] == ["1", "7"]

    def test_a_weight_scales_the_reported_similarity(self, pgvector_graph):
        g = _pg_corpus(pgvector_graph)
        hits = g.vector_search(Near("pgdoc", PG_QUERY, weight=0.5), k=2, where={"type": "doc"})
        assert [h["similarity"] for h in hits] == pytest.approx([0.5, 0.4], abs=1e-6)

    def test_every_search_turns_on_the_iterative_scan(self, pgvector_graph):
        """Per CONNECTION, not per handle: a pooled connection handed
        back and reissued has been reset in between, so caching this the
        way the version check is cached would leave later searches
        under-returning."""
        g = _pg_corpus(pgvector_graph)
        statements = []

        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):       # noqa: ARG001
            statements.append(statement)

        try:
            g.vector_search(Near("pgdoc", PG_QUERY), k=2)
            g.vector_search(Near("pgdoc", PG_QUERY), k=2)
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        assert statements.count("SET hnsw.iterative_scan = strict_order") == 2

    def test_the_version_is_probed_once_per_handle(self, pgvector_graph):
        """The check is a round trip, and it catches a misconfiguration
        that can only be fixed between processes -- paying for it before
        every search would tax the ordinary path forever."""
        g = _pg_corpus(pgvector_graph)
        g._pgvector_version = None
        statements = []

        @event.listens_for(g.engine, "before_cursor_execute")
        def record(conn, cursor, statement, *args):       # noqa: ARG001
            statements.append(statement)

        try:
            g.vector_search(Near("pgdoc", PG_QUERY), k=2)
            first = sum("pg_extension" in s for s in statements)
            statements.clear()
            g.vector_search(Near("pgdoc", PG_QUERY), k=2)
            second = sum("pg_extension" in s for s in statements)
        finally:
            event.remove(g.engine, "before_cursor_execute", record)
        # ensure_available() reads pg_extension twice -- the installed
        # version, then whether its schema is on the search_path -- and
        # what this test pins is that the whole check happens ONCE per
        # handle, not that it costs one statement. The second search
        # must pay nothing.
        assert first > 0
        assert second == 0
        assert g._pgvector_version >= (0, 8, 0)

    def test_the_refusals_reach_a_caller_through_vector_search_itself(self, pgvector_graph):
        """The offline class asserts the messages; this asserts they are
        raised on the live path too, before any SQL runs -- a refusal
        that only fires in build_search_query() would be no refusal at
        all for the caller who never calls it."""
        g = _pg_corpus(pgvector_graph)
        with pytest.raises(ValueError, match="ONE field per search"):
            g.vector_search(Near("pgdoc", PG_QUERY), Near("pgnote", PG_QUERY), k=2)
        with pytest.raises(ValueError, match="does not support boost="):
            g.vector_search(Near("pgdoc", PG_QUERY), k=2, boost=Boost("priority"))
        with pytest.raises(ValueError, match="needs a positive Near weight"):
            g.vector_search(Near("pgdoc", PG_QUERY, weight=-1.0), k=2)


class TestPgvectorSearchManyLive:
    def test_each_query_gets_its_own_result_list(self, pgvector_graph):
        """One statement, one list per query, in the order they were
        given -- the grouping is what the batch is for."""
        g = _pg_corpus(pgvector_graph)
        results = g.vector_search_many(
            [Near("pgdoc", [1.0, 0.0, 0.0]), Near("pgdoc", [0.0, 1.0, 0.0])],
            k=2, where={"type": "doc"})
        assert [[h["id"] for h in one] for one in results] == [["1", "3"], ["4", "2"]]
        assert results[0][0]["similarities"] == {"pgdoc": pytest.approx(1.0, abs=1e-6)}

    def test_a_query_matching_nothing_still_gets_its_own_empty_list(self, pgvector_graph):
        """Positional correspondence is the contract: dropping the empty
        one would shift every later result onto the wrong query."""
        g = _pg_corpus(pgvector_graph)
        results = g.vector_search_many(
            [Near("pgdoc", [1.0, 0.0, 0.0], min_similarity=0.99),
             Near("pgdoc", [0.0, 0.0, 1.0], min_similarity=0.99),
             Near("pgdoc", [0.0, 1.0, 0.0], min_similarity=0.99)],
            k=5, where={"type": "doc"})
        assert [[h["id"] for h in one] for one in results] == [["1"], [], ["4"]]

    def test_missing_vectors_are_missing_for_every_query_in_the_batch(self, pgvector_graph):
        g = _pg_corpus(pgvector_graph)
        results = g.vector_search_many([Near("pgdoc", PG_QUERY)] * 2, k=10)
        for one in results:
            ids = {h["id"] for h in one}
            assert "6" not in ids and "8" not in ids


class TestPgvectorTraversalLive:
    def test_start_near_and_keep_rank_the_seed(self, pgvector_graph):
        g = _pg_corpus(pgvector_graph)
        g.add_edges([{"start_id": 1, "end_id": 4, "kind": "cites"},
                     {"start_id": 3, "end_id": 5, "kind": "cites"}])
        result = g.traverse(Start(where={"type": "doc"}, near=Near("pgdoc", PG_QUERY), keep=2),
                            Hop(via={"kind": "cites"}))
        assert sorted(n["id"] for n in result.nodes) == ["1", "3", "4", "5"]

    def test_a_seed_never_ranks_a_row_with_no_usable_vector(self, pgvector_graph):
        """keep= is a beam, so a NULL or directionless vector taking a
        slot would push a real match out of the walk entirely."""
        g = _pg_corpus(pgvector_graph)
        result = g.traverse(Start(near=Near("pgdoc", PG_QUERY), keep=8), Hop(optional=True))
        assert sorted(n["id"] for n in result.nodes) == ["1", "2", "3", "4", "5", "7"]

    def test_via_near_and_via_keep_choose_the_edges(self, pgvector_graph):
        """The beam ranks EDGES, which is a different table, a different
        alias and a different code path from the node one."""
        g = _pg_migrated(pgvector_graph)
        g.add_nodes([{"id": i, "name": f"n{i}"} for i in range(1, 4)])
        g.add_edges([{"start_id": 1, "end_id": 2, "kind": "k"},
                     {"start_id": 1, "end_id": 3, "kind": "k"}])
        with g.engine.connect() as conn:
            by_end = dict(conn.execute(text("SELECT end_id, id FROM edges")).all())
        g.set_vectors(edges=[{"id": by_end[2], "pgedge": [1.0, 0.0, 0.0]},
                             {"id": by_end[3], "pgedge": [-1.0, 0.0, 0.0]}])
        result = g.traverse(Start(where={"name": "n1"}),
                            Hop(via_near=Near("pgedge", PG_QUERY), via_keep=1))
        assert sorted(n["id"] for n in result.nodes) == ["1", "2"]

    def test_an_aggregate_over_a_ranked_start_runs_the_same_way(self, pgvector_graph):
        """aggregate() has its own session entry point, so the scan
        setting has to be applied there too -- a walk that ranks
        correctly under traverse() and silently under-returns under
        aggregate() would be the worst kind of difference."""
        g = _pg_corpus(pgvector_graph)
        totals = g.aggregate(Start(near=Near("pgdoc", PG_QUERY), keep=3),
                             aggregates={"n": Count()})
        assert totals == {"n": 3}


class TestTheTwoBackendsAgreeLive:
    """Parity is the claim this backend makes: same data, same query,
    same answer -- only faster and approximate. Ranking is what a caller
    consumes, so ranking is what these assert.

    Distinct FIELD NAMES per backend on purpose: the vec_* column is
    shared process-wide, and one column has one type."""

    @staticmethod
    def _vectors():
        # Distinct similarities against the query below -- no ties, so a
        # float difference between the two backends can never flip the
        # ORDER and make this test flap.
        return [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0],
                [-0.6, 0.8, 0.0], [-1.0, 0.0, 0.0]]

    def _both(self, fresh_graph, pgvector_graph):
        fresh_graph.define_vectors(nodes=[Vector("exdoc", 3)], edges=[Vector("exedge", 3)])
        fresh_graph.migrate_vectors()
        pgvector_graph.define_vectors(nodes=[Vector("pgdoc", 3)], edges=[Vector("pgedge", 3)])
        pgvector_graph.migrate_vectors()
        for graph, field in ((fresh_graph, "exdoc"), (pgvector_graph, "pgdoc")):
            graph.add_nodes([{"id": i, "type": "doc", "name": f"n{i}"}
                             for i in range(1, len(self._vectors()) + 1)])
            graph.set_vectors(nodes=[{"id": i + 1, field: vector}
                                     for i, vector in enumerate(self._vectors())])
            graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "cites"},
                             {"start_id": 1, "end_id": 3, "kind": "cites"}])
        return (fresh_graph, "exdoc"), (pgvector_graph, "pgdoc")

    def test_a_search_ranks_the_same_ids_in_the_same_order(self, fresh_graph, pgvector_graph):
        (exact, ex_field), (approx, pg_field) = self._both(fresh_graph, pgvector_graph)
        query = [1.0, 0.0, 0.0]
        exact_hits = exact.vector_search(Near(ex_field, query), k=5)
        approx_hits = approx.vector_search(Near(pg_field, query), k=5)
        assert [h["id"] for h in exact_hits] == [h["id"] for h in approx_hits]
        assert [h["similarity"] for h in approx_hits] == pytest.approx(
            [h["similarity"] for h in exact_hits], abs=1e-6)

    def test_a_filtered_and_floored_search_agrees_too(self, fresh_graph, pgvector_graph):
        (exact, ex_field), (approx, pg_field) = self._both(fresh_graph, pgvector_graph)
        query = [0.0, 1.0, 0.0]
        assert [h["id"] for h in exact.vector_search(
            Near(ex_field, query, min_similarity=0.5), k=5, where={"type": "doc"})] == \
            [h["id"] for h in approx.vector_search(
                Near(pg_field, query, min_similarity=0.5), k=5, where={"type": "doc"})]

    def test_a_ranked_traversal_reaches_the_same_nodes(self, fresh_graph, pgvector_graph):
        (exact, ex_field), (approx, pg_field) = self._both(fresh_graph, pgvector_graph)
        query = [1.0, 0.0, 0.0]
        walk = lambda graph, field: sorted(               # noqa: E731
            node["id"] for node in graph.traverse(
                Start(near=Near(field, query), keep=2),
                Hop(via={"kind": "cites"}, optional=True)).nodes)
        assert walk(exact, ex_field) == walk(approx, pg_field)


# ---------------------------------------------------------------------
# Contracts the mutation run showed nothing was holding
# ---------------------------------------------------------------------

class TestJsonVectorKeysReachTheSpec:
    """Every documented JSON key must land on the right Start/Hop field
    with the right default. Nothing asserted this, so a mutant could
    drop `boost`, misspell `via_near`, or look up `VIA_KEEP` and the
    whole suite stayed green -- the spec grammar and the parser would
    have drifted in silence, which the CLAUDE.md convention forbids."""

    def test_every_start_key_lands(self):
        start, _ = spec_to_traversal({"start": {
            "where": {"type": "doc"}, "label": "s",
            "near": {"field": "summary", "vector": [1.0, 0.0, 0.0]},
            "keep": 7, "boost": {"property": "score", "weight": 0.25},
        }})
        assert start.where == {"type": "doc"} and start.label == "s"
        assert repr(start.near) == "Near('summary', 3 dims)"
        assert start.keep == 7
        assert repr(start.boost) == "Boost('score', weight=0.25)"

    def test_every_hop_key_lands(self):
        _, hops = spec_to_traversal({"start": {"where": {"a": 1}}, "hops": [{
            "where": {"b": 2}, "via": {"kind": "k"}, "hops": [1, 3],
            "direction": "backward", "optional": True, "label": "h",
            "near": {"field": "summary", "vector": [1.0, 0.0, 0.0]}, "keep": 4,
            "via_near": {"field": "rel", "vector": [0.0, 1.0, 0.0]}, "via_keep": 2,
            "boost": {"property": "score", "weight": 0.5},
        }]})
        hop = hops[0]
        assert (hop.min_hops, hop.max_hops) == (1, 3)
        assert hop.direction == "backward" and hop.optional is True and hop.label == "h"
        assert repr(hop.near) == "Near('summary', 3 dims)" and hop.keep == 4
        assert repr(hop.via_near) == "Near('rel', 3 dims)" and hop.via_keep == 2
        assert repr(hop.boost) == "Boost('score', weight=0.5)"

    def test_absent_vector_keys_stay_absent(self):
        start, hops = spec_to_traversal({"start": {"where": {"a": 1}}, "hops": [{}]})
        assert (start.near, start.keep, start.boost) == (None, None, None)
        assert (hops[0].near, hops[0].keep, hops[0].via_near, hops[0].via_keep,
                hops[0].boost) == (None, None, None, None, None)

    @pytest.mark.parametrize("spec,message", [
        ({"start": {"where": {"a": 1}}, "top_k": 3}, r"unknown traversal spec keys \['top_k'\]"),
        ({"start": {"where": {"a": 1}, "limit": 2}}, r'unknown "start" keys \[\'limit\'\]'),
        ({"start": {"wehre": {"a": 1}}}, r'unknown "start" keys \[\'wehre\'\]'),
        ({"start": {"where": {"a": 1}}, "hops": [{"filter": {}}]},
         r'unknown "hops" entry keys \[\'filter\'\]'),
    ])
    def test_unknown_keys_are_refused_by_name(self, spec, message):
        """top_k/limit/filter are the names a model reaches for, and
        each was silently ignored -- answering a different question."""
        with pytest.raises(ValueError, match=message):
            spec_to_traversal(spec)


class TestParseBoostContract:
    @pytest.mark.parametrize("spec,expected", [
        ({"property": "s"}, "Boost('s', weight=1.0)"),
        ({"property": "s", "weight": 0.25}, "Boost('s', weight=0.25)"),
        ({"property": "s", "default": -1.0}, "Boost('s', weight=1.0, default=-1.0)"),
        ({"property": "s", "weight": 2.0, "default": 0.5},
         "Boost('s', weight=2.0, default=0.5)"),
    ])
    def test_defaults_and_keys(self, spec, expected):
        """weight defaults to 1.0 and default to 0.0; both were unpinned,
        so a mutant could ship weight=2.0 for every JSON boost."""
        assert repr(parse_boost(spec)) == expected

    def test_list_form_and_its_empty_refusal(self):
        parsed = parse_boost([{"property": "a"}, {"property": "b", "weight": 2.0}])
        assert [repr(one) for one in parsed] == ["Boost('a', weight=1.0)",
                                                 "Boost('b', weight=2.0)"]
        with pytest.raises(ValueError, match=r'^"boost" is an empty list'):
            parse_boost([])

    def test_messages_are_exact(self):
        """Anchored and full: XX-padding mutants hid inside looser
        matches, and these messages are the whole interface for a
        caller assembling JSON by hand."""
        with pytest.raises(ValueError, match=r'^a boost spec needs "property"'):
            parse_boost({})
        with pytest.raises(TypeError,
                           match=r'^"boost" must be an object or a list of objects -- got int'):
            parse_boost(42)

    def test_callable_boost_repr_names_itself(self):
        assert repr(Boost(lambda p: p, 1.0)) == "Boost(<callable>, weight=1.0)"


class TestVectorCallerNamesArePinned:
    """Every refusal leads with the CALL, so a traceback says which one
    to fix. Each name below was mutable with the suite still green."""

    #: Every caller label this class asserts on. The test below reads the
    #: real call sites out of the source and compares, so a NEW entry
    #: point fails here until someone pins it -- rather than surfacing
    #: two mutation rounds later, which is how the last four were found.
    PINNED = {
        "vector_search()", "vector_search_many()",
        "get_vectors()", "set_vectors()", "stale_vectors()", "embed_stale()",
        "traverse_json()", "aggregate_json()", "vector_search_json()",
        "traversal spec", '"start"', '"hops" entry', "vector search spec",
    }

    def test_every_caller_label_in_the_source_is_pinned_here(self):
        """The durable half of this class.

        Five helpers take a caller label from eleven call sites, and the
        per-site assertions below can only ever cover the sites that
        existed when they were written. Comparing against the source
        makes an unpinned twelfth site a failure, and a removed one a
        failure too -- so this set cannot rot in either direction."""
        declared = declared_caller_labels()
        assert declared - self.PINNED == set(), \
            f"caller labels with no assertion in this class: {sorted(declared - self.PINNED)}"
        assert self.PINNED - declared == set(), \
            f"pinned labels no longer in the source: {sorted(self.PINNED - declared)}"

    def test_get_vectors_names_itself(self):
        with pytest.raises(ValueError, match=r"^get_vectors\(\) needs vector fields"):
            offline().get_vectors(node_ids=[1])

    def test_set_vectors_names_itself(self, fresh_graph):
        """Reached only against a live graph: set_vectors() opens its
        transaction before it consults the registry."""
        with pytest.raises(ValueError, match=r"^set_vectors\(\) needs vector fields"):
            fresh_graph.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 2.0, 3.0]}])

    def test_search_names_itself(self, vg):
        with pytest.raises(ValueError, match=r"^vector_search\(\): boost="):
            vg.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0]), boost=[])

    def test_search_many_names_itself(self, vg):
        with pytest.raises(TypeError, match=r"^vector_search_many\(\): queries must be"):
            vg.build_vector_search_many_query([])

    @pytest.mark.parametrize("bad,exception", [([], ValueError), ("score", TypeError)])
    def test_search_many_names_itself_on_boosts_too(self, vg, bad, exception):
        """Each call site passes its own name to validate_boosts(), so
        each one needs its own assertion -- the batch site was reachable
        only through a message no test read, and a bad boost= there
        reported no caller at all."""
        with pytest.raises(exception, match=r"^vector_search_many\(\): boost="):
            vg.build_vector_search_many_query([Near("summary", [1.0, 0.0, 0.0])], boost=bad)

    def test_search_many_names_itself_on_nears_too(self, vg):
        """validate_nears() is the THIRD helper the batch path hands its
        own name to, after _check_k and validate_boosts. Each is a
        separate argument at a separate call site, so each needs its own
        assertion -- pinning two of the three left the last free."""
        with pytest.raises(ValueError, match=r"^vector_search_many\(\): two Near specs"):
            vg.build_vector_search_many_query(
                [[Near("summary", [1.0, 0.0, 0.0]), Near("summary", [0.0, 1.0, 0.0])]], k=2)

    def test_the_json_search_spec_names_itself_on_an_unknown_key(self, vg):
        """`vector search spec` is the only thing telling a caller WHICH
        JSON shape rejected their key -- traverse_json, aggregate_json
        and this one all route through _check_keys with nothing but that
        label to tell them apart."""
        with pytest.raises(ValueError, match=r"^unknown vector search spec keys \['top_k'\]"):
            vector_search_json(vg, {"near": {"field": "summary", "vector": [1.0, 0.0, 0.0]},
                                    "top_k": 5})

    @pytest.mark.parametrize("caller,run", [
        ("traverse_json()", lambda g, s: traverse_json(g, s)),
        ("aggregate_json()", lambda g, s: aggregate_json(
            g, {**s, "aggregates": {"n": {"fn": "count"}}})),
        ("vector_search_json()",
         lambda g, s: vector_search_json(g, {"near": s["start"]["near"]})),
    ])
    def test_json_refusals_name_the_call(self, vg, caller, run):
        spec = {"start": {"near": {"field": "summary", "vector": [1.0, 0.0, 0.0]}, "keep": 1}}
        with pytest.raises(ValueError,
                           match=rf"^{re.escape(caller)}: near=.*cannot come from a tool call"):
            run(vg, spec)

    @pytest.mark.parametrize("caller,run", [
        ("vector_ddl()", lambda g: g.vector_ddl()),
        ("pgvector_exit_ddl()", lambda g: g.pgvector_exit_ddl()),
        ("migrate_vectors()", lambda g: g.migrate_vectors()),
        ("stale_vectors()", lambda g: g.stale_vectors()),
    ])
    def test_undeclared_handle_refusals_name_the_call(self, caller, run):
        """Four calls share one sentence about calling define_vectors()
        first, and only the caller half tells you which of them you
        reached. Asserting the shared tail leaves every one of these
        names free to be anything."""
        with pytest.raises(ValueError, match=rf"^{re.escape(caller)} needs vector fields"):
            run(offline())

    @pytest.mark.parametrize("caller,run", [
        ("vector_search()", lambda g: g.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), k=0)),
        ("vector_search_many()", lambda g: g.build_vector_search_many_query(
            [Near("summary", [1.0, 0.0, 0.0])], k=0)),
    ])
    def test_bad_k_names_the_call(self, vg, caller, run):
        """`k` is spelled the same on both, so "k must be a positive
        integer" alone does not say which call to go fix."""
        with pytest.raises(ValueError, match=rf"^{re.escape(caller)}: k must be"):
            run(vg)


class TestBuilderDelegation:
    """The Graph methods are thin wrappers, and nothing exercised their
    keywords -- every argument but `queries` could be dropped or
    replaced with None and the suite stayed green."""

    def test_search_many_builder_forwards_every_argument(self, vg):
        sql = norm(vg.build_vector_search_many_query(
            [[Near("rel", [1.0, 0.0, 0.0])]], target="edges", k=3,
            where={"kind": "cites"}, boost=Boost("weight", 0.5)), literal_binds=True)
        assert "FROM edges" in sql            # target=
        assert "LIMIT 3" in sql               # k=
        assert '{"kind": "cites"}' in sql     # where=
        assert "* 0.5" in sql                 # boost=

    def test_search_builder_forwards_every_argument(self, vg):
        sql = norm(vg.build_vector_search_query(
            Near("rel", [1.0, 0.0, 0.0]), target="edges", k=3,
            where={"kind": "cites"}, boost=Boost("weight", 0.5)), literal_binds=True)
        assert "FROM edges" in sql and "LIMIT 3" in sql
        assert '{"kind": "cites"}' in sql and "* 0.5" in sql

    def test_pgvector_index_default_is_hnsw(self, vg):
        """The default is the whole choice this function makes."""
        assert any("USING hnsw" in s for s in vg.pgvector_exit_ddl())

    @pytest.mark.parametrize("build", [
        lambda g: g.build_vector_search_query(Near("summary", [1.0, 0.0, 0.0])),
        lambda g: g.build_vector_search_many_query([Near("summary", [1.0, 0.0, 0.0])]),
    ])
    def test_the_default_k_is_ten(self, vg, build):
        """Every test passed `k` explicitly, so the default both
        builders document could be any number at all. It is a promise
        the README and both docstrings make."""
        assert "LIMIT 10" in norm(build(vg), literal_binds=True)

    @pytest.mark.parametrize("build", [
        lambda g: build_search_query(g, Near("summary", [1.0, 0.0, 0.0])),
        lambda g: build_search_many_query(g, [Near("summary", [1.0, 0.0, 0.0])]),
    ])
    def test_the_same_default_holds_one_layer_down(self, vg, build):
        """Each default is written TWICE -- once on the Graph method and
        once on the function behind it -- so the tests above pin only
        half of each pair, and the halves are free to drift apart. The
        method always passes k explicitly, which is exactly what hides
        a changed default underneath it."""
        assert "LIMIT 10" in norm(build(vg), literal_binds=True)

    def test_the_pgvector_index_default_holds_one_layer_down(self, vg):
        """Same pair, same drift: Graph.pgvector_exit_ddl() passes
        index= down, so only its own default was ever read."""
        assert any("USING hnsw" in s for s in pgvector_exit_ddl(vg))


# ---------------------------------------------------------------------
# Text, embedded on the way in and on the way out
# ---------------------------------------------------------------------

def counting_embedder(dimensions: int = 3, log: list = None):
    """A fake embedding client that records every batch it is handed.

    A plain callable, so it goes down Embedder's last dispatch branch
    and exercises no provider-specific code. The vector it returns
    encodes the text's first character, which is enough to tell two
    embeddings apart without pretending to be a model."""
    log = [] if log is None else log

    def embed(texts):
        log.append(list(texts))
        return [[float(ord(t[0])), 0.0, 0.0][:dimensions] + [0.0] * (dimensions - 3)
                for t in texts]
    embed.log = log
    return embed


class TestVectorSourceAndEmbedder:
    def test_source_defaults_to_the_fields_own_name(self):
        """The gap this closes: Vector("title") named a column and read
        nothing, so "which property does it embed" had no answer at
        all. It is the property of the same name."""
        assert Vector("title", 3).source == "title"

    def test_source_can_point_at_another_property(self):
        assert Vector("summary", 3, source="abstract").source == "abstract"

    @pytest.mark.parametrize("bad", ["", 5, []])
    def test_source_must_be_a_non_empty_string(self, bad):
        with pytest.raises(ValueError, match="source= is the PROPERTY name"):
            Vector("summary", 3, source=bad)

    def test_a_client_is_wrapped_once_into_an_embedder(self):
        field = Vector("summary", 3, embed=counting_embedder())
        assert isinstance(field.embed, Embedder)
        # And an Embedder passed straight in is kept, not re-wrapped --
        # double wrapping is what Embedder(Embedder(...)) refuses.
        made = Embedder(counting_embedder())
        assert Vector("summary", 3, embed=made).embed is made

    def test_an_embedder_sized_for_another_field_is_refused(self):
        """Two numbers, one of which would have to be ignored. Silently
        picking either leaves half the caller's configuration inert."""
        from hopai import Embedder
        with pytest.raises(ValueError, match="its embedder is built with dimensions=768"):
            Vector("summary", 3, embed=Embedder(counting_embedder(), dimensions=768))

    def test_one_embedder_serves_fields_of_different_sizes(self):
        """The flip side: an embedder with no dimensions of its own is
        the ordinary case, and pinning it to the first field it met
        would break the second."""
        shared = Embedder(counting_embedder())
        assert Vector("a", 3, embed=shared).dimensions == 3
        assert Vector("b", 8, embed=shared).dimensions == 8
        assert shared.dimensions is None


class TestNearText:
    def test_text_and_vector_are_alternatives(self):
        for kwargs, word in (({}, "neither"),
                             ({"query": [1.0, 0.0, 0.0], "text": "x"}, "both")):
            with pytest.raises(TypeError, match=f"vector or text to embed, not {word}"):
                Near("summary", **kwargs)

    @pytest.mark.parametrize("bad", ["", "   ", 5])
    def test_text_must_say_something(self, bad):
        with pytest.raises((ValueError, TypeError),
                           match="the text to embed must be a non-empty string"):
            Near("summary", text=bad)

    def test_a_bare_string_is_text_to_embed(self):
        """The terse form. A str and a sequence of numbers can never be
        confused for one another, so the second argument takes either
        and the caller does not have to remember a keyword."""
        assert Near("summary", "raft consensus").text == "raft consensus"
        assert Near("summary", "raft consensus").vector == ()
        # ...and the same spec written explicitly is the same spec.
        assert repr(Near("summary", "raft consensus")) \
            == repr(Near("summary", text="raft consensus"))

    def test_a_sequence_in_the_same_slot_is_still_a_vector(self):
        spec = Near("summary", [0.1, 0.4, 0.9])
        assert spec.text is None and len(spec.vector) == 3

    @pytest.mark.parametrize("looks_like", ["[0.1, 0.2]", "(1, 2)", "  [1,2]  "])
    def test_a_serialized_vector_is_refused_rather_than_embedded(self, looks_like):
        """The one case where the two forms could be confused, and it
        would be silent: embedding the literal characters '[0.1, 0.2]'
        ranks against whatever that phrase means to the model, with a
        confident score attached and nothing to notice."""
        with pytest.raises(ValueError, match="looks like a serialized vector"):
            Near("summary", looks_like)

    def test_text_is_the_way_to_embed_such_a_string_on_purpose(self):
        """The guard above must not make a legitimate string
        unreachable -- text= is unambiguous by construction."""
        assert Near("summary", text="[0.1, 0.2]").text == "[0.1, 0.2]"

    def test_ordinary_brackets_inside_a_sentence_are_not_a_vector(self):
        """The guard is anchored at both ends, so prose that merely
        mentions brackets still embeds."""
        assert Near("summary", "a paper about [databases]").text is not None

    def test_repr_shows_the_text_rather_than_a_dimension_count(self):
        """An unresolved Near has no dimensions yet, and "0 dims" would
        read as a bug in the vector rather than a spec awaiting one."""
        assert repr(Near("summary", text="raft")) == "Near('summary', text='raft')"

    def test_text_is_embedded_with_the_fields_own_embedder(self, vg):
        log = []
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder(log=log))])
        sql = norm(vg.build_vector_search_query(Near("summary", text="apple"), k=1),
                   literal_binds=True)
        assert log == [["apple"]]
        # ord('a') == 97: the embedder's answer is what reached the SQL,
        # so this is the whole round trip, not just the call.
        assert "97" in sql

    def test_a_field_with_no_embedder_says_which_declaration_to_change(self, vg):
        with pytest.raises(ValueError, match=r"vector_search\(\): field 'summary' on nodes "
                                             r"was given text, but declares no embedder"):
            vg.build_vector_search_query(Near("summary", text="apple"), k=1)

    def test_resolving_leaves_the_original_spec_alone(self, vg):
        """A Near is a value: reusing one across two graphs whose fields
        embed differently must not hand the second graph the first
        graph's embedding."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder())])
        spec = Near("summary", text="apple")
        vg.build_vector_search_query(spec, k=1)
        assert spec.text == "apple" and spec.vector == ()

    def test_a_text_query_is_still_checked_against_the_declared_size(self, vg):
        """An embedder answering with the wrong width is a
        configuration mistake, not a query that should run and rank
        nothing."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=lambda texts: [[1.0, 0.0]])])
        with pytest.raises(ValueError, match="has 2 dimensions, the field is defined with 3"):
            vg.build_vector_search_query(Near("summary", text="apple"), k=1)

    def test_two_text_specs_on_one_field_raise_before_either_is_embedded(self, vg):
        """The duplicate-field refusal comes first on purpose: a
        provider call for a query that is about to be rejected is a
        round trip and a bill for nothing."""
        log = []
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder(log=log))])
        with pytest.raises(ValueError, match="two Near specs both rank field 'summary'"):
            vg.build_vector_search_query(
                Near("summary", text="a"), Near("summary", text="b"), k=1)
        assert log == []

    def test_many_queries_cost_one_provider_call_per_field(self, vg):
        """vector_search_many() exists to turn N round trips into one.
        Resolving each query's text on its own would put all N back,
        just against a different server."""
        log = []
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder(log=log))])
        vg.build_vector_search_many_query(
            [Near("summary", text="apple"), Near("summary", text="banana"),
             Near("summary", text="cherry")], k=1)
        assert log == [["apple", "banana", "cherry"]]

    def test_resolution_carries_every_knob_across(self, vg):
        """The resolved Near is a NEW object, so weight, min_similarity
        and missing have to be copied onto it by hand -- and dropping
        any one of them re-ranks the results in silence. Mutation found
        all three unasserted at once."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder()),
                                 Vector("title", 3, embed=counting_embedder())])
        sql = norm(vg.build_vector_search_query(
            Near("summary", text="apple", weight=0.25, min_similarity=0.75,
                 missing="zero"),
            Near("title", text="banana"), k=1), literal_binds=True)
        assert "0.25" in sql                       # weight=
        assert "0.75" in sql                       # min_similarity=
        assert "coalesce" in sql.lower()           # missing="zero"

    def test_a_batch_may_mix_text_and_vectors(self, vg):
        """Queries have to agree on SHAPE -- same fields, weights,
        thresholds -- and how each one arrived is not part of that. So
        a caller holding one embedding already and wanting another
        embedded is a legal batch, and the resolver has to deal the one
        answer it asked for back to the one query that asked."""
        log = []
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder(log=log))])
        sql = norm(vg.build_vector_search_many_query(
            [Near("summary", text="apple"),
             Near("summary", [5.0, 6.0, 7.0]),
             Near("summary", text="banana")], k=1), literal_binds=True)
        assert log == [["apple", "banana"]]        # only the texts were sent
        assert re.search(r"'0'.*?97.*?'1'.*?5\.0.*?'2'.*?98", sql, re.S)

    def test_the_batched_resolver_names_the_call_it_serves(self, vg):
        """_resolve_query_texts() is reached only from
        vector_search_many(), and its refusals are the first thing a
        caller sees when a field cannot embed -- with nothing but the
        label to say which of the search calls they were in.

        It raises through TWO helpers, and each takes the label
        separately: _field() for a name that is not declared at all,
        _embedder() for one that is but cannot embed itself."""
        with pytest.raises(ValueError,
                           match=r"^vector_search_many\(\): field 'summary' on nodes "
                                 r"was given text"):
            vg.build_vector_search_many_query([Near("summary", text="apple")], k=1)
        with pytest.raises(ValueError,
                           match=r"^vector_search_many\(\): no vector field 'ghost'"):
            vg.build_vector_search_many_query([Near("ghost", text="apple")], k=1)

    def test_queries_keep_their_own_text_in_order(self, vg):
        """One batched answer has to be dealt back to the query it came
        from -- zipping it wrongly is a silent mis-ranking."""
        vg.define_vectors(nodes=[Vector("summary", 3, embed=counting_embedder())])
        sql = norm(vg.build_vector_search_many_query(
            [Near("summary", text="apple"), Near("summary", text="banana")], k=1),
            literal_binds=True)
        # ord('a')=97 for query 0, ord('b')=98 for query 1.
        assert re.search(r"'0'.*?97.*?'1'.*?98", sql, re.S)


class TestGraphVectorWrappersForwardEveryArgument:
    """Graph's vector methods are one-line delegations into vectors.py,
    and the mutation runs kept picking a different one apiece: `boost`
    dropped from vector_search_many (twice -- once on Graph, once on
    AsyncGraph), `edge_fields` dropped from drop_vectors. Same defect,
    different member each round, because every prior test called each
    wrapper with the defaults for everything it was not itself about.

    So this closes the FAMILY rather than its members: each wrapper is
    called with a non-default, individually recognizable value for every
    argument it accepts, and every one of those has to arrive at the
    function underneath. Nothing connects -- the delegate is recorded
    and replaced, which is also why a new wrapper joining this table
    costs one line."""

    #: {Graph method: the arguments to pass}. Every VALUE here has to
    #: arrive at the delegate; which of them travel positionally and
    #: which by keyword is the wrapper's business, not this test's.
    CALLS = {
        "drop_vectors": {"node_fields": ["nf"], "edge_fields": ["ef"]},
        "get_vectors": {"node_ids": ["ni"], "edge_ids": ["ei"],
                        "node_fields": ["nf"], "edge_fields": ["ef"]},
        "stale_vectors": {"node_fields": ["nf"], "edge_fields": ["ef"],
                          "limit": 7, "after": "cur"},
        "embed_stale": {"node_fields": ["nf"], "edge_fields": ["ef"],
                        "limit": 7, "batch": 3},
        "set_vectors": {"nodes": ["N"], "edges": ["E"]},
        "vector_search_many": {"queries": ["Q"], "target": "edges", "k": 3,
                               "where": {"w": 1}, "boost": "B", "rerank": "R"},
        "load_vectors": {"connection": "C"},
    }

    #: The two names that differ: Graph says vector_*, vectors.py does
    #: not repeat the word it is already a module about.
    DELEGATE = {"vector_search_many": "search_many", "vector_search": "search"}

    @staticmethod
    def _record(monkeypatch, name: str) -> dict:
        seen: dict = {}

        def recorder(*args, **kwargs):
            seen["passed"] = [*args, *kwargs.values()]
            return "recorded"

        monkeypatch.setattr(f"hopai.vectors.{name}", recorder)
        return seen

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_every_argument_reaches_vectors_py(self, offline_graph, monkeypatch, name):
        arguments = self.CALLS[name]
        seen = self._record(monkeypatch, self.DELEGATE.get(name, name))
        getattr(offline_graph, name)(**arguments)
        for keyword, value in arguments.items():
            assert value in seen["passed"], f"{name}() dropped {keyword}={value!r}"

    def test_vector_search_forwards_its_varargs_and_options(self, offline_graph, monkeypatch):
        """Separate because `near` is *args here and a list one layer
        down -- the wrapper's one piece of real work, and the shape a
        table cannot express."""
        seen = self._record(monkeypatch, "search")
        near = Near("summary", [1.0, 0.0, 0.0])
        offline_graph.vector_search(near, target="edges", k=3, where={"w": 1}, boost="B",
                                    rerank="R")
        assert [near] in seen["passed"]
        for value in ("edges", 3, {"w": 1}, "B", "R"):
            assert value in seen["passed"]
