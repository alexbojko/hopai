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
        property would filter on it and get nothing."""
        from hopai import NodeType, Property

        g = offline()
        g.define_schema(nodes=[NodeType("doc", properties=[Property("title", "string")])])
        g.define_vectors(nodes=[Vector("summary", 1536)], edges=[Vector("rel", 8)])
        dumped = json.dumps(g.tool_schemas())
        assert "title" in dumped                      # declared properties DO appear
        for leaked in ("summary", "vec_summary", "embedding"):
            assert leaked not in dumped, leaked

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
        }, allow_vectors=True)
        assert json.loads(json.dumps(result)) == result
        assert [h["id"] for h in result["results"]] == ["1", "3"]

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

    def test_json_form_matches_the_python_form(self, vg):
        python = vg.build_vector_search_query(
            Near("summary", [1.0, 0.0, 0.0]), boost=Boost("score", 0.5, default=-1.0))
        parsed = parse_boost({"property": "score", "weight": 0.5, "default": -1.0})
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
        assert boosted[0]["similarity"] == pytest.approx(0.8 + 0.9, abs=1e-6)

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
        g = _migrated(fresh_graph)
        g.add_nodes([{"id": 1, "score": 0.0}, {"id": 2, "score": 5.0}, {"id": 3, "n": "target"}])
        g.set_vectors(nodes=[{"id": 1, "docvec": [1.0, 0.0, 0.0]},
                             {"id": 2, "docvec": [0.0, 1.0, 0.0]}])
        g.add_edges([{"start_id": 1, "end_id": 3, "kind": "k"},
                     {"start_id": 2, "end_id": 3, "kind": "k"}])
        result = g.traverse(Start(near=Near("docvec", QUERY), keep=1,
                                  boost=Boost("score", 1.0)), Hop(via={"kind": "k"}))
        # Node 2 loses on similarity and wins on the boost.
        assert {n["id"] for n in result.nodes} == {"2", "3"}


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
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS hopai_beam_scope CASCADE"))
            conn.execute(text("CREATE SCHEMA hopai_beam_scope"))
        meta.create_all(engine)
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
                               "where": {"w": 1}, "boost": "B"},
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
        offline_graph.vector_search(near, target="edges", k=3, where={"w": 1}, boost="B")
        assert [near] in seen["passed"]
        for value in ("edges", 3, {"w": 1}, "B"):
            assert value in seen["passed"]
