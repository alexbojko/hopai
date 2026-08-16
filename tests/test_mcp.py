"""
The MCP server: which tools exist at each permission level, what their
schemas advertise, and the refusals that stand between a model and an
invented embedding.

Most of this needs neither a database nor the MCP SDK. `ToolSpec.call`
is an ordinary function whose parameters are the tool's arguments, so
the whole tool surface is testable by calling it -- the same reason
hopai.mcp separates tools() from build_server(). The TestServer*
classes need the SDK and skip without it; the *Live classes need
PostgreSQL, like the rest of the suite.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re

import pytest
from sqlalchemy.exc import OperationalError

from hopai import (
    AGGREGATE_TOOL_SCHEMA, CypherError, EdgeType, Graph,
    INGEST_TOOL_SCHEMA, Hop, MUTATE_TOOL_SCHEMA, NodeType, Property, Start,
    TRAVERSE_TOOL_SCHEMA, Unique, Vector,
)
from hopai.mcp import DEFAULT_KEEP, SERVER_INSTRUCTIONS, ToolSpec, _seed, build_parser
from hopai.mcp import build_server, main, tools

try:                                    # the extra is optional, and so is testing against it
    import mcp as _sdk
except ImportError:                     # pragma: no cover - depends on the environment
    _sdk = None

needs_sdk = pytest.mark.skipif(_sdk is None,
                               reason="the MCP SDK is not installed -- pip install hopai[mcp]")

# CONSTRUCTING an SDK server does not survive mutmut, and one failing test
# there costs the whole mutation run. mutmut executes the suite IN ITS OWN
# PROCESS, from a `mutants/` copy, with PY_IGNORE_IMPORTMISMATCH=1 -- so a
# module can end up loaded twice under two identities. The MCP 2.x server
# constructor derives an AES-GCM key on the way up, and `cryptography`'s
# Rust binding then rejects its own `SHA256()` with "Expected instance of
# hashes.HashAlgorithm". Nothing in hopai is involved: the same tests pass
# under pytest run inside `mutants/` directly.
#
# It matters because mutmut's stats pass runs with -x: that one error
# aborts the run before a single mutant is checked, and the PR report
# reads `0/N killed, no survivors` -- which looks like a clean sweep and
# is the opposite (CLAUDE.md's triage rule calls this out by name). The
# rest of this file still runs under mutmut and is what kills mutants in
# hopai/mcp.py; the handful of lines only these tests reach are reported
# as `no_tests` rather than as killed, which is the honest answer.
#
# MUTANT_UNDER_TEST is mutmut's own marker, set for every phase it runs.
survives_mutmut = pytest.mark.skipif(
    "MUTANT_UNDER_TEST" in os.environ,
    reason="building an SDK server dies inside cryptography under mutmut's in-process "
           "runner; see the comment in tests/test_mcp.py")


def offline() -> Graph:
    return Graph("postgresql+psycopg2://offline:offline@127.0.0.1:1/offline")


def embedder(dimensions: int = 3):
    """A stand-in for a real embedding function: deterministic, cheap,
    and it records what it was asked to embed."""
    calls = []

    def embed(text: str) -> list:
        calls.append(text)
        return [float(len(text) % 7 + 1), 0.5, -0.25][:dimensions]

    embed.calls = calls
    return embed


def named(graph: Graph, **options) -> dict:
    return {spec.name: spec for spec in tools(graph, **options)}


def parameter_names(schema: dict) -> set:
    """Every parameter name anywhere in a JSON Schema, nested included --
    the same walk tests/test_vectors.py uses on the static schemas."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for name, child in (node.get("properties") or {}).items():
                found.add(name)
                walk(child)
            for key in ("items", "additionalProperties"):
                walk(node.get(key))
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(schema)
    return found


def _object_schema() -> dict:
    return {"type": "object", "properties": {}, "required": []}


class _Stub:
    """Stands in for whatever a stubbed library call returns, for the
    tests that assert on the ARGUMENTS reaching it. The handlers then
    serialize the result, so it has to answer to_dict()."""

    def to_dict(self) -> dict:
        return {}


@pytest.fixture()
def vector_graph() -> Graph:
    g = offline()
    g.define_vectors(nodes=[Vector("summary", 3)], edges=[Vector("rel", 3)])
    return g


def field_embedder(dimensions: int = 3):
    """A field-level `embed=` client, which is NOT the same contract as
    serve(embed=): Vector(..., embed=) hands its client a LIST of texts
    and expects a list of vectors, because it batches. serve(embed=)
    takes one string. Passing one where the other is wanted is the
    mistake this separate helper exists to keep visible."""
    def embed(texts):
        return [[float(len(t) % 7 + 1), 0.5, -0.25][:dimensions] for t in texts]
    return embed


@pytest.fixture
def embedded_graph() -> Graph:
    """A graph whose FIELDS can embed, so `near`/`via_near` text needs
    nothing from the server -- the configuration where the two ways in
    are genuinely independent."""
    g = offline()
    g.define_vectors(nodes=[Vector("summary", 3, embed=field_embedder())],
                     edges=[Vector("rel", 3, embed=field_embedder())])
    return g


# ---------------------------------------------------------------------
# What gets registered
# ---------------------------------------------------------------------

class TestToolInventory:
    def test_default_offers_reads_and_writes_but_no_ddl(self):
        assert set(named(offline())) == {
            "describe_graph", "traverse_graph", "aggregate_graph", "cypher",
            "infer_schema", "ingest_graph", "define_schema",
        }

    def test_read_only_registers_no_write_tool(self):
        """The permission gate is which tools EXIST, not a check inside
        a handler: a tool a model cannot see is one it cannot be argued
        into calling, and one whose refusal cannot be worded badly."""
        assert set(named(offline(), read_only=True)) == {
            "describe_graph", "traverse_graph", "aggregate_graph", "cypher", "infer_schema",
        }

    def test_mutations_are_opt_in(self):
        """Creating rows and destroying them are not the same power: a
        delete matches by filter and does not come back, so the default
        write server does not get one. Without this the flag could
        collapse into `not read_only` and every writing server would
        quietly gain a delete tool."""
        assert "mutate_graph" not in named(offline())
        assert "mutate_graph" in named(offline(), allow_mutations=True)

    def test_mutations_and_read_only_contradict_each_other(self):
        with pytest.raises(ValueError, match="contradict"):
            tools(offline(), read_only=True, allow_mutations=True)

    def test_ddl_is_opt_in(self):
        """enforce_schema runs ALTER TABLE. Being allowed to write rows
        is not the same permission as being allowed to change what the
        database will accept."""
        assert "enforce_schema" not in named(offline())
        assert "enforce_schema" in named(offline(), allow_ddl=True)

    def test_ddl_and_read_only_contradict_each_other(self):
        with pytest.raises(ValueError, match="contradict"):
            tools(offline(), read_only=True, allow_ddl=True)

    def test_search_appears_only_with_an_embedder(self, vector_graph):
        """Vectors never come from the model, so similarity is offered
        only when the OPERATOR wired up something to embed text with."""
        assert "search_similar" not in named(vector_graph)
        assert "search_similar" in named(vector_graph, embed=embedder())

    def test_an_embedder_without_vector_fields_is_refused(self):
        """Nothing to search: vector fields are declared per handle and
        do not travel with the DSN, so a server started without them
        would offer a search tool that fails on every call."""
        with pytest.raises(ValueError, match="define_vectors"):
            tools(offline(), embed=embedder())

    def test_a_non_callable_embedder_is_refused(self, vector_graph):
        with pytest.raises(TypeError, match="callable"):
            tools(vector_graph, embed=[0.1, 0.2])

    def test_strict_schema_without_a_schema_is_refused_at_startup(self):
        """Refusing here means the operator learns at start-up rather
        than through every Cypher call failing later."""
        with pytest.raises(ValueError, match="strict_schema"):
            tools(offline(), strict_schema=True)


class TestManyGraphs:
    """One server, several graphs -- because `Graph` is a cheap handle
    and `in_graph()` shares the engine's pool, so the alternative was
    making an operator run one process per graph to get back what the
    library gives away."""

    @staticmethod
    def two() -> dict:
        base = offline()
        return {"docs": base.in_graph("docs"), "crm": base.in_graph("crm")}

    def test_one_graph_never_mentions_graphs_at_all(self):
        """The single-graph server is the common case and must stay
        exactly as it was: an argument that has one legal value is noise
        in front of a model, and the schemas here are hand-written, so
        nothing else would notice it appearing."""
        for spec in tools(offline(), allow_ddl=True, allow_mutations=True):
            assert "graph" not in spec.parameters["properties"], spec.name

    def test_several_graphs_require_the_name_on_every_tool(self):
        """Required, not defaulted: with several graphs an omitted
        `graph` has no safe reading. Falling back to one of them answers
        a question about another -- and for a write, puts the rows
        there. `graph` is exactly the argument a model forgets."""
        specs = tools(self.two(), allow_ddl=True, allow_mutations=True)
        assert specs, "expected the usual inventory"
        for spec in specs:
            if spec.name == "list_graphs":
                assert spec.parameters["properties"] == {}, "the way IN cannot need a name"
                continue
            key = spec.parameters["properties"].get("graph")
            assert key and key["enum"] == ["docs", "crm"], spec.name
            assert "graph" in spec.parameters["required"], spec.name

    def test_list_graphs_exists_only_when_there_is_a_list(self):
        assert "list_graphs" not in named(offline())
        assert "list_graphs" in named(self.two())

    def test_list_graphs_is_the_index_a_required_name_needs(self):
        """Requiring a name everywhere else creates a chicken and egg:
        this is the one call that answers "which names are there?" and
        the only one that cannot itself take a graph."""
        graphs = self.two()
        graphs["docs"].define_schema(nodes=[NodeType("paper"), NodeType("author")])
        graphs["docs"].define_vectors(nodes=[Vector("summary", 3)])
        listed = named(graphs)["list_graphs"].call()["graphs"]
        assert [entry["graph"] for entry in listed] == ["docs", "crm"]
        assert listed[0]["node_types"] == ["author", "paper"]
        assert listed[0]["vector_fields"] == {"nodes": ["summary"], "edges": []}
        assert listed[1]["schema_declared"] is False and listed[1]["node_types"] is None
        # the reminder rides in the RESULT too, where a model is looking
        # when it picks a name -- a renamed key would drop it silently
        assert "graph` argument" in named(graphs)["list_graphs"].call()["note"]

    def test_an_unnamed_call_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="every call names one -- pass graph="):
            named(self.two())["describe_graph"].call()

    @pytest.mark.parametrize("tool, arguments", [
        ("describe_graph", {}),
        ("traverse_graph", {"start": {"where": {"type": "paper"}}}),
        ("aggregate_graph", {"start": {"where": {}}, "aggregates": {"n": {"fn": "count"}}}),
        ("cypher", {"query": "MATCH (a) RETURN count(a)"}),
        ("infer_schema", {}),
        ("ingest_graph", {"nodes": [{"id": 1}]}),
        ("mutate_graph", {"operations": [{"op": "delete_nodes", "where": {"type": "paper"}}]}),
        ("define_schema", {"schema": {"nodes": {}, "edges": []}}),
        ("enforce_schema", {}),
    ])
    def test_every_tool_routes_the_NAMED_graph(self, tool, arguments):
        """The argument has to reach Served.pick() in every handler, not
        most of them. A handler that passed None instead would quietly
        use the wrong graph on a single-graph server and refuse
        everything on a multi-graph one -- the exact failure the
        required name exists to prevent, and one that survives a
        DB-free suite unless asserted here.

        `graph="nope"` proves the argument arrived: the refusal names it
        back, and it is raised before any connection is opened."""
        spec = {s.name: s for s in tools(self.two(), allow_ddl=True, allow_mutations=True)}[tool]
        with pytest.raises(ValueError,
                           match=rf"{tool}: this server does not serve a graph named 'nope'"):
            spec.call(graph="nope", **arguments)

    def test_a_named_graph_is_the_one_used(self):
        described = named(self.two())["describe_graph"].call(graph="crm")
        assert described["graph"] == "crm"
        # ...and the OTHER names ride along, so a model that reached for
        # describe_graph first still learns what else it may ask about
        # without a second call. A mutant renaming this key dropped that
        # in silence -- the tool answered, just about one graph only.
        assert described["graphs"] == ["docs", "crm"]

    def test_an_unserved_name_is_refused_with_the_list(self):
        """Never answered from the default: reporting one graph's rows
        as another's is the single bug multi-graph scoping can produce,
        and `_scoped()` exists to stop exactly that."""
        with pytest.raises(ValueError, match=r"does not serve a graph named 'crn'.*"
                                             r"it serves \['docs', 'crm'\]"):
            named(self.two())["describe_graph"].call(graph="crn")

    def test_one_graph_still_needs_no_name(self):
        """The single-graph server keeps its surface: nothing to choose,
        nothing to forget, no argument."""
        assert named(offline())["describe_graph"].call()["graph"] == "default"

    def test_each_graph_keeps_its_own_schema(self):
        """in_graph() carries neither schema nor vector fields on
        purpose -- a different graph is allowed a different shape -- so
        the server holds handles, not one handle and a list of names."""
        graphs = self.two()
        graphs["docs"].define_schema(nodes=[NodeType("paper")])
        graphs["crm"].define_schema(nodes=[NodeType("account")])
        describe = named(graphs)["describe_graph"]
        assert set(describe.call(graph="docs")["schema"]["nodes"]) == {"paper"}
        assert set(describe.call(graph="crm")["schema"]["nodes"]) == {"account"}

    def test_descriptions_name_the_graphs_instead_of_one_vocabulary(self):
        """With several graphs there is no single vocabulary to append,
        and appending the default's would present one graph's type names
        as every graph's. This library supports thousands of graphs, so
        a per-graph summary in every description is not an option
        either -- they are named, and describe_graph has the rest."""
        graphs = self.two()
        graphs["docs"].define_schema(nodes=[NodeType("paper")])
        for spec in tools(graphs):
            assert "'docs', 'crm'" in spec.description, spec.name
            assert "Node types: paper" not in spec.description, spec.name

    def test_strict_schema_needs_one_on_every_served_graph(self):
        """A per-call `graph` argument would otherwise reach a graph
        that cannot be strict, and the refusal would name Cypher rather
        than the missing schema."""
        graphs = self.two()
        graphs["docs"].define_schema(nodes=[NodeType("paper")])
        with pytest.raises(ValueError, match=r"\['crm'\] has none"):
            tools(graphs, strict_schema=True)

    def test_the_search_field_enum_is_the_union_across_graphs(self):
        """One schema serves every graph, so the enum is the union;
        naming a field the CHOSEN graph does not declare is refused per
        call, by that graph's own field list."""
        graphs = self.two()
        graphs["docs"].define_vectors(nodes=[Vector("summary", 3)])
        graphs["crm"].define_vectors(nodes=[Vector("notes", 3)])
        specs = named(graphs, embed=embedder())
        start = specs["traverse_graph"].parameters["properties"]["start"]
        assert start["properties"]["search_field"]["enum"] == ["notes", "summary"]
        with pytest.raises(ValueError, match=r"defined: \['notes'\]"):
            specs["search_similar"].call(query="x", field="summary", graph="crm")

    def test_a_seeded_traversal_embeds_against_the_graph_it_was_given(self):
        """`start.search` must resolve its field on the CHOSEN graph:
        handing _seed the wrong graph -- or no embedder -- searches
        somewhere the caller did not ask for. Refused before any
        connection, so this needs no database."""
        graphs = self.two()
        graphs["docs"].define_vectors(nodes=[Vector("summary", 3)])
        spec = named(graphs, embed=embedder())["traverse_graph"]
        with pytest.raises(ValueError,
                           match="traverse_graph: no vector fields are defined for nodes"):
            spec.call(start={"search": "anything"}, graph="crm")

    def test_an_embedder_needs_vectors_on_only_one_of_them(self):
        graphs = self.two()
        graphs["docs"].define_vectors(nodes=[Vector("summary", 3)])
        specs = named(graphs, embed=embedder())
        assert "search_similar" in specs
        # ...and the graph without them says so rather than ranking nothing
        with pytest.raises(ValueError, match="no vector fields are defined for nodes"):
            specs["search_similar"].call(query="x", graph="crm")

    @pytest.mark.parametrize("bad, message", [
        ({}, "non-empty {name: Graph} mapping"),
        ([], "non-empty {name: Graph} mapping"),
        ("docs", "non-empty {name: Graph} mapping"),
        ({"docs": "not a graph"}, "graph 'docs' must be a Graph, got str"),
        ({"": offline()}, "a graph name must be a non-empty string"),
    ])
    def test_a_malformed_registry_is_refused_by_name(self, bad, message):
        """The message, not just the type: an operator wiring up a
        server reads it once, at start-up, and a mutant that blanked
        one survived because only the exception class was asserted."""
        with pytest.raises((TypeError, ValueError), match=re.escape(message)):
            tools(bad)


# ---------------------------------------------------------------------
# What the model is told
# ---------------------------------------------------------------------

class TestToolSchemas:
    @pytest.mark.parametrize("name, static", [
        ("traverse_graph", TRAVERSE_TOOL_SCHEMA),
        ("aggregate_graph", AGGREGATE_TOOL_SCHEMA),
        ("mutate_graph", MUTATE_TOOL_SCHEMA),
    ])
    def test_the_advertised_schema_is_hopais_own(self, name, static):
        """The SDK would derive `{"type": "object"}` from `start: dict`
        and leave the model to guess `where`, `via`, `hops` and
        `direction`. hopai ships hand-written schemas that spell those
        out and are kept in step with the parsers; this is the test that
        they are what actually reaches a client. mutate_graph is the one
        with a `oneOf` per operation -- the part a derived schema loses
        most, since `operations: list` says nothing about which four
        shapes an entry may take."""
        assert named(offline(), allow_mutations=True)[name].parameters == static["parameters"]

    def test_ingest_adds_the_merge_keys_to_the_static_schema(self):
        """Update, not just create: without merge_nodes_on a model can
        only ever insert, and re-running an ingestion duplicates rows."""
        parameters = named(offline())["ingest_graph"].parameters
        assert set(INGEST_TOOL_SCHEMA["parameters"]["properties"]) <= set(parameters["properties"])
        assert "merge_nodes_on" in parameters["properties"]
        assert "merge_edges_on" in parameters["properties"]

    def test_every_advertised_parameter_is_one_the_handler_accepts(self):
        """A schema is a promise. An advertised parameter no handler
        takes is a promise broken at call time, with a TypeError the
        model cannot act on -- and the schemas are hand-written here, so
        nothing else would catch it."""
        graph = offline()
        graph.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)])
        # EVERY configuration, because a tool that is not registered is a
        # tool this cannot check: run without an embedder and allow_ddl,
        # search_similar and enforce_schema went unexamined -- and
        # mutants renaming search_similar's `query`, `k`, `target` and
        # `where`, and enforce_schema's `dry_run` and `endpoints`, all
        # survived on exactly that gap. Two vector fields so the
        # search_field key exists to be checked at all.
        for spec in tools(graph, embed=embedder(), allow_ddl=True, allow_mutations=True):
            accepted = set(inspect.signature(spec.call).parameters)
            assert set(spec.parameters["properties"]) <= accepted, spec.name

    def test_required_parameters_are_exactly_the_handlers_required_ones(self):
        for spec in tools(offline(), allow_mutations=True):
            needed = {name for name, p in inspect.signature(spec.call).parameters.items()
                      if p.default is inspect.Parameter.empty}
            assert set(spec.parameters.get("required", [])) == needed, spec.name

    def test_a_required_parameter_may_exceed_the_signature_but_never_the_reverse(self):
        """`graph` is required only when several graphs are served, and
        one handler serves both configurations -- so its signature
        cannot say it and Served.pick() enforces it instead. The rule
        that still has to hold: everything advertised is accepted, and
        everything the signature demands is advertised."""
        base = offline()
        for spec in tools({"docs": base.in_graph("docs"), "crm": base.in_graph("crm")}):
            accepted = set(inspect.signature(spec.call).parameters)
            required = set(spec.parameters.get("required", []))
            needed = {name for name, p in inspect.signature(spec.call).parameters.items()
                      if p.default is inspect.Parameter.empty}
            assert required <= accepted, spec.name
            assert needed <= required, spec.name

    def test_no_tool_advertises_a_vector_parameter(self, vector_graph):
        """The invariant tests/test_vectors.py pins for the static tool
        schemas, restated for every tool this server registers -- with
        an embedder configured, which is the only configuration where a
        similarity parameter could plausibly appear.

        What must never appear is a place to PUT FLOATS: a model asked
        for an embedding invents one, and an invented embedding finds
        confidently wrong neighbors. Everything else in the similarity
        family is advertised on purpose -- `near` and `via_near` carry
        the model's TEXT, `keep`/`via_keep` are counts, `boost` names a
        property. A model supplies those as truthfully as it supplies a
        filter, and refusing them put semantic search out of a tool
        call's reach entirely.

        So the forbidden set is two names, not seven, and the test still
        walks $defs/anyOf -- `vector` hiding one level down inside the
        near definition is exactly the regression this catches."""
        forbidden = {"vector", "embedding"}
        for spec in tools(vector_graph, embed=embedder(), allow_ddl=True, allow_mutations=True):
            assert not parameter_names(spec.parameters) & forbidden, spec.name

    def test_the_near_definition_offers_text_and_no_vector(self, vector_graph):
        """The other half of the same rule: `near` being advertised is
        only safe while the thing it advertises has no float-shaped
        slot. A near spec a model can fill in must offer `text` and
        must not offer `vector`."""
        for spec in tools(vector_graph, embed=embedder()):
            near = spec.parameters.get("$defs", {}).get("near")
            if near is None:
                continue
            assert "text" in near["properties"], spec.name
            assert "vector" not in near["properties"], spec.name

    @pytest.mark.parametrize("tool, method, arguments, forwarded", [
        ("ingest_graph", "ingest",
         {"nodes": [{"id": 1}], "merge_nodes_on": ["email"], "merge_edges_on": ["ref"]},
         {"merge_nodes_on": ["email"], "merge_edges_on": ["ref"]}),
        ("enforce_schema", "enforce_schema",
         {"dry_run": False, "endpoints": True}, {"endpoints": True}),
    ])
    def test_an_optional_argument_reaches_the_call_it_names(self, tool, method, arguments,
                                                           forwarded, monkeypatch):
        """An advertised argument that parses and is then dropped on the
        way to the library fails in silence and in the direction nobody
        wants: `merge_edges_on` omitted turns an upsert back into an
        insert, and `endpoints` omitted leaves the endpoint-type trigger
        off while reporting success. Both mutants -- the argument
        dropped, and `endpoints=None` -- survived because the only tests
        that could see them need a database.

        The library call is stubbed rather than run, which is what makes
        this checkable with nothing running: what is asserted is the
        handoff, not what happens after it."""
        seen = {}
        monkeypatch.setattr(f"hopai.core.Graph.{method}",
                            lambda self, *a, **kw: seen.update(kw) or _Stub())
        graph = offline()
        graph.define_schema(nodes=[NodeType("person")])
        named(graph, allow_ddl=True)[tool].call(**arguments)
        for name, value in forwarded.items():
            assert seen.get(name) == value, name

    def test_descriptions_carry_this_graphs_vocabulary(self):
        """Same reason Graph.tool_schemas() exists: a model that has
        been told the type names does not guess them, and a guessed
        type name returns an empty result rather than an error."""
        g = offline()
        g.define_schema(nodes=[NodeType("person", properties=[Property("email", "string")]),
                               NodeType("company")],
                        edges=[EdgeType("works_at", "person", "company")])
        for spec in tools(g):
            if spec.name in ("traverse_graph", "aggregate_graph", "cypher",
                             "ingest_graph", "define_schema"):
                assert "person" in spec.description, spec.name

    def test_building_tools_never_edits_the_module_constants(self, vector_graph):
        """tool_schemas() hands back deep copies and the search keys are
        added to those. Mutating the shared constant would leak a
        `search` parameter into every OTHER integration in the process
        -- including ones with no embedder behind it."""
        before = json.dumps(TRAVERSE_TOOL_SCHEMA, sort_keys=True)
        tools(vector_graph, embed=embedder())
        assert json.dumps(TRAVERSE_TOOL_SCHEMA, sort_keys=True) == before

    def test_the_one_way_in_sentence_is_replaced_by_the_two_way_one(self, vector_graph):
        """The static schema points a model at `near` -- the field
        embeds the text, and that needs nothing from this server. An
        `embed` callable adds a SECOND way in, so the sentence is
        replaced rather than appended to: a description naming one route
        while the parameters offer two is how a model picks neither.

        This used to assert the static sentence said meaning-based
        lookup was impossible. It is not impossible any more; what the
        server adds is the fallback for fields carrying no embedder."""
        plain = named(vector_graph)["traverse_graph"].description
        seeded = named(vector_graph, embed=embedder())["traverse_graph"].description
        assert "`near`" in plain and "start.search" not in plain
        assert "`near`" in seeded and "start.search" in seeded
        assert "not both" in seeded

    def test_a_seed_set_still_has_to_come_from_somewhere(self, vector_graph):
        """The static schema requires `where` on start. Search is a
        second way in, not a way to ask for every node in the graph, so
        the requirement becomes a choice between the two rather than
        disappearing."""
        start = named(vector_graph,
                      embed=embedder())["traverse_graph"].parameters["properties"]["start"]
        assert "required" not in start
        assert start["anyOf"] == [{"required": ["where"]}, {"required": ["search"]}]

    def test_edge_only_vectors_offer_search_but_not_a_seed(self):
        """A seed ranks NODE vectors. A graph with only edge vector
        fields can still be searched, but advertising `start.search`
        there would put a parameter in front of a model that fails on
        every use. (Found by a surviving mutant.)"""
        g = offline()
        g.define_vectors(edges=[Vector("rel", 3)])
        specs = named(g, embed=embedder())
        assert "search_similar" in specs
        start = specs["traverse_graph"].parameters["properties"]["start"]
        assert "search" not in start["properties"]
        assert "start.search" not in specs["traverse_graph"].description

    def test_aggregation_can_be_seeded_by_meaning_too(self):
        """"How many papers cite anything about retrieval" is the same
        question as the traversal, counted. Wiring the seed into only
        one of the two tools is a silent half-feature."""
        g = offline()
        g.define_vectors(nodes=[Vector("summary", 3)])
        spec = named(g, embed=embedder())["aggregate_graph"]
        assert "search" in spec.parameters["properties"]["start"]["properties"]
        assert "start.search" in spec.description

    @pytest.mark.parametrize("broken, message", [
        ({"name": ""}, "a tool needs a name"),
        ({"description": None}, "has no description"),
        ({"description": "   "}, "has no description"),
        ({"parameters": {"properties": {}}}, "must be a JSON Schema object"),
        ({"call": "not callable"}, "must be callable"),
    ])
    def test_a_tool_that_cannot_describe_itself_is_not_built(self, broken, message):
        """Checked on the thing, not in a test, because the SDK ships
        whatever it is handed: an undescribed tool reaches the model as
        one it can only guess at. A mutant that blanked list_graphs's
        description got all the way to a registered tool."""
        fields = {"name": "t", "description": "d", "parameters": _object_schema(),
                  "call": lambda: None, **broken}
        with pytest.raises((ValueError, TypeError), match=re.escape(message)):
            ToolSpec(**fields)

    def test_search_similar_offers_no_field_when_there_is_one(self):
        """Same rule as `search_field`: an argument with one legal value
        is noise. Pinned on both sides, because the boundary is a
        comparison a mutant can slide by one."""
        one = offline()
        one.define_vectors(nodes=[Vector("summary", 3)])
        spec = named(one, embed=embedder())["search_similar"]
        assert "field" not in spec.parameters["properties"]
        assert inspect.signature(spec.call).parameters["k"].default == 10

    def test_search_similar_advertises_the_fields_that_exist(self):
        """The enum is what stops a model inventing a field name. It is
        built from the registry, and a mutation that read the registry
        under a target name that does not exist -- leaving the enum
        short -- survived until this asserted the contents."""
        g = offline()
        g.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)],
                         edges=[Vector("rel", 3)])
        spec = named(g, embed=embedder())["search_similar"]
        assert spec.parameters["properties"]["field"]["enum"] == ["rel", "summary", "title"]
        assert spec.parameters["required"] == ["query"]

        # two is already a choice: the boundary is a comparison, and a
        # mutant that slid it to `> 2` kept three fields working
        two = offline()
        two.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)])
        assert "field" in named(two, embed=embedder())["search_similar"].parameters["properties"]

    def test_search_refusals_name_the_tool_and_the_graphs_fields(self):
        """Both refusals carry the caller: "which of my calls was this?"
        is the first thing a model has to answer."""
        g = offline()
        g.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)])
        spec = named(g, embed=embedder())["search_similar"]
        with pytest.raises(ValueError, match=r"search_similar: no vector field 'sumary'"):
            spec.call(query="x", field="sumary")
        with pytest.raises(ValueError, match="search_similar: this graph defines several"):
            spec.call(query="x")

    def test_the_advertised_enums_are_values_the_handler_takes(self, vector_graph):
        """An `enum` is the tightest promise a schema makes: a model
        picks from it verbatim and nothing else. `target` advertising
        "NODES" while the handler branches on "nodes" is a tool that
        fails on every well-formed call -- and the mutant that wrote it
        survived, because every test until now passed `target` by hand
        or left it out.

        Asserted by CALLING with each advertised value, not by
        comparing the list to a second copy of itself."""
        spec = named(vector_graph, embed=embedder())["search_similar"]
        for value in spec.parameters["properties"]["target"]["enum"]:
            # reaches the database that is not there, which is proof the
            # value was understood rather than rejected as unknown
            with pytest.raises(OperationalError):
                spec.call(query="x", target=value)

    def test_search_names_itself_when_the_graph_is_unknown(self):
        graphs = {"docs": offline().in_graph("docs"), "crm": offline().in_graph("crm")}
        graphs["docs"].define_vectors(nodes=[Vector("summary", 3)])
        with pytest.raises(ValueError, match="search_similar: this server does not serve"):
            named(graphs, embed=embedder())["search_similar"].call(query="x", graph="nope")

    @pytest.mark.parametrize("options", [{}, {"allow_ddl": True}, {"read_only": True},
                                         {"allow_mutations": True}, {"embed": embedder()}])
    @pytest.mark.parametrize("many", [False, True])
    def test_every_built_schema_is_a_well_formed_object_schema(self, options, many,
                                                              vector_graph):
        """The tools this module writes itself build their schemas
        rather than copying hopai's. A malformed one is not rejected by
        anything -- the SDK ships whatever it is given, and the model
        gets a schema it cannot satisfy.

        Every configuration, because each one builds a parameter the
        others do not: `search_similar` exists only with an embedder,
        the `graph` key only with several graphs, and mutants broke
        both of those -- one writing a `type` of "STRING", one renaming
        the `description` key so it vanished."""
        if many:
            vector_graph = {"docs": vector_graph, "crm": vector_graph.in_graph("crm")}
        json_schema_types = {"string", "number", "integer", "boolean",
                             "object", "array", "null"}
        for spec in tools(vector_graph, **options):
            assert spec.parameters["type"] == "object", spec.name
            assert isinstance(spec.parameters["properties"], dict), spec.name
            assert isinstance(spec.parameters.get("required", []), list), spec.name
            def well_formed(node, where):
                """Nested too: an `items` or a sub-object schema reaches
                the model just as surely as a top-level one, and a
                mutant blanked the `properties` of define_schema's
                edge entries."""
                for key in ("items", "additionalProperties"):
                    if isinstance(node.get(key), dict):
                        well_formed(node[key], f"{where}.{key}")
                if node.get("type") == "object" and "properties" in node:
                    assert isinstance(node["properties"], dict), where
                    assert isinstance(node.get("required", []), list), where
                    for child, spec_ in node["properties"].items():
                        assert isinstance(spec_, dict), f"{where}.{child}"
                        well_formed(spec_, f"{where}.{child}")

            for name, prop in spec.parameters["properties"].items():
                where = f"{spec.name}.{name}"
                well_formed(prop, where)
                # A JSON Schema type, not merely a truthy string: "STRING"
                # is not one, and a mutant that wrote it survived here.
                assert prop.get("anyOf") or prop.get("type") in json_schema_types, where
                # And a description, because a parameter without one is a
                # parameter a model guesses at -- which is the whole
                # argument for hand-writing these schemas. A mutant
                # renaming the `description` KEY drops it in silence.
                assert prop.get("description", "").strip(), where

    def test_the_keys_this_module_injects_are_well_formed_too(self, vector_graph):
        """The check above stops at the outermost properties, and the
        keys this module adds to a COPIED schema are nested one level
        down: `search`, `keep` and `search_field` live inside `start`.
        Mutants that gave `keep` a `type` of "STRING" and renamed
        `search_field`'s `description` KEY survived because of exactly
        that -- and a model reads a nested parameter as literally as a
        top-level one.

        Nested rather than deep-walked because the static schemas are
        not this module's to police: TRAVERSE_TOOL_SCHEMA's own
        `hops.items.direction` carries an enum and no description, which
        is json_api.py's call to make."""
        json_schema_types = {"string", "number", "integer", "boolean",
                             "object", "array", "null"}
        graphs = {"docs": vector_graph, "crm": vector_graph.in_graph("crm")}
        graphs["crm"].define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)])
        seen = set()
        for spec in tools(graphs, embed=embedder()):
            if spec.name == "list_graphs":    # the way IN cannot need a name
                continue
            injected = {"graph": spec.parameters["properties"].get("graph")}
            start = spec.parameters["properties"].get("start")
            if start is not None:
                for key in ("search", "keep", "search_field"):
                    injected[f"start.{key}"] = (start.get("properties") or {}).get(key)
            for where, schema in injected.items():
                # PRESENT, not merely well-formed if present. Skipping a
                # missing key made this test pass for the mutant that
                # renamed `keep` to `KEEP`: the key it looked up was
                # gone, so there was nothing to check and nothing to
                # object to. A rename is exactly what has to fail here --
                # the handler still takes `keep`, so the model is being
                # offered a name that will be rejected.
                assert schema is not None, f"{spec.name}.{where} is not advertised"
                assert schema.get("type") in json_schema_types, f"{spec.name}.{where}"
                assert schema.get("description", "").strip(), f"{spec.name}.{where}"
                seen.add(where)
        # and the loop above has to have HAD something to check: every
        # injected key reached, or this asserts nothing at all
        assert seen == {"graph", "start.search", "start.keep", "start.search_field"}

    def test_search_field_is_offered_only_when_the_choice_is_real(self):
        """One declared field needs no argument. Several make the choice
        the caller's, because ranking against the wrong field returns
        confident nonsense instead of an error."""
        one = offline()
        one.define_vectors(nodes=[Vector("summary", 3)])
        start = named(one, embed=embedder())["traverse_graph"].parameters["properties"]["start"]
        assert "search_field" not in start["properties"]

        two = offline()
        two.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)])
        start = named(two, embed=embedder())["traverse_graph"].parameters["properties"]["start"]
        assert start["properties"]["search_field"]["enum"] == ["summary", "title"]


# ---------------------------------------------------------------------
# The similarity rule
# ---------------------------------------------------------------------

class TestVectorsNeverComeFromTheModel:
    @pytest.mark.parametrize("tool, arguments", [
        ("traverse_graph", {}),
        ("aggregate_graph", {"aggregates": {"n": {"fn": "count"}}}),
    ])
    def test_a_start_near_is_refused(self, tool, arguments):
        """The whole reason this server embeds text itself. Without this
        the tool would happily rank against invented floats and return a
        plausible subgraph, which is the worst thing this library can
        produce."""
        spec = named(offline())[tool]
        # the refusal names the TOOL that made it, not just the rule:
        # "which of my calls was this?" is the first thing a model has
        # to answer, and a mutation blanking the caller survived. Both
        # tools, because they pass their own name to _seed and a mutant
        # mangled aggregate_graph's while traverse_graph's was pinned.
        with pytest.raises(ValueError, match=rf"{tool}: near=.*cannot come from"):
            spec.call(start={"near": {"field": "summary", "vector": [0.1, 0.2, 0.3]}},
                      **arguments)

    @pytest.mark.parametrize("key", ["near", "via_near"])
    @pytest.mark.parametrize("tool, arguments", [
        ("traverse_graph", {}),
        ("aggregate_graph", {"aggregates": {"n": {"fn": "count"}}}),
    ])
    def test_hop_vector_keys_are_refused_too(self, key, tool, arguments):
        """Hops are passed through to json_api verbatim, and the seeded
        path calls it with allow_vectors=True -- so the hops have to be
        refused here or the invariant would hold only for `start`.

        Only the two keys that CARRY a near spec can smuggle floats in;
        `keep`/`via_keep`/`boost` hold an integer or a property name and
        are pinned to the opposite behavior just below.

        Both tools name themselves in the refusal, and both are checked:
        a mutant that mangled aggregate_graph's caller name survived
        while traverse_graph's was pinned. "Which of my calls was this?"
        is the first thing a model has to answer, and the answer being
        right for one of two tools is how it stops being reliable."""
        spec = named(offline())[tool]
        with pytest.raises(ValueError, match=rf"{tool}: {key}=.*cannot come from"):
            spec.call(start={"where": {"type": "person"}},
                      hops=[{key: {"field": "summary", "vector": [1.0, 2.0, 3.0]}}],
                      **arguments)

    @pytest.mark.parametrize("hop", [
        {"near": {"field": "summary", "text": "raft"}, "keep": 3},
        {"via_near": {"field": "rel", "text": "cites"}, "via_keep": 3},
        {"near": {"field": "summary", "text": "raft"}, "keep": 3,
         "boost": {"property": "rank"}},
    ])
    def test_the_countable_similarity_keys_are_not_refused(self, embedded_graph, hop):
        """The half that stops the refusal creeping back outwards.
        `keep`, `via_keep` and `boost` were refused as a family with
        `near`, which put semantic search out of a tool call's reach
        entirely -- they hold an integer and a property name, which a
        model supplies as truthfully as it supplies a filter.

        Each is paired with the near it ranks, because `keep` alone is
        "keep the top 3 of WHAT" and hopai refuses it on its own -- the
        first draft of this test asserted the opposite and was wrong
        about the code rather than the other way round.

        The server has no embedder here on purpose: the FIELDS do, so
        this pins that `near` text needs nothing from serve(embed=).

        OperationalError is the assertion, and it has to be that
        specific: reaching the connection that is not there proves the
        spec was accepted and compiled, where pytest.raises(Exception)
        would pass for the refusal this test exists to rule out."""
        spec = named(embedded_graph)["traverse_graph"]
        with pytest.raises(OperationalError):
            spec.call(start={"where": {"type": "person"}}, hops=[hop])

    def test_a_seeded_server_still_refuses_an_invented_near(self, vector_graph):
        spec = named(vector_graph, embed=embedder())["traverse_graph"]
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            spec.call(start={"near": {"field": "summary", "vector": [1, 2, 3]}})

    @pytest.mark.parametrize("tool, arguments", [
        ("traverse_graph", {}),
        ("aggregate_graph", {"aggregates": {"n": {"fn": "count"}}}),
    ])
    def test_the_embedding_this_server_made_is_allowed_through(self, vector_graph, tool,
                                                               arguments):
        """The other side of the invariant, and the half a refusal-only
        test cannot see. `refuse_vectors()` runs on what the MODEL sent;
        the call that follows passes allow_vectors=True because by then
        the `near` is this server's own. A mutant flipping that to False
        made every search-seeded call refuse its own embedding -- and
        survived on aggregate_graph, which no test had ever seeded.

        Both tools, because they inject separately. OperationalError is
        the assertion, and it has to be that specific: `pytest.raises
        (Exception)` plus "not the refusal" passes for ANY failure, so a
        handler mutated into a TypeError reads as a success. Reaching
        the connection that is not there is what proves the whole call
        was built -- the embedding made, the spec assembled, the query
        compiled -- with only the database missing."""
        spec = named(vector_graph, embed=embedder())[tool]
        with pytest.raises(OperationalError):
            spec.call(start={"search": "graph databases"}, **arguments)

    def test_search_field_without_search_is_refused(self):
        """`search_field` says which vector field the SEARCH TEXT ranks
        against, so without `search` it configures nothing. `keep` used
        to be refused here beside it and must not be: it is how many of
        `near`'s ranked nodes to keep, and that is a spec traverse_json()
        has always accepted."""
        spec = named(offline())["traverse_graph"]
        with pytest.raises(ValueError, match="start.search_field"):
            spec.call(start={"where": {"type": "person"}, "search_field": "summary"})

    def test_keep_without_search_is_now_a_near_spec_not_a_refusal(self, embedded_graph):
        spec = named(embedded_graph)["traverse_graph"]
        with pytest.raises(OperationalError):
            spec.call(start={"near": {"field": "summary", "text": "raft"}, "keep": 10})

    def test_search_and_near_together_are_refused_rather_than_one_dropped(self, vector_graph):
        """Both rank the seed set, and _seed() writes `near` -- so
        accepting both would silently discard whichever it overwrote.
        Rule 4: refuse and name the rewrite."""
        spec = named(vector_graph, embed=embedder())["traverse_graph"]
        with pytest.raises(ValueError, match="only one of them can"):
            spec.call(start={"search": "graph databases",
                             "near": {"field": "summary", "text": "raft"}})

    def test_search_without_an_embedder_names_the_fix(self):
        spec = named(offline())["traverse_graph"]
        with pytest.raises(ValueError, match="embed="):
            spec.call(start={"search": "quantum computing"})

    def test_search_becomes_a_real_embedding_of_the_text(self, vector_graph):
        embed = embedder()
        seeded = _seed(vector_graph, embed, {"search": "graph databases"}, "traverse_graph")
        assert embed.calls == ["graph databases"]
        assert seeded["near"] == {"field": "summary", "vector": embed("graph databases")}
        assert seeded["keep"] == DEFAULT_KEEP
        assert "search" not in seeded

    def test_keep_overrides_the_default_seed_size(self, vector_graph):
        seeded = _seed(vector_graph, embedder(), {"search": "x", "keep": 4}, "t")
        assert seeded["keep"] == 4

    def test_an_empty_search_is_refused_rather_than_embedded(self, vector_graph):
        with pytest.raises(ValueError, match="non-empty string"):
            _seed(vector_graph, embedder(), {"search": "   "}, "t")

    def test_an_ambiguous_field_is_refused(self):
        g = offline()
        g.define_vectors(nodes=[Vector("summary", 3), Vector("title", 3)])
        with pytest.raises(ValueError, match="aggregate_graph: this graph defines several"):
            _seed(g, embedder(), {"search": "x"}, "aggregate_graph")

    def test_an_unknown_field_lists_the_real_ones(self, vector_graph):
        with pytest.raises(ValueError, match=r"defined: \['summary'\]"):
            _seed(vector_graph, embedder(), {"search": "x", "search_field": "sumary"}, "t")

    def test_a_start_that_is_not_an_object_says_so(self):
        with pytest.raises(TypeError, match="traverse_graph: `start` must be an object, got list"):
            _seed(offline(), None, ["type", "person"], "traverse_graph")

    def test_searching_a_target_with_no_vectors_names_the_target(self):
        """Node and edge vector fields are separate namespaces, so
        "there is a field called summary" is not an answer to "can I
        search edges"."""
        g = offline()
        g.define_vectors(nodes=[Vector("summary", 3)])
        with pytest.raises(ValueError, match="no vector fields are defined for edges"):
            named(g, embed=embedder())["search_similar"].call(query="x", target="edges")

    def test_an_empty_query_is_refused_before_it_is_embedded(self, vector_graph):
        with pytest.raises(ValueError, match="non-empty string"):
            named(vector_graph, embed=embedder())["search_similar"].call(query="")

    def test_a_reworded_static_description_fails_loudly(self, vector_graph):
        """The replacement above is a string match against another
        module's constant. If that sentence is reworded, this is what
        says so -- rather than a server that silently keeps telling
        models it cannot search by meaning while offering start.search."""
        from hopai.mcp import _with_search

        with pytest.raises(RuntimeError, match="update _NO_MEANING"):
            _with_search({"name": "traverse_graph", "description": "reworded",
                          "parameters": {"properties": {"start": {"properties": {}}}}},
                         vector_graph, "the sentence that used to be there")


# ---------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------

class TestCypherTool:
    def test_a_read_only_server_refuses_a_write_before_running_it(self):
        """Classified by parsing, not by executing and hoping the
        database says no -- this graph has no database behind it and the
        refusal still lands."""
        with pytest.raises(ValueError, match="read-only"):
            named(offline(), read_only=True)["cypher"].call(
                query="CREATE (a:person {email: 'a@x.com'})")

    def test_read_only_refuses_writes_and_only_writes(self):
        """`read_only and is_write`, not `or`: a server that refused
        every query would be useless in exactly the configuration most
        people deploy. The read gets past the gate and fails on the
        schema instead, which is the proof."""
        g = offline()
        g.define_schema(nodes=[NodeType("person")])
        spec = named(g, read_only=True, strict_schema=True)["cypher"]
        with pytest.raises(CypherError, match="unknown label 'persn'"):
            spec.call(query="MATCH (a:persn) RETURN count(a)")

    def test_a_writing_server_still_refuses_a_delete(self):
        """The gap this closes: hopai gained DELETE, and `read_only`
        was the only gate in front of Cypher. A server started to let an
        agent CREATE would have picked up DETACH DELETE with it --
        classified as a write and waved through -- which is not a
        sentence anyone said. Refused before a connection is opened."""
        with pytest.raises(ValueError, match="does not allow deleting"):
            named(offline())["cypher"].call(query="MATCH (a:person) DETACH DELETE a")

    def test_allow_mutations_lets_the_same_query_through_the_gate(self):
        """The other half: a flag that refused everything would be
        indistinguishable from one that was never read. OperationalError
        rather than "some exception that is not the refusal" -- the
        looser form passes for any failure at all, including a handler
        mutated into a TypeError. This one proves the query was
        translated and compiled, with only the database missing."""
        spec = named(offline(), allow_mutations=True)["cypher"]
        with pytest.raises(OperationalError):
            spec.call(query="MATCH (a:person) DETACH DELETE a")

    def test_a_read_only_server_refuses_a_delete_as_a_delete(self):
        """read_only and allow_mutations cannot both be set, so a
        read-only server has to refuse a DELETE through the mutation
        gate -- checked first for exactly this reason. Without it a
        DELETE would fall through to the write branch and be described
        to the model as a write it could ask to have enabled."""
        with pytest.raises(ValueError, match="does not allow deleting"):
            named(offline(), read_only=True)["cypher"].call(
                query="MATCH (a:person) DETACH DELETE a")

    def test_the_description_says_which_way_the_mutation_gate_is_set(self):
        """Both wordings, because a model that reads "refused" and one
        that reads "supported" ask for different things. A description
        naming DELETE unconditionally would send the first one into a
        refusal loop."""
        assert "are refused" in named(offline())["cypher"].description
        assert "DETACH DELETE" in named(offline(), allow_mutations=True)["cypher"].description

    def test_strict_schema_reaches_the_query(self):
        """The flag is forwarded, not just accepted. Without this a
        server started with --strict-schema would quietly answer a
        query naming a label the schema has never heard of, with the
        empty result that spelling mistake deserves and no hint that
        it was one. Refused at translation, so no database is needed.
        (Found by a surviving mutant.)"""
        g = offline()
        g.define_schema(nodes=[NodeType("person")])
        with pytest.raises(CypherError, match="unknown label 'persn'"):
            named(g, strict_schema=True)["cypher"].call(
                query="MATCH (a:persn) RETURN count(a)")

    def test_the_description_says_which_half_is_available(self):
        """Both halves: what a read may do is as load-bearing as what a
        write may do, and it is the half a read-only server has left."""
        read_only = named(offline(), read_only=True)["cypher"].description
        assert "READ-ONLY" in read_only
        assert "MATCH with one linear chain of hops" in read_only
        assert "CREATE and MERGE write" in named(offline())["cypher"].description


class TestDescribeGraph:
    def test_node_vectors_alone_are_enough_to_search(self):
        """Asserted on a nodes-only graph: with fields on both targets,
        a mutant reading the wrong one still finds the other and the
        flag comes out right by accident."""
        g = offline()
        g.define_vectors(nodes=[Vector("summary", 3)])
        described = named(g, embed=embedder())["describe_graph"].call()
        assert described["search_by_meaning"] is True

    def test_it_reports_the_permissions_it_was_started_with(self, vector_graph):
        described = named(vector_graph, embed=embedder(), read_only=True)["describe_graph"].call()
        assert described["writes_allowed"] is False
        assert described["ddl_allowed"] is False
        assert described["search_by_meaning"] is True
        assert described["seed_traversal_by_meaning"] is True
        assert described["vector_fields"] == {"nodes": {"summary": 3}, "edges": {"rel": 3}}

    def test_searching_and_seeding_are_reported_as_the_two_things_they_are(self):
        """Edge-only vectors can be SEARCHED but cannot SEED a
        traversal, which ranks node vectors. Reported as one flag, that
        graph was told it could not search by meaning at all -- and a
        mutation making the two interchangeable survived, which is how
        the conflation surfaced."""
        edges_only = offline()
        edges_only.define_vectors(edges=[Vector("rel", 3)])
        described = named(edges_only, embed=embedder())["describe_graph"].call()
        assert described["search_by_meaning"] is True
        assert described["seed_traversal_by_meaning"] is False

    def test_neither_is_claimed_without_an_embedder(self, vector_graph):
        """With vector fields but no embedder there is still nothing to
        search BY: the model sends text and only the operator's callable
        turns it into a vector. Asserted on a graph that HAS fields,
        since one without them reports False either way."""
        described = named(vector_graph)["describe_graph"].call()
        assert described["search_by_meaning"] is False
        assert described["seed_traversal_by_meaning"] is False

    def test_it_says_when_there_is_no_schema_to_describe(self):
        described = named(offline())["describe_graph"].call()
        assert described["schema"] is None
        assert "infer_schema" in described["note"]

    def test_the_dangerous_arguments_default_to_the_safe_side(self):
        """dry_run on, endpoints off, save on: each is the reading or
        the reversible choice, and each is a default a model reaches by
        omitting the argument."""
        specs = named(offline(), allow_ddl=True)
        enforce = inspect.signature(specs["enforce_schema"].call).parameters
        assert enforce["dry_run"].default is True
        assert enforce["endpoints"].default is False
        assert inspect.signature(specs["define_schema"].call).parameters["save"].default is True

    def test_it_says_a_server_without_mutations_cannot_delete(self):
        """A model that cannot find a delete tool will otherwise try to
        emulate one -- with a Cypher DELETE, or by "updating" a node to
        look deleted. hopai HAS a delete now, so the absence is this
        server's choice and has to be reported as one."""
        described = named(offline())["describe_graph"].call()
        assert described["deletes_and_updates_allowed"] is False
        assert any("No delete and no update" in line for line in described["refusals"])

    def test_it_reports_the_delete_semantics_once_they_are_reachable(self):
        """The two refusals that only exist once mutate_graph does:
        telling a model about all=true and detach=true while it has no
        way to delete is advice for a tool it cannot see."""
        described = named(offline(), allow_mutations=True)["describe_graph"].call()
        assert described["deletes_and_updates_allowed"] is True
        assert any("all=true" in line for line in described["refusals"])
        assert any("detach=true" in line for line in described["refusals"])
        assert not any("No delete and no update" in line for line in described["refusals"])

    def test_a_declared_schema_arrives_as_json_and_as_a_diagram(self):
        g = offline()
        g.define_schema(nodes=[NodeType("person")])
        described = named(g)["describe_graph"].call()
        assert described["schema"]["nodes"] == {"person": {"type": "object", "properties": {}}}
        assert described["schema_mermaid"].startswith("flowchart LR")


# ---------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------

class TestCommandLine:
    def test_the_dsn_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("HOPAI_DSN", "postgresql+psycopg2://u:p@localhost/db")
        assert build_parser().parse_args([]).dsn == "postgresql+psycopg2://u:p@localhost/db"

    def test_no_dsn_anywhere_names_the_fix(self, monkeypatch, capsys):
        monkeypatch.delenv("HOPAI_DSN", raising=False)
        with pytest.raises(SystemExit):
            main([])
        assert "--dsn or set HOPAI_DSN" in capsys.readouterr().err

    def test_vector_fields_are_declared_on_the_command_line(self):
        """They are per-handle, not stored in the database like a saved
        schema, so a CLI server has no other way to know about them."""
        chosen, target, field = build_parser().parse_args(
            ["--vector", "nodes:summary:1536"]).vector[0]
        assert (chosen, target, field.name, field.dimensions) == (None, "nodes", "summary", 1536)
        chosen, target, field = build_parser().parse_args(
            ["--vector", "crm:edges:rel:8"]).vector[0]
        assert (chosen, target, field.name, field.dimensions) == ("crm", "edges", "rel", 8)

    @pytest.mark.parametrize("value, expected", [
        ("summary:1536", "--vector takes TARGET:NAME:DIMENSIONS"),
        ("nodes:summary", "--vector takes TARGET:NAME:DIMENSIONS"),
        ("a:b:rows:summary:8", "--vector takes TARGET:NAME:DIMENSIONS"),
        ("rows:summary:8", "target must be 'nodes' or 'edges'"),
        ("nodes:summary:many", "dimensions must be a positive integer"),
    ])
    def test_a_malformed_vector_spec_says_which_part_is_wrong(self, value, expected, capsys):
        """Each message, not one shared substring: argparse prints the
        flag and its metavar in the usage line whatever the message
        says, so asserting on those passes even when the message itself
        has been blanked -- which is how three mutants survived."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--vector", value])
        assert expected in capsys.readouterr().err

    @pytest.mark.parametrize("value, expected", [
        ("no_colon_here", "--embed takes MODULE:FUNCTION"),
        ("nosuchmodule_xyz:embed", "cannot import"),
        ("json:no_such_attribute", "no attribute"),
        ("json:__doc__", "not callable"),
    ])
    def test_a_bad_embed_reference_says_which_part_is_wrong(self, value, expected, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--embed", value])
        assert expected in capsys.readouterr().err

    def test_the_embed_reference_resolves_to_the_function(self):
        loaded = build_parser().parse_args(["--embed", "json:dumps"]).embed
        assert loaded is json.dumps

    def test_the_module_is_everything_before_the_FIRST_colon(self, capsys):
        """`partition`, not `rpartition`: a dotted path never contains a
        colon, so the split is unambiguous only from the left. With
        rpartition, `a:b:c` asks for a module named `a:b` -- and the
        error would then name something the operator did not type,
        which is the one job an error message has here. Both spellings
        fail on this input; only one of them fails honestly."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--embed", "no_such_module_xyz:b:c"])
        assert "'no_such_module_xyz'" in capsys.readouterr().err

    def test_defaults_are_the_cautious_ones(self):
        args = build_parser().parse_args(["--dsn", "postgresql://x/y"])
        assert (args.transport, args.host, args.read_only, args.allow_ddl,
                args.allow_mutations) == ("stdio", "127.0.0.1", False, False, False)
        # the rest of them too: a default that drifts to None surfaces
        # as a server bound somewhere else, or named something else
        assert (args.port, args.path, args.name) == (8000, "/mcp", "hopai")
        assert args.graph == [] and args.load_schema is True
        # serve() carries its own copies of these, for callers who never
        # go through the parser at all
        from hopai.mcp import serve
        defaults = inspect.signature(serve).parameters
        assert (defaults["port"].default, defaults["path"].default,
                defaults["host"].default) == (8000, "/mcp", "127.0.0.1")

    def test_an_unknown_transport_is_rejected_at_the_command_line(self):
        """Not left to serve(): the CLI knows the two it accepts, and
        finding out after a server is built is later than necessary."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--dsn", "postgresql://x/y", "--transport", "grpc"])

    @pytest.mark.parametrize("transport", ["stdio", "http"])
    def test_both_transports_are_accepted_by_the_same_gate(self, transport):
        """The half a refusal-only test cannot see. `choices` is a
        literal pair, and mutants that misspelled one of them made that
        transport unusable while `--transport grpc` went on being
        rejected -- so the test above passed and the CLI was broken."""
        assert build_parser().parse_args(
            ["--dsn", "postgresql://x/y", "--transport", transport]).transport == transport

    def test_graph_is_repeatable_and_the_first_is_the_handle_that_opens_the_pool(
            self, monkeypatch):
        """The CLI half of many-graphs-one-pool: `--graph` appends, the
        first Graph opens the engine and in_graph() shares it, and the
        order given is the order served."""
        served = {}
        monkeypatch.setattr("hopai.mcp.serve",
                            lambda graph, **options: served.update(graph=graph, **options))
        main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
              "--no-load-schema", "--graph", "docs", "--graph", "crm",
              "--vector", "crm:nodes:notes:8"])
        assert list(served["graph"]) == ["docs", "crm"]
        # the keys come from the flags; the HANDLES have to agree with them
        assert [g.graph for g in served["graph"].values()] == ["docs", "crm"]
        assert served["graph"]["docs"].engine is served["graph"]["crm"].engine
        # ...and a prefixed --vector lands on that graph alone
        assert served["graph"]["crm"].vectors["nodes"]["notes"].dimensions == 8
        assert served["graph"]["docs"].vectors is None

    def test_without_graph_every_graph_in_the_database_is_served(self, monkeypatch, capsys):
        """The default is full access, because the DSN already is: a
        process holding these credentials can read every graph in the
        database, so declining to enumerate them protects nothing.

        What the old default did instead was serve the graph literally
        named 'default' -- so a server pointed at a database whose rows
        live in 'docs' and 'crm' answered "nothing here", confidently
        and about graphs it had never been told to look at."""
        served = {}
        monkeypatch.setattr("hopai.core.Graph.graphs", lambda self: ["crm", "docs"])
        monkeypatch.setattr("hopai.mcp.serve",
                            lambda graph, **options: served.update(graph=graph, **options))
        main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
              "--no-load-schema"])
        assert list(served["graph"]) == ["crm", "docs"]
        # the HANDLES have to agree with the names, and share the pool
        assert [g.graph for g in served["graph"].values()] == ["crm", "docs"]
        assert served["graph"]["crm"].engine is served["graph"]["docs"].engine
        # said out loud on stderr, which is a stdio client's log: an
        # operator who did not mean to serve everything should find out
        # at start-up rather than from a model naming someone else's graph
        assert "serving every graph in this database" in capsys.readouterr().err

    def test_graph_restricts_and_never_looks_the_others_up(self, monkeypatch):
        """`--graph` is the opt-in restriction, and it has to be a real
        one: discovering first and then filtering would still reach a
        database the operator may have scoped the credentials for. The
        exploding stub is the assertion -- if discovery ran, this fails."""
        served = {}
        monkeypatch.setattr("hopai.core.Graph.graphs",
                            lambda self: pytest.fail("--graph must not enumerate"))
        monkeypatch.setattr("hopai.mcp.serve",
                            lambda graph, **options: served.update(graph=graph, **options))
        main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
              "--no-load-schema", "--graph", "docs"])
        assert list(served["graph"]) == ["docs"]

    def test_an_empty_database_still_starts(self, monkeypatch):
        """Nothing has been written yet, so there is nothing to
        discover. Refusing to start would make the server useless
        exactly when someone is setting it up -- and with one graph
        served, no tool mentions graphs at all."""
        served = {}
        monkeypatch.setattr("hopai.core.Graph.graphs", lambda self: [])
        monkeypatch.setattr("hopai.mcp.serve",
                            lambda graph, **options: served.update(graph=graph, **options))
        main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
              "--no-load-schema"])
        assert list(served["graph"]) == ["default"]

    def test_a_database_that_cannot_be_listed_names_the_fix(self, capsys):
        """Discovery needs a connection, and the DSN may be wrong or the
        database down. The message has to name `--graph`, which is the
        way to start without looking anything up -- otherwise the
        operator is told only that something failed."""
        with pytest.raises(SystemExit):
            main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                  "--no-load-schema"])
        error = capsys.readouterr().err
        assert "cannot list the graphs in this database" in error
        assert "--graph" in error

    def test_a_graph_named_twice_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            main(["--dsn", "postgresql://x/y", "--graph", "docs", "--graph", "docs"])
        assert "given more than once" in capsys.readouterr().err

    def test_a_vector_for_an_unserved_graph_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            main(["--dsn", "postgresql://x/y", "--graph", "docs",
                  "--vector", "crm:nodes:notes:8"])
        assert "this server does not serve" in capsys.readouterr().err

    def test_main_declares_the_vector_fields_and_forwards_every_option(self, monkeypatch):
        """Everything main() does between parsing and serving, with the
        serving stubbed. Every flag is asserted, not a sample: an option
        that parses and is then dropped on the way to serve() fails
        silently and in the direction nobody wants -- `--host` binding
        somewhere else, `--allow-ddl` off (or on) without anyone saying
        so. And the vector fields have to reach the Graph, or the search
        tools they enable have nothing to rank."""
        served = {}
        monkeypatch.setattr("hopai.mcp.serve",
                            lambda graph, **options: served.update(graph=graph, **options))
        assert main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                     "--graph", "default",
                     "--no-load-schema", "--vector", "nodes:summary:1536",
                     "--vector", "edges:rel:8", "--transport", "http", "--host", "0.0.0.0",
                     "--port", "9999", "--path", "/graph", "--name", "kg",
                     "--allow-ddl", "--allow-mutations", "--embed", "json:dumps"]) == 0
        assert set(served["graph"]) == {"default"}
        only = served["graph"]["default"]
        assert only.vectors["nodes"]["summary"].dimensions == 1536
        assert only.vectors["edges"]["rel"].dimensions == 8
        assert served["transport"] == "http"
        assert (served["host"], served["port"], served["path"]) == ("0.0.0.0", 9999, "/graph")
        assert (served["name"], served["read_only"], served["allow_ddl"]) == ("kg", False, True)
        assert served["allow_mutations"] is True
        assert (served["embed"], served["strict_schema"]) == (json.dumps, False)

    def test_an_unreachable_saved_schema_is_reported_not_fatal(self, monkeypatch, capsys):
        """A graph that never called save_schema() is the normal case,
        and the server is perfectly useful without one -- so it says why
        it has no schema on stderr (which is a stdio client's log) and
        carries on."""
        monkeypatch.setattr("hopai.mcp.serve", lambda graph, **options: None)
        monkeypatch.setattr("hopai.core.Graph.load_schema",
                            lambda self: (_ for _ in ()).throw(ValueError("no saved schema")))
        main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
              "--graph", "default"])
        assert "serving 'default' without a declared schema" in capsys.readouterr().err


class TestServeArguments:
    def test_an_unknown_transport_is_refused_before_anything_starts(self):
        """Named here rather than left to the SDK: 'sse' and 'grpc' are
        both plausible guesses, and the SDK's own error arrives after a
        server has been built."""
        from hopai.mcp import serve

        with pytest.raises(ValueError, match="'stdio' or 'http'"):
            serve(offline(), transport="grpc")

    @staticmethod
    def _stub(monkeypatch) -> dict:
        """serve() without an SDK or a socket: record what it would have
        built and run, and return instead of blocking on a transport."""
        seen = {}

        class FakeServer:
            def run(self, transport, **kwargs):
                seen["ran"] = (transport, kwargs)

        def build(graph, **options):
            seen["options"] = options
            return FakeServer()

        monkeypatch.setattr("hopai.mcp._sdk", lambda: (None, None, 2))
        monkeypatch.setattr("hopai.mcp.build_server", build)
        return seen

    def test_the_bind_defaults_reach_the_server_that_is_built(self, monkeypatch):
        """The defaults are asserted through the CALL, not through
        inspect.signature(serve): mutmut wraps every function in a
        `trampoline(*args, **kwargs)`, so a signature assertion reads the
        wrapper and is blind to a mutated default by construction. Six
        mutants moving the port off 8000, the path off /mcp and the host
        off 127.0.0.1 all survived a test that checked the signature.

        The host one is the reason this matters rather than being
        tidiness: serve()'s own docstring says a graph an agent may
        write to is not a thing to put on 0.0.0.0 by accident, and
        nothing was checking that it did not."""
        from hopai.mcp import serve

        seen = self._stub(monkeypatch)
        serve(offline())
        assert seen["options"]["http"] == {"host": "127.0.0.1", "port": 8000,
                                           "streamable_http_path": "/mcp"}
        assert seen["ran"] == ("stdio", {})

    @pytest.mark.parametrize("transport, expected", [("stdio", "stdio"),
                                                     ("http", "streamable-http")])
    def test_both_transports_pass_the_gate_and_reach_their_own_runner(
            self, monkeypatch, transport, expected):
        """The half the refusal test cannot see. `transport not in
        ("stdio", "http")` still rejects "grpc" when either literal is
        misspelled, so four mutants broke a working transport while the
        refusal went on passing -- the same shape as the CLI `choices`
        pair one class up."""
        from hopai.mcp import serve

        seen = self._stub(monkeypatch)
        serve(offline(), transport=transport, host="0.0.0.0", port=9001, path="/x")
        assert seen["ran"][0] == expected
        assert seen["options"]["http"]["port"] == 9001


# ---------------------------------------------------------------------
# Registration against the real SDK
# ---------------------------------------------------------------------

def advertised(tool) -> dict:
    """The input schema of a listed tool, under whichever name this SDK
    version gives the field (`inputSchema` in 1.x, `input_schema` in
    2.0 -- the same JSON on the wire either way)."""
    return getattr(tool, "inputSchema", None) or tool.input_schema


@needs_sdk
@survives_mutmut
class TestServerRegistration:
    def test_every_tool_reaches_the_client_with_hopais_schema(self, vector_graph):
        """The end of the chain the rest of this file tests in pieces:
        what a client actually lists. _register() replaces the schema
        the SDK derives from the handler's Python signature, and if a
        future SDK stops honouring that, this is what fails."""
        server = build_server(vector_graph, embed=embedder(), allow_ddl=True,
                              allow_mutations=True)
        listed = {tool.name: tool for tool in asyncio.run(server.list_tools())}
        specs = named(vector_graph, embed=embedder(), allow_ddl=True, allow_mutations=True)
        assert set(listed) == set(specs)
        for name, tool in listed.items():
            assert advertised(tool) == specs[name].parameters, name
            assert tool.description == specs[name].description, name

    def test_a_call_round_trips_through_the_server(self):
        server = build_server(offline(), read_only=True)
        result = asyncio.run(server.call_tool("describe_graph", {}))
        content = result.content if hasattr(result, "content") else result
        assert json.loads(content[0].text)["writes_allowed"] is False

    def test_an_error_reaches_the_client_with_its_message_intact(self):
        """These messages name the fix, which is the whole reason a
        model can correct itself. A wrapper that replaced them with
        "tool failed" would cost that."""
        server = build_server(offline(), read_only=True)
        try:
            result = asyncio.run(server.call_tool("cypher", {"query": "CREATE (a:person)"}))
        except Exception as exc:                     # mcp 1.x raises, 2.0 returns is_error
            assert "read-only" in str(exc)
        else:
            content = result.content if hasattr(result, "content") else result
            assert "read-only" in content[0].text

    def test_the_server_carries_instructions_for_the_client(self):
        """Server-level guidance is the one place to say "call
        describe_graph first" -- no single tool schema can."""
        server = build_server(offline())
        assert server.instructions == SERVER_INSTRUCTIONS
        assert "describe_graph first" in SERVER_INSTRUCTIONS


# ---------------------------------------------------------------------
# Against a real database
# ---------------------------------------------------------------------

class TestReadToolsLive:
    def test_traverse_returns_the_subgraph_the_python_api_returns(self, graph):
        spec = named(graph, read_only=True)["traverse_graph"]
        through_mcp = spec.call(start={"where": {"type": "leaf"}},
                                hops=[{"via": {"kind": "knows"}}])
        direct = graph.traverse(Start(where={"type": "leaf"}), Hop(via={"kind": "knows"}))
        assert {n["id"] for n in through_mcp["nodes"]} == {n["id"] for n in direct.nodes}
        assert through_mcp["edges"]

    def test_aggregate_counts_the_last_steps_nodes(self, graph):
        spec = named(graph, read_only=True)["aggregate_graph"]
        assert spec.call(start={"where": {"type": "leaf"}},
                         hops=[{"via": {"kind": "knows"}}],
                         aggregates={"reached": {"fn": "count"}}) == {"reached": 2}

    def test_cypher_reads(self, graph):
        result = named(graph, read_only=True)["cypher"].call(
            query="MATCH (a:leaf)-[:knows]->(b) RETURN count(DISTINCT b)")
        assert result == {"count": 2}

    def test_describe_can_count_the_nodes(self, graph):
        described = named(graph, read_only=True)["describe_graph"].call(counts=True)
        assert described["counts"]["nodes"] == 7

    def test_infer_schema_passes_the_sample_through(self, graph):
        """The percentage has to reach infer_schema: a model asking for
        a sample of a large graph and silently getting a full scan is a
        scan nobody chose."""
        result = named(graph, read_only=True)["infer_schema"].call(sample_percent=100)
        assert result["report"]["sampled"] == 100

    def test_infer_schema_observes_without_declaring(self, graph):
        result = named(graph, read_only=True)["infer_schema"].call()
        assert set(result["schema"]["nodes"]) == {"leaf", "hub"}
        assert result["adopted"] is False
        assert result["report"]["untyped_nodes"] == 2
        assert graph.schema is None


class TestWriteToolsLive:
    def test_ingest_creates_nodes_and_edges_in_one_call(self, fresh_graph):
        spec = named(fresh_graph)["ingest_graph"]
        written = spec.call(
            nodes=[{"id": 1, "type": "person", "email": "a@x.com"},
                   {"id": 2, "type": "company", "name": "Acme"}],
            edges=[{"start": {"email": "a@x.com"}, "end_id": 2, "kind": "works_at"}],
        )
        assert (written["nodes"], written["edges"]) == (2, 1)
        assert named(fresh_graph)["aggregate_graph"].call(
            start={"where": {"type": "person"}}, aggregates={"n": {"fn": "count"}}) == {"n": 1}

    def test_merge_keys_make_the_second_run_an_update(self, fresh_graph):
        """Without merge_nodes_on a model can only insert, and an agent
        re-running its own ingestion silently doubles the graph."""
        fresh_graph.define_constraints(nodes=[Unique("email")])
        spec = named(fresh_graph)["ingest_graph"]
        rows = [{"type": "person", "email": "a@x.com", "name": "Alice"}]
        spec.call(nodes=rows, merge_nodes_on=["email"])
        spec.call(nodes=[{"type": "person", "email": "a@x.com", "name": "Alicia"}],
                  merge_nodes_on=["email"])
        found = named(fresh_graph)["traverse_graph"].call(start={"where": {"email": "a@x.com"}})
        assert len(found["nodes"]) == 1
        assert found["nodes"][0]["properties"]["name"] == "Alicia"

    def test_mutate_updates_and_deletes_in_one_ordered_transaction(self, fresh_graph):
        """The tool is a translation into Graph.mutate() and nothing
        else -- so what this pins is that the model's `operations` list
        arrives whole, in order, and that the counts come back as the
        four separate numbers a caller needs (one total would make "3"
        mean three of something unstated)."""
        named(fresh_graph)["ingest_graph"].call(
            nodes=[{"id": 1, "type": "draft", "title": "a"},
                   {"id": 2, "type": "draft", "title": "b"},
                   {"id": 3, "type": "person"}],
            edges=[{"start_id": 3, "end_id": 1, "kind": "wrote"}])
        result = named(fresh_graph, allow_mutations=True)["mutate_graph"].call(operations=[
            {"op": "update_nodes", "where": {"type": "draft"},
             "set": {"status": "archived"}},
            {"op": "delete_edges", "where": {"kind": "wrote"}},
            {"op": "delete_nodes", "where": {"type": "draft", "title": "b"}},
        ])
        assert result["updated_nodes"] == 2
        assert (result["deleted_edges"], result["deleted_nodes"]) == (1, 1)
        left = named(fresh_graph)["traverse_graph"].call(start={"where": {"type": "draft"}})
        assert [n["properties"]["status"] for n in left["nodes"]] == ["archived"]

    def test_a_filterless_delete_refuses_rather_than_emptying_the_graph(self, fresh_graph):
        """The library's own refusal, reaching the model through the
        tool rather than being caught and softened by it. A server that
        swallowed it would turn the one unrecoverable mistake into a
        successful-looking call."""
        named(fresh_graph)["ingest_graph"].call(nodes=[{"id": 1, "type": "person"}])
        with pytest.raises((ValueError, TypeError), match="all="):
            named(fresh_graph, allow_mutations=True)["mutate_graph"].call(
                operations=[{"op": "delete_nodes"}])
        assert named(fresh_graph)["aggregate_graph"].call(
            start={"where": {}}, aggregates={"n": {"fn": "count"}}) == {"n": 1}

    def test_cypher_deletes_once_the_server_is_allowed_to(self, fresh_graph):
        """The other route to the same power: the gate is in front of
        the Cypher tool as well as being the reason mutate_graph exists,
        and an agent that found DELETE working through one and refused
        through the other would be right to be confused."""
        named(fresh_graph)["ingest_graph"].call(
            nodes=[{"id": 1, "type": "person", "email": "a@x.com"},
                   {"id": 2, "type": "company"}],
            edges=[{"start_id": 1, "end_id": 2, "kind": "works_at"}])
        result = named(fresh_graph, allow_mutations=True)["cypher"].call(
            query="MATCH (a:person {email: 'a@x.com'}) DETACH DELETE a")
        assert (result["deleted_nodes"], result["deleted_edges"]) == (1, 1)
        assert named(fresh_graph)["aggregate_graph"].call(
            start={"where": {"type": "person"}}, aggregates={"n": {"fn": "count"}}) == {"n": 0}

    def test_define_schema_declares_and_persists_it(self, fresh_graph):
        """Saved, not just declared: the next process to serve this
        graph loads the contract instead of being told about it again."""
        document = {
            "nodes": {"person": {"type": "object",
                                 "properties": {"email": {"type": "string"}},
                                 "required": ["email"]}},
            "edges": [{"kind": "knows", "source": "person", "target": "person",
                       "properties": {"type": "object", "properties": {}}}],
        }
        result = named(fresh_graph)["define_schema"].call(schema=document)
        assert result["saved"] is True
        assert result["schema"]["nodes"]["person"]["required"] == ["email"]
        assert fresh_graph.in_graph(fresh_graph.graph).load_schema().to_json() == result["schema"]

    def test_a_malformed_schema_document_is_refused(self, fresh_graph):
        with pytest.raises((ValueError, TypeError)):
            named(fresh_graph)["define_schema"].call(
                schema={"nodes": {"person": {"type": "object",
                                             "properties": {"age": {"type": "integer"}}}}})

    def test_enforce_dry_runs_by_default(self, fresh_graph):
        """The default has to be the reading one: ADD CONSTRAINT stops
        at the first bad row, while the dry run returns the whole work
        list -- and a model that has to opt in to the destructive
        version cannot reach it by omission."""
        fresh_graph.add_nodes([{"id": 1, "type": "person"},
                               {"id": 2, "type": "person", "email": 42}])
        fresh_graph.define_schema(nodes=[NodeType("person", properties=[
            Property("email", "string", required=True)])])
        spec = named(fresh_graph, allow_ddl=True)["enforce_schema"]

        report = spec.call()
        assert report["dry_run"] is True and report["clean"] is False
        assert report["rules"] and report["rules"][0]["rows"]
        # the prose summary too: it is what a model reads before the
        # rules array, and a mutant renaming the KEY dropped it in
        # silence while every other assertion here still passed
        assert report["summary"]

        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            spec.call(dry_run=False)

    def test_enforce_returns_the_constraints_now_in_force(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"}])
        fresh_graph.define_schema(nodes=[NodeType("person", properties=[
            Property("email", "string", required=True)])])
        names = named(fresh_graph, allow_ddl=True)["enforce_schema"].call(dry_run=False)
        assert names["dry_run"] is False and names["constraints"]


class TestManyGraphsLive:
    def test_two_graphs_stay_invisible_to_each_other_through_the_tools(self, fresh_graph):
        """The `graph` argument is not a label -- it selects the handle,
        and every read and write goes through that handle's
        `_scoped()`. Writing into one graph and finding it in the other
        would be the one bug multi-graph scoping can produce."""
        graphs = {"docs": fresh_graph, "crm": fresh_graph.in_graph("crm")}
        specs = named(graphs)

        specs["ingest_graph"].call(nodes=[{"id": 1, "type": "paper", "title": "graphs"}],
                                   graph="docs")
        specs["ingest_graph"].call(nodes=[{"id": 2, "type": "account", "name": "acme"}],
                                   graph="crm")

        in_docs = specs["traverse_graph"].call(start={"where": {"type": "paper"}},
                                               graph="docs")
        assert [n["id"] for n in in_docs["nodes"]] == ["1"]
        assert specs["traverse_graph"].call(
            start={"where": {"type": "paper"}}, graph="crm")["nodes"] == []
        assert specs["aggregate_graph"].call(
            start={"where": {"type": "account"}}, aggregates={"n": {"fn": "count"}},
            graph="crm") == {"n": 1}

    def test_a_schema_saved_through_one_graph_is_that_graphs_alone(self, fresh_graph):
        """define_schema persists per graph_id, so the second server
        process loads each graph's own contract -- not the first one it
        happens to find."""
        graphs = {"docs": fresh_graph, "crm": fresh_graph.in_graph("crm")}
        specs = named(graphs)
        document = {"nodes": {"paper": {"type": "object", "properties": {}}}, "edges": []}
        specs["define_schema"].call(schema=document, graph="docs")

        assert set(specs["describe_graph"].call(graph="docs")["schema"]["nodes"]) == {"paper"}
        assert specs["describe_graph"].call(graph="crm")["schema"] is None
        with pytest.raises(ValueError, match="no saved schema for graph 'crm'"):
            fresh_graph.in_graph("crm").load_schema()


class TestSearchLive:
    def test_text_in_becomes_similarity_out(self, fresh_graph):
        """The whole point of the embed callable: the model sends words,
        the server sends vectors, and the two never swap places."""
        fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1, "type": "doc", "title": "graphs"},
                               {"id": 2, "type": "doc", "title": "cooking"}])
        fresh_graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]},
                                       {"id": 2, "summary": [0.0, 1.0, 0.0]}])

        embed = embedder()
        results = named(fresh_graph, embed=embed)["search_similar"].call(
            query="graph databases", k=1)["results"]
        assert embed.calls == ["graph databases"]
        assert results[0]["id"] == "1"
        # no vector comes back: 6KB of floats has no business in a tool result
        assert "summary" not in json.dumps(results)

    def test_a_traversal_can_be_seeded_from_meaning(self, fresh_graph):
        fresh_graph.define_vectors(nodes=[Vector("summary", 3)])
        fresh_graph.migrate_vectors()
        fresh_graph.add_nodes([{"id": 1, "type": "doc"}, {"id": 2, "type": "doc"},
                               {"id": 3, "type": "author"}])
        fresh_graph.add_edges([{"start_id": 1, "end_id": 3, "kind": "by"},
                               {"start_id": 2, "end_id": 3, "kind": "by"}])
        fresh_graph.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]},
                                       {"id": 2, "summary": [0.0, 1.0, 0.0]}])

        spec = named(fresh_graph, embed=lambda text: [1.0, 0.0, 0.0])["traverse_graph"]
        found = spec.call(start={"search": "graphs", "keep": 1}, hops=[{"via": {"kind": "by"}}])
        assert {n["id"] for n in found["nodes"]} == {"1", "3"}
