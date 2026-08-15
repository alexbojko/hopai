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

import pytest

from hopai import (
    AGGREGATE_TOOL_SCHEMA, CypherError, EdgeType, Graph,
    INGEST_TOOL_SCHEMA, Hop, NodeType, Property, Start, TRAVERSE_TOOL_SCHEMA, Unique, Vector,
)
from hopai.mcp import DEFAULT_KEEP, SERVER_INSTRUCTIONS, _seed, build_parser, build_server
from hopai.mcp import main, tools

try:                                    # the extra is optional, and so is testing against it
    import mcp as _sdk
except ImportError:                     # pragma: no cover - depends on the environment
    _sdk = None

needs_sdk = pytest.mark.skipif(_sdk is None,
                               reason="the MCP SDK is not installed -- pip install hopai[mcp]")


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


@pytest.fixture()
def vector_graph() -> Graph:
    g = offline()
    g.define_vectors(nodes=[Vector("summary", 3)], edges=[Vector("rel", 3)])
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


# ---------------------------------------------------------------------
# What the model is told
# ---------------------------------------------------------------------

class TestToolSchemas:
    @pytest.mark.parametrize("name, static", [
        ("traverse_graph", TRAVERSE_TOOL_SCHEMA),
        ("aggregate_graph", AGGREGATE_TOOL_SCHEMA),
    ])
    def test_the_advertised_schema_is_hopais_own(self, name, static):
        """The SDK would derive `{"type": "object"}` from `start: dict`
        and leave the model to guess `where`, `via`, `hops` and
        `direction`. hopai ships hand-written schemas that spell those
        out and are kept in step with the parsers; this is the test that
        they are what actually reaches a client."""
        assert named(offline())[name].parameters == static["parameters"]

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
        for spec in tools(offline(), allow_ddl=False):
            accepted = set(inspect.signature(spec.call).parameters)
            assert set(spec.parameters["properties"]) <= accepted, spec.name

    def test_required_parameters_are_exactly_the_handlers_required_ones(self):
        for spec in tools(offline()):
            needed = {name for name, p in inspect.signature(spec.call).parameters.items()
                      if p.default is inspect.Parameter.empty}
            assert set(spec.parameters.get("required", [])) == needed, spec.name

    def test_no_tool_advertises_a_vector_parameter(self, vector_graph):
        """The invariant tests/test_vectors.py pins for the static tool
        schemas, restated for every tool this server registers -- with
        an embedder configured, which is the only configuration where a
        similarity parameter could plausibly appear.

        `search` and `keep` are the deliberate exception and are not in
        this set: `search` takes the model's TEXT, which the server
        embeds itself, and `keep` is a count. What must never appear is
        a place to PUT floats -- a model asked for an embedding invents
        one, and an invented embedding finds confidently wrong
        neighbors."""
        forbidden = {"near", "via_near", "via_keep", "boost", "vector", "embedding"}
        for spec in tools(vector_graph, embed=embedder(), allow_ddl=True):
            assert not parameter_names(spec.parameters) & forbidden, spec.name

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

    def test_the_cannot_search_by_meaning_sentence_is_replaced_not_kept(self, vector_graph):
        """The static schema tells a model this tool cannot find nodes
        by meaning, which stops it guessing property values for a
        semantic question. With an embedder that sentence is false, and
        a description carrying both instructions is worse than either:
        a model reading the refusal will not use the feature next to
        it."""
        plain = named(vector_graph)["traverse_graph"].description
        seeded = named(vector_graph, embed=embedder())["traverse_graph"].description
        assert "cannot find nodes by meaning" in plain
        assert "cannot find nodes by meaning" not in seeded
        assert "start.search" in seeded

    def test_a_seed_set_still_has_to_come_from_somewhere(self, vector_graph):
        """The static schema requires `where` on start. Search is a
        second way in, not a way to ask for every node in the graph, so
        the requirement becomes a choice between the two rather than
        disappearing."""
        start = named(vector_graph,
                      embed=embedder())["traverse_graph"].parameters["properties"]["start"]
        assert "required" not in start
        assert start["anyOf"] == [{"required": ["where"]}, {"required": ["search"]}]

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
    def test_a_start_near_is_refused(self):
        """The whole reason this server embeds text itself. Without this
        the tool would happily rank against invented floats and return a
        plausible subgraph, which is the worst thing this library can
        produce."""
        spec = named(offline())["traverse_graph"]
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            spec.call(start={"near": {"field": "summary", "vector": [0.1, 0.2, 0.3]}})

    @pytest.mark.parametrize("key", ["near", "keep", "via_near", "via_keep", "boost"])
    def test_hop_vector_keys_are_refused_too(self, key):
        """Hops are passed through to json_api verbatim, and the seeded
        path calls it with allow_vectors=True -- so the hops have to be
        refused here or the invariant would hold only for `start`."""
        spec = named(offline())["traverse_graph"]
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            spec.call(start={"where": {"type": "person"}}, hops=[{key: 3}])

    def test_a_seeded_server_still_refuses_an_invented_near(self, vector_graph):
        spec = named(vector_graph, embed=embedder())["traverse_graph"]
        with pytest.raises(ValueError, match="cannot come from a tool call"):
            spec.call(start={"near": {"field": "summary", "vector": [1, 2, 3]}})

    def test_keep_without_search_is_refused(self):
        spec = named(offline())["traverse_graph"]
        with pytest.raises(ValueError, match="start.keep"):
            spec.call(start={"where": {"type": "person"}, "keep": 10})

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
        with pytest.raises(ValueError, match="name the one to search"):
            _seed(g, embedder(), {"search": "x"}, "t")

    def test_an_unknown_field_lists_the_real_ones(self, vector_graph):
        with pytest.raises(ValueError, match=r"defined: \['summary'\]"):
            _seed(vector_graph, embedder(), {"search": "x", "search_field": "sumary"}, "t")

    def test_a_start_that_is_not_an_object_says_so(self):
        with pytest.raises(TypeError, match="must be an object"):
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

    def test_delete_is_refused_with_the_librarys_own_message(self):
        """There is no delete API anywhere in hopai, and this server
        does not invent one. The message names what to do instead."""
        with pytest.raises(CypherError, match="no delete API"):
            named(offline())["cypher"].call(query="MATCH (a) DETACH DELETE a")

    def test_the_description_says_which_half_is_available(self):
        assert "READ-ONLY" in named(offline(), read_only=True)["cypher"].description
        assert "CREATE and MERGE write" in named(offline())["cypher"].description


class TestDescribeGraph:
    def test_it_reports_the_permissions_it_was_started_with(self, vector_graph):
        described = named(vector_graph, embed=embedder(), read_only=True)["describe_graph"].call()
        assert described["writes_allowed"] is False
        assert described["ddl_allowed"] is False
        assert described["search_by_meaning"] is True
        assert described["vector_fields"] == {"nodes": {"summary": 3}, "edges": {"rel": 3}}

    def test_it_says_when_there_is_no_schema_to_describe(self):
        described = named(offline())["describe_graph"].call()
        assert described["schema"] is None
        assert "infer_schema" in described["note"]

    def test_it_names_the_missing_delete_api(self):
        """A model that cannot find a delete tool will otherwise try to
        emulate one -- with a Cypher DELETE, or by "updating" a node to
        look deleted."""
        described = named(offline())["describe_graph"].call()
        assert any("No delete" in line for line in described["refusals"])

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
        target, field = build_parser().parse_args(["--vector", "nodes:summary:1536"]).vector[0]
        assert (target, field.name, field.dimensions) == ("nodes", "summary", 1536)

    @pytest.mark.parametrize("value", ["summary:1536", "nodes:summary", "rows:summary:8",
                                       "nodes:summary:many"])
    def test_a_malformed_vector_spec_is_rejected_with_the_form(self, value, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--vector", value])
        assert "--vector" in capsys.readouterr().err

    @pytest.mark.parametrize("value, expected", [
        ("no_colon_here", "MODULE:FUNCTION"),
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

    def test_defaults_are_the_cautious_ones(self):
        args = build_parser().parse_args(["--dsn", "postgresql://x/y"])
        assert (args.transport, args.host, args.read_only, args.allow_ddl) == (
            "stdio", "127.0.0.1", False, False)

    def test_main_declares_the_vector_fields_and_serves(self, monkeypatch):
        """Everything main() does between parsing and serving, with the
        serving stubbed: the vector fields have to reach the Graph, or
        the search tools it just enabled have nothing to rank."""
        served = {}
        monkeypatch.setattr("hopai.mcp.serve",
                            lambda graph, **options: served.update(graph=graph, **options))
        assert main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline",
                     "--no-load-schema", "--vector", "nodes:summary:1536",
                     "--transport", "http", "--port", "9999", "--read-only"]) == 0
        assert served["graph"].vectors["nodes"]["summary"].dimensions == 1536
        assert (served["transport"], served["port"], served["read_only"]) == ("http", 9999, True)

    def test_an_unreachable_saved_schema_is_reported_not_fatal(self, monkeypatch, capsys):
        """A graph that never called save_schema() is the normal case,
        and the server is perfectly useful without one -- so it says why
        it has no schema on stderr (which is a stdio client's log) and
        carries on."""
        monkeypatch.setattr("hopai.mcp.serve", lambda graph, **options: None)
        monkeypatch.setattr("hopai.core.Graph.load_schema",
                            lambda self: (_ for _ in ()).throw(ValueError("no saved schema")))
        main(["--dsn", "postgresql+psycopg2://offline:offline@127.0.0.1:1/offline"])
        assert "serving without a declared schema" in capsys.readouterr().err


class TestServeArguments:
    def test_an_unknown_transport_is_refused_before_anything_starts(self):
        """Named here rather than left to the SDK: 'sse' and 'grpc' are
        both plausible guesses, and the SDK's own error arrives after a
        server has been built."""
        from hopai.mcp import serve

        with pytest.raises(ValueError, match="'stdio' or 'http'"):
            serve(offline(), transport="grpc")


# ---------------------------------------------------------------------
# Registration against the real SDK
# ---------------------------------------------------------------------

def advertised(tool) -> dict:
    """The input schema of a listed tool, under whichever name this SDK
    version gives the field (`inputSchema` in 1.x, `input_schema` in
    2.0 -- the same JSON on the wire either way)."""
    return getattr(tool, "inputSchema", None) or tool.input_schema


@needs_sdk
class TestServerRegistration:
    def test_every_tool_reaches_the_client_with_hopais_schema(self, vector_graph):
        """The end of the chain the rest of this file tests in pieces:
        what a client actually lists. _register() replaces the schema
        the SDK derives from the handler's Python signature, and if a
        future SDK stops honouring that, this is what fails."""
        server = build_server(vector_graph, embed=embedder(), allow_ddl=True)
        listed = {tool.name: tool for tool in asyncio.run(server.list_tools())}
        specs = named(vector_graph, embed=embedder(), allow_ddl=True)
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

        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            spec.call(dry_run=False)

    def test_enforce_returns_the_constraints_now_in_force(self, fresh_graph):
        fresh_graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"}])
        fresh_graph.define_schema(nodes=[NodeType("person", properties=[
            Property("email", "string", required=True)])])
        names = named(fresh_graph, allow_ddl=True)["enforce_schema"].call(dry_run=False)
        assert names["dry_run"] is False and names["constraints"]


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
