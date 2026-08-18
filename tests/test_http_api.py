"""
hopai.http_api -- the JSON HTTP API `hopai-api` serves.

The point of the module is that a browser can reach the graph, so the
tests drive it as a browser does: through Starlette's TestClient, over
real requests, against a real database. What is asserted is the FRONT
END -- routing, the permission gate, CORS, and the mapping from hopai's
refusals to status codes -- and not traversal semantics, which are
covered exhaustively against Graph elsewhere and reached here through
exactly the same json_api functions the MCP server calls.

The refusal-to-400 mapping gets the most attention, because it is the
part a UI depends on and the part that is easy to get quietly wrong: a
handler that turns "delete with no filter refuses" into a 500 with
"internal error" has thrown away the sentence that tells the user what
to do, and nothing about the response says so.
"""

from __future__ import annotations

import pytest

pytest.importorskip("starlette", reason='the HTTP API needs pip install "hopai[http]"')
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

from starlette.testclient import TestClient          # noqa: E402

from hopai import Start, Vector                      # noqa: E402
from hopai.http_api import Served, build_app         # noqa: E402


def dsn_of(graph) -> str:
    """main() takes a DSN string, and the fixtures hand out a Graph."""
    return graph.engine.url.render_as_string(hide_password=False)


def client(graph, **options) -> TestClient:
    return TestClient(build_app(graph, **options))


@pytest.fixture
def seeded(fresh_graph):
    fresh_graph.add_nodes([{"id": 1, "type": "person", "name": "Alice"},
                           {"id": 2, "type": "person", "name": "Bob"},
                           {"id": 3, "type": "company", "name": "Initech"}])
    fresh_graph.add_edges([{"start_id": 1, "end_id": 2, "kind": "knows"},
                           {"start_id": 2, "end_id": 3, "kind": "works_at"}])
    return fresh_graph


class TestReading:
    def test_health_names_the_graphs_it_serves(self, seeded):
        body = client(seeded).get("/health").json()
        assert body["status"] == "ok"
        assert body["graphs"] == [seeded.graph]

    def test_graphs_reports_the_permissions_it_was_started_with(self, seeded):
        body = client(seeded, allow_mutations=True).get("/graphs").json()
        assert body["permissions"] == {"writes": True, "mutations": True, "ddl": False}

    def test_traverse_returns_the_subgraph(self, seeded):
        answer = client(seeded).post("/traverse", json={
            "start": {"where": {"type": "person"}},
            "hops": [{"via": {"kind": "knows"}, "hops": 1}],
        })
        assert answer.status_code == 200
        assert {node["properties"]["name"] for node in answer.json()["nodes"]} == \
            {"Alice", "Bob"}

    def test_aggregate_returns_the_number(self, seeded):
        answer = client(seeded).post("/aggregate", json={
            "start": {"where": {"type": "person"}},
            "aggregates": {"n": {"fn": "count"}},
        })
        assert answer.json() == {"n": 2}

    def test_the_graph_key_is_routing_and_not_part_of_the_spec(self, seeded):
        """json_api refuses unknown spec keys on purpose, so `graph` has
        to be removed rather than passed through -- otherwise naming the
        graph would make every request a 400."""
        answer = client(seeded).post("/traverse", json={
            "graph": seeded.graph, "start": {"where": {"type": "company"}}})
        assert answer.status_code == 200
        assert len(answer.json()["nodes"]) == 1


class TestRefusalsReachTheCaller:
    def test_a_refusal_is_a_400_carrying_the_sentence_that_names_the_fix(self, seeded):
        """hopai refuses unknown keys because `top_k`/`limit`/`filter` are
        what a caller reaches for, and silently dropping one answers a
        different question. That sentence is the useful part of the
        response and has to survive the trip."""
        answer = client(seeded).post("/traverse", json={
            "start": {"where": {"type": "person"}, "limit": 5}})
        assert answer.status_code == 400
        assert "limit" in answer.json()["error"]

    def test_an_unserved_graph_is_refused_with_the_list(self, seeded):
        answer = client(seeded).post("/traverse", json={
            "graph": "nope", "start": {"where": {}}})
        assert answer.status_code == 400
        assert seeded.graph in answer.json()["error"]

    def test_a_body_that_is_not_json_says_so(self, seeded):
        answer = client(seeded).post("/traverse", content=b"not json",
                                     headers={"content-type": "application/json"})
        assert answer.status_code == 400
        assert "must be JSON" in answer.json()["error"]

    def test_a_json_array_body_is_refused_by_type(self, seeded):
        answer = client(seeded).post("/traverse", json=[1, 2, 3])
        assert answer.status_code == 400
        assert "JSON object" in answer.json()["error"]

    def test_an_invented_vector_is_refused_like_everywhere_else(self, seeded):
        """The invariant the whole library holds: floats a caller did not
        get from an embedding model do not travel through a JSON front
        end without saying allow_vectors."""
        seeded.define_vectors(nodes=[Vector("summary", 3)])
        answer = client(seeded).post("/search", json={
            "near": {"field": "summary", "vector": [0.1, 0.2, 0.3]}})
        assert answer.status_code == 400
        assert "cannot come from a tool call" in answer.json()["error"]


class TestPermissionsDecideWhichRoutesExist:
    def test_read_only_does_not_mount_ingest(self, seeded):
        """A route that is not permitted is not mounted, so it 404s
        rather than 403s -- there is nothing to talk past, and no handler
        that has to remember to check."""
        assert client(seeded, read_only=True).post("/ingest", json={}).status_code == 404

    def test_the_default_mounts_ingest_and_not_mutate(self, seeded):
        api = client(seeded)
        assert api.post("/ingest", json={"document": {"nodes": [{"id": 9}]}}).status_code == 200
        assert api.post("/mutate", json={}).status_code == 404

    def test_allow_mutations_mounts_mutate(self, seeded):
        answer = client(seeded, allow_mutations=True).post("/mutate", json={
            "document": {"operations": [
                {"op": "update_nodes", "where": {"type": "person"},
                 "set": {"seen": True}}]}})
        assert answer.status_code == 200
        assert answer.json()["updated_nodes"] == 2

    def test_ingest_actually_writes(self, seeded):
        """A 200 only says the handler returned. The row is the claim."""
        answer = client(seeded).post("/ingest", json={
            "document": {"nodes": [{"id": 42, "type": "person", "name": "Carol"}]}})
        assert answer.json() == {"nodes": 1, "edges": 0}
        assert len(seeded.traverse(Start(where={"name": "Carol"})).nodes) == 1

    def test_read_only_and_allow_mutations_contradict_at_construction(self, seeded):
        with pytest.raises(ValueError, match="contradicts"):
            Served(seeded, read_only=True, allow_mutations=True)

    def test_serving_nothing_refuses_rather_than_starting_empty(self):
        """An API with no graphs answers every request with "no graph
        named ..." -- a server that started and cannot work."""
        with pytest.raises(ValueError, match="non-empty"):
            Served({})

    def test_several_graphs_make_graph_required_and_say_which(self, seeded):
        """Picking one for the caller would answer a question about a
        graph they did not name."""
        api = client({"a": seeded, "b": seeded.in_graph("b")})
        answer = api.get("/schema")
        assert answer.status_code == 400
        assert "['a', 'b']" in answer.json()["error"]


class TestSchema:
    def test_it_reports_no_schema_as_null_rather_than_erroring(self, seeded):
        body = client(seeded).get("/schema").json()
        assert body == {"graph": seeded.graph, "schema": None}

    def test_a_declared_schema_comes_back(self, seeded):
        from hopai import NodeType
        seeded.define_schema(nodes=[NodeType("person")])
        body = client(seeded).get("/schema").json()
        assert "person" in str(body["schema"])


class TestAnUnexpectedErrorIsNotLeaked:
    def test_a_500_carries_no_traceback(self, seeded, monkeypatch, capsys):
        """The 400 path exists to pass hopai's sentences through. This
        one is its opposite: an error hopai did not raise on purpose says
        nothing about the internals, and the traceback goes to the
        server log where an operator can read it."""
        import hopai.http_api as module

        def explode(*args, **kwargs):
            raise RuntimeError("connection string: postgres://user:hunter2@db")

        monkeypatch.setattr(module, "traverse_json", explode)
        answer = client(seeded).post("/traverse", json={"start": {}})
        assert answer.status_code == 500
        assert answer.json() == {"error": "internal error -- see the server log",
                                 "type": "RuntimeError"}
        assert "hunter2" not in answer.text
        assert "hunter2" in capsys.readouterr().err      # the log DOES get it


class TestCypherIsGatedByClassificationNotByRouting:
    """One endpoint covers read, write and mutate, so registration cannot
    gate it -- it classifies the query first and refuses by permission
    before opening a connection, exactly as the MCP tool does."""

    def test_a_read_runs(self, seeded):
        answer = client(seeded).post("/cypher", json={
            "query": "MATCH (a {type: 'person'}) RETURN a"})
        assert answer.status_code == 200
        assert len(answer.json()["nodes"]) == 2

    def test_a_write_is_refused_on_a_read_only_api(self, seeded):
        answer = client(seeded, read_only=True).post("/cypher", json={
            "query": "CREATE (a:person {name: 'Dave'})"})
        assert answer.status_code == 400
        assert "read-only" in answer.json()["error"]

    def test_a_delete_is_refused_without_allow_mutations(self, seeded):
        answer = client(seeded).post("/cypher", json={
            "query": "MATCH (a {type: 'person'}) DETACH DELETE a"})
        assert answer.status_code == 400
        assert "--allow-mutations" in answer.json()["error"]

    def test_an_empty_query_is_refused_before_anything_runs(self, seeded):
        answer = client(seeded).post("/cypher", json={"query": "   "})
        assert answer.status_code == 400


class TestCORS:
    def test_no_origin_is_allowed_by_default(self, seeded):
        """There is no authentication here, so allowing any origin to
        POST /ingest is a decision an operator makes out loud rather than
        inherits."""
        answer = client(seeded).get("/health", headers={"Origin": "http://evil.test"})
        assert "access-control-allow-origin" not in answer.headers

    def test_a_named_origin_is_allowed(self, seeded):
        answer = client(seeded, cors=["http://localhost:3000"]).get(
            "/health", headers={"Origin": "http://localhost:3000"})
        assert answer.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_star_allows_the_page_that_a_ui_is_served_from(self, seeded):
        answer = client(seeded, cors=["*"]).get(
            "/health", headers={"Origin": "http://localhost:5173"})
        assert answer.headers["access-control-allow-origin"] == "*"


class TestSearch:
    def test_text_is_embedded_by_the_field(self, seeded):
        """The better of the two paths: the field's own client, so the
        query embedding comes from the model that wrote the stored ones.
        Needs nothing from the server."""
        from hopai import Embedder
        seeded.define_vectors(nodes=[Vector(
            "summary", 3, embed=Embedder(lambda texts: [[1.0, 0.0, 0.0] for _ in texts]))])
        seeded.migrate_vectors()
        seeded.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]},
                                  {"id": 2, "summary": [0.0, 1.0, 0.0]}])
        answer = client(seeded).post("/search", json={
            "near": {"field": "summary", "text": "anything"}, "k": 1})
        assert answer.status_code == 200
        assert answer.json()["results"][0]["id"] == "1"

    def test_the_server_embedder_is_the_fallback_for_a_field_with_none(self, seeded):
        """A field declared by --vector has a size and no client, which is
        every CLI-configured server that did not also name a provider.
        The server's own embed= answers there, the same shape and the
        same order as the MCP server's start.search."""
        seeded.define_vectors(nodes=[Vector("summary", 3)])
        seeded.migrate_vectors()
        seeded.set_vectors(nodes=[{"id": 1, "summary": [1.0, 0.0, 0.0]},
                                  {"id": 2, "summary": [0.0, 1.0, 0.0]}])
        api = client(seeded, embed=lambda text: [0.0, 1.0, 0.0])
        answer = api.post("/search", json={
            "near": {"field": "summary", "text": "anything"}, "k": 1})
        assert answer.status_code == 200
        assert answer.json()["results"][0]["id"] == "2"

    def test_a_field_with_no_embedder_and_no_server_one_refuses_by_name(self, seeded):
        seeded.define_vectors(nodes=[Vector("summary", 3)])
        seeded.migrate_vectors()
        answer = client(seeded).post("/search", json={
            "near": {"field": "summary", "text": "anything"}})
        assert answer.status_code == 400
        assert "declares no embedder" in answer.json()["error"]


class TestGraphDataIsOneCallForAWholeGraph:
    """The viewer's one read. `Start()` alone returns no edges and a hop
    alone prunes isolated nodes as dead ends -- only the OPTIONAL hop is
    both, and a page that lost either would be quietly wrong rather than
    broken."""

    def test_it_returns_every_node_and_every_edge(self, seeded):
        body = client(seeded).get("/graph-data").json()
        assert {node["id"] for node in body["nodes"]} == {"1", "2", "3"}
        assert len(body["edges"]) == 2
        assert body["truncated"] is False

    def test_an_isolated_node_survives(self, seeded):
        """OPTIONAL is the whole reason this is not a plain hop: a node
        with no edges is exactly what a viewer must still draw."""
        seeded.add_nodes([{"id": 4, "type": "note", "name": "Loose"}])
        body = client(seeded).get("/graph-data").json()
        assert "4" in {node["id"] for node in body["nodes"]}

    def test_edges_carry_the_id_the_page_edits_by(self, seeded):
        """A page repoints and deletes BY ID. An edge without one is a
        line it can draw and nothing it can act on."""
        body = client(seeded).get("/graph-data").json()
        assert all(edge.get("id") for edge in body["edges"])

    def test_limit_drops_dangling_edges_with_the_nodes(self, seeded):
        """Capping nodes and keeping their edges would leave lines
        pointing at nothing. The result is a smaller graph, not a
        broken one -- and `truncated` says it happened."""
        body = client(seeded).get("/graph-data?limit=1").json()
        kept = {node["id"] for node in body["nodes"]}
        assert len(kept) == 1
        assert body["truncated"] is True
        assert all(edge["start_id"] in kept and edge["end_id"] in kept
                   for edge in body["edges"])


class TestTheExplorerPageIsServed:
    def test_slash_returns_the_page(self, seeded):
        answer = client(seeded).get("/")
        assert answer.status_code == 200
        assert answer.headers["content-type"].startswith("text/html")
        assert "<title>hopai · graph explorer</title>" in answer.text

    def test_no_ui_leaves_the_json_endpoints_alone(self, seeded):
        api = client(seeded, ui=False)
        assert api.get("/").status_code == 404
        assert api.get("/health").status_code == 200

    def test_the_page_is_self_contained(self, seeded):
        """No CDN: the whole point of shipping it in the wheel is that it
        works on a laptop with no internet and inside a container with no
        egress. An external <script> or <link> would make the page a
        network dependency the library promises not to have."""
        import re
        page = client(seeded).get("/").text
        assert not re.search(r'<(script|link)[^>]+(src|href)="(https?:)?//', page)

    def test_the_page_ships_in_the_package(self):
        """Declared as package-data in pyproject.toml. Without that line
        `pip install hopai[http]` installs the module and not the file,
        and / 500s on a server that looked fine in the repository."""
        from hopai.http_api import UI_PAGE
        assert UI_PAGE.exists() and UI_PAGE.name == "index.html"


class TestEditingByIdIsGatedLikeEveryOtherWrite:
    """The page's two destructive powers. Repointing is a WRITE (it moves
    an edge that already exists); deleting is a MUTATION -- the same
    split mcp.py makes, for the same reason: creating a row and
    destroying one are not the same permission."""

    def test_repoint_needs_writes(self, seeded):
        assert client(seeded, read_only=True).post(
            "/edges/repoint", json={"id": 1}).status_code == 404

    def test_deleting_needs_allow_mutations(self, seeded):
        api = client(seeded)          # the default: writes, no mutations
        assert api.post("/nodes/delete", json={"ids": [1]}).status_code == 404
        assert api.post("/edges/delete", json={"ids": [1]}).status_code == 404

    def test_repoint_moves_one_end_and_leaves_the_other(self, seeded):
        edge = client(seeded).get("/graph-data").json()["edges"][0]
        answer = client(seeded).post("/edges/repoint",
                                     json={"id": edge["id"], "end_id": 3})
        assert answer.status_code == 200
        assert answer.json()["updated_edges"] == 1
        after = next(e for e in client(seeded).get("/graph-data").json()["edges"]
                     if e["id"] == edge["id"])
        assert (after["start_id"], after["end_id"]) == (edge["start_id"], "3")

    def test_repoint_without_an_id_refuses_rather_than_guessing(self, seeded):
        answer = client(seeded).post("/edges/repoint", json={"end_id": 3})
        assert answer.status_code == 400
        assert "id" in answer.json()["error"]

    def test_deleting_with_no_ids_refuses(self, seeded):
        """An empty list is what an empty selection looks like. Deleting
        on it is the unrecoverable version of a no-op."""
        api = client(seeded, allow_mutations=True)
        for route in ("/nodes/delete", "/edges/delete"):
            answer = api.post(route, json={"ids": []})
            assert answer.status_code == 400
            assert "non-empty" in answer.json()["error"]

    def test_deleting_a_node_by_id_takes_its_edges_with_it(self, seeded):
        answer = client(seeded, allow_mutations=True).post(
            "/nodes/delete", json={"ids": [2], "detach": True})
        assert answer.json() == {"deleted_nodes": 1, "deleted_edges": 2}
        assert {n["id"] for n in client(seeded).get("/graph-data").json()["nodes"]} == {"1", "3"}

    def test_deleting_an_attached_node_without_detach_refuses_by_name(self, seeded):
        """Cascading instead would leave exactly the corruption the
        composite foreign key exists to prevent, so the refusal names
        the flag rather than doing it anyway."""
        answer = client(seeded, allow_mutations=True).post(
            "/nodes/delete", json={"ids": [2]})
        assert answer.status_code == 400
        assert "detach" in answer.json()["error"]

    def test_deleting_an_edge_by_id_leaves_its_endpoints(self, seeded):
        api = client(seeded, allow_mutations=True)
        edge = api.get("/graph-data").json()["edges"][0]
        assert api.post("/edges/delete", json={"ids": [edge["id"]]}).json() == {
            "deleted_edges": 1}
        assert len(api.get("/graph-data").json()["nodes"]) == 3


class TestTheCommandLine:
    """`hopai-api`'s parsing, with nothing served. main() is where the
    graphs, the vector fields and the embedder are assembled, and every
    mistake it can catch is one an operator makes at 3am with a compose
    file -- so each refusal is asserted by the sentence it prints, not
    just by the exit code."""

    def _main(self, monkeypatch, argv, **stub):
        """Run main() with serve() replaced. What is under test is
        everything BEFORE the server starts."""
        from hopai import http_api
        captured = {}

        def fake_serve(graphs, **options):
            captured["graphs"] = graphs
            captured["options"] = options

        monkeypatch.setattr(http_api, "serve", fake_serve)
        for name, value in stub.items():
            monkeypatch.setattr(http_api, name, value)
        code = http_api.main(argv)
        return code, captured

    def test_it_serves_every_graph_it_was_named(self, seeded, monkeypatch):
        code, captured = self._main(monkeypatch, [
            "--dsn", dsn_of(seeded), "--graph", seeded.graph, "--no-load-schema"])
        assert code == 0
        assert sorted(captured["graphs"]) == [seeded.graph]

    def test_no_dsn_names_both_ways_to_give_one(self, monkeypatch, capsys):
        monkeypatch.delenv("HOPAI_DSN", raising=False)
        with pytest.raises(SystemExit):
            self._main(monkeypatch, [])
        assert "--dsn or set HOPAI_DSN" in capsys.readouterr().err

    def test_a_vector_naming_an_unserved_graph_refuses(self, seeded, monkeypatch, capsys):
        """Silently ignoring it would leave the field undeclared and
        every search on it answering "no such field" much later."""
        with pytest.raises(SystemExit):
            self._main(monkeypatch, ["--dsn", dsn_of(seeded), "--graph", seeded.graph,
                                     "--vector", "elsewhere:nodes:bio:3"])
        assert "does not serve" in capsys.readouterr().err

    def test_a_vector_field_is_declared_on_the_graph_it_names(self, seeded, monkeypatch):
        _, captured = self._main(monkeypatch, [
            "--dsn", dsn_of(seeded), "--graph", seeded.graph, "--no-load-schema",
            "--vector", "nodes:bio:4"])
        graph = captured["graphs"][seeded.graph]
        assert graph.vectors["nodes"]["bio"].dimensions == 4

    def test_ui_is_on_by_default_and_no_ui_turns_it_off(self, seeded, monkeypatch):
        _, on = self._main(monkeypatch, ["--dsn", dsn_of(seeded), "--graph", seeded.graph,
                                         "--no-load-schema"])
        _, off = self._main(monkeypatch, ["--dsn", dsn_of(seeded), "--graph", seeded.graph,
                                          "--no-load-schema", "--no-ui"])
        assert on["options"]["ui"] is True
        assert off["options"]["ui"] is False

    def test_permissions_reach_the_app_as_given(self, seeded, monkeypatch):
        _, captured = self._main(monkeypatch, [
            "--dsn", dsn_of(seeded), "--graph", seeded.graph, "--no-load-schema",
            "--allow-mutations", "--allow-ddl"])
        assert captured["options"]["allow_mutations"] is True
        assert captured["options"]["allow_ddl"] is True

    def test_cors_comes_from_the_flag_then_the_environment_then_nothing(
            self, seeded, monkeypatch):
        from hopai.http_api import _origins

        class Args:
            cors = None
        monkeypatch.delenv("HOPAI_API_CORS", raising=False)
        assert _origins(Args()) == []
        monkeypatch.setenv("HOPAI_API_CORS", "http://a.test, http://b.test")
        assert _origins(Args()) == ["http://a.test", "http://b.test"]
        Args.cors = ["http://c.test"]
        assert _origins(Args()) == ["http://c.test"]

    def test_an_unreadable_database_says_to_name_the_graphs_instead(
            self, monkeypatch, capsys):
        """`--graph` is the way past a database this process cannot
        enumerate, so the refusal names it rather than only reporting
        the driver error."""
        with pytest.raises(SystemExit):
            self._main(monkeypatch,
                       ["--dsn", "postgresql+psycopg2://nobody:x@127.0.0.1:1/none"])
        assert "Pass --graph" in capsys.readouterr().err

    def test_an_embed_provider_that_cannot_be_built_fails_before_the_database(
            self, monkeypatch, capsys):
        """Exit 2 with the sentence, not a traceback -- and BEFORE any
        connection, so a missing key is not diagnosed as a database
        problem."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            self._main(monkeypatch, ["--dsn", "postgresql+psycopg2://u:p@127.0.0.1:1/x",
                                     "--embed-provider", "openai",
                                     "--embed-model", "text-embedding-3-small"])
        assert "--embed-provider openai needs" in capsys.readouterr().err
