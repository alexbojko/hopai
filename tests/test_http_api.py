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
