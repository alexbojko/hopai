"""
benchmarks/comparison.py

Neo4j and Apache AGE, running the same 14 traversal queries against the
same data, so hopai's numbers have something to be compared to.

    docker compose --profile compare up -d
    python bench_hopai.py --data-dir ./data --dsn ... \
        --neo4j-url http://localhost:7474 --age-dsn postgresql://...:5433/agebench

WHY THESE TWO. AGE is the "just put a graph extension on the Postgres you
already have" answer and Neo4j is the "use a real graph database" answer.
hopai's whole premise is that neither is needed for this workload, and a
premise like that is worth nothing without the other two systems in the
same table on the same machine.

NO NEW DEPENDENCIES, even here. Neo4j is driven through its HTTP query
API with urllib rather than the `neo4j` package; AGE is a PostgreSQL
extension, so psycopg2 -- already required -- reaches it.

WHAT IS AND IS NOT COMPARABLE:

  - Every system loads the SAME generated CSVs and answers the same
    question, and the runner checks the answers agree. A speed
    comparison between systems returning different results is not a
    comparison.
  - Timings are wall-clock around query execution. Neo4j's exclude
    client start-up (the HTTP API is already connected); AGE's are
    measured the same way hopai's are, through psycopg2.
  - A query that overruns its budget is reported DNF, never as a large
    number. In this workload that outcome is not hypothetical.

ONE KNOWN, REAL DISAGREEMENT -- Q6, mixed backward then forward. Cypher
enforces RELATIONSHIP UNIQUENESS inside a single MATCH: the same edge may
not be traversed twice in one pattern, so `(h)<-[]-(m)-[]->(x)` cannot
return x = h by walking back down the edge it arrived on, and both Neo4j
and AGE answer 0. hopai has no such rule -- its hops are independent
steps, so the walk goes back up and h is matched, answering 1.

Neither is wrong; they are different questions. It is recorded here, and
flagged by the report, because it also means `traverse_cypher()` of that
pattern does not answer what Neo4j would answer -- which is worth knowing
before treating the Cypher front end as a drop-in.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

#: Cypher for each query in the traversal suite, keyed by the id
#: bench_hopai.py uses. Standard Cypher -- what Neo4j runs. AGE's dialect
#: needs the rewrites in AGE_OVERRIDES below.
CYPHER = {
    "Q1": "MATCH (a:Node {type:'leaf'})-[:EDGE]->(m:Node {flag:1}) RETURN count(DISTINCT m) AS n",
    "Q2": "MATCH (h:Node {type:'hub'})<-[:EDGE]-(d:Node) RETURN count(DISTINCT d) AS n",
    "Q3": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
          "RETURN count(DISTINCT m) AS n",
    "Q4": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) RETURN count(DISTINCT a) AS n",
    "Q5": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
          "MATCH (m)-[:EDGE*1..3]->(h:Node {type:'hub'}) RETURN count(DISTINCT h) AS n",
    "Q6": "MATCH (h:Node {type:'hub'})<-[:EDGE]-(m:Node)-[:EDGE]->(x:Node) "
          "RETURN count(DISTINCT x) AS n",
    "Q7": "MATCH p=(a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
          "WHERE all(r IN relationships(p) WHERE r.tag IN ['p1','p2']) "
          "RETURN count(DISTINCT m) AS n",
    "Q8": "MATCH (a:Node)-[:EDGE]->(b:Node) WHERE a.type IN ['leaf','hub'] "
          "RETURN count(DISTINCT b) AS n",
    # NULL-safe negation: `a.type <> 'leaf'` alone drops nodes with no
    # type at all, which is the trap hopai's containment NOT avoids.
    "Q9": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) "
          "WHERE a.type IS NULL OR a.type <> 'leaf' RETURN count(DISTINCT a) AS n",
    "Q10": "MATCH (a:Node {type:'leaf'}) WHERE a.priority > 5 RETURN count(a) AS n",
    "Q11": "MATCH (a:Node {type:'leaf'}) WHERE a.priority >= 5 AND a.priority <= 15 "
           "RETURN count(a) AS n",
    "Q12": "MATCH (a:Node) WHERE (a.type IN ['leaf','hub']) AND a.type = 'leaf' "
           "RETURN count(a) AS n",
    "Q13": "MATCH (h:Node {type:'hub'})<-[:EDGE]-(a:Node) "
           "WHERE a.type IS NULL OR a.type <> 'leaf' "
           "OPTIONAL MATCH (a)-[:EDGE]->(d:Node {type:'leaf'}) "
           "RETURN count(DISTINCT a) AS n",
    "Q14": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..12]-(a:Node) RETURN count(DISTINCT a) AS n",
}


# Aggregations, Q15-Q29. Every query in the suite has a Cypher
# equivalent so no cell in the comparison table is empty -- a blank cell
# reads as "the other system could not do this", which is a claim, and
# usually a false one.
#
# `WITH DISTINCT a` before the aggregate is load-bearing: hopai
# aggregates over the DISTINCT nodes the last hop matched, while a bare
# Cypher aggregate counts once per PATH. Without it a node reachable
# three ways is summed three times and the two systems answer different
# questions.
CYPHER.update({
    "Q15": "MATCH (a:Node {type:'leaf'}) RETURN count(a) AS n",
    "Q16": "MATCH (a:Node {type:'leaf'}) WHERE a.priority > 5 RETURN avg(a.priority) AS n",
    "Q17": "MATCH (a:Node {type:'leaf'}) "
           "RETURN min(a.priority) AS lo, max(a.priority) AS hi",
    "Q18": "MATCH (a:Node {type:'leaf'})-[:EDGE]->(m:Node {flag:1}) "
           "RETURN count(DISTINCT m) AS n",
    "Q19": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
           "RETURN count(DISTINCT m) AS n",
    "Q20": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) "
           "WHERE a.priority IS NOT NULL RETURN count(DISTINCT a) AS n",
    "Q21": "MATCH (f:Node {flag:1})<-[:EDGE*1..4]-(a:Node) WITH DISTINCT a "
           "RETURN count(a) AS n, sum(a.priority) AS total, avg(a.priority) AS mean, "
           "min(a.priority) AS lo, max(a.priority) AS hi",
    "Q22": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..3]-(a:Node) "
           "WHERE a.type IS NULL OR a.type <> 'leaf' RETURN count(DISTINCT a) AS n",
    "Q23": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..4]-(a:Node) "
           "RETURN count(DISTINCT a.priority) AS n",
    "Q24": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..4]-(a:Node) "
           "WITH DISTINCT a.priority AS p RETURN sum(p) AS total, avg(p) AS mean",
    "Q25": "MATCH (a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
           "MATCH (m)-[:EDGE*1..3]->(h:Node {type:'hub'}) RETURN count(DISTINCT h) AS n",
    "Q26": "MATCH p=(a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
           "WHERE all(r IN relationships(p) WHERE r.tag IN ['p1','p2']) "
           "RETURN count(DISTINCT m) AS n",
    "Q27": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..12]-(a:Node) "
           "RETURN count(DISTINCT a) AS n",
    "Q28": "MATCH (h:Node {type:'hub'})<-[:EDGE*1..12]-(a:Node) WITH DISTINCT a "
           "RETURN count(a) AS n, count(a.priority) AS have, "
           "count(DISTINCT a.priority) AS distinct_p, sum(a.priority) AS total, "
           "avg(a.priority) AS mean, min(a.priority) AS lo, max(a.priority) AS hi",
    "Q29": "MATCH (a:Node)-[:EDGE*1..2]->(b:Node) "
           "WHERE (a.type IN ['leaf','hub']) AND a.type = 'leaf' "
           "RETURN count(DISTINCT b) AS n",
})


#: AGE 1.5/1.7 has no `all()` over relationships(p), so Q7 needs the
#: UNWIND+CASE rewrite. Documented in benchmarks/README.md because it is
#: also the query where AGE's answer historically disagreed.
AGE_OVERRIDES = {
    # AGE has no all() over relationships(p) -- same rewrite as Q7
    "Q26": "MATCH p=(a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
           "UNWIND relationships(p) AS r "
           "WITH m, collect(CASE WHEN r.tag IN ['p1','p2'] THEN 1 ELSE 0 END) AS ok "
           "WHERE NOT 0 IN ok RETURN count(DISTINCT m) AS n",
    "Q7": "MATCH p=(a:Node {type:'leaf'})-[:EDGE*1..4]->(m:Node {flag:1}) "
          "UNWIND relationships(p) AS r "
          "WITH m, collect(CASE WHEN r.tag IN ['p1','p2'] THEN 1 ELSE 0 END) AS ok "
          "WHERE NOT 0 IN ok RETURN count(DISTINCT m) AS n",
}


def _as_number(value):
    """agtype comes back as text. An aggregate may be a count or a mean,
    so parse int first and fall back to float -- `int("12.49")` raises,
    which sank a whole run on the one query that returns an average."""
    if value is None:
        return None
    text = str(value).strip('"')
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


class Neo4jRunner:
    """Neo4j over its HTTP query API -- no driver dependency."""

    name = "Neo4j"

    def __init__(self, url: str, user: str = "neo4j", password: str = "benchpassword",
                 database: str = "neo4j"):
        self.endpoint = f"{url.rstrip('/')}/db/{database}/tx/commit"
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.headers = {"Content-Type": "application/json", "Authorization": f"Basic {token}"}

    def _post(self, statement: str, timeout: float):
        body = json.dumps({"statements": [{"statement": statement}]}).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers=self.headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        if payload.get("errors"):
            raise RuntimeError(payload["errors"][0].get("message", "neo4j error"))
        return payload["results"]

    def load(self, node_count: int = 0, edge_count: int = 0) -> None:
        """Load the same CSVs, then index the properties queries filter
        on -- an unindexed comparison would measure the missing index."""
        # Batched: a single DETACH DELETE over a million-node graph
        # builds one transaction bigger than the heap and dies with a
        # bare "Failed to commit transaction".
        self._post("MATCH (n) CALL { WITH n DETACH DELETE n } "
                   "IN TRANSACTIONS OF 10000 ROWS", timeout=3600)
        self._post("CREATE INDEX node_id IF NOT EXISTS FOR (n:Node) ON (n.id)", timeout=300)
        self._post("LOAD CSV FROM 'file:///nodes.csv' AS row "
                   "CALL { WITH row CREATE (n:Node {id: toInteger(row[0]), props: row[1]}) } "
                   "IN TRANSACTIONS OF 10000 ROWS", timeout=3600)
        # properties arrive as a JSON string; expand the three the suite
        # filters on into real properties so Neo4j can index them
        self._post("MATCH (n:Node) WHERE n.props CONTAINS '\"type\": \"leaf\"' "
                   "SET n.type = 'leaf' ", timeout=3600)
        self._post("MATCH (n:Node) WHERE n.props CONTAINS '\"type\": \"hub\"' "
                   "SET n.type = 'hub'", timeout=3600)
        self._post("MATCH (n:Node) WHERE n.props CONTAINS '\"flag\": 1' SET n.flag = 1",
                   timeout=3600)
        self._post("MATCH (n:Node) WHERE n.props CONTAINS 'priority' "
                   "SET n.priority = toInteger(split(split(n.props, '\"priority\": ')[1], '}')[0])",
                   timeout=3600)
        for statement in (
            "CREATE INDEX node_type IF NOT EXISTS FOR (n:Node) ON (n.type)",
            "CREATE INDEX node_flag IF NOT EXISTS FOR (n:Node) ON (n.flag)",
            "CREATE INDEX node_priority IF NOT EXISTS FOR (n:Node) ON (n.priority)",
        ):
            self._post(statement, timeout=600)
        self._post("LOAD CSV FROM 'file:///edges.csv' AS row "
                   "CALL { WITH row "
                   "MATCH (a:Node {id: toInteger(row[0])}), (b:Node {id: toInteger(row[1])}) "
                   "CREATE (a)-[:EDGE {tag: row[2]}]->(b) } IN TRANSACTIONS OF 10000 ROWS",
                   timeout=7200)

    def run(self, qid: str, budget_s: float) -> tuple:
        statement = CYPHER.get(qid)
        if statement is None:
            return None, None
        t0 = time.perf_counter()
        try:
            results = self._post(statement, timeout=budget_s)
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError):
            return None, "DNF"
        elapsed = (time.perf_counter() - t0) * 1000
        rows = results[0].get("data", [])
        answer = rows[0]["row"][0] if rows and rows[0].get("row") else None
        return elapsed, answer


class AgeRunner:
    """Apache AGE, reached with psycopg2 -- it is a Postgres extension."""

    name = "Apache AGE"
    GRAPH = "benchgraph"

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        import psycopg2

        conn = psycopg2.connect(self.dsn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;")
        return conn

    def load(self, node_count: int = 0, edge_count: int = 0) -> None:
        """Build the graph from the same CSVs.

        AGE has no bulk loader and creating a million vertices one
        `CREATE` at a time is not a benchmark, it is a wait. So the rows
        are staged with COPY and inserted straight into the label tables
        AGE keeps underneath -- which is how AGE's own docs suggest bulk
        loading, and takes under a second here.

        The edge insert needs each endpoint's internal graphid, so the
        external ids are mapped once into an indexed table first. Note
        `properties->'"id"'`: agtype subscripting takes a quoted key, and
        the plain `->>'id'` spelling fails with "Expected agtype value".
        """
        conn = self._connect()
        with conn.cursor() as cur:
            if self._graph_exists(cur):
                cur.execute(f"SELECT drop_graph('{self.GRAPH}', true)")
            cur.execute(f"SELECT create_graph('{self.GRAPH}')")
            cur.execute(f"SELECT create_vlabel('{self.GRAPH}', 'Node')")
            cur.execute(f"SELECT create_elabel('{self.GRAPH}', 'EDGE')")

            cur.execute("DROP TABLE IF EXISTS stage_nodes, stage_edges, idmap")
            cur.execute("CREATE TABLE stage_nodes (id bigint, props text)")
            cur.execute("CREATE TABLE stage_edges (start_id bigint, end_id bigint, tag text)")
            # server-side COPY: the CSVs are mounted into the container
            cur.execute("COPY stage_nodes FROM '/data/nodes.csv' WITH (FORMAT csv)")
            cur.execute("COPY stage_edges FROM '/data/edges.csv' WITH (FORMAT csv)")

            # the generator writes properties as a JSON blob; expand the
            # three the suite filters on into real agtype properties, or
            # every filter would have to parse a string
            cur.execute(f'''
                INSERT INTO {self.GRAPH}."Node" (properties)
                SELECT format('{{"id": %s, "type": %s, "flag": %s, "priority": %s}}',
                              s.id,
                              coalesce('"'||(s.props::json->>'type')||'"', 'null'),
                              coalesce(s.props::json->>'flag', 'null'),
                              coalesce(s.props::json->>'priority', 'null'))::agtype
                FROM stage_nodes s
            ''')
            cur.execute(f'''
                CREATE TABLE idmap AS
                SELECT (properties->'"id"')::text::bigint AS ext_id, id AS gid
                FROM {self.GRAPH}."Node"
            ''')
            cur.execute("CREATE INDEX ON idmap (ext_id)")
            cur.execute(f'''
                INSERT INTO {self.GRAPH}."EDGE" (start_id, end_id, properties)
                SELECT a.gid, b.gid, format('{{"tag": "%s"}}', s.tag)::agtype
                FROM stage_edges s
                JOIN idmap a ON a.ext_id = s.start_id
                JOIN idmap b ON b.ext_id = s.end_id
            ''')
            cur.execute(f'ANALYZE {self.GRAPH}."Node"')
            cur.execute(f'ANALYZE {self.GRAPH}."EDGE"')
        conn.close()

    @staticmethod
    def _graph_exists(cur) -> bool:
        cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = 'benchgraph'")
        return bool(cur.fetchone()[0])

    def run(self, qid: str, budget_s: float) -> tuple:
        statement = AGE_OVERRIDES.get(qid, CYPHER.get(qid))
        if statement is None:
            return None, None
        import psycopg2

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{int(budget_s * 1000)}ms'")
                t0 = time.perf_counter()
                try:
                    cur.execute(
                        f"SELECT * FROM cypher('{self.GRAPH}', $${statement}$$) AS (n agtype)")
                    rows = cur.fetchall()
                except psycopg2.errors.QueryCanceled:
                    return None, "DNF"
                except psycopg2.Error:
                    return None, "ERROR"
                elapsed = (time.perf_counter() - t0) * 1000
            return elapsed, _as_number(rows[0][0] if rows else None)
        finally:
            conn.close()
