"""
hopai -- a knowledge graph in the PostgreSQL you already run.

Traversal, ingestion and real constraints on two ordinary tables. No
graph database, no extension, no new operational dependency.

SET UP -- idempotent, so both calls belong in your start-up path:

    from hopai import Graph, Unique, Required, PropertyType

    graph = Graph("postgresql+psycopg2://user:pass@host/db")
    graph.create_schema()
    graph.define_constraints(nodes=[Required("type"), Unique("email"),
                                    PropertyType("age", "number")])

WRITE -- a row with a `properties` key is nested; any other row is flat,
and every key that is not an identity key is a property:

    graph.add_nodes([{"id": 1, "type": "person", "email": "a@x.com"}])
    graph.add_edges([{"start_id": 1, "end": {"email": "b@x.com"}, "kind": "knows"}])
    graph.merge_nodes([{"type": "person", "email": "a@x.com"}], on=["email"])

...or the same thing in Cypher:

    graph.cypher("CREATE (a:person {email: 'a@x.com'})-[:knows]->(b:person {email: 'b@x.com'})")
    graph.cypher("MERGE (a:person {email: 'a@x.com'}) ON CREATE SET a.name = 'Alice'")

READ -- the same traversal in three interchangeable notations:

    from hopai import Start, Hop, traverse_json, traverse_cypher

    graph.traverse(                                     # Python
        Start(where={"type": "person"}),
        Hop(where={"active": True}, via={"kind": "friend"}, hops=(1, 4)),
    )
    traverse_json(graph, {                              # JSON in, JSON out
        "start": {"where": {"type": "person"}},
        "hops": [{"where": {"active": True}, "via": {"kind": "friend"},
                  "hops": [1, 4]}],
    })
    graph.cypher('''                                    # Cypher
        MATCH (a:person)-[:friend*1..4]->(b {active: true}) RETURN b
    ''')

graph.cypher() returns a Subgraph for a query that reads and an
IngestResult for one that writes.

A result carries every node and edge on a matching chain:

    result.nodes           # list[{"id": ..., "properties": {...}}]
    result.edges           # list[{"start_id": ..., "end_id": ..., "properties": {...}}]
    result.to_networkx()   # in-memory graph, if you have networkx installed

FOR TOOL-CALLING MODELS: TRAVERSE_TOOL_SCHEMA and INGEST_TOOL_SCHEMA are
JSON Schemas ready to hand to a function-calling definition, covering
reading and writing respectively.
"""

from .constraints import (
    Check, Col, ConstraintViolation, Index, PropertyType, Required, Unique,
)
from .core import Graph, Subgraph
from .cypher import (
    CypherError, cypher_to_operations, cypher_to_traversal, traverse_cypher,
)
from .filters import AND, BETWEEN, GT, GTE, LT, LTE, NOT, OR, parse_filter
from .hop import Hop, Start
from .ingest import INGEST_TOOL_SCHEMA, IngestResult
from .json_api import TRAVERSE_TOOL_SCHEMA, spec_to_traversal, traverse_json
from .models import Edge, Node

__version__ = "0.1.0"

__all__ = [
    "Graph", "Subgraph", "Start", "Hop",
    "OR", "AND", "NOT", "GT", "GTE", "LT", "LTE", "BETWEEN", "parse_filter",
    "traverse_json", "spec_to_traversal", "TRAVERSE_TOOL_SCHEMA",
    "traverse_cypher", "cypher_to_traversal", "cypher_to_operations", "CypherError",
    "Unique", "Required", "Check", "Index", "PropertyType", "Col", "ConstraintViolation",
    "IngestResult", "INGEST_TOOL_SCHEMA",
    "Node", "Edge",
]
