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

AGGREGATE -- a number instead of a subgraph, computed in the database,
over the distinct nodes the last step matched; the same three notations:

    from hopai import Count, Sum, Avg, Min, Max, aggregate_json

    graph.aggregate(                                    # Python
        Start(where={"type": "person"}),
        Hop(via={"kind": "friend"}, hops=(1, 4)),
        aggregates={"friends": Count(), "avg_age": Avg("age")},
    )                                                   # {"friends": 42, "avg_age": 31.5}
    aggregate_json(graph, {                             # JSON in, JSON out
        "start": {"where": {"type": "person"}},
        "hops": [{"via": {"kind": "friend"}, "hops": [1, 4]}],
        "aggregates": {"friends": {"fn": "count"}},
    })
    graph.cypher('''                                    # Cypher
        MATCH (a:person)-[:friend*1..4]->(b) RETURN count(DISTINCT b)
    ''')

VECTOR SEARCH -- exact cosine similarity on plain real[] columns, no
pgvector, no extension; declare named fields, migrate, store, search,
or seed a traversal from similarity:

    from hopai import Vector, Near

    graph.define_vectors(nodes=[Vector("summary", 1536)])
    graph.migrate_vectors()                       # ALTER TABLE, idempotent
    graph.set_vectors(nodes=[{"id": 1, "summary": embedding}])
    graph.vector_search(Near("summary", query_embedding), k=10,
                        where={"type": "person"})
    graph.traverse(Start(near=Near("summary", query_embedding), k=25),
                   Hop(via={"kind": "cites"}))

Several Near specs combine into one weighted score (multivector
search). See hopai/vectors.py for the storage model, the cost model,
and every deliberate refusal.

FOR TOOL-CALLING MODELS: TRAVERSE_TOOL_SCHEMA, AGGREGATE_TOOL_SCHEMA and
INGEST_TOOL_SCHEMA are JSON Schemas ready to hand to a function-calling
definition, covering reading, aggregating and writing respectively.
(Vector search has no tool schema on purpose -- a model asked for an
embedding invents one; vectors.py explains.)

SCHEMA -- declare the shape of the graph (node types, edge kinds, typed
properties) as plain dataclasses or NodeType/EdgeType primitives, read
it back as dataclasses, JSON Schema, networkx or pydantic models, and
optionally have Postgres enforce it on every write path:

    graph.define_schema(nodes=[Person, Company], edges=[WorksAt])
    graph.schema_json        # for a system prompt or tool result
    graph.enforce_schema()   # CHECK constraints; violations raise
                             # ConstraintViolation

See hopai/schema.py for both notations and the annotation mapping.
"""

from .aggregates import Avg, Count, Max, Min, Sum, parse_aggregate
from .constraints import (
    Check, Col, ConstraintViolation, Index, PropertyType, Required, Unique,
)
from .core import Graph, Subgraph
from .cypher import (
    CypherError, aggregate_cypher, cypher_to_aggregation, cypher_to_operations,
    cypher_to_traversal, traverse_cypher,
)
from .filters import AND, BETWEEN, GT, GTE, LT, LTE, NOT, OR, parse_filter
from .hop import Hop, Start
from .ingest import INGEST_TOOL_SCHEMA, IngestResult
from .json_api import (
    AGGREGATE_TOOL_SCHEMA, TRAVERSE_TOOL_SCHEMA, aggregate_json, spec_to_aggregation,
    spec_to_traversal, traverse_json, vector_search_json,
)
from .models import Edge, Node
from .schema import EdgeType, GraphSchema, InferenceReport, NodeType, Property, TypeConflict
from .vectors import Boost, Near, Vector, parse_near

# The trailing annotation is load-bearing: release-please updates this
# file with its GENERIC updater, which rewrites the version only on lines
# carrying this marker. Without it the file is left untouched and
# hopai.__version__ silently drifts from the released version. There is
# no "python" extra-file type -- the python strategy only auto-updates a
# file literally named version.py.
__version__ = "0.0.1"  # x-release-please-version

__all__ = [
    "Graph", "Subgraph", "Start", "Hop",
    "OR", "AND", "NOT", "GT", "GTE", "LT", "LTE", "BETWEEN", "parse_filter",
    "Count", "Sum", "Avg", "Min", "Max", "parse_aggregate",
    "traverse_json", "spec_to_traversal", "TRAVERSE_TOOL_SCHEMA",
    "aggregate_json", "spec_to_aggregation", "AGGREGATE_TOOL_SCHEMA",
    "traverse_cypher", "cypher_to_traversal", "cypher_to_operations", "CypherError",
    "aggregate_cypher", "cypher_to_aggregation",
    "Unique", "Required", "Check", "Index", "PropertyType", "Col", "ConstraintViolation",
    "GraphSchema", "NodeType", "EdgeType", "Property", "InferenceReport", "TypeConflict",
    "Vector", "Near", "Boost", "parse_near", "vector_search_json",
    "IngestResult", "INGEST_TOOL_SCHEMA",
    "Node", "Edge",
]
