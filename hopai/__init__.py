"""
hopai -- graph traversal on plain PostgreSQL.

    from hopai import Graph, Start, Hop, OR, AND, NOT, GT, GTE, LT, LTE, BETWEEN

    graph = Graph("postgresql+psycopg2://user:pass@host/db")

    result = graph.traverse(
        Start(where={"type": "person"}),
        Hop(where={"active": True}, via={"kind": "friend"}, hops=(1, 4)),
        Hop(where={"type": "company"}, hops=3),
    )

    result.nodes           # list[{"id": ..., "properties": {...}}]
    result.edges           # list[{"start_id": ..., "end_id": ..., "properties": {...}}]
    result.to_networkx()   # in-memory graph, if you have networkx installed

For callers that want JSON in, JSON out (LLM tool calls, an HTTP
handler, config-driven traversal):

    from hopai import traverse_json
    traverse_json(graph, {
        "start": {"where": {"type": "person"}},
        "hops": [{"where": {"active": True}, "hops": [1, 4]}],
    })
"""

from .core import Graph, Subgraph
from .filters import AND, BETWEEN, GT, GTE, LT, LTE, NOT, OR, parse_filter
from .hop import Hop, Start
from .json_api import TRAVERSE_TOOL_SCHEMA, spec_to_traversal, traverse_json
from .models import Edge, Node

__version__ = "0.1.0"

__all__ = [
    "Graph", "Subgraph", "Start", "Hop",
    "OR", "AND", "NOT", "GT", "GTE", "LT", "LTE", "BETWEEN", "parse_filter",
    "traverse_json", "spec_to_traversal", "TRAVERSE_TOOL_SCHEMA",
    "Node", "Edge",
]
