"""
benchmarks/generate_graph.py

Generates a synthetic graph for benchmarking: a sparse random background
DAG (models incidental, mostly-irrelevant connectivity) plus a
deliberately structured "hub" subgraph -- a widely-shared node with many
direct and transitive dependents, fanning out across several depth
levels -- which is the shape that actually stresses a graph engine (real
fan-in/fan-out), unlike a purely random graph.

Usage:
    python generate_graph.py --nodes 1000000 --seed 42 --out-dir ./data

Produces two CSVs (nodes.csv, edges.csv) suitable for bulk COPY into a
hopai-shaped schema (see hopai/models.py), plus a summary of the
hub structure's known shape (depth counts) for writing benchmark
assertions against.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def generate(n_nodes: int, seed: int, out_dir: Path, hub_id: int | None = None,
             fanin_per_level=(15, 5, 3, 3, 2, 2, 2), edges_per_node: float = 3.0):
    """Build the graph and stream it to CSV.

    `edges_per_node` is the average out-degree of the background DAG, and
    it is the knob that decides whether this is a graph worth
    benchmarking. At the original 0.8 a million nodes produced only
    810k edges -- fewer edges than nodes, which is a forest of short
    chains, not a graph. Traversal cost is driven by fan-out, so a
    density below 1 measures almost nothing. The default 3.0 gives a
    million nodes about three million edges.

    Edges stream to disk as they are generated rather than accumulating
    in a list: at these densities the list alone is hundreds of
    megabytes, and nothing here needs them all in memory at once.
    """
    random.seed(seed)
    hub_id = hub_id or (n_nodes // 2)
    if edges_per_node < 0:
        raise ValueError(f"edges_per_node must be >= 0, got {edges_per_node}")

    out_dir.mkdir(parents=True, exist_ok=True)
    edge_count = 0
    # the reverse index the hub labelling needs, kept only for the hub
    # subgraph rather than the whole graph -- a full one at three million
    # edges is a dict nobody needs
    parents_of: dict[int, list[int]] = {}

    # not a `with`: the writer is used by the nested emit() across two
    # separate generation phases, and closed explicitly once both are
    # done. contextlib.ExitStack would buy nothing here.
    edge_file = open(out_dir / "edges.csv", "w", newline="")  # noqa: SIM115
    edge_writer = csv.writer(edge_file)

    def emit(a: int, b: int, tag: str, remember: bool = False) -> None:
        nonlocal edge_count
        edge_writer.writerow([a, b, tag])
        edge_count += 1
        if remember:
            parents_of.setdefault(b, []).append(a)

    # background DAG: every node points at `edges_per_node` randomly
    # nearby earlier nodes on average, modelling incidental references.
    # The fractional part is spent as a probability so a density of 2.5
    # means "two edges, and a third half the time".
    whole, fraction = int(edges_per_node), edges_per_node % 1
    for i in range(2, n_nodes + 1):
        for _ in range(whole + (1 if random.random() < fraction else 0)):
            j = max(1, i - random.randint(1, 40))
            if j != i:
                emit(i, j, f"p{random.randint(1, 5)}", remember=True)

    # deliberately structured hub: fans out to `fanin_per_level[d]` new
    # nodes at each depth level, all tagged 'p1' so they're identifiable
    # against the background noise
    next_id = n_nodes + 1
    current_level = [hub_id]
    depth_counts = {0: 1}
    for depth, fanin in enumerate(fanin_per_level, start=1):
        new_level = []
        for target in current_level:
            for _ in range(fanin):
                emit(next_id, target, "p1", remember=True)
                new_level.append(next_id)
                next_id += 1
        depth_counts[depth] = len(new_level)
        current_level = new_level

    total_nodes = next_id - 1

    edge_file.close()

    # tag hub-subgraph nodes by depth (leaf/mid labels the benchmark
    # filters on), walking backwards from the hub
    seen_edges_by_target = parents_of

    hub_nodes_by_depth: dict[int, list[int]] = {0: [hub_id]}
    frontier = [hub_id]
    visited = {hub_id}
    depth = 0
    while frontier and depth < len(fanin_per_level):
        depth += 1
        next_frontier = []
        for node in frontier:
            for parent in seen_edges_by_target.get(node, []):
                if parent not in visited:
                    visited.add(parent)
                    next_frontier.append(parent)
        hub_nodes_by_depth[depth] = next_frontier
        frontier = next_frontier

    tags: dict[int, dict] = {hub_id: {"type": "hub"}}
    for depth, nodes in hub_nodes_by_depth.items():
        if depth == max(hub_nodes_by_depth):
            for nd in nodes:
                tags[nd] = {"type": "leaf", "priority": nd % 20}
        elif depth == len(fanin_per_level) // 2:
            for nd in nodes:
                tags[nd] = {"flag": 1}

    with open(out_dir / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        for i in range(1, total_nodes + 1):
            props = tags.get(i, {})
            w.writerow([i, str(props).replace("'", '"') if props else "{}"])

    summary = {
        "total_nodes": total_nodes,
        "total_edges": edge_count,
        "edges_per_node": round(edge_count / total_nodes, 2),
        "hub_id": hub_id,
        "depth_counts": depth_counts,
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edges-per-node", type=float, default=3.0,
                    help="average out-degree of the background DAG (default 3.0). "
                         "Below 1 the result is a forest of short chains, not a graph")
    ap.add_argument("--out-dir", type=str, default="./data")
    args = ap.parse_args()

    summary = generate(args.nodes, args.seed, Path(args.out_dir),
                       edges_per_node=args.edges_per_node)
    print("Generated graph:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
