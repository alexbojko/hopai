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
             fanin_per_level=(15, 5, 3, 3, 2, 2, 2)):
    random.seed(seed)
    hub_id = hub_id or (n_nodes // 2)

    edges: list[tuple[int, int, str]] = []

    # sparse background DAG: each node i (i > 1) gets an edge to a
    # randomly-nearby earlier node with 80% probability, modeling
    # incidental references
    for i in range(2, n_nodes + 1):
        if random.random() < 0.8:
            offset = random.randint(1, 40)
            j = max(1, i - offset)
            if j != i:
                tag = f"p{random.randint(1, 5)}"
                edges.append((i, j, tag))

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
                edges.append((next_id, target, "p1"))
                new_level.append(next_id)
                next_id += 1
        depth_counts[depth] = len(new_level)
        current_level = new_level

    total_nodes = next_id - 1

    # tag hub-subgraph nodes by depth (leaf/mid labels used by the
    # benchmark scripts), via a reverse-edge lookup from the hub
    seen_edges_by_target: dict[int, list[int]] = {}
    for a, b, _ in edges:
        seen_edges_by_target.setdefault(b, []).append(a)

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

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        for i in range(1, total_nodes + 1):
            props = tags.get(i, {})
            w.writerow([i, str(props).replace("'", '"') if props else "{}"])

    with open(out_dir / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        for a, b, tag in edges:
            w.writerow([a, b, tag])

    summary = {
        "total_nodes": total_nodes,
        "total_edges": len(edges),
        "hub_id": hub_id,
        "depth_counts": depth_counts,
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="./data")
    args = ap.parse_args()

    summary = generate(args.nodes, args.seed, Path(args.out_dir))
    print("Generated graph:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
