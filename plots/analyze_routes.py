"""
Structural analysis of transit route designs.

Computes coverage, distance, edge structure, interchange, and comparative
overlap metrics for all methods defined in figure_routes.py. Output feeds
directly into figure captions.

Usage:
    cd AlphaTransit && python plots/analyze_routes.py
    cd AlphaTransit && python plots/analyze_routes.py --alpha 1_0
    cd AlphaTransit && python plots/analyze_routes.py --csv /tmp/routes.csv
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import pandas as pd

from figure_routes import (
    METHODS,
    NETWORKS_DIR,
    load_links,
    load_nodes,
    load_routes,
)

TRANSIT_CENTER = "96"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_demand(network: str = "bloomington") -> pd.DataFrame:
    path = NETWORKS_DIR / network / f"{network}_demand_standard.csv"
    return pd.read_csv(path, dtype={"orig": str, "dest": str})


def build_edge_lengths(links: pd.DataFrame) -> Dict[FrozenSet, float]:
    """Map undirected edge (frozenset) -> length in metres."""
    lengths: Dict[FrozenSet, float] = {}
    for _, row in links.iterrows():
        key = frozenset([str(row["start"]), str(row["end"])])
        lengths[key] = row["length"]
    return lengths


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------
def route_edges(route: List[str]) -> List[FrozenSet]:
    """Return ordered list of undirected edges for one route."""
    return [frozenset([route[i], route[i + 1]]) for i in range(len(route) - 1)]


def all_route_edges(routes: List[List[str]]) -> Tuple[List[Set[FrozenSet]], Counter]:
    """Return per-route edge sets and a global edge counter."""
    edge_sets: List[Set[FrozenSet]] = []
    counter: Counter = Counter()
    for r in routes:
        es = set(route_edges(r))
        edge_sets.append(es)
        counter.update(es)
    return edge_sets, counter


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def coverage_metrics(
    routes: List[List[str]], total_nodes: int
) -> Dict[str, object]:
    served = set()
    node_route_count: Counter = Counter()
    for r in routes:
        served.update(r)
        node_route_count.update(set(r))  # count each node once per route

    nodes_3plus = sum(1 for n, c in node_route_count.items() if c >= 3)
    return {
        "unique_nodes": len(served),
        "node_cov_pct": 100.0 * len(served) / total_nodes,
        "nodes_3plus": nodes_3plus,
        "nodes_3plus_pct": 100.0 * nodes_3plus / len(served) if served else 0.0,
    }


def distance_metrics(
    routes: List[List[str]], edge_lengths: Dict[FrozenSet, float]
) -> Dict[str, float]:
    total = 0.0
    for r in routes:
        for e in route_edges(r):
            total += edge_lengths.get(e, 0.0)
    return {"total_dist_km": total / 1000.0}


def edge_metrics(counter: Counter) -> Dict[str, object]:
    unique = len(counter)
    total_traversals = sum(counter.values())
    shared = {e for e, c in counter.items() if c > 1}
    shared_traversals = sum(counter[e] for e in shared)
    max_mult = max(counter.values()) if counter else 0
    return {
        "unique_edges": unique,
        "shared_edges": len(shared),
        "shared_edge_pct": 100.0 * len(shared) / unique if unique else 0.0,
        "max_multiplicity": max_mult,
        "overlap_conc_pct": 100.0 * shared_traversals / total_traversals
        if total_traversals
        else 0.0,
    }


def interchange_metrics(routes: List[List[str]]) -> Dict[str, int]:
    node_route_count: Counter = Counter()
    for r in routes:
        node_route_count.update(set(r))
    interchange = sum(1 for c in node_route_count.values() if c > 1)
    return {"interchange_nodes": interchange}


def hub_routes(routes: List[List[str]], hub: str = TRANSIT_CENTER) -> int:
    return sum(1 for r in routes if hub in r)


# ---------------------------------------------------------------------------
# Comparative overlap
# ---------------------------------------------------------------------------
def jaccard(a: Set[FrozenSet], b: Set[FrozenSet]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def combined_edge_set(edge_sets: List[Set[FrozenSet]]) -> Set[FrozenSet]:
    out: Set[FrozenSet] = set()
    for s in edge_sets:
        out |= s
    return out


def exclusive_nodes(
    routes_a: List[List[str]], routes_b: List[List[str]]
) -> Tuple[int, int]:
    """Return (nodes only in A, nodes only in B)."""
    a = set()
    b = set()
    for r in routes_a:
        a.update(r)
    for r in routes_b:
        b.update(r)
    return len(a - b), len(b - a)


# ---------------------------------------------------------------------------
# Analyze one method
# ---------------------------------------------------------------------------
def analyze(
    name: str,
    routes: List[List[str]],
    total_nodes: int,
    edge_lengths: Dict[FrozenSet, float],
) -> Dict[str, object]:
    edge_sets, counter = all_route_edges(routes)
    row: Dict[str, object] = {"method": name}
    row.update(coverage_metrics(routes, total_nodes))
    row.update(distance_metrics(routes, edge_lengths))
    row.update(edge_metrics(counter))
    row.update(interchange_metrics(routes))
    row["hub_routes"] = hub_routes(routes)
    row["num_routes"] = len(routes)
    row["_edge_sets"] = edge_sets  # kept for comparative phase, dropped before output
    row["_routes"] = routes
    return row


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_table(rows: List[Dict[str, object]]) -> None:
    hdr = (
        f"{'Method':<22s} {'Nodes':>5s} {'Cov%':>6s} {'Dist':>7s} "
        f"{'UEdge':>5s} {'Shrd%':>6s} {'MaxM':>4s} {'OvlC%':>6s} "
        f"{'Xfer':>4s} {'3+Rt%':>6s} {'Hub':>6s}"
    )
    sep = "-" * len(hdr)
    print(f"\n{'STRUCTURAL ROUTE ANALYSIS':=^{len(hdr)}}")
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['method']:<22s} "
            f"{r['unique_nodes']:>5d} "
            f"{r['node_cov_pct']:>5.1f}% "
            f"{r['total_dist_km']:>6.1f}k "
            f"{r['unique_edges']:>5d} "
            f"{r['shared_edge_pct']:>5.1f}% "
            f"{r['max_multiplicity']:>4d} "
            f"{r['overlap_conc_pct']:>5.1f}% "
            f"{r['interchange_nodes']:>4d} "
            f"{r['nodes_3plus_pct']:>5.1f}% "
            f"{r['hub_routes']:>2d}/{r['num_routes']:<2d}"
        )
    print(sep)


def print_comparative(
    rows: List[Dict[str, object]],
    rw_edges: Set[FrozenSet],
    at_edges: Set[FrozenSet],
    rw_routes: List[List[str]],
    at_routes: List[List[str]],
) -> None:
    print(f"\n{'COMPARATIVE OVERLAP':=^70}")
    hdr = f"{'Method':<22s} {'J(RealWorld)':>12s} {'J(AlphaTr.)':>12s} {'ExclVsRW':>10s} {'ExclVsAT':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        edges = combined_edge_set(r["_edge_sets"])
        j_rw = jaccard(edges, rw_edges)
        j_at = jaccard(edges, at_edges)
        exc_rw_a, exc_rw_b = exclusive_nodes(r["_routes"], rw_routes)
        exc_at_a, exc_at_b = exclusive_nodes(r["_routes"], at_routes)
        print(
            f"{r['method']:<22s} "
            f"{j_rw:>11.3f}  "
            f"{j_at:>11.3f}  "
            f"{exc_rw_a:>4d}/{exc_rw_b:<4d} "
            f"{exc_at_a:>4d}/{exc_at_b:<4d}"
        )


def print_findings(rows: List[Dict[str, object]]) -> None:
    print(f"\n{'KEY FINDINGS':=^70}")
    best_cov = max(rows, key=lambda r: r["node_cov_pct"])
    least_dist = min(rows, key=lambda r: r["total_dist_km"])
    most_dist = max(rows, key=lambda r: r["total_dist_km"])
    least_overlap = min(rows, key=lambda r: r["shared_edge_pct"])
    most_overlap = max(rows, key=lambda r: r["shared_edge_pct"])
    most_xfer = max(rows, key=lambda r: r["interchange_nodes"])
    max_mult = max(rows, key=lambda r: r["max_multiplicity"])

    findings = [
        f"Highest node coverage: {best_cov['method']} ({best_cov['unique_nodes']} nodes, {best_cov['node_cov_pct']:.1f}%)",
        f"Shortest total distance: {least_dist['method']} ({least_dist['total_dist_km']:.1f} km)",
        f"Longest total distance: {most_dist['method']} ({most_dist['total_dist_km']:.1f} km)",
        f"Lowest shared-edge %: {least_overlap['method']} ({least_overlap['shared_edge_pct']:.1f}%)",
        f"Highest shared-edge %: {most_overlap['method']} ({most_overlap['shared_edge_pct']:.1f}%)",
        f"Most interchange nodes: {most_xfer['method']} ({most_xfer['interchange_nodes']})",
        f"Highest max edge multiplicity: {max_mult['method']} ({max_mult['max_multiplicity']}x)",
    ]
    for f in findings:
        print(f"  * {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Structural route analysis")
    parser.add_argument("--csv", type=str, default=None, help="Save results CSV")
    parser.add_argument(
        "--alpha", choices=["0_3", "1_0"], default="0_3", help="Alpha value"
    )
    args = parser.parse_args()

    nodes_df = load_nodes()
    links_df = load_links()
    edge_lengths = build_edge_lengths(links_df)
    total_nodes = len(nodes_df)
    path_key = f"path_{args.alpha}"

    rows: List[Dict[str, object]] = []
    for m in METHODS:
        routes = load_routes(m[path_key])
        row = analyze(m["name"], routes, total_nodes, edge_lengths)
        rows.append(row)

    print_table(rows)

    # Comparative: find Real-World and AlphaTransit rows
    rw_row = next((r for r in rows if r["method"] == "Real-World"), None)
    at_row = next((r for r in rows if r["method"] == "AlphaTransit"), None)
    if rw_row and at_row:
        rw_edges = combined_edge_set(rw_row["_edge_sets"])
        at_edges = combined_edge_set(at_row["_edge_sets"])
        print_comparative(rows, rw_edges, at_edges, rw_row["_routes"], at_row["_routes"])

    print_findings(rows)

    if args.csv:
        out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        pd.DataFrame(out).to_csv(args.csv, index=False)
        print(f"\nSaved: {args.csv}")


if __name__ == "__main__":
    main()
