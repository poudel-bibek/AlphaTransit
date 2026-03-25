"""
Convert AlphaTransit's Bloomington data to Holliday et al.'s Mumford format.

Produces three files in datasets/bloomington/:
  - BloomingtonCoords.txt    : first line = n_nodes, then x y per node (0-indexed)
  - BloomingtonTravelTimes.txt : n x n matrix of travel times in MINUTES (0 = no edge)
  - BloomingtonDemand.txt     : n x n symmetric demand matrix

Holliday's code reads these via:
  - np.genfromtxt(CoordsPath, skip_header=1) -> (n, 2) array
  - np.genfromtxt(TravelTimesPath) -> (n, n) array, times in minutes, * 60 -> seconds
  - np.genfromtxt(DemandPath) -> (n, n) array
"""

import csv
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOOMINGTON_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "..", "networks", "bloomington"
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets", "bloomington")


def load_nodes(path):
    """Load nodes CSV. Returns dict: node_name (int) -> (x, y)."""
    nodes = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = int(row["name"])
            x = float(row["x"])
            y = float(row["y"])
            nodes[name] = (x, y)
    return nodes


def load_links(path):
    """Load links CSV. Returns list of (start, end, length, speed) tuples."""
    links = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row["start"])
            end = int(row["end"])
            length = float(row["length"])
            speed = float(row["free_flow_speed"])
            links.append((start, end, length, speed))
    return links


def load_demand(path):
    """Load demand CSV. Returns list of (orig, dest, volume) tuples."""
    demand = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = int(row["orig"])
            dest = int(row["dest"])
            volume = float(row["volume"])
            demand.append((orig, dest, volume))
    return demand


def convert():
    nodes_path = os.path.join(BLOOMINGTON_DIR, "bloomington_nodes_standard.csv")
    links_path = os.path.join(BLOOMINGTON_DIR, "bloomington_links_standard.csv")
    demand_path = os.path.join(
        BLOOMINGTON_DIR, "bloomington_demand_standard.csv"
    )

    # Load data
    nodes = load_nodes(nodes_path)
    links = load_links(links_path)
    demand_list = load_demand(demand_path)

    # Sort node names and create mapping: original_name -> 0-indexed
    sorted_names = sorted(nodes.keys())
    n = len(sorted_names)
    name_to_idx = {name: idx for idx, name in enumerate(sorted_names)}

    print(f"Loaded {n} nodes, {len(links)} links, {len(demand_list)} OD pairs")

    # --- Coords file ---
    # Format: first line = n_nodes, then one line per node with "x y"
    coords_path = os.path.join(OUTPUT_DIR, "BloomingtonCoords.txt")
    with open(coords_path, "w") as f:
        f.write(f"{n}\n")
        for name in sorted_names:
            x, y = nodes[name]
            f.write(f"{x:.1f} {y:.1f}\n")
    print(f"Written {coords_path}")

    # --- Travel times matrix ---
    # n x n matrix, entry (i,j) = travel time in MINUTES from i to j
    # Holliday's floyd_warshall expects 0 on diagonal, inf for no edge.
    # travel_time = length / speed (in seconds), then / 60 -> minutes
    tt_matrix = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(tt_matrix, 0.0)
    for start, end, length, speed in links:
        i = name_to_idx[start]
        j = name_to_idx[end]
        travel_time_min = (length / speed) / 60.0  # seconds -> minutes
        tt_matrix[i, j] = travel_time_min
        tt_matrix[j, i] = travel_time_min  # bidirectional

    tt_path = os.path.join(OUTPUT_DIR, "BloomingtonTravelTimes.txt")
    np.savetxt(tt_path, tt_matrix, fmt="%.6f", delimiter=" ")
    print(f"Written {tt_path}")

    n_edges = np.sum(tt_matrix > 0)
    print(f"  Non-zero entries in travel time matrix: {n_edges} "
          f"({n_edges // 2} bidirectional edges)")

    # --- Demand matrix ---
    # n x n symmetric matrix. Symmetrize: D'[i,j] = (D[i,j] + D[j,i]) / 2
    demand_matrix = np.zeros((n, n), dtype=np.float64)
    for orig, dest, volume in demand_list:
        i = name_to_idx[orig]
        j = name_to_idx[dest]
        demand_matrix[i, j] = volume

    # Symmetrize
    demand_sym = (demand_matrix + demand_matrix.T) / 2.0

    demand_path_out = os.path.join(OUTPUT_DIR, "BloomingtonDemand.txt")
    np.savetxt(demand_path_out, demand_sym, fmt="%.4f", delimiter=" ")
    print(f"Written {demand_path_out}")

    total_demand_orig = np.sum(demand_matrix)
    total_demand_sym = np.sum(demand_sym)
    n_nonzero = np.sum(demand_sym > 0)
    print(f"  Original total demand: {total_demand_orig:.0f}")
    print(f"  Symmetrized total demand: {total_demand_sym:.0f}")
    print(f"  Non-zero OD pairs (symmetric): {n_nonzero}")

    # --- Node mapping file (for later route conversion) ---
    mapping_path = os.path.join(OUTPUT_DIR, "node_mapping.csv")
    with open(mapping_path, "w") as f:
        f.write("original_name,mumford_idx\n")
        for name in sorted_names:
            f.write(f"{name},{name_to_idx[name]}\n")
    print(f"Written {mapping_path}")

    # --- Summary ---
    print(f"\nConversion complete. Files in {OUTPUT_DIR}/")
    print(f"  Nodes: {n}")
    print(f"  Edges: {n_edges // 2}")
    print(f"  Transit center (node 96) -> Mumford index: "
          f"{name_to_idx.get(96, 'NOT FOUND')}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    convert()
