"""
Title: Helper functions to standardize data.
Author: Anonynomus for submission.

DATA SOURCES:
================================================================================

1. Sioux Falls Network:
   - Original source: LeBlanc, L.J., Morlok, E.K., Pierskalla, W.P. (1975). 
     "An efficient approach to solving the road network equilibrium traffic assignment problem." 
     Transportation Research
   - Repository: https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
   - Network size: 24 nodes, 76 links
   - Notes: 
      - Classic traffic assignment benchmark. 
      - Artificial coordinates.

2. Mumford 3 Network:
   - Original source: Mumford, Christine L. (2013). 
     "New heuristic and evolutionary operators for the multi-objective urban transit routing problem." 
     IEEE Congress on Evolutionary Computation (CEC)
   - Repository: https://github.com/RenatoArbex/TransitNetworkDesign
   - Network size: 127 nodes, 425 links
   - Notes: 
       - Synthetic network based on real-world characteristics.
       - Transit time is recorded in minutes (with "Inf" between nodes that are not connected).

2. Rivera City Network:
   - Original source: Mauttone, Antonio (2005). 
     "Optimización de recorridos y frecuencias en sistemas de transporte público urbano colectivo." 
     Master's Thesis, Universidad de la República, Uruguay.
   - Repository: https://github.com/RenatoArbex/TransitNetworkDesign
   - Network size: 84 nodes, 143 links  
   - Notes: 
       - Real-world network from Rivera, Uruguay (border city). Geographic coordinates.

4. Laval Network:
   - Original source: Holliday, A., El-Geneidy, A., Dudek, G. (2024).
     "Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning."
     arXiv preprint https://arxiv.org/html/2404.05894v2
   - Repository: Obtained from email correspondence with the authors.
   - Network size: 632 nodes, 1971 links
   - Notes: 
       - Real-world network from Laval, Canada

STANDARDIZED FORMAT:
================================================================================

1. {network_name}_nodes.csv
   - name: str (node identifier, e.g., "1", "2", "3")
   - x: float (x-coordinate in meters)
   - y: float (y-coordinate in meters)

2. {network_name}_links.csv  
   - name: str (link identifier, e.g., "link_1_2")
   - start: str (start node name)
   - end: str (end node name)
   - length: float (link length in meters)
   - u: float (free flow speed in m/s)

3. {network_name}_demand.csv
   - orig: str (origin node name)
   - dest: str (destination node name) 
   - volume: float (vehicles/hour)
"""

import numpy as np
import pandas as pd

def convert_lat_long_to_meters(nodes_csv: str) -> None:
    """
    Convert node coordinates (latitude, longitude) to meters.
    TODO: Complete this.
    """
    df = pd.read_csv(nodes_csv)
    df["x"] = df["x"].apply(lambda x: x * 1000)
    df["y"] = df["y"].apply(lambda x: x * 1000)
    df.to_csv(nodes_csv, index=False)


