"""
Title: 
Author: Anonynomus for submission.

Standardized versions of commonly used transit network design benchmark datasets.
All datasets have been converted to a consistent format suitable for reinforcement learning applications using UXSim.

================================================================================
DATA SOURCES AND CITATIONS:
================================================================================

1. Sioux Falls Network:
   - Original source: LeBlanc, L.J., Morlok, E.K., Pierskalla, W.P. (1975). 
     "An efficient approach to solving the road network equilibrium traffic assignment problem." 
     Transportation Research
   - Data repository: https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
   - Network size: 24 nodes, 76 links
   - Notes: Classic traffic assignment benchmark. Coordinates are artificial.

2. Rivera City Network:
   - Original source: Mauttone, Antonio (2005). 
     "Optimización de recorridos y frecuencias en sistemas de transporte público urbano colectivo." 
     Master's Thesis, Universidad de la República, Uruguay.
   - Data repository: https://github.com/RenatoArbex/TransitNetworkDesign
   - Network size: 84 nodes, 143 links  
   - Notes: Real-world network from Rivera, Uruguay (border city). Geographic coordinates.

3. Mumford 3 Network:
   - Original source: Mumford, Christine L. (2013). 
     "New heuristic and evolutionary operators for the multi-objective urban transit routing problem." 
     IEEE Congress on Evolutionary Computation (CEC)
   - Data repository: https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/Outline.html
   - Network size: 127 nodes, 425 links
   - Notes: 
       - Synthetic network based on real-world characteristics.
       - Transit time is recorded in minutes (with "Inf" between nodes that are not connected).

4. Laval Network:
   - Original source: Holliday, A., El-Geneidy, A., Dudek, G. (2024).
     "Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning."
     arXiv preprint https://arxiv.org/html/2404.05894v2
   - Data repository: Obtained from email correspondence with the authors.
   - Network size: 632 nodes, 1971 links
   - Notes: Real-world network from Laval, Canada

================================================================================
STANDARDIZED FORMAT:
================================================================================

All datasets are provided in three CSV files per network:

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
   
   Optional parameters (UXSim will use defaults if not provided):
   - kappa: float (jam density in vehicles/m) [default: 0.2]
   - merge_priority: float (merging priority at intersections) [default: 1.0]

3. {network_name}_demand.csv
   - orig: str (origin node name)
   - dest: str (destination node name) 
   - start_t: float (demand start time in seconds)
   - end_t: float (demand end time in seconds)
   - q: float (flow rate in vehicles/s)

================================================================================
NOTES:
================================================================================

**Coordinate Systems:**
- Sioux Falls: Artificial coordinate system (units preserved from original)
- Rivera City: Converted from lat/lon to UTM-like projection in meters
- Mumford 3: Euclidean coordinate system (units preserved from original)
- Laval: Geographic coordinates converted to local projection

**Missing Data Handling:**
- Link lengths: Calculated from Euclidean distance when not provided
- Free flow speeds: Estimated from length/travel_time when available, otherwise set to 13.89 m/s (50 km/h)

**Demand Scaling:**
- All demands converted to vehicles/second 

**Time Horizon Flexibility:**
Time horizons can be adjusted during simulation setup.
The demand flow rates (q) are in vehicles/second, so you can adjust the simulation horizon by changing start_t and end_t:

================================================================================

The timing (start and end time for a given flow) can vary based on horizon set
1 vehicle per secon

"""