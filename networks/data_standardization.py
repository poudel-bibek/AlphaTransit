"""
Helper functions to standardize data 

1. Sioux Falls Network:
   - Original source: LeBlanc, L.J., Morlok, E.K., Pierskalla, W.P. (1975). 
     "An efficient approach to solving the road network equilibrium traffic assignment problem." 
     Transportation Research
   - Repository: https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
   - Network size: 24 nodes, 76 links
   - Notes: 
      - Classic traffic assignment benchmark. 
      - Artificial coordinates.
      - Nodes: 
         - Download the lat, long version of nodes data from the repo (this file may be present as tntp format in the repo). 
         - Researchers (e.g., Chakirov and Fourie, 2014) matched each abstract node to a major real-world intersection in Sioux Falls using city maps, GIS data.
         - i.e. Node coordinates were generated artificially to reproduce the diagram shown in the paper.
         - The coordinates are in longitude, latitude format.
         - Convert to (x, y) in meters.
      - Links:
         - From the repo, download the _net.tntp file links data. 
         - Contains init_node, term_node, capacity, length, free_flow_time (in minutes), b, power, speed, toll, link_type
         - free_flow_speed is calculated using free_flow_time and length .
         - Convert to (start, end, length, free_flow_speed)

      - Demand:
         - From the repo, the header "ton" lists OD flows as daily vehicle trips, with a total OD flow of 360,600 vehicles per day across all pairs. 
         - Convert to vehicles per hour using a peak hour factor.

2. Bloomington Network:
   - Original source: This network is being released by us. 
   - Network size: 143 nodes, 240 links
   - Notes: 
      - The coordinates, links, and demand are from the real-world.
      - Nodes: 
         - Coordinates are in longitude, latitude format.
         - Convert to (x, y) in meters.
      - Links:
         - The length of the links are obtained from Google Maps. 
         - There are a total of i.e., missing link names: Link names 
      - Existing real-world routes: 
         - 16 total existing routes. 
         - Source 1: https://www.google.com/maps/d/u/0/viewer?mid=1hABSC6s2MoTnnfVmxy1JPLFAvcfkPZs&ll=39.174579022786%2C-86.5382328329941&z=15
         - Source 2: https://www.transit.land/operators/o-dnfq-bloomingtontransit
      - Demand:
         - F
         
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
   - free_flow_speed: float (m/s)

3. {network_name}_demand.csv
   - orig: str (origin node name)
   - dest: str (destination node name) 
   - volume: float (vehicles/hour)
"""

import numpy as np
import pandas as pd

class Helpers: 
   def __init__(self):
      pass

   def convert_lat_long_to_meters(self, nodes_csv: str, nodes_map: dict) -> None:
      """
      For Nodes: Convert coordinates (longitude, latitude) to (x, y) in meters.
      
      Every spherical -> rectangular reprojection is going to have some error. 
      The code implements a local tangent plane approximation for conversion. 
      This method treats the Earth's surface as approximately flat. It should be fine over small areas (~100km).
      """
      
      df = pd.read_csv(nodes_csv)
      
      # Find which columns contain longitude and latitude using nodes_map
      lon_col = None
      lat_col = None
      name_col = None

      for col, coord_type in nodes_map.items():
          if coord_type == 'lon':
              lon_col = col
          elif coord_type == 'lat':
              lat_col = col
          elif coord_type == 'name':
              name_col = col
      
      if lon_col is None or lat_col is None:
          raise ValueError(f"nodes_map must contain both 'lon' and 'lat' mappings. Got: {nodes_map}")
      
      lon = df[lon_col].values
      lat = df[lat_col].values
      
      ref_lat = np.mean(lat)
      ref_lon = np.mean(lon)
      
      # Meters per degree at reference latitude
      m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(ref_lat)) + 1.175 * np.cos(4 * np.radians(ref_lat)) - 0.0023 * np.cos(6 * np.radians(ref_lat))
      m_per_deg_lon = 111412.84 * np.cos(np.radians(ref_lat)) - 93.5 * np.cos(3 * np.radians(ref_lat)) + 0.118 * np.cos(5 * np.radians(ref_lat))
      
      # Convert deltas to meters
      df["x"] = (lon - ref_lon) * m_per_deg_lon
      df["y"] = (lat - ref_lat) * m_per_deg_lat
      
      if name_col:
         df[name_col] = df[name_col].astype(str)  # Convert before renaming
      
      df = df.rename(columns={name_col: 'name'})
      df = df[['name', 'x', 'y']]

      output_file = nodes_csv.replace('.csv', '_standard.csv')
      df.to_csv(output_file, index=False)
      print(f"Converted coordinates saved to: {output_file}")

   def convert_ton_to_hourly(self, demand_csv: str, demand_map: dict, factor: float = 0.06) -> None:
      """
      For Demand: Convert daily vehicle trips to vehicles per hour using a peak hour factor.
      In urban planning, the peak hour often carries about 6-12% of the day's total traffic due to rush hours.
      Using Default factor=0.06, it assumes it's 6% of daily demand, common for urban networks.
      """
      df = pd.read_csv(demand_csv)
      
      rename_dict = {}
      for orig_col, std_col in demand_map.items():
          if orig_col in df.columns:
              rename_dict[orig_col] = std_col
          else:
              raise ValueError(f"Column '{orig_col}' from demand_map not found in CSV.")
      
      df = df.rename(columns=rename_dict)
      
      required_cols = ['orig', 'dest', 'volume']
      missing = [col for col in required_cols if col not in df.columns]
      if missing:
          raise ValueError(f"Missing required columns after rename: {missing}")
      
      df['volume'] *= factor
      
      output_file = demand_csv.replace('.csv', '_standard.csv')
      df.to_csv(output_file, index=False)
      print(f"Converted demand saved to: {output_file}")

   def convert_links_csv_tntp(self, links_tntp: str, nodes_standard_csv: str) -> None:
      """
      For Links: Parse the TNTP file for topology and compute Euclidean lengths, and derive free_flow_speed in m/s.
      - Recalculates length from coordinates rather than using the TNTP length field (4th column)
         - The original TNTP lengths in SiouxFalls_net.tntp are not meant to represent real-world distances
      - Assumes nodes have already been standardized to (x, y) in meters.
      - Calculates free_flow_speed from length and free_flow_time.
      """
      # Parse TNTP file
      links = []
      with open(links_tntp, 'r') as f:
          started = False
          for line in f:
              line = line.strip()
              if not line:
                  continue
              if line.startswith('<'):
                  continue  # metadata
              if line.startswith('~'):
                  started = True
                  continue  # header
              if started:
                  fields = line.rstrip(';').split()
                  if len(fields) == 10:
                      start, end, cap, len_, fft, b, power, speed, toll, type_ = fields
                      links.append({'start': str(start), 'end': str(end), 'free_flow_time': float(fft)})

      print(f"Found {len(links)} links") 

      # Load georeferenced nodes in meters
      df_nodes = pd.read_csv(nodes_standard_csv, dtype={'name': str})
      node_pos = {str(row['name']): (row['x'], row['y']) for _, row in df_nodes.iterrows()}

      # print(f"Loaded {len(node_pos)} nodes: {sorted(node_pos.keys())}")

      # Compute for each link
      data = []
      for link in links:
          start = link['start']
          end = link['end']
          if start not in node_pos or end not in node_pos:
              raise ValueError(f"Missing node positions for link {start}-{end}")
          x1, y1 = node_pos[start]
          x2, y2 = node_pos[end]
          length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
          fft_seconds = link['free_flow_time'] * 60  # minutes to seconds
          free_flow_speed = length / fft_seconds if fft_seconds > 0 else np.nan
          name = f"{start}-{end}"
          data.append({'name': name, 'start': start, 'end': end, 'length': length, 'free_flow_speed': free_flow_speed})

      df_links = pd.DataFrame(data)
      output_file = links_tntp.replace('_net.tntp', '_links_standard.csv')
      df_links.to_csv(output_file, index=False)
      print(f"Converted links saved to: {output_file}")

   def convert_sioux_falls(self, nodes_csv: str, demand_csv: str, links_tntp: str):
      """
      For Nodes: Assuming columns: node_name, longitude, latitude
      """
      nodes_map = {'Node': 'name', 'X': 'lon', 'Y': 'lat'}
      self.convert_lat_long_to_meters(nodes_csv, nodes_map)

      demand_map = {'O': 'orig', 'D': 'dest', 'Ton': 'volume'}
      self.convert_ton_to_hourly(demand_csv, demand_map, factor=0.06)

      nodes_standard_csv = nodes_csv.replace('.csv', '_standard.csv')
      self.convert_links_csv_tntp(links_tntp, nodes_standard_csv)

### Sioux Falls Network ###
# sioux_falls = Helpers()
# sioux_falls_nodes_csv = "./sioux_falls/sioux_falls_nodes.csv"
# sioux_falls_links_tntp = "./sioux_falls/sioux_falls_net.tntp"
# sioux_falls_demand_csv = "./sioux_falls/sioux_falls_demand.csv"
# sioux_falls.convert_sioux_falls(nodes_csv=sioux_falls_nodes_csv, demand_csv=sioux_falls_demand_csv, links_tntp=sioux_falls_links_tntp)

### Bloomington Network ###
bloomington = Helpers()
bloomington_nodes_csv = "./bloomington/bloomington_nodes.csv"
bloomington_links_tntp = "./bloomington/bloomington_links.csv"
bloomington_demand_csv = "./bloomington/bloomington_demand.csv"
bloomington.convert_bloomington(nodes_csv=bloomington_nodes_csv, demand_csv=bloomington_demand_csv, links_tntp=bloomington_links_tntp)


