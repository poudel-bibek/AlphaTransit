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
   - Network size: 143 nodes, 243 links
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
         - Source 3: https://bloomingtontransit.com/gtfs/
      - Demand:
         - First download the raw data from the source.
         - Source: https://lehd.ces.census.gov/data/lodes/LODES8/in/od/in_od_main_JT00_2022.csv.gz
         - JT00 means all jobs.
         - h_geocode and w_geocode are are home (origin) and work (destination) geocodes.
         - S000 i.e., total number of jobs/ commuters (all ages, earnings, industries, etc.). This is the primary demand metric for O-D flow.
         - We ignore all other columns such as (SA01, SA02..), (SE01, SE02..), etc. and the createdate column
         - Area of interest: Monroe County, Indiana which has a FIPS code of 105 (full GEOID prefix is 18105)
         - Source: https://www.geocod.io/geoids/indiana/monroe-county-18105/
         - We keep only the rows where rows both the home and work blocks are within Monroe County.
         - Mapping census block to nodes:
            - Source for 2022 data: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
            - We use census blocks to map to nodes (instead of tracts). Blocks are smaller units (building blocks) are are designed to be relatively permanent.
            - 

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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon, Circle
from pyproj import Transformer
from matplotlib.collections import LineCollection
import contextily as ctx
from matplotlib.ticker import FuncFormatter
from scipy.spatial import ConvexHull

# Import plotting functions
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
   sys.path.insert(0, str(ROOT))

from plots.network_diagnostics import (
   plot_demand_viz,
   plot_unified,
   plot_unified_2,
   plot_bloomington_base,
   plot_validation_mapping,
)

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
      print(f"\nNodes DF:\n{df.head()}")

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
      
      transformer = Transformer.from_crs('epsg:4326', 'epsg:32616', always_xy=True)
      df["x"], df["y"] = transformer.transform(lon, lat)
      
      df['x'] = df['x'].round(1)
      df['y'] = df['y'].round(1)
      
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
      PEAK_HOUR_FACTOR = 0.06

      nodes_map = {'Node': 'name', 'X': 'lon', 'Y': 'lat'}
      self.convert_lat_long_to_meters(nodes_csv, nodes_map)

      demand_map = {'O': 'orig', 'D': 'dest', 'Ton': 'volume'}
      self.convert_ton_to_hourly(demand_csv, demand_map, factor=PEAK_HOUR_FACTOR)

      nodes_standard_csv = nodes_csv.replace('.csv', '_standard.csv')
      self.convert_links_csv_tntp(links_tntp, nodes_standard_csv)


   def convert_bloomington(self, nodes_csv: str, links_csv: str, demand_lodes_csv: str):
      """
      For Nodes: Assuming columns: node_name, longitude, latitude
        - Convert to (x, y) in meters.
      For edges:
        - Convert length from miles to meters.

      For demand:
        - Convert
      """

      PEAK_HOUR_FACTOR = 0.11

      # Non-Commuter Trip Expansion Factor
      OTHER_TRIPS_FACTOR = 1.50 # Since the data only has commuter (work trips), include 100% of other trips

      # The original nodes file contains name, lon, lat
      nodes_map = {'name': 'name', 'lon': 'lon', 'lat': 'lat'}
      self.convert_lat_long_to_meters(nodes_csv, nodes_map)
      
      # Links conversion (assume columns: name, start, end, length)
      df_links = pd.read_csv(links_csv)
      df_links['length'] *= 1609.34  # Miles to meters

      df_links['length'] = df_links['length'].round(1)

      # If adding a free_flow_speed column
      if 'free_flow_speed' not in df_links.columns:
          df_links['free_flow_speed'] = 16.67  # Default m/s (60 km/h); adjust as needed

      output_links = links_csv.replace('.csv', '_standard.csv')
      df_links.to_csv(output_links, index=False)
      print(f"Converted links saved to: {output_links}")

      #########
      # Demand data filtering
      # Step 1: Remove irrelevant columns
      columns_to_keep = ['h_geocode', 'w_geocode', 'S000'] # S000 means all types of jobs (all ages, earnings, industries, etc.).
      df_demand = pd.read_csv(demand_lodes_csv, usecols=columns_to_keep)
      
      # Step 2: The data contains geocodes for entire Indiana. 
      # Filter and keep only the rows in our area of interest (Monroe County, Indiana)
      # i.e., both home and work blocks are within Monroe County.
      df_demand = df_demand[df_demand['h_geocode'].astype(str).str.startswith('18105') & df_demand['w_geocode'].astype(str).str.startswith('18105')]
      print(f"\nDemand data after filtering:\n{df_demand.head()}")

      # Step 3: Define the area of interest (shrinking, processing centroids)
      # a. Download the census block shapefile for 2022 and load it 
      gdf_blocks = gpd.read_file('./bloomington/tl_2022_18_tabblock20/tl_2022_18_tabblock20.shp')
      print("\nColumns in gdf_blocks:", gdf_blocks.columns.tolist())
      print("Unique COUNTYFP20 values:", gdf_blocks['COUNTYFP20'].unique())
      print("COUNTYFP20 value counts:\n", gdf_blocks['COUNTYFP20'].value_counts())

      # b. Extract shapes of blocks in Monroe County
      gdf_monroe = gdf_blocks[gdf_blocks['COUNTYFP20'] == '105']
      print(f"Extracted {len(gdf_monroe)} blocks in Monroe County\n")
      print("Sample of gdf_monroe (first 5 rows):\n", gdf_monroe.head())

      # c. Upon several iterations, we found that the entire county is still a bit large. 
      # So first project to UTM 16N (EPSG:32616) to avoid geographic CRS warning and ensure metric accuracy
      # Then, we shrink the area by 40% from all directions 
      gdf_monroe = gdf_monroe.to_crs(epsg=32616)
      print(f"\nProjected CRS: {gdf_monroe.crs}\n")  # Log the new CRS for confirmation

      bounds = gdf_monroe.total_bounds  # [minx, miny, maxx, maxy]
      width = bounds[2] - bounds[0]
      height = bounds[3] - bounds[1]
      print(f"Bounds:{bounds}, \nWidth: {width}, Height: {height}")

      # The inset sizes are chosen empirically/ iteratively.
      inset_left = 0.22 * width
      inset_right = 0.4 * width
      inset_top = 0.37 * height
      inset_bottom = 0.32 * height
      shrunk_bounds = [bounds[0] + inset_left, bounds[1] + inset_bottom, bounds[2] - inset_right, bounds[3] - inset_top]

      # Calculate and print area in square kilometers
      shrunk_width = shrunk_bounds[2] - shrunk_bounds[0]
      shrunk_height = shrunk_bounds[3] - shrunk_bounds[1]
      area_sqm = shrunk_width * shrunk_height
      area_sqkm = area_sqm / 1_000_000
      print(f"\nShrunk area: {area_sqkm:.2f} square kilometers\n")
      print(f"\nShrunk bounds: {shrunk_bounds}\n Width: {shrunk_width}, Height: {shrunk_height}")

      # d. Within the  shrunk bounds, identify the centroid of each block (x, y) and not (long, lat)
      gdf_monroe['centroid'] = gdf_monroe.geometry.centroid  # Temp for filtering
      gdf_monroe = gdf_monroe[(gdf_monroe['centroid'].x >= shrunk_bounds[0]) & 
                              (gdf_monroe['centroid'].x <= shrunk_bounds[2]) &
                              (gdf_monroe['centroid'].y >= shrunk_bounds[1]) & 
                              (gdf_monroe['centroid'].y <= shrunk_bounds[3])]
      print(f"\nAfter shrinking: {len(gdf_monroe)} blocks")
      
      print("\nAfter centroids:, columns:", gdf_monroe.columns.tolist())
      gdf_monroe.head()

      # Plot some sample blocks and their centroids
      fig, axs = plt.subplots(2, 5, figsize=(25, 10)) 
      axs = axs.flat  # Flatten the axes
      for idx, (df_index, row) in enumerate(gdf_monroe.head(10).iterrows()):
         ax = axs[idx]
         
         # Plot polygon (handles full geometry)
         x, y = row['geometry'].exterior.xy
         ax.fill(x, y, alpha=0.3, color='lightblue')  # Shade the shape
         ax.plot(x, y, 'b-', label='Polygon edges')
         
         # Plot centroid
         cx, cy = row['centroid'].x, row['centroid'].y
         ax.plot(cx, cy, 'ro', label='Centroid')
         
         ax.set_title(f"Block {idx +1} (GEOID: {row['GEOID20']})")
         ax.set_xlabel('Easting (m)' if not gdf_monroe.crs.is_geographic else 'Longitude')
         ax.set_ylabel('Northing (m)' if not gdf_monroe.crs.is_geographic else 'Latitude')
         ax.legend()

      plt.tight_layout()
      plt.savefig('./bloomington/blocks_centroids_grid.png')
      # plt.show()

      # Step 4: Map census block to nodes
      # At this point we have: 
      # - gdf_monroe with tons of columns. I only need GEOID20, centroid, and geometry
      # - df_demand with h_geocode, w_geocode, S000
      gdf_monroe = gdf_monroe[['GEOID20', 'centroid', 'geometry']]
      print(f"\nAfter selecting columns:\n{gdf_monroe.head()}")
      
      # For each row in df_demand, for both h_geocode and w_geocode:
      # First, look at the respective row in gdf_monroe and get the centroid. 
      # Then get the node that is geographically closest to the centroid.
      # Then in the df_demand file, add 2 columns: orig_node and dest_node.
      
      # Helper
      # We have nodes originally in long, lat in csv file. Load it. 
      nodes_long_lat_df = pd.read_csv(nodes_csv, usecols=[0,1,2], names=['name', 'lon', 'lat'], header=0, dtype={'name': str})
      print(f"\nNodes long, lat:\n{nodes_long_lat_df.head()}")
      
      # Project the nodes to UTM 16N (EPSG:32616)
      transformer = Transformer.from_crs('epsg:4326', 'epsg:32616', always_xy=True)
      nodes_long_lat_df['x'], nodes_long_lat_df['y'] = transformer.transform(nodes_long_lat_df['lon'].values, nodes_long_lat_df['lat'].values)
      nodes_long_lat_df = nodes_long_lat_df[['name', 'x', 'y', 'lon', 'lat']]  # Reorder columns
      print(f"\nAfter projection:\n{nodes_long_lat_df.head()}")

      def closest_node(centroid_x, centroid_y):
          distances = (nodes_long_lat_df['x'] - centroid_x)**2 + (nodes_long_lat_df['y'] - centroid_y)**2
          min_idx = distances.idxmin()
          return nodes_long_lat_df.loc[min_idx, 'name']

      # This is rather inefficient, but it works.
      for _, row in df_demand.iterrows():
         h_geocode = str(row['h_geocode'])
         w_geocode = str(row['w_geocode'])
         
         # The demand may contain geo_ids that are outside our area of interest.
         # gdf_monroe contains geoid of blocks that are within our area of interest.
         h_filtered = gdf_monroe[gdf_monroe['GEOID20'] == h_geocode]['centroid']
         w_filtered = gdf_monroe[gdf_monroe['GEOID20'] == w_geocode]['centroid']

         # If both work and home are not within our area of interest, skip.
         if len(h_filtered) == 0 or len(w_filtered) == 0:
              print(f"\nWarning: Missing centroid for h_geocode {h_geocode} or w_geocode {w_geocode}")
              continue
         else:
            # print(f"\nh_filtered: {h_filtered}, w_filtered: {w_filtered}")

            h_point = h_filtered.values[0]
            w_point = w_filtered.values[0]

            print(f"h_point: {h_point}, w_point: {w_point}")

            # We must also exclude if the orig_node and dest_node are the same.
            orig_node = closest_node(h_point.x, h_point.y)
            dest_node = closest_node(w_point.x, w_point.y)
            if orig_node != dest_node:
               df_demand.loc[_, 'orig'] = orig_node
               df_demand.loc[_, 'dest'] = dest_node
            else:
               print(f"Warning: orig_node and dest_node are the same for row {_}")
               continue

      print(f"\nAfter mapping:\n{df_demand.head()}")
      print(f"Unique orig_nodes: {df_demand['orig'].unique()}, Total: {len(df_demand['orig'].unique())}")
      print(f"Unique dest_nodes: {df_demand['dest'].unique()}, Total: {len(df_demand['dest'].unique())}")
      print(f"Number of rows: {len(df_demand)}")
      print(f"Number of None/ NaN in orig_node: {df_demand['orig'].isna().sum()}")
      print(f"Number of None/ NaN in dest_node: {df_demand['dest'].isna().sum()}")

      # Step 5: Aggregate and save data.
      # The same pair of orig and dest must be summed up. And the columns should just be orig, dest, and volume.
      # Filter out any rows where orig_node or dest_node might be None/NaN before aggregation
      df_demand = df_demand.dropna(subset=['orig', 'dest'])

      # Save mapping info for validation plot before aggregation
      mapping_info = df_demand[['h_geocode', 'w_geocode', 'orig', 'dest']].copy()

      # Rename S000 column to volume for consistency with standard format
      df_demand = df_demand.rename(columns={'S000': 'volume'})

      # Now aggregate by summing the volume for duplicate OD pairs
      df_demand = df_demand.groupby(['orig', 'dest']).agg({'volume': 'sum'}).reset_index()
      print(f"\nAfter aggregation:\n{df_demand.head()}")

      # Apply the peak hour factor
      df_demand['volume'] = np.ceil(df_demand['volume'] * PEAK_HOUR_FACTOR * ( 1 + OTHER_TRIPS_FACTOR)).astype(int)

      output_csv = './bloomington/bloomington_demand_standard.csv'
      df_demand.to_csv(output_csv, index=False)
      print(f"Saved aggregated demand to: {output_csv}")

      # Print demand statistics
      print("\n" + "="*60)
      print("DEMAND STATISTICS (vehicles/hour)")
      print("="*60)
      
      # Calculate trip generation (outgoing) and attraction (incoming)
      trip_generation = df_demand.groupby('orig')['volume'].sum()
      trip_attraction = df_demand.groupby('dest')['volume'].sum()
      
      # Find highest origin demand
      max_origin_node = trip_generation.idxmax()
      max_origin_value = trip_generation.max()
      print(f"\nHighest Origin Demand:")
      print(f"  Node: {max_origin_node}")
      print(f"  Volume: {max_origin_value} vehicles/hour")
      
      # Find highest destination demand
      max_dest_node = trip_attraction.idxmax()
      max_dest_value = trip_attraction.max()
      print(f"\nHighest Destination Demand:")
      print(f"  Node: {max_dest_node}")
      print(f"  Volume: {max_dest_value} vehicles/hour")
      
      # Total demand
      total_demand = df_demand['volume'].sum()
      print(f"\nTotal Network Demand: {total_demand} vehicles/hour")
      print(f"Number of O-D pairs: {len(df_demand)}")
      print("="*60 + "\n")

      ######
      # Step 6: Validation plot - Show centroid to node mapping
      # Create a validation plot showing how census block centroids map to network nodes
      # Load the processed data
      nodes_df = pd.read_csv('./bloomington/bloomington_nodes_standard.csv', dtype={'name': str})
      links_df = pd.read_csv('./bloomington/bloomington_links_standard.csv', dtype={'start': str, 'end': str})

      print("\nPlotting ... ")
      print(f"\nLoaded {len(nodes_df)} nodes and {len(links_df)} links for plotting")
      print(f"Sample node data:\n{nodes_df.head()}")
      print(f"Sample link data:\n{links_df.head()}")

      # Create the publication-ready plot
      fig, ax = plot_bloomington_base(nodes_df, links_df, plot_node_ids=False)
      plt.savefig('./bloomington/base_network.png', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
      print("Saved base network plot to './bloomington/base_network.png'")

      
      mapped_nodes, total_centroids = plot_validation_mapping(
          ax=ax, 
          nodes_df=nodes_df, 
          gdf_monroe=gdf_monroe, 
          mapping_info=mapping_info,
          mode='contour' # 'contour', 'lines', or 'blocks'
      )
      
      ax.set_title(f'Bloomington Network: Centroid-to-Node Mapping ({len(mapped_nodes)} nodes, {total_centroids} centroids)')
      plt.tight_layout()
      plt.savefig('./bloomington/validation_mapping.png', bbox_inches='tight')
      print(f"Saved validation plot to './bloomington/validation_mapping.png'")

      # Step 7: Demand visualization
      fig_demand, (ax1, ax2) = plot_demand_viz('./bloomington/bloomington_nodes_standard.csv', './bloomington/bloomington_links_standard.csv', output_csv)
      plt.savefig('./bloomington/demand_visualization.png', bbox_inches='tight')
      print("Saved demand visualization to './bloomington/demand_visualization.png'")

      fig_unified, (ax1, ax2, ax3) = plot_unified('./bloomington/bloomington_nodes_standard.csv', './bloomington/bloomington_links_standard.csv', output_csv)
      plt.savefig('./bloomington/unified_visualization.png', bbox_inches='tight')
      print("Saved unified visualization to './bloomington/unified_visualization.png'")

      # Create new unified plot with combined origins/destinations and routes
      fig_unified_2, (ax1_2, ax2_2, ax3_2) = plot_unified_2(
          './bloomington/bloomington_nodes_standard.csv',
          './bloomington/bloomington_links_standard.csv',
          output_csv,
          routes_json='./bloomington/bloomington_existing_routes.json'
      )
      plt.savefig('./bloomington/unified_visualization_2.png', bbox_inches='tight')
      print("Saved unified visualization 2 (with routes) to './bloomington/unified_visualization_2.png'")

if __name__ == "__main__":
   ### Bloomington Network ###
   bloomington = Helpers()
   bloomington_nodes_csv = "./bloomington/bloomington_nodes.csv"
   bloomington_links_csv = "./bloomington/bloomington_links.csv"
   bloomington_demand_lodes_csv = "./bloomington/in_od_main_JT00_2022.csv"
   bloomington.convert_bloomington(nodes_csv=bloomington_nodes_csv, links_csv=bloomington_links_csv, demand_lodes_csv=bloomington_demand_lodes_csv)


   ### Sioux Falls Network ###
   # sioux_falls = Helpers()
   # sioux_falls_nodes_csv = "./sioux_falls/sioux_falls_nodes.csv"
   # sioux_falls_links_tntp = "./sioux_falls/sioux_falls_net.tntp"
   # sioux_falls_demand_csv = "./sioux_falls/sioux_falls_demand.csv"
   # sioux_falls.convert_sioux_falls(nodes_csv=sioux_falls_nodes_csv, demand_csv=sioux_falls_demand_csv, links_tntp=sioux_falls_links_tntp)
