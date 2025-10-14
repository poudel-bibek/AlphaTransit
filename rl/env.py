import os
import random
import numpy as np
import networkx as nx
import pandas as pd
import gymnasium as gym
from uxsim import World
from pathlib import Path

from collections import defaultdict
from uxsim.BusHandler import BusHandler
from rl.env_utils import plot_network_demand_and_path
from typing import Any, Dict, Optional, Tuple, List 

class TransitEnv(gym.Env):
    """
    Transit environment for RL.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        The order of operations performed in the constructor matters. 
        Be careful while modifying.

        We treat two way streets as two separate directed edges.
        """
        self.config = dict(config)
        self.horizon = self.config.get("horizon")
        self.delta_t = self.config.get("delta_t") # deltat = deltan * reaction_time
        self.delta_n = self.config.get("delta_n")

        self.network_dir = Path(__file__).resolve().parents[1] / "networks" / self.config.get("network")
    
        # Important params: 
        self.service_frequency_mode = self.config.get("service_frequency_mode")
        self.STOP_SPACING = self.config.get("stop_spacing")
        self.BUS_CAPACITY = self.config.get("bus_capacity")
        self.STOP_DURATION = self.config.get("stop_duration")
        self.alpha = self.config.get("alpha")
        self.arrival_window = self.config.get("arrival_window")
        self.unserved_as_cars = self.config.get("unserved_as_cars") 
        self.comfort_threshold = self.config.get("comfort_threshold")
        self.radius = self.config.get("radius")
        self.random_path_init = self.config.get("random_path_init")
        
        # Constraints:
        self.NUM_ROUTES = self.config.get("num_routes")
        self.MAX_ROUTE_LENGTH = self.config.get("max_route_length")
        self.MIN_ROUTE_LENGTH = self.config.get("min_route_length")
        self.world = None

        # Multi-route management:
        self.all_routes = []           # List of completed routes [[route1], [route2], ...]
        self.current_route = []        # Currently active route being built
        self.current_route_index = 0   # Index of route currently being built (0 to NUM_ROUTES-1)
        self.is_baseline = False       # Flag to indicate if this is baseline evaluation mode
        self.N_NODE_FEATURES = 12
        self.N_EDGE_FEATURES = 2

        # Lookup dicts for efficient queries:
        # Only work on standardized data.
        nodes_csv = self.network_dir / f"{self.config.get('network')}_nodes_standard.csv"
        links_csv = self.network_dir / f"{self.config.get('network')}_links_standard.csv"
        demand_csv = self.network_dir / f"{self.config.get('network')}_demand_standard.csv"

        df_nodes = pd.read_csv(nodes_csv, dtype={"name": str})
        df_links = pd.read_csv(links_csv, dtype={"name": str, "start": str, "end": str})
        df_demand = pd.read_csv(demand_csv, dtype={"orig": str, "dest": str}) # Read as strings so node names remain consistent with the World (nodes are named as strings)

        # Sort node names numerically, even though they are strings
        self.node_list = sorted(list(df_nodes["name"].unique()), key=lambda x: int(x))
        self.n_nodes = len(self.node_list)
        self.n_edges = int(len(df_links))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}
        
        # Cache demand DataFrame to avoid reading CSV every step
        self.demand_df_cached = df_demand

        # Compute adjacency matrix (assuming undirected graph since roads are bidirectional):
        # Use dict with sets for faster lookups during training
        adj_temp = defaultdict(set)
        for _, row in df_links.iterrows():
            # Force to strings to match node identifiers
            x, y = str(row["start"]), str(row["end"])
            adj_temp[x].add(y)
            adj_temp[y].add(x)  # Add reverse direction for undirected graph
        
        # Convert to dict for better performance
        self.adj = {node: neighbors for node, neighbors in adj_temp.items()}
        
        # Demand matrix: 
        self.od_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for _, row in df_demand.iterrows():
            if row["orig"] in self.node_to_idx and row["dest"] in self.node_to_idx:
                i = self.node_to_idx[row["orig"]]
                j = self.node_to_idx[row["dest"]]
                self.od_matrix[i, j] += row["volume"]

        # Demand vectors:
        self.demand_out = self.od_matrix.sum(axis=1) # Trips starting from node i
        self.demand_in = self.od_matrix.sum(axis=0) # Trips ending at node i

        # Normalization constants: 
        self.max_demand = max(self.demand_out.max(), self.demand_in.max()) 
        self.min_demand = min(self.demand_out.min(), self.demand_in.min())
                
        self.max_degree = max(len(neighbors) for neighbors in self.adj.values())
        self.min_degree = min(len(neighbors) for neighbors in self.adj.values()) 
        
        self.min_x = df_nodes["x"].min()
        self.max_x = df_nodes["x"].max()
        self.min_y = df_nodes["y"].min()
        self.max_y = df_nodes["y"].max()
        self.min_length = df_links["length"].min()
        self.max_length = df_links["length"].max()
        self.min_free_flow_speed = df_links["free_flow_speed"].min() # u = free_flow_speed
        self.max_free_flow_speed = df_links["free_flow_speed"].max()
        
        # Edge index and features:
        edge_index_list = []
        edge_attr_list = []
        self.link_lengths = {}
        for _, row in df_links.iterrows():
            x, y = str(row['start']), str(row['end'])
            if x in self.node_to_idx and y in self.node_to_idx:
                i, j = self.node_to_idx[x], self.node_to_idx[y]
                edge_index_list.append([i, j]) # Add the i, j indices of the edge

                length = row['length']
                self.link_lengths[(x, y)] = length # We need to add un-normalized route lengths.
                self.link_lengths[(y, x)] = length # Store in both directions for easy lookup

                # Now edge attributes: Normalize to [0,1] and add
                length_norm = (length - self.min_length) / (self.max_length - self.min_length) 
                # In bloomington, all links have the same free_flow_speed.
                speed_norm = (row['free_flow_speed'] - self.min_free_flow_speed) / (self.max_free_flow_speed - self.min_free_flow_speed) if self.max_free_flow_speed != self.min_free_flow_speed else 1.0
                edge_attr_list.append([length_norm, speed_norm]) # Add the normalized edge attributes

        self.edge_index = np.array(edge_index_list).T.astype(np.int64) # Transpose, shape (2, E)
        self.edge_features = np.array(edge_attr_list, dtype=np.float32) # Shape (E, 3)

        # For efficiency, pre-compute a number of things required in state during init (once):
        self.node_coordinates_norm = np.zeros((self.n_nodes, 2), dtype=np.float32)
        self.node_degrees_norm = np.zeros(self.n_nodes, dtype=np.float32)

        for node_name in self.node_list:
            node_idx = self.node_to_idx[node_name]
            node_data = df_nodes.loc[df_nodes["name"] == node_name].iloc[0]

            # Min max normalize
            self.node_coordinates_norm[node_idx, 0] = (node_data["x"] - self.min_x) / (self.max_x - self.min_x)
            self.node_coordinates_norm[node_idx, 1] = (node_data["y"] - self.min_y) / (self.max_y - self.min_y)
            
            # degree 
            degree = len(self.adj.get(node_name, [])) # If not found in adj, return empty list
            self.node_degrees_norm[node_idx] = (degree - self.min_degree) / (self.max_degree - self.min_degree)

        
    @property
    def observation_space(self) -> gym.Space:
        """
        State (size is network dependent and fixed per selected network): 
        Need to encode the following: 
        - The network graph: 
            - Nodes with node features (x, y)
            - Links with link features (length, free_flow_speed)
            - Total demand (raw) with demand features (orig, dest, start_t, end_t, q)
        - Current path (partial path so far):
            - Path nodes
        - Budget state:
            - Remaining number of nodes to add.
        
        ##########
        1. For each node in the network (12): 
            Static (5): 
                - coordinates (x, y) - min-max normalized
                - degree - divide by max degree in the network
                - d_out: sum of all O-D flows emanating from node i  - divide by max demand in the network
                - d_in: sum of all O-D flows arriving at node i - divide by max demand in the network
            
            Dynamic:
                - Related to demand (4):  
                    
                    - d_out_current_route: sum of all O-D flows emanating from node i to nodes within the current route - divide by max demand in the network
                        - "If I add node i to the current route, how much demand from i could be served by the current route nodes?"

                    - d_in_current_route: sum of all O-D flows arriving at node i from nodes within the current route - divide by max demand in the network
                        - "If I add node i to the current route, how much demand from nodes in current route could be reach node i?"

                    - d_out_completed_routes: sum of all O-D flows emanating from node i to nodes within all completed routes - divide by max demand in the network   
                        - "If I add node i to the current route, how much demand from i could be served by the completed route nodes?"
                        - Ridership potential of each node i.e., if I add this node 5 to the current route, how many passengers who start from node 5 want to go to nodes that are already in the completed routes?
                    
                    - d_in_completed_routes: sum of all O-D flows arriving at node i from nodes within all completed routes - divide by max demand in the network
                        - "If I add node i to the current route, how much demand from nodes in completed routes could be reach node i?"
                         - Attractiveness of the node i.e., if I add this node 5 to the current route, how many passengers from the completed route nodes want to go to node 5?
                
                - Related to route:
                    - Related to current route (2):

                        - in_current_route_flag: binary (1 if node is in the current route, else 0). 
                        
                        - is_valid_next: binary (1 if adjacent to current frontier and not in current route, else 0)
                            - inform the policy about the validity of next nodes to select
                            - Not allowed if:
                                - Node is already in the current route.
                                - Node is not adjacent to the current frontier.

                    - Related to completed routes (1):
                        - in_completed_routes_flag: A single node can be in multiple routes.
                            - A fraction to indicate how many completed routes the node is in. 0.0 = not in any completed routes. 1.0 = in all completed routes.
                            - a value like 1/3 would mean that this is a potential node to expand as a transfer node.

        2. For each edge in the network: 
            - Edge index (to indicate connectivity):
                - For policy networks like GATv2, edge index is required. edge_index = a compact list of directed edges using node indices.
                (https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html)
                - Shape (2, E) where E is the number of edges in the graph.
                - If the road is bidirectional, include both directions: 
                    - e.g., nodes = ["1","5","8"] → indices 0,1,2
                    - if edges 1→5, 5→1, 5→8, and 8→5 then edge_index: src: [0,1,1,2], dst: [1,0,2,1]
            - Edge features:
                - Since the policy network is a GAT, edge features can be provided.
                - Edge features (2):
                    - length
                    - free_flow_speed (u)

        3. Episode completion (self.NUM_ROUTES):
            - For example if self.NUM_ROUTES = 3,[0.5, 0.9, 0.1] means route 1 was terminated after 50% completion, route 2 90% and current route 3 is 10% complete.
            - Fractions indicate the completion of each route i.e., steps completed upto max_route_length.
        
        - Total: 12 + 2 + 3 = 17
        - The edge features injected at each GAT layer
        - Episode completion features injected at the readout layer.

        Notes: 
        - d_out_path and d_in_path are "path-aware" demand service vectors
        - How to enforce the action mask?
            # Option 1: Enforce it simulation level i.e., by simply disallowing the action (when it is selected). However, this is not a good idea for several reasons.
                1. Hurts learning stability: 
                - PPO computes log_probs from policy's distribution, if samples are rejected (and env substitutes the action for another) then the log_probs are not valid.
                
                2. Hurts sample efficiency.
                - For discrete action spaces, entropy is calculated over the full action set.
                    - Lets say there were 24 actions (N=24) among which only 3 were valid, entropy for categorial policy is -sum_0^(N)(p_i * log(p_i))
                    
                    - When there is no mask, earlier in the learning process, policy network makes all logits similar (uniform distribution across actions).
                    - i.e., each action gets (1/24) of the probability mass. However (21/24)= 87.5% of the probability mass is wasted (not useful for learning).
                    - Most samples hit invalid actions, policy gradient increases probability of actions that cannot execute.
                    - P.S. 21 out of 24 nodes being invalid is a realistic scenario.
                    
                    - On the other hand, when a mask is used, each valid action initially gets (1/3) of the probability mass and all of the mass (100%) is useful for learning.
                    - Every sample is actionable and gradients focus among feasible choices. 
                    
                    - When there is waste, more samples are needed to learn.
                
                3. Leads to incorrect credit assignment: 
                - The reward is credited to the actions the policy did not choose (this injects bias and noise in the learning process).

            # Option 2: Enforce using an action_mask (without agent awareness).
                - Keep action space fixed: Discrete(N) where N is the number of nodes in the network.
                - When constructing a categorical distribution from the policy output, mask out the logits related to invalid actions i.e., logits[invalid] = -inf (or a large negative)
                - So that this action has a near zero probability of being sampled this masked distribution.
                - Compute log probabilities from the masked distribution. Use the masked distribution in the act, evaluate as necessary.
                - Print a warning if invalid action slips through (In practice, with proper masking, this shouldn't happen and I dont expect this, but due to incorrect implementation, it could happen).
                - However, this is also a problem for several reasons:
                    - The policy is not aware of the mask (because it is not a part of the observation space).
                    - Without a negative feedback from the reward function, it does not truly know (and learn) that it assigned a higher logit to an invalid action.
                        - Since we assigned a zero probability to the invalid action, this action is never sampled and the policy has no "practice" of selecting it.
            
            # Option 3: Supply validity as a part of node feature in the observation space.
                - Keep action space fixed: Discrete(N) where N is the number of nodes in the network.
                - Use a binary flag (feature is_valid_next) in the observation to inform the policy about the feasible set.
                - Add a large negative penalty in the reward function for selecting an invalid action and truncate the episode based on these conditions: 
                    - New node selected is not connected to the current frontier 
                    - New node is connected to the current frontier but is already in the path.
                    - New node is the same as the current frontier.
                - Theoritically this is good. But requires a lot of samples to learn to select valid actions.

            # Option 4: A combination of option 2 and 3.
                - However: 
                    - If I am doing action masking, invalid actions are simply not sampled. 
                    - So what is the point of including option 3 with option 2?
                        - Episodes are not going to get truncated.
                        - Agent is not going to get a reward for bad actions anyway.. so its not going to learn to select valid actions.
                    - We need a solution that is scalable to large networks. Option 3 is not scalable. Option 2 is scalable. Option 4 is not scalable.
                
            # Finally, Option 2 is chosen and some components of option 3 are included (without the reward penalty i.e., just including the is_valid_next flag).
            - This is the way to go, validated: https://arxiv.org/pdf/2006.14171

        Important:
        - edge_index dtype/ordering: Use int64 here so converting to torch tensors naturally yields torch.long, as required by PyG. Keep columns aligned with edge_features rows so edge_attr[i] corresponds to edge_index[:, i].
        - edge_features normalization: Perform per-edge normalization (e.g., length_norm, u_norm, t_ff_norm scaled to [0,1]) to keep feature magnitudes comparable and stabilize attention/optimization.
        """
        
        return gym.spaces.Dict({
            "node_features": gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_nodes, self.N_NODE_FEATURES), dtype=np.float32),  # +1 for is_valid_next
            "edge_index": gym.spaces.Box(low=0, high=self.n_nodes - 1, shape=(2, self.n_edges), dtype=np.int64), # int64 so torch.from_numpy(...).long() matches PyG expectations
            "edge_features": gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_edges, self.N_EDGE_FEATURES), dtype=np.float32), # Per-edge features must be normalized to [0,1] prior to insertion (e.g., length_norm, u_norm)
            "route_progress": gym.spaces.Box(low=0.0, high=1.0, shape=(self.NUM_ROUTES,), dtype=np.float32), # Normalized to 0-1
        })
        
    @property
    def action_space(self) -> gym.Space:
        """
        Option 1: Simultaneously advance all routes at once by 1 node.
            - Action space would increase i.e., 3 x N 
        Option 2: Sequentially advance each route by 1 node.
            - Action space remains N
            - It's easier to learn (credit assignment per route is cleaner i.e., allows the reward to immediately reflect on overlaps, transfers.)

        For each route, select the next node to add to the path (starting from a single node).
        The agent builds linearly expanding paths for each route i.e., at each step, chooses one more neighbor to add at the end of current path.
        The neighbor is a "frontier" node and is adjacent (connected by an edge) to the current end node of the route i.e., no skipping allowed.
        Further, no self loops or back-tracking (visiting of nodes already in the route) are allowed.

        Notes: 
        - The action space is naturally discrete
        - Node IDs are arbitrary labels, we cant choose what to add based on node IDs 
        - For example gym.spaces.Discrete(3) means choose ids 0, 1, 2
        - This RL method needs to be generalizable to larger networks (in which the number of nodes can be very large) 
        - This does not mean that a policy trained on single network must be generalizable to another, we can train policy for each network separately but the algo in general must work in various networks.
                
        Based on how the agent operates, the action space seems to be variable sized per step:
        - At node with 3 neighbors has 3 discrete choices, which one with 10 has 10 discrete choices.
        
        Solution 1: 
        - Set a fixed sized maximum cap (top-k): e.g., 5; select them based on heuristic such as demand emanating/ distance to current frontier.

        Solution 2 (This is used, with action masking): 
        - Large action space: Discrete(N) where N is the number of nodes in the network.
        - i.e., Select a node index (0 to N-1) to add to the path.
        
        --------------

        Adding Frequency of Service (FOS) to the action space:
        - Although the stand-alone route design by itself is a well formed problem, incorporating FOS is important: 
            - FOS has direct impact on : 
                - Passenger satisfaction: wait time, total travel time
                - Operator cost
                - Network wide effects such as congestion, emissions, etc.
        - The current env has "partial route with dense reward" setup i.e., every step is rewarded.
            - However, learning frequency on a partial routes may be mis-leading to the full routes.
                - Initially, the route might just be [1 -> 5], this short segment only serves a single O-D pair. 
                  The agent may learn that setting a high frequency is beneficial for reducing wait times, etc. 
                - But later, when routes are longer, high frequencies might induce congestion and waste.
                - A frequency of 6 buses/hour is a bad choice for a 2-node route but might be an excellent choice for a 10-node route. 
                  The agent would have to learn a completely different frequency preference for step 1, step 2, step 3.. which is extremely difficult.
                - Using (next node, frequency) as the action space would also increase the action space significantly. Increasing the difficulty of learning.
                - 
            - Solution: 


        """
        return gym.spaces.Discrete(self.n_nodes)
    
    def _initialize_current_route(self, use_random: bool = False, avoid_completed_routes: bool = True) -> list:
        """
        Initialize route design at: 
        - The highest demand node i.e., node from which highest demand emanates.
        - Random node.
        Option to avoid starting from nodes that are already present in the completed routes.
        """
        
        all_nodes = list(self.demand_df_cached["orig"].unique())
        if avoid_completed_routes:
            # Flatten self.all_routes into a set of nodes already used
            completed_nodes = set(node for route in self.all_routes for node in route)
            choice_nodes = [node for node in all_nodes if node not in completed_nodes]
        else:
            choice_nodes = all_nodes
        
        if use_random:
            choice = random.choice(choice_nodes)
            print(f"Initializing route randomly at node: {choice}")
            return [choice]
            
        else: 
            # Rank all nodes by highest demand emanating from them (total volume leaving each node)
            demand_df_grouped = self.demand_df_cached.groupby("orig").sum(numeric_only=True).reset_index()
            demand_df_ranked = demand_df_grouped.sort_values("volume", ascending=False)
            
            # Choose the highest ranking node from available choice_nodes
            for _, row in demand_df_ranked.iterrows():
                candidate_node = row["orig"]
                if candidate_node in choice_nodes:
                    choice = candidate_node
                    print(f"Initializing route at highest available demand node: {choice}")
                    return [choice]

    def _allocate_demand_by_service(self, world: World, method: str = "volume", unserved_as_cars: bool = False) -> None:
        """
        Assigns mode-specific demands based on the current bus route:
        - for OD pairs served by the route (both O and D on route)
            - Its necessary to have both O and D, a bus route that cannot connect OD's is meaningless.
            - Since buses go back and forth in the route, we dont have to enforce that O comes before D.
        - bus_passenger gets alpha * q, car gets (1 - alpha) * q
        - alpha: Modal split parameter for served O-D pairs (proportion taking bus)
        - alpha=1.0 means 100% of SERVED demand goes to bus

        - Dealing with demand that is not served by the route (UNSERVED demand):
            - If alpha < 1.0: Allocate unserved demand to cars if unserved_as_cars=True, otherwise ignore it
            - If alpha = 1.0: Always ignore unserved demand (no cars created for unserved O-D pairs)
            - Notes:
                - When alpha=1.0, unserved demand is ignored regardless of unserved_as_cars setting
                - Some nodes (associated O-D pairs) may not be served because of 1) partial route construction 2) no coverage even at full route construction.

        --------------
        - TODO: radius: Radius within each node to consider for demand allocation.
            - how far are passengers willing to walk to the bus stop? In kilometers i.e., 0.5 = 500m
            - Both during boarding and alighting. 
            - TODO Notes: 
                - Requires modifications to BusHandler.py to support this. 
                - The links file has lengths in meters, will be helpful.

        The flow-based demand csv expects: 
            - Mandatory: orig, dest, start_t, end_t, mode (its optional in sim; mandatory in the RL env to split with alpha) 
            - per-second flow rate (q) vehicles/s
        The volume-based demand csv expects: 
            - Mandatory: orig, dest, volume
            - The volume will be spread over the time window (start_t, end_t) by adddemand.
            - Advantage of using volume is that it can be spread across the entire sim horizon.
            - The sim expects volume to be specified in total vehicle spread over time. But in the data, it is specified in per/hour (transformation done below).
        """
        # Use cached DataFrame instead of reading CSV every step
        demand_df = self.demand_df_cached
        print(f"Loading {len(demand_df)} demand records...")  
        
        total_demand = 0 
        bus_demand = 0 
        car_demand = 0 

        if method == "volume":
            for _, row in demand_df.iterrows():
                orig, dest, volume_per_hour = str(row["orig"]), str(row["dest"]), row["volume"]
                total_volume = volume_per_hour * (self.horizon / 3600) # multiply by "how many hours" in horizon
                # print(f"Orig: {orig}, Dest: {dest}, Volume: {volume_per_hour}, Total Volume: {total_volume}")
                if volume_per_hour <= 0 or orig == dest:
                    continue

                is_served = self._is_od_served(orig, dest)
                bus_volume_float = total_volume * self.alpha if is_served else 0
                car_volume_float = total_volume - bus_volume_float if (unserved_as_cars and self.alpha < 1.0) else 0

                # Use Poisson sampling to convert fractional volumes to integers while preserving expected totals
                # This prevents systematic loss from flooring small per-OD volumes to zero
                bus_passengers = np.random.poisson(bus_volume_float) if bus_volume_float > 0 else 0
                car_passengers = np.random.poisson(car_volume_float) if car_volume_float > 0 else 0

                # Alternative: Use ceiling to convert fractional volumes to integers while avoiding systematic loss
                # This prevents small per-OD volumes from being floored to zero
                # np.random.poisson() for statistically correct sampling, but ceiling is simpler and deterministic
                # bus_passengers = int(np.ceil(bus_volume_float)) if bus_volume_float > 0 else 0
                # car_passengers = int(np.ceil(car_volume_float)) if car_volume_float > 0 else 0

                total_demand += total_volume
                bus_demand += bus_volume_float  # Keep float for logging/analytics
                car_demand += car_volume_float  # Keep float for logging/analytics

                # Spread passengers over horizon the first X% of simulation horizon
                # Car
                arrival_window = int(self.horizon * self.arrival_window)
                if car_passengers > 0:
                    world.adddemand(orig=orig, dest=dest, t_start=0, t_end=arrival_window,
                                    volume=car_passengers, mode="vehicle")

                # Spread bus passengers if served
                if bus_passengers > 0:
                    world.adddemand(orig=orig, dest=dest, t_start=0, t_end=arrival_window,
                                    volume=bus_passengers, mode="bus_passenger")

        # Not used right now for the sake of data standardization.
        # elif method == "flow":
        #     for _, row in demand_df.iterrows():

        print(f"Total demand: {total_demand}, Bus demand: {bus_demand}, Car demand: {car_demand}")
        return world

    def _is_od_served(self, orig: str, dest: str) -> bool:
        """
        Check if O-D pair can be served by the complete transit system (all routes + transfers).
        
        Uses BusHandler's transit_graph which includes:
        - All completed routes + current route being built
        - Transfer connections between overlapping routes
        - NetworkX pathfinding for multi-route journeys
        
        Returns True if any path exists from origin to destination.
        """
        if hasattr(self.world, 'bus_handler') and self.world.bus_handler and hasattr(self.world.bus_handler, 'transit_graph'):
            transit_graph = self.world.bus_handler.transit_graph
            if transit_graph and orig in transit_graph.nodes and dest in transit_graph.nodes:
                return nx.has_path(transit_graph, orig, dest)
                
        return False

    def _get_state(self, ) -> Dict[str, Any]:
        """
        Build observation state as a dict
        Normalize as necessary.
        
        The agent is designing route for a bus, however, both demands in the state are total demands.
        - Total demand does not reflect what can be influenced by the agent.
        - However, since the total demand is split by a fixed alpha, higher total demand means higher bus demand as well. 
        - So the current setup should work out.
        - Or we can show only the capturable demand in the path-aware demands?
        - TODO: add this alpha multuplier as optional argument later.
        """
        
        # Nodes and node features:
        node_features = np.zeros((self.n_nodes, self.N_NODE_FEATURES), dtype=np.float32)  # +1 for is_valid_next

        # Static node features (0-4):
        node_features[:, 0] = self.node_coordinates_norm[:, 0] # x
        node_features[:, 1] = self.node_coordinates_norm[:, 1] # y
        node_features[:, 2] = self.node_degrees_norm # degree
        node_features[:, 3] = self.demand_out / self.max_demand # d_out
        node_features[:, 4] = self.demand_in / self.max_demand # d_in
        
        # Dymanic node features (5-6, route-aware demands to current route):
        current_route_indices = np.array([self.node_to_idx[node] for node in self.current_route]) # Set of nodes in the path
        demand_out_current_route = self.od_matrix[:, current_route_indices].sum(axis=1) # Sum of all O-D flows emanating from node i to nodes within the path
        demand_in_current_route = self.od_matrix[current_route_indices, :].sum(axis=0) # Sum of all O-D flows arriving at node i from nodes within the path
        node_features[:, 5] = demand_out_current_route / self.max_demand # d_out_current_route
        node_features[:, 6] = demand_in_current_route / self.max_demand # d_in_current_route
        
        # Dynamic node features (7-8, route-aware demands to completed routes)
        # NOTE: If the node is not connected to the completed routes, it cannot be served yet. So would be 0.
        completed_routes_indices_set = set()  # Unique indices across all completed routes
        for route in self.all_routes:
            completed_routes_indices_set.update(self.node_to_idx[node] for node in route)  # Generator for efficiency, no list/array needed

        completed_routes_indices = np.array(list(completed_routes_indices_set)) if completed_routes_indices_set else np.array([])

        demand_out_completed_routes = self.od_matrix[:, completed_routes_indices].sum(axis=1) if len(completed_routes_indices) > 0 else np.zeros(self.n_nodes)  # Sum of all O-D flows emanating from node i to nodes within the path
        demand_in_completed_routes = self.od_matrix[completed_routes_indices, :].sum(axis=0) if len(completed_routes_indices) > 0 else np.zeros(self.n_nodes)  # Sum of all O-D flows arriving at node i from nodes within the path

        # Apply connectivity check (reuse existing current_route_indices from earlier in _get_state)
        current_route_indices_set = set(current_route_indices)  # O(N) but N small; indices are ints
        # If current route overlaps with any completed indices, it means the current route is connected to completed routes.
        # In this case, demands to/from completed are meaningful (can be served via transfers).
        # Otherwise, set to 0 across all nodes (no service to completed possible yet).
        is_connected = bool(current_route_indices_set & completed_routes_indices_set)
        if not is_connected:
            demand_out_completed_routes = np.zeros(self.n_nodes)
            demand_in_completed_routes = np.zeros(self.n_nodes)

        node_features[:, 7] = demand_out_completed_routes / self.max_demand # d_out_completed_routes
        node_features[:, 8] = demand_in_completed_routes / self.max_demand # d_in_completed_routes

        # Node features that are Binary flags/ fractions (9-11):
        # 9: in_current_route flag
        node_features[current_route_indices, 9] = 1.0 
        
        # 10: is_valid_next flag
        frontier = self.current_route[-1]
        route_set = set(self.current_route)  # O(1) lookup
        valid_neighbors = self.adj.get(frontier, set()) - route_set  # Set difference with safe lookup
        valid_indices = [self.node_to_idx[node] for node in valid_neighbors]
        if len(valid_indices) > 0:
            node_features[valid_indices, 10] = 1.0 

        # 11: in_completed_routes flag
        # Expressed as fraction.
        completed_count_per_node = np.zeros(self.n_nodes, dtype=np.float32)
        for route in self.all_routes:
            for node in route:
                idx = self.node_to_idx[node]
                completed_count_per_node[idx] += 1.0

        node_features[:, 11] = completed_count_per_node / len(self.all_routes) if len(self.all_routes) > 0 else 0.0 # Guard against division by zero when no completed routes

        # Edge index and edge features dont dynamically change. Already set in __init__.
        # Route progress for self.NUM_ROUTES (including completed and current route)
        route_progress = np.zeros(self.NUM_ROUTES, dtype=np.float32)
        for i, route in enumerate(self.all_routes):
            route_progress[i] = len(route) / self.MAX_ROUTE_LENGTH
        print(f"\nAll routes: {self.all_routes}, length: {len(self.all_routes)}\n")
        if self.current_route_index < self.NUM_ROUTES:  # Only if there's a current route being built
            route_progress[self.current_route_index] = len(self.current_route) / self.MAX_ROUTE_LENGTH
        
        # Return the state as a dict
        state: Dict[str, Any] = {
            "node_features": node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "route_progress": route_progress
        }
        
        return state
    
    def reset( self, ) -> None:
        """
        TODO: Can I just sample the action space and get a random initial state?
        """

        self.all_routes = []           # Reset completed routes
        self.current_route_index = 0   # Start with first route
        self.current_route = self._initialize_current_route(use_random=self.random_path_init)
        state = self._get_state() # initial state
        return state, {}
    
    def build_world(self, network: str) -> World:
        """
        generate nodes from csv:
        - expects: name, x, y
        generate links from csv:
        - expects: link_name, start, end, length, free_flow_speed
        """
        if not network:
            raise ValueError("Network name must be provided in config['network']")

        nodes_csv = self.network_dir / f"{network}_nodes_standard.csv"
        links_csv = self.network_dir / f"{network}_links_standard.csv"
        demand_csv = self.network_dir / f"{network}_demand_standard.csv"

        for path in (nodes_csv, links_csv, demand_csv):
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")

        world = World(
            name=network,
            deltan=int(self.delta_n),
            reaction_time=float(self.delta_t),
            tmax=self.horizon,
            print_mode=0,
            save_mode=0,
            show_mode=0,
            random_seed=self.config.get("seed"),
        )
        # Populate network from CSVs
        world.generate_Nodes_from_csv(str(nodes_csv))

        df_links = pd.read_csv(links_csv, dtype={"name": str, "start": str, "end": str})
        # Making use of links data other than free_flow_speed
        for _, row in df_links.iterrows():
            world.addLink(
                name=row['name'],
                start_node=row['start'],
                end_node=row['end'],
                length=row['length'],
                free_flow_speed=row['free_flow_speed']
            )

            # Add reverse link for undirected behavior (since these are not one-way roads)
            reverse_name = f"{row['name']}_reverse"
            world.addLink(
                name=reverse_name,
                start_node=row['end'],
                end_node=row['start'],
                length=row['length'],
                free_flow_speed=row['free_flow_speed']
            )

        # Required for adding bus passenger demand via world.adddemand(..., mode="bus_passenger")
        world.set_bus_handler(BusHandler)

        return world

    def load_demand_for_plotting(self, world: World) -> None:
        """
        Load demand data for plotting purposes only.
        This is a simplified version that loads all demand as car traffic.
        """
        demand_csv = self.network_dir / f"{self.config.get('network')}_demand_standard.csv"
        df_demand = pd.read_csv(demand_csv, dtype={"orig": str, "dest": str})

        for _, row in df_demand.iterrows():
            orig, dest, volume = str(row["orig"]), str(row["dest"]), row["volume"]
            if volume > 0 and orig != dest:
                # Spread volume over simulation horizon
                total_volume = volume * (self.horizon / 3600)  # convert per-hour to total
                world.adddemand(orig=orig, dest=dest, t_start=0, t_end=self.horizon,
                               volume=total_volume, mode="vehicle")

    def _get_service_frequency(self, route: List[str], all_routes: List[List[str]] = None) -> int:
        """
        Calculates the bus service frequency for a given route based on the selected mode.

        The frequency represents the number of buses dispatched per hour.

        Modes:
        -------
        - "fixed": Returns a constant, predefined service frequency.

        - "max_load": Calculates frequency based on the peak passenger load on the busiest segment of the route (normalized for route overlaps).
            - The formula used is: ceil(Q_max_norm / (comfort_threshold * capacity)) where Q_max_norm is the normalized peak segment demand (per hour and not over the entire horizon).
            - i.e., How often buses should run on a route to handle the peak passenger demand while keeping buses comfortable, accounting for load sharing between overlapping routes
            - If Q_k,max_unnormalized = 200 pph, C_k = 40, and delta_max = 0.8, and 2 routes serve the busiest segment, then:
              Q_k,max_normalized = 200 / 2 = 100 pph, f_k = ceil(100 / (0.8 * 40)) = ceil(3.125) = 4 buses/hour
            - Normalization process:
                1. Calculate which segments are served by how many routes across all routes
                2. For each segment in the current route, divide its load by the number of routes serving that segment

            - Notes:
                - Uses graph-based pathfinding to accurately model passenger journeys, including transfers across multiple routes
                - For each O-D pair served by the transit network, finds the actual path passengers take using shortest path
                - Distributes passenger load across ALL segments they traverse on each route. Eg. if a passenger travels segments B->C->D->E on a route, their load is added to all segments (B-C, C-D, D-E), not just transfer points
                - The problem with a naive approach (without normalization) is that a segment may end up having a large load (due to multiple route overlaps) and the bus frequency is set to this very high value (due to this large load).
                - But that max load will be served by a number of routes overlapping on that segment. So, frequency should be normalized by the number of routes serving that segment.
                - i.e., Normalization prevents overestimation of frequency when multiple routes serve the same segments, ensuring fair load distribution.
        """

        if self.service_frequency_mode == "fixed":
            return 10 # A fixed value

        elif self.service_frequency_mode == "max_load":

            # 1. Build a complete transit graph from all known routes for accurate pathfinding
            temp_transit_graph = nx.Graph()
            if all_routes:
                for r in all_routes:
                    if len(r) > 1:
                        nx.add_path(temp_transit_graph, r)

            # 2. Create a quick lookup for node positions on the current route
            route_indices = {node: idx for idx, node in enumerate(route)}
            segment_loads = np.zeros(len(route) - 1, dtype=np.float64)

            # 3. Calculate segment coverage across all routes for normalization
            if all_routes is not None:
                all_routes_segments = {}  # segment -> count of routes that serve it

                # Calculate segments for all routes including current route
                all_routes_with_current = all_routes + [route]
                for r in all_routes_with_current:
                    r_indices = {node: idx for idx, node in enumerate(r)}
                    for i in range(len(r) - 1):
                        seg_key = (min(r[i], r[i+1]), max(r[i], r[i+1]))  # Normalize segment order
                        all_routes_segments[seg_key] = all_routes_segments.get(seg_key, 0) + 1

                # Convert to segment index mapping for current route
                route_segments_coverage = {}
                for i in range(len(route) - 1):
                    seg_key = (min(route[i], route[i+1]), max(route[i], route[i+1]))
                    route_segments_coverage[i] = all_routes_segments.get(seg_key, 1)

            # 4. Calculate per-hour passenger load for each segment using accurate pathfinding
            for _, row in self.demand_df_cached.iterrows():
                orig, dest = str(row["orig"]), str(row["dest"])

                # Check if this O-D pair can be served by the current transit network
                if (orig in temp_transit_graph and dest in temp_transit_graph and
                    nx.has_path(temp_transit_graph, orig, dest)):

                    passenger_volume = float(row["volume"]) * float(self.alpha)
                    if passenger_volume <= 0:
                        continue

                    # Use graph-based pathfinding to find the actual path this passenger will take
                    full_path = nx.shortest_path(temp_transit_graph, orig, dest)

                    # Check if this route is part of the passenger's journey
                    route_in_path = any(node in route_indices for node in full_path)

                    if route_in_path:
                        # Extract the portion of the path that uses this specific route
                        route_path_portion = []
                        for node in full_path:
                            if node in route_indices:
                                route_path_portion.append(node)

                        # If passenger travels on this route, determine which segments they use
                        if len(route_path_portion) > 1:
                            # Convert route nodes to indices and find segments traversed
                            route_indices_in_path = [route_indices[node] for node in route_path_portion]
                            min_idx = min(route_indices_in_path)
                            max_idx = max(route_indices_in_path)

                            # Add passenger load to all segments they traverse on this route
                            for seg_idx in range(min_idx, max_idx):
                                normalization_factor = route_segments_coverage.get(seg_idx, 1)
                                segment_loads[seg_idx] += passenger_volume / normalization_factor

            # Compute the comfortable capacity per departure and ensure the threshold is well defined.
            comfort_capacity = float(self.comfort_threshold * self.BUS_CAPACITY)

            # Calculate frequency based on max load principle
            max_segment_load = segment_loads.max()  # Identify the bottleneck load
            print(f"DEBUG: Route {route[:3]}... max_load={max_segment_load:.1f} pph, capacity={comfort_capacity:.1f}")
            frequency = int(np.ceil(max_segment_load / comfort_capacity))
            frequency = max(1, frequency)  # Minimum 1 bus per hour

            return frequency
            
    def _apply_action(self) -> Dict[str, bool]:
        """
        Invoked internally by step
        - Add stops based on STOP_SPACING.
        - Add necessary vehicles to the world.
            - Add buses based on the current path and SERVICE_FREQUENCY.
        - Check for route completion conditions and handle transitions

        Args:
            baseline: If True, skip route completion checks (for baseline evaluation)

        Returns a dict with route completion signals:
        - route_completed: Route was successfully completed (met min length)
        - route_forced_end: Route ended due to no valid moves (may be short)
        - ep_done: All routes have been processed
        """

        # Check for route completion conditions BEFORE creating buses (skip for baselines)
        route_completed, route_forced_end, ep_done = False, False, False

        if not self.is_baseline:
            # Route reached maximum length - complete it successfully
            if len(self.current_route) >= self.MAX_ROUTE_LENGTH:
                self.all_routes.append(self.current_route[:])  # Store completed route
                print(f"Route {self.current_route_index} completed successfully: {self.current_route}")
                route_completed = True

            # No valid moves left - route is incomplete/forced to end
            elif len(self._get_valid_indices()) == 0:
                print(f"Route {self.current_route_index} forced to end (no valid moves, incomplete: {len(self.current_route)} < {self.MAX_ROUTE_LENGTH})")
                route_forced_end = True

        # Simulate both completed and current partial route being built (only if current route has at least 2 nodes)
        routes_to_simulate = self.all_routes.copy()  # Create a copy, not a reference!
        if not self.is_baseline and len(self.current_route) > 1:
            routes_to_simulate.append(self.current_route)
        
        for route_idx, route in enumerate(routes_to_simulate):
            # Determine bus stops based on STOP_SPACING for this route
            bus_stops = route[::self.STOP_SPACING]
            bus_name = f"bus_route_{route_idx}"
            bus = self.world.addVehicle(
                orig=route[0], 
                dest=route[1],  # Always start with next node as dest (consistent with UXSim)
                departure_time= 0, # First bus starts at time 0
                name=bus_name, # unique name for each bus
                mode="bus"
            )

            # Pass all routes including current for normalization
            all_routes_for_freq = self.all_routes + [self.current_route] if self.current_route and len(self.current_route) > 1 else self.all_routes
            service_frequency_route = self._get_service_frequency(route, all_routes_for_freq)
            print(f"\nService frequency for route {route_idx}: {service_frequency_route}\n")
            print(f"DEBUG: service_frequency_route type: {type(service_frequency_route)}, value: {service_frequency_route}")

            # Set the bus route - this will create SERVICE_FREQUENCY number of buses:
            buses = bus.set_bus_route(
                route=route,
                stops=bus_stops,  # Use the calculated bus stops for this route
                is_circular=False, # Do not make routes circular 
                capacity=self.BUS_CAPACITY,
                stop_duration=self.STOP_DURATION,
                service_frequency=service_frequency_route, # This creates multiple buses for the frequency
                sim_horizon=self.horizon # Pass simulation horizon for proper bus spacing
            )

            route_status = "completed" if route in self.all_routes else "current"
            print(f"Route {route_idx} ({route_status}) bus '{bus_name}': set_bus_route created additional {len(buses) - 1} buses")
                
        # Print bus summary AFTER all buses are created
        # all_buses = [v for v in self.world.VEHICLES.values() if hasattr(v, 'mode') and v.mode == 'bus']
        # print(f"\nTotal buses in simulation: {len(all_buses)} - {[b.name for b in all_buses]}")
        
        # Build transit graph AFTER all buses are created
        self.world.bus_handler.build_transit_graph()
        
        # Add demand AFTER buses and transit graph are ready
        self.world = self._allocate_demand_by_service(self.world, unserved_as_cars=self.unserved_as_cars)
        
        return {
            'route_completed': route_completed,
            'route_forced_end': route_forced_end,
            'ep_done': ep_done
        }

    def _get_final_metrics(self, handler: BusHandler) -> Dict[str, Any]:
        """
        Get metrics after the sim has been run.
        - Average bus speed (m/s, calculated per bus and averaged)
            - This includes time the bus waits at the stops. 
        - Average bus utilization (%, calculated per bus and averaged)
        - A list containing travel time for each passenger (both passengers who completed the trips and ongoing trips)
        - A list containing waiting time for each passenger (passengers who completed the trips, are in ongoing trips, and still waiting at the end of the sim)
        - Count of passengers waiting at the end of the sim
        - Total waiting time for passengers who are still waiting at stops at the end of the sim
        - Count of passengers who completed their trip
        - Count of passengers who on-boarded the bus
        
        - Passengers are not pre-allocated to a bus ahead of time. They are loaded on the bus when it physically arrives at the stop.
        """

        # 1. Average bus speed: 
        # Per-bus (Total distance traveled/ total operating time),operating time depends on the time bus was spawned.
        # Per-bus average speed when actually moving (excludes time spent stopped at bus stops)
        bus_speeds = []
        all_buses = [v for v in self.world.VEHICLES.values() if hasattr(v, 'mode') and v.mode == 'bus']
        for bus in all_buses:
            total_distance = bus.distance_traveled
            total_operating_time = self.world.TIME - bus.departure_time_in_second
            # print(f"\nBus: {bus.name}, total_distance: {total_distance}, total_operating_time: {total_operating_time} = {self.world.TIME} - {bus.departure_time_in_second}")
            
            # Only include buses that have actually started operating and moved
            if total_operating_time > 0 and total_distance > 0:
                speed = total_distance / total_operating_time
                bus_speeds.append(speed)

        avg_bus_speed_mps = np.mean(bus_speeds) if bus_speeds else 0.0
        
        # 2. Average bus utilization
        bus_utilizations = {}
        utilization_sum = 0.0
        buses_with_data = 0  # Count only buses with journey data
        
        for bus_name, journey in handler.bus_route_journeys.items():
            if not journey:
                continue
            
            buses_with_data += 1  # Count this bus as having data
            bus_utilizations[bus_name] = []
            for i , stop_info in enumerate(journey):

                # bus capacity 
                bus_capacity = stop_info['bus_capacity']
                # capacity after passengers boarded/ alighted
                capacity_after = stop_info['capacity_after']

                utilization = capacity_after / bus_capacity
                bus_utilizations[bus_name].append((f"stop_{i}", utilization))
            
            average_utilization_this_bus = np.mean([utilization for _, utilization in bus_utilizations[bus_name]])
            utilization_sum += average_utilization_this_bus

        avg_bus_utilization_pct = (utilization_sum / buses_with_data * 100) if buses_with_data > 0 else 0.0  # Use buses_with_data instead of len(all_buses)

        # 3. Travel time distributions (wait time, in-vehicle time (movement time), total travel time)
        wait_time_dstr = []
        movement_time_dstr = []
        travel_time_dstr = [] # wait time + in-vehicle time
        all_passengers_total_wait_time = 0.0

        # Handle COMPLETED trips (from passenger_stats, not buses)
        # These logs will automatically include wait and travel time across multiple legs.
        trips_with_transfers = 0
        for p_stats in handler.passenger_stats:
            wait_time = p_stats['total_wait_time']
            all_passengers_total_wait_time += wait_time

            wait_time_dstr.append(wait_time)
            movement_time_dstr.append(p_stats['total_in_vehicle_time'])
            travel_time_dstr.append(p_stats['total_travel_time'])

            # Count trips that require transfers (not the total number of transfers)
            if p_stats.get('num_transfers', 0) > 0:
                trips_with_transfers += 1
            
        # Handle PARTIAL/ONGOING trips.
        # looking at passengers who are currently on a bus. Use journey_log from new BusHandler structure.
        on_going_count_sim_end = 0
        for bus in all_buses:
            for passenger in bus.passengers:
                
                # Looking at all prior legs of the journey.
                # Passengers whose wait time is completed (not None)
                completed_waits = [log for log in passenger.journey_log if log['type'] == 'wait' and log['end'] is not None]
                wait_time = sum(log['end'] - log['start'] for log in completed_waits)
                
                # Calculate total movement time: ALL completed rides + current ongoing ride
                completed_rides = [log for log in passenger.journey_log if log['type'] == 'ride' and log['end'] is not None]
                completed_movement_time = sum(log['end'] - log['start'] for log in completed_rides)
                
                current_ride = next((log for log in reversed(passenger.journey_log) if log['type'] == 'ride' and log['end'] is None), None)
                
                if current_ride:
                    current_ride_time = self.world.TIME - current_ride['start']
                    total_movement_time = completed_movement_time + current_ride_time
                    
                    all_passengers_total_wait_time += wait_time
                    wait_time_dstr.append(wait_time)
                    movement_time_dstr.append(total_movement_time)  # Now includes all legs
                    travel_time_dstr.append(wait_time + total_movement_time)
                    on_going_count_sim_end += 1
                

        # Handle STILL-WAITING (not onboarded) i.e., passengers whose departure time has arrived but are still waiting for the bus 
        # They only have wait time and have no in-vehicle time; no point in adding travel time)
        total_waiting_time_sim_end = 0.0
        for waiting_list in handler.waiting_passengers.values():
            for passenger in waiting_list:
                
                current_wait = next((log for log in reversed(passenger.journey_log) if log['type'] == 'wait' and log['end'] is None), None)
                
                if current_wait:
                    wait_time = self.world.TIME - current_wait['start']
                    wait_time_dstr.append(wait_time)
                    total_waiting_time_sim_end += wait_time

        # 5. From above, also waiting passengers at the end of the sim
        waiting_count_sim_end = sum(len(passengers) for passengers in handler.waiting_passengers.values()) # At the end of the sim, how many passengers are still waiting at stops.

        # 6. Completed trips
        completed_count = len(handler.passenger_stats) # This works because the stats are only added when passengers alight the bus.
        
        # 7. Onboarded passengers
        total_onboarded_count = completed_count + on_going_count_sim_end

        # 8. Transfer rate calculation (% of completed trips requiring transfers)
        transfer_rate_pct = 100 * (trips_with_transfers / completed_count) if completed_count > 0 else 0.0

        return {'avg_bus_speed_mps': avg_bus_speed_mps,
                'avg_bus_utilization_pct': avg_bus_utilization_pct,
                'travel_time_dstr': travel_time_dstr,
                'waiting_time_dstr': wait_time_dstr,
                'movement_time_dstr': movement_time_dstr, # movement time
                'waiting_count_sim_end': waiting_count_sim_end,
                'total_waiting_time_sim_end': total_waiting_time_sim_end,
                'completed_count': completed_count,
                'total_onboarded_count': total_onboarded_count,
                'total_wait_time': all_passengers_total_wait_time,
                'transfer_rate': transfer_rate_pct,
                'fleet_size': len(all_buses)} # Total number of buses deployed

    def _get_initial_metrics(self) -> Dict[str, float]:
        """
        Compute metrics (which can be computed before the sim starts):
        - Route length (meters). Since we use heuristic baselines like shortest path, this should be part of the metrics .
        - Count of wanting to onboard:
            - The goal is to serve all demand
            - This measures the demand that can be served by current routes.
            - If routes are designed well, this should be close to the total demand.
        """

        # 1. Route length - sum of all completed routes plus current route
        route_length = 0.0

        # Add length of all completed routes
        for route in self.all_routes:
            if len(route) >= 2:
                for i in range(len(route) - 1):
                    route_length += self.link_lengths.get((str(route[i]), str(route[i+1])), 0.0)

        # Add length of current route (if it has at least 2 nodes and not baseline)
        if not self.is_baseline and len(self.current_route) >= 2:
            for i in range(len(self.current_route) - 1):
                route_length += self.link_lengths.get((str(self.current_route[i]), str(self.current_route[i+1])), 0.0)

        # 2. Wanting to onboard
        wanting_to_onboard = 0
        for _, row in self.demand_df_cached.iterrows():
            orig, dest, volume_per_hour = str(row["orig"]), str(row["dest"]), row["volume"]
            total_volume = volume_per_hour * (self.horizon / 3600) # Multiply by number of hours in horizon

            if volume_per_hour > 0 and orig != dest and self._is_od_served(orig, dest):
                wanting_to_onboard += total_volume * self.alpha  # Potential bus passengers

        # 3. Total demand (may or may not lie on the routes)
        total_demand = 0
        for _, row in self.demand_df_cached.iterrows():
            orig, dest, volume_per_hour = str(row["orig"]), str(row["dest"]), row["volume"]
            total_volume = volume_per_hour * (self.horizon / 3600) # Multiply by number of hours in horizon

            if volume_per_hour > 0 and orig != dest:
                total_demand += total_volume

        return {'route_length': route_length, 'wanting_to_onboard': wanting_to_onboard, 'total_demand': total_demand}
    
    def _calculate_route_overlap_ratio(self) -> float:
        """
        Calculate the ratio of overlapped segments in routes.
        Logic: 
        - pair-wise node overlap across all routes.
        Returns:
            - Ratio of overlapped segments [0-1]
            - 0.0 = No overlaps
            - 1.0 = All segments overlap
        """
        total_routes = self.all_routes + [self.current_route]
        num_routes = len(total_routes)

        # No overlap possible if there is only one route
        if num_routes <= 1:
            return 0.0
        
        overlap_sum = 0.0
        num_pairs = 0

        for i in range(num_routes):
            for j in range(i + 1, num_routes):
                set_i = set(total_routes[i])
                set_j = set(total_routes[j])
                common = len(set_i & set_j)
                min_len = min(len(set_i), len(set_j))
                pair_ratio = common / min_len if min_len > 0 else 0.0
                overlap_sum += pair_ratio
                num_pairs += 1

        return overlap_sum / num_pairs if num_pairs > 0 else 0.0
    
    def _step_until(self, until_t: int, print_metrics: bool = True) -> Dict[str, int]:
        """
        Run the simulation until the given time and collect metrics related to performance of the route.

        Notes: 
        - Pending passengers → passengers that have been created by demand but not yet boarded a bus (start time has not arrived)
        - Waiting passengers → passengers that have been created by demand but not yet boarded a bus (waiting for a bus to arrive)
        - Onboarded passengers → passengers that have been boarded a bus and are currently traveling on it
        - Completed passengers → passengers that have reached their destination and have completed their trip
        - Data to be collected to plot Distributions: 
            - Waiting time 
                - distribution can be used to determine the quality of waiting time (how equitable is the distribution)
                - passengers who completed trips + currently traveling + still waiting at sim end.
            - Travel time 
                - passengers who completed trips + currently traveling in buses
        - Service rate: % of passengers who "lie on the routes and wanted to board" that were able to get to their destination.
        - Demand coverage: % of passengers who "are part of the demand and may or may not lie on the routes" that were able to onboard (if they onboard, given enough time, they will get to their destination).
        """

        handler = self.world.bus_handler
        current_route_str = [str(node) for node in self.current_route]

        initial_metrics = self._get_initial_metrics()
        self.world.exec_simulation(until_t=until_t)
        final_metrics = self._get_final_metrics(handler)

        # Calculations 
        total_wait_time_minutes = (1/60) * final_metrics['total_wait_time'] # How much time did the passengers who completed the trips, or who are currently traveling, spent waiting for the bus to arrive.
        wait_time_sim_end_minutes = (1/60) * final_metrics['total_waiting_time_sim_end'] # At the end of the simulation, how much time did the passengers who are still waiting at stops, spent waiting for the bus to arrive.
        movement_time_minutes = (1/60) * sum(final_metrics['movement_time_dstr']) # For trip completion how much time was spent in-vehicle for passengers who completed the trip + still in the bus at the end of sim.
        total_travel_time_minutes = (1/60) * sum(final_metrics['travel_time_dstr']) # Movement time + waiting time for passengers who completed the trip + still in the bus at the end of sim .
        
        # Service rate calculation (used in both metrics dict and print statements)
        service_rate_pct = 100 * (final_metrics['completed_count'] / initial_metrics['wanting_to_onboard']) if initial_metrics['wanting_to_onboard'] > 0 else 0.0 # What % of demand was fulfilled.
        
        # Per-passenger averages 
        total_combined_wait_time = total_wait_time_minutes + wait_time_sim_end_minutes
        avg_wait_time_minutes = total_combined_wait_time / len(final_metrics['waiting_time_dstr']) if len(final_metrics['waiting_time_dstr']) > 0 else 0.0
        avg_movement_time_minutes = movement_time_minutes / len(final_metrics['movement_time_dstr']) if len(final_metrics['movement_time_dstr']) > 0 else 0.0
        avg_travel_time_minutes = sum(final_metrics['travel_time_dstr']) / len(final_metrics['travel_time_dstr']) / 60 if len(final_metrics['travel_time_dstr']) > 0 else 0.0
        
        # Onboarding rate calculation
        onboard_rate_pct = 100 * (final_metrics['total_onboarded_count'] / initial_metrics['wanting_to_onboard']) if initial_metrics['wanting_to_onboard'] > 0 else 0.0
        
        # Route efficiency calculation (passengers completed per km of route)
        route_efficiency_passengers_per_km = 1000 * (final_metrics['completed_count'] / initial_metrics['route_length']) if initial_metrics['route_length'] > 0 else 0.0
        
        # Other components
        demand_coverage_potential_pct = 100 * (initial_metrics['wanting_to_onboard'] / initial_metrics['total_demand']) if initial_metrics['total_demand'] > 0 else 0.0
        demand_coverage_actual_pct = 100 * (final_metrics['total_onboarded_count'] / initial_metrics['total_demand']) if initial_metrics['total_demand'] > 0 else 0.0
        route_overlap_ratio = self._calculate_route_overlap_ratio()

        # Calculate node coverage percentage
        all_routes_nodes = set()
        for route in self.all_routes:
            all_routes_nodes.update(route)
        if len(self.current_route) > 1:  # Only add current route if it has more than just the starting node
            all_routes_nodes.update(self.current_route)
        node_coverage_pct = 100 * (len(all_routes_nodes) / self.n_nodes) if self.n_nodes > 0 else 0.0

        metrics = {
            # Waiting metrics (all times in seconds for consistency).
            'total_wait_time': final_metrics['total_wait_time'], # Total wait time for completed/traveling passengers (seconds)
            'wait_time_sim_end': final_metrics['total_waiting_time_sim_end'], # Wait time for passengers still waiting at sim end (seconds)
            'sim_end_waiting_passengers_count': final_metrics['waiting_count_sim_end'], # Count of passengers still waiting at sim end

            # Travel metrics (all times in seconds for consistency).
            'wanting_to_onboard': initial_metrics['wanting_to_onboard'], # Count of passengers whose O-D is served by route
            'total_onboarded_count': final_metrics['total_onboarded_count'], # Count of passengers who boarded buses
            'completed_trip_passengers_count': final_metrics['completed_count'], # Count of passengers who completed trips
            'movement_time': sum(final_metrics['movement_time_dstr']), # Total in-vehicle time for completed/traveling passengers (seconds)
            'total_travel_time': sum(final_metrics['travel_time_dstr']), # Total travel time for completed/traveling passengers (seconds)
            
            # per passenger averages
            'avg_wait_time': avg_wait_time_minutes * 60, # Average wait time per passenger (seconds)
            'avg_movement_time': avg_movement_time_minutes * 60, # Average in-vehicle time per passenger (seconds)
            
            # Data collected (raw distributions in seconds).
            'waiting_time_dstr': final_metrics['waiting_time_dstr'], # Waiting time distribution for all passengers (seconds)
            'movement_time_dstr': final_metrics['movement_time_dstr'], # In-vehicle time distribution for completed/traveling passengers (seconds)
            'travel_time_dstr': final_metrics['travel_time_dstr'], # Total travel time distribution for completed/traveling passengers (seconds)
            
            # Reward components
            'demand_coverage_potential': demand_coverage_potential_pct, # Percentage of demand that wants to board buses (wanting/total)
            'demand_coverage_actual': demand_coverage_actual_pct, # Percentage of demand that boarded buses (onboarded/total)
            'avg_travel_time': avg_travel_time_minutes * 60, # Average travel time per passenger (seconds)
            'route_overlap_ratio': route_overlap_ratio, # Ratio of overlapped segments [0-1]

            # Route metrics.
            'route_length': initial_metrics['route_length'], # Route length (meters)
            'bus_utilization': final_metrics['avg_bus_utilization_pct'], # Average bus utilization (percentage)
            'average_bus_speed': final_metrics['avg_bus_speed_mps'], # Average bus speed (meters/second)
            'fleet_size': final_metrics['fleet_size'], # Total number of buses deployed (count)
            'service_rate': service_rate_pct, # Percentage of demand fulfilled (completed/wanting)
            'onboard_rate': onboard_rate_pct, # Percentage of demand that boarded buses (onboarded/wanting)
            'transfer_rate': final_metrics['transfer_rate'], # Percentage of trips requiring transfers
            'route_efficiency': route_efficiency_passengers_per_km, # Passengers completed per km of route (passengers/km)
            'node_coverage': node_coverage_pct, # Percentage of nodes covered by routes (0-100%)
        }

        if print_metrics:
            print("\n" + "="*70)
            print("SIMULATION METRICS")
            print("="*70)
            
            # Route Information
            print("\nROUTE INFORMATION:")
            print(f"   All Routes:                {self.all_routes}")
            print(f"   Current Route:               {' → '.join(current_route_str)}")
            print(f"   Route Length:             {metrics['route_length']/1000:.2f} km")
            print(f"   Average Bus Speed:        {metrics['average_bus_speed']:.2f} m/s ({metrics['average_bus_speed']*3.6:.1f} km/h)")
            print(f"   Bus Utilization:          {metrics['bus_utilization']:.1f}%")
            print(f"   Fleet Size:               {metrics['fleet_size']} buses")
            
            # Passenger Counts
            print("\nPASSENGER COUNTS:")
            print(f"   Wanting to Onboard:       {metrics['wanting_to_onboard']:,} passengers")
            print(f"   Total Onboarded:          {metrics['total_onboarded_count']:,} passengers")
            print(f"   Completed Trips:          {metrics['completed_trip_passengers_count']:,} passengers")
            print(f"   Still Waiting at End:     {metrics['sim_end_waiting_passengers_count']:,} passengers")
            
            # Time Metrics - Aggregated
            print("\nAGGREGATE TIME METRICS:")
            print(f"   Simulation Duration:      {until_t:,} seconds")
            print(f"   Total Wait Time:          {total_combined_wait_time:.1f} minutes")
            print(f"   │  ├─ Completed/Traveling: {total_wait_time_minutes:.1f} minutes")
            print(f"   │  └─ Still Waiting:       {wait_time_sim_end_minutes:.1f} minutes")
            print(f"   Total Movement Time:      {movement_time_minutes:.1f} minutes (in-vehicle only)")
            print(f"   Total Travel Time:        {total_travel_time_minutes:.1f} minutes (wait + movement)")
            
            # Time Metrics - Per Passenger Averages
            print("\nPER-PASSENGER AVERAGES:")
            print(f"   Average Wait Time:        {avg_wait_time_minutes:.1f} minutes")
            print(f"   Average Movement Time:    {avg_movement_time_minutes:.1f} minutes")
            print(f"   Average Travel Time:      {avg_travel_time_minutes:.1f} minutes")
            
            # Performance Summary
            print("\nPERFORMANCE SUMMARY:")
            print(f"   Passengers Served:        {service_rate_pct:.1f}% ({metrics['completed_trip_passengers_count']} / {metrics['wanting_to_onboard']})")
            print(f"   Boarding Success:         {onboard_rate_pct:.1f}% ({metrics['total_onboarded_count']} / {metrics['wanting_to_onboard']})")
            print(f"   Transfer Rate:            {metrics['transfer_rate']:.1f}% ")
            print(f"   Node Coverage:            {node_coverage_pct:.1f}% ({len(all_routes_nodes)} / {self.n_nodes} nodes)")
            print(f"   Route Efficiency:         {route_efficiency_passengers_per_km:.2f} passengers/km")
            
            print("="*70)

        return metrics
    
    def _plot_metrics(self, metrics: Dict[str, int]) -> None:
        """
        # TODO: Complete this.
        Plot: 
        - Distribution of waiting time
        - Distribution of travel time

        """

        # Distribution Statistics moved from print_metrics:
        print("\n DISTRIBUTION STATISTICS:")
        if len(metrics['waiting_time_dstr']) > 0:
            wait_times = np.array(metrics['waiting_time_dstr'])
            print(f"   Wait Times (n={len(wait_times)}):  min={wait_times.min():.1f}s, max={wait_times.max():.1f}s, mean={wait_times.mean():.1f}s, std={wait_times.std():.1f}s")
        
        if len(metrics['movement_time_dstr']) > 0:
            movement_times = np.array(metrics['movement_time_dstr'])
            print(f"   Movement Times (n={len(movement_times)}): min={movement_times.min():.1f}s, max={movement_times.max():.1f}s, mean={movement_times.mean():.1f}s, std={movement_times.std():.1f}s")
        
        if len(metrics['travel_time_dstr']) > 0:
            travel_times = np.array(metrics['travel_time_dstr'])
            print(f"   Travel Times (n={len(travel_times)}):   min={travel_times.min():.1f}s, max={travel_times.max():.1f}s, mean={travel_times.mean():.1f}s, std={travel_times.std():.1f}s")
        
        pass

    def compute_reward(self, sim_result: Dict[str, int], action_signals: Dict[str, bool]) -> float:
        """
        Design routes that serve the full demand (100% coverage) with high passenger travel efficiency.
        In addition to route performance, we will use some components to help the agent design better routes i.e., complete routes, minimal overlaps.
        
        ---------
        Encourage: 
            1. Higher demand coverage potential.
                - This encourages agent to select include O-D pairs that have high demand.
            2. Higher service rate (demand coverage actual/ demand coverage potential)
                - This is related to frequency of service as well.
            
        Penalize: 
        1. Total travel time (in-vehicle time + wait time) i.e., we want higher passenger travel efficiency.
        2. Overlap between routes: some minimal overlaps are necessary for transfers (and 100% demand coverage)
            but high overlaps mean duplication of service (waste of resources).
        2. Forced end: 
            - Early termination: Each route should be designed upto the full max_route_length. This represents poor planning from agent.
            - When invalid (masked) action slips through.

        TODO: Should I add a final reward at the end of the episode? Based on final performace metrics?
        

        As of now do not care about, operator side metrics like: 
        - route_efficiency: Passengers served per km of route (passengers/km)
            - Prevents wastefully long routes with few passengers (encourages compact, high-demand route design, prevents arbitrarily long routes)
        - bus_utilization: Average bus capacity utilization (0-100%)
            - Forces agent to choose routes with actual demand, preventing gaming via low-demand routes
        - fleet cost: 
            - Some multiple of len(all_routes) * SERVICE_FREQUENCY 
        ---------
    
        reward = β₀ × demand_coverage_potential + β₁ × service_rate - β₂ × travel_time - β₃ × overlap_penalty - β₄ × forced_end_penalties
        
        - Units are normalized: 
            - demand_coverage_potential: [0 to 1] (divide by 100)
            - demand_coverage_actual: [0 to 1] (divide by 100)
            - avg_total_travel_time: [0 to perhaps >1] (normalize by max_travel_time = 1 hour)
            - route_overlap_ratio: [0 to 1] (normalize by max_route_length i.e., max overlap = entire route overlap)
            - forced_end_penalty: [0 to 1] 
                - For early termination: 1 - (current route ended at what length/ max_route_length)
                - TODO: For invalid action: set to 1. Not done for now (the probability is very low)
        ---------
        TODO: On further normalizing the reward: 
        - Applying the Welford Normalization to the returns (not the absolute reward values).
        - Normalizing raw rewards can be a problem, example: 
            - Episode 1: Total travel time = 1000s → reward = -1000
            - Episode 100: Total travel time = 500s → reward = -500
            - Without normalization: Clear improvement! (-500 > -1000)
            - With normalization: Both might map to ~0 (relative to running mean)
            - The agent can't tell it's improving!
        ---------
        Potential pitfalls: 
        1. If the rewards are too small like 0.0001, then gradients are too small to be effective.
        2. An improper reward formulation could lead to reward hacking: 
           
        ---------

        Max reward:
        - 100% demand coverage: +50
        - 100% service rate: +30
        - 0 travel time penalty: +0  
        - No overlaps: +0
        - No forced ends: +0
        → Total: +50
        
        Min reward:
        - 0% demand coverage: +0 (If the demand coverage is 0 then perhaps routes terminated early and overlaps are low)
        - 0% service rate: +0
        - Max travel time penalty: -30
        - Overlap penalty: -5
        - forced ends: -15
        → Total: -50
        
        Typical range: [-50, +80] 
        """
        
        BETA_0 = 50.0      # Demand coverage component (demand served)
        BETA_1 = 30.0      # Service rate component (demand coverage actual/ demand coverage potential)
        BETA_2 = 30.0      # Travel time penalty (passenger efficiency)  
        BETA_3 = 20.0      # Route overlap penalty
        BETA_4 = 15.0      # Forced end penalty
        MAX_TOTAL_TRAVEL_TIME = 3600.0   # 1 hour max expected travel time (seconds)

        demand_coverage_potential = sim_result['demand_coverage_potential'] / 100.0 # Percentage [0-1] 
        demand_coverage_actual = sim_result['demand_coverage_actual'] / 100.0 # Percentage [0-1] 
        avg_travel_time = sim_result['avg_travel_time']  # Average total travel time per passenger (seconds)
        route_overlap_ratio = sim_result['route_overlap_ratio']  # Ratio of overlapped segments [0-1]
        
        # Normalize components to [0-1] scale
        demand_coverage_potential_norm = demand_coverage_potential # Already [0-1] from sim_result
        demand_coverage_actual_norm = demand_coverage_actual # Already [0-1] from sim_result
        avg_travel_time_norm = min(avg_travel_time / MAX_TOTAL_TRAVEL_TIME, 1.0)  # [0-1] capped at 1
        overlap_penalty_norm = route_overlap_ratio  
        
        # Calculate reward components
        demand_coverage_component = BETA_0 * demand_coverage_potential_norm
        # Prevent division by zero (initially demand_coverage_potential_norm could be zero)
        service_rate_component = BETA_1 * (demand_coverage_actual_norm / demand_coverage_potential_norm) if demand_coverage_potential_norm > 0 else 0.0
        travel_time_penalty = BETA_2 * avg_travel_time_norm  
        overlap_penalty = BETA_3 * overlap_penalty_norm

        # Penalties for poor forced end
        forced_end_penalty = 0.0
        if action_signals['route_forced_end']:
            # Calculate penalty based on how early the route was terminated
            current_route_length = len(self.current_route)
            early_termination_ratio = 1.0 - (current_route_length / self.MAX_ROUTE_LENGTH) 
            forced_end_penalty = BETA_4 * early_termination_ratio  

        # TODO: Add invalid action penalty

        total_reward = (demand_coverage_component + service_rate_component - travel_time_penalty - overlap_penalty - forced_end_penalty)
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"\tDemand coverage component: +{demand_coverage_component:.2f} (β₀={BETA_0} × {demand_coverage_potential:.1f}%)")
        print(f"\tService rate component: +{service_rate_component:.2f} (β₁={BETA_1} × {demand_coverage_actual:.1f}%/{demand_coverage_potential:.1f}%)")
        print(f"\tTravel time penalty: -{travel_time_penalty:.2f} (β₁={BETA_1} × {avg_travel_time:.0f}s)")
        print(f"\tOverlap penalty: -{overlap_penalty:.2f} (β₂={BETA_3} × {route_overlap_ratio:.3f})")
        print(f"\tForced end penalty: -{forced_end_penalty:.2f} (β₃={BETA_4} × early_termination_ratio)")
        
        return total_reward    

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Return (obs, reward, route_done, ep_done, info).
        Run the simulation on the current route and get metrics.

        Two-tier termination system:
            - route_done: Current route is completed (max length reached or no valid moves). Just used for logging. 
            - ep_done: Episode is completed (all NUM_ROUTES routes built). True episode termination signal.

        Truncation: 
         - This does not indicate a truncation of the episode. It is a truncation of current route. Just used for logging.
        """

        # 1. Extend the current route
        action_node = self.idx_to_node[action]
        self.current_route = [str(node) for node in self.current_route] + [action_node]
        print(f"Route {self.current_route_index} extended: {self.current_route}")

        # 2. Build_world needs to happen every step.
        # i.e., add the network and the classified demand (bus vs car).
        self.world = self.build_world(self.config.get("network"))
        
        # 3. spawn necessary buses, set routes, and handle route completion
        action_signals = self._apply_action()
        
        # 4. Run the full simulation upto horizon end.
        sim_result = self._step_until(self.horizon)
        
        # 5. Compute reward with termination signals
        reward = self.compute_reward(sim_result, action_signals)
        
        # 6. Extract route completion signals from _apply_action  
        sim_result['route_completed'] = action_signals['route_completed']  # Route completed gracefully 
        sim_result['route_forced_end'] = action_signals['route_forced_end']  # Route got stuck (bad truncation)
        
        # 7. If route completed this step, init next route NOW (after sim/ reward)
        terminated = False  # Default to continue episode
        if action_signals['route_completed'] or action_signals['route_forced_end']:
            self.current_route_index += 1
            if self.current_route_index < self.NUM_ROUTES:
                self.current_route = self._initialize_current_route(use_random=self.random_path_init)
                print(f"Starting route {self.current_route_index}: {self.current_route}")
            else:
                print("All routes processed!")
                terminated = True  # Episode is done when all routes are processed

        return self._get_state(), reward, terminated, None, sim_result # Truncation is not used.
    
    def _get_valid_indices(self) -> list:
        """
        For a given frontier node, get the indices of valid next nodes.
        - When this is called, the action has not yet been applied i.e. current_path does not have the action attached.
        """
        frontier = self.current_route[-1]
        route_set = set(self.current_route)  # O(1) lookup

        # 1. Get all neighbors of frontier (with safe lookup)
        valid_neighbors = self.adj.get(frontier, set()) 
        
        # 2. Remove nodes that are already in the path
        valid_neighbors = valid_neighbors - route_set  

        # 3. Get the indices of valid next nodes
        valid_indices = [self.node_to_idx[node] for node in valid_neighbors]
        return valid_indices

    def render(self, save_dir: str, render_name: str) -> None:
        """
        - Visualize network + all routes.
        - Episode simulation gif.
        """

        all_routes_to_display = self.all_routes.copy()
        # Only add current_route if it's not already in all_routes (avoid duplicates)
        if self.current_route and len(self.current_route) > 1 and self.current_route not in all_routes_to_display:
            all_routes_to_display.append(self.current_route)

        output_loc = os.path.join(os.path.join(save_dir, "images"), render_name)
        plot_network_demand_and_path(self.world, all_routes_to_display, output_loc)

