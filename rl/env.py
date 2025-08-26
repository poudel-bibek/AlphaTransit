import os
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from uxsim import World
from pathlib import Path

from collections import defaultdict
from uxsim.BusHandler import BusHandler
from rl.env_utils import plot_network_demand_and_path
from typing import Any, Dict, Optional, Tuple

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
        self.SERVICE_FREQUENCY = self.config.get("service_frequency")
        self.STOP_SPACING = self.config.get("stop_spacing")
        self.BUS_CAPACITY = self.config.get("bus_capacity")
        self.STOP_DURATION = self.config.get("stop_duration")
        self.alpha = self.config.get("alpha")
        self.radius = self.config.get("radius")
        self.random_path_init = self.config.get("random_path_init")
        
        # Constraints:
        self.MAX_PATH_LENGTH = self.config.get("max_path_length")
        self.MIN_PATH_LENGTH = self.config.get("min_path_length")
        self.world, self.current_path = None, []
        self.N_NODE_FEATURES = 9
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

        # Compute adjacency matrix (assuming directed graph):
        # Use dict with sets for faster lookups during training
        adj_temp = defaultdict(set)
        for _, row in df_links.iterrows():
            # Force to strings to match node identifiers
            x, y = str(row["start"]), str(row["end"])
            adj_temp[x].add(y)
        
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
                speed_norm = (row['free_flow_speed'] - self.min_free_flow_speed) / (self.max_free_flow_speed - self.min_free_flow_speed) 
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
        1. For each node in the network: 
            - coordinates (x, y) - min-max normalized
            - degree - divide by max degree in the network
            - d_out: sum of all O-D flows emanating from node i  - divide by max demand in the network
            - d_in: sum of all O-D flows arriving at node i - divide by max demand in the network
            - d_out_path: sum of all O-D flows emanating from node i to nodes within the path - divide by max demand in the network   
                - "If I add node i to the path, how much demand from i could be served by the existing path nodes?"
                - Ridership potential of each node i.e., if I add this node 5 to the path, how many passengers who start from node 5 want to go to nodes that are already in the current path?
            - d_in_path: sum of all O-D flows arriving at node i from nodes within the path - divide by max demand in the network
                - "If I add node i to the path, how much demand from existing path nodes could be reach node i?"
                - Attractiveness of the node i.e., if I add this node 5 to the path, how many passengers from the current path nodes want to go to node 5?
            - in_path flag: binary (1 if node is in the path, else 0)
            - is_valid_next: binary (1 if adjacent to current frontier and not in path, else 0)
                - inform the policy about the validity of next nodes to select
                - Not allowed if:
                    - Node is already in the path.
                    - Node is not adjacent to the current frontier.
        2. Edges: 
            - Edge index (to indicate connectivity):
                - For policy networks like GATv2, edge index is required. edge_index = a compact list of directed edges using node indices.
                (https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html)
                - Shape (2, E) where E is the number of edges in the graph.
                - If the road is bidirectional, include both directions: 
                    - e.g., nodes = ["1","5","8"] → indices 0,1,2
                    - if edges 1→5, 5→1, 5→8, and 8→5 then edge_index: src: [0,1,1,2], dst: [1,0,2,1]
            - Edge features:
                - Since the policy network is a GAT, edge features can be provided.
                - Edge features:
                    - length
                    - free_flow_speed (u)

        3. Steps left until max_route_length is reached. (Constraint)
        
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
            "steps_left": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), # Normalized to 0-1
        })
        
    @property
    def action_space(self) -> gym.Space:
        """
        Select the next node to add to the path (starting from a single node).
        The agent builds a linearly expanding path i.e., at each step, chooses one more neighbor to add at the end of current path.
        The neighbor is a "frontier" node and is adjacent (connected by an edge) to the current end node of the route i.e., no skipping allowed.
        Further, no loops or back-tracking (visiting of nodes already in the route) are allowed.

        Notes: 
        - The action space is naturally discrete
        - Node IDs are arbitrary labels, we cant choose what to add based on node IDS? 
        - For example gym.spaces.Discrete(3) means choose ids 0, 1, 2
        - This RL method needs to be generalizable to larger networks (in which the number of nodes can be very large) 
        - This does not mean that a policy trained on single network must be generalizable to another, we can train policy for each network separately but the algo in general must work in various networks.
                
        Based on how the agent operates, the action space seems to be variable sized per step:
        - At node with 3 neighbors has 3 discrete choices, which one with 10 has 10 discrete choices.
        - But this is not practical. 
        
        Solution 1: 
        - Set a fixed sized maximum cap (top-k): e.g., 5; select them based on heuristic such as demand emanating/ distance to current frontier.

        Solution 2 (This one used, with action masking): 
        - Large action space: Discrete(N) where N is the number of nodes in the network.
        - i.e., Select a node index (0 to N-1) to add to the path.
        """
        return gym.spaces.Discrete(self.n_nodes)
    
    def _initialize_current_path(self, use_random: bool = False) -> list:
        """
        Various strategies can be used. 
        Current one: Initialize at the highest demand node i.e., node from which highest demand emanates.
        Other option: random i.e., pick a random node.
        """
        demand_df = self.demand_df_cached # Use cached DataFrame 

        if use_random:
            choice = random.choice(list(demand_df["orig"].unique()))
            print(f"Initializing route randomly at node: {choice}")
            return [choice]
            
        else: 
            demand_df_grouped = demand_df.groupby("orig").sum(numeric_only=True).reset_index()
            highest_demand_node = demand_df_grouped.loc[demand_df_grouped["volume"].idxmax()]
            choice = highest_demand_node["orig"]
            print(f"Initializing route at node: {choice}")
            return [choice]

    def _allocate_demand_by_service(self, world: World, demand_csv: str, current_path: list = None, method: str = "volume") -> None:
        """
        Assigns mode-specific demands based on the current bus route:
        - for OD pairs served by the route (both O and D on route)
            - Its necessary to have both O and D, a bus route that cannot connect OD's is meaningless.
            - Since buses go back and forth in the route, we dont have to enforce that O comes before D.
        - bus_passenger gets alpha * q, car gets (1 - alpha) * q
        - alpha: Modal split parameter for served O-D pairs (proportion taking bus)
        - TODO: radius: Radius within each node to consider for demand allocation.
            - how far are passengers willing to walk to the bus stop? In kilometers i.e., 0.5 = 500m
            - Both during boarding and alighting. 
            - TODO Notes: 
                - Requires modifications to BusHandler.py to support this. 
                - The links file has lengths in meters, will be helpful.

        Caveats: 
            - The number of nodes (where demand exists) that is being served changed dynamically as the route is being constructed. 
            - But the observation space is fixed. 
            - Solution: Do not input the partial route demand as part of the state.
                - Other solutions possible.

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
        
        # current_path will have at least one node (where it starts) i.e., it wont have O-D pair then.
        current_path_str = [str(node) for node in current_path] # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
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
                
                is_served = orig in current_path_str and dest in current_path_str
                bus_volume = total_volume * self.alpha if is_served else 0
                car_volume = total_volume - bus_volume
                
                total_demand += total_volume
                bus_demand += bus_volume
                car_demand += car_volume
                
                # Spread car volume over horizon
                world.adddemand(orig=orig, dest=dest, t_start=0, t_end=self.horizon,
                                volume=car_volume, mode="vehicle")
                
                # Spread bus volume if served
                if bus_volume > 0:
                    world.adddemand(orig=orig, dest=dest, t_start=0, t_end=self.horizon,
                                    volume=bus_volume, mode="bus_passenger")

        # Not used right now for the sake of data standardization.
        # elif method == "flow":
        #     for _, row in demand_df.iterrows():
            
        #         orig, dest = str(row["orig"]), str(row["dest"])
        #         start_t, end_t, flow_rate = row["start_t"], row["end_t"], row["q"]
        #         if self._is_od_served(orig, dest, current_path_str):
        #             bus_demand += flow_rate * self.alpha
        #             car_demand += flow_rate * (1 - self.alpha)
        #             total_demand += flow_rate

        #             # Add demand to the world:
        #             world.adddemand(orig=orig, 
        #                             dest=dest, 
        #                             t_start=start_t, 
        #                             t_end=end_t, 
        #                             flow=flow_rate * (1 - self.alpha), 
        #                             mode="vehicle")

        #             world.adddemand(orig=orig, 
        #                             dest=dest, 
        #                             t_start=start_t, 
        #                             t_end=end_t, 
        #                             flow=flow_rate * self.alpha, 
        #                             mode="bus_passenger")
        #         else: 
        #             car_demand += flow_rate
        #             total_demand += flow_rate

        #             # Add demand to the world:
        #             world.adddemand(orig=orig, 
        #                             dest=dest, 
        #                             t_start=start_t, 
        #                             t_end=end_t, 
        #                             flow=flow_rate, 
        #                             mode="vehicle")
        else:
            raise ValueError(f"Invalid method: {method}")

        print(f"Total demand: {total_demand}, Bus demand: {bus_demand}, Car demand: {car_demand}")

        return world

    def _is_od_served(self, orig: str, dest: str, current_path_str: list) -> bool:
        """
        Check if both items in the OD pair are served by the route.
        TODO: implement node radius within the route. Current version: nodes lies in the route.
        """
        if len(current_path_str) < 2: 
            return False
        return orig in current_path_str and dest in current_path_str

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
        
        # Dymanic node features (5-6, path-aware demands):
        path_indices = np.array([self.node_to_idx[node] for node in self.current_path]) # Set of nodes in the path
        demand_out_path = self.od_matrix[:, path_indices].sum(axis=1) # Sum of all O-D flows emanating from node i to nodes within the path
        demand_in_path = self.od_matrix[path_indices, :].sum(axis=0) # Sum of all O-D flows arriving at node i from nodes within the path
        
        node_features[:, 5] = demand_out_path / self.max_demand # d_out_path
        node_features[:, 6] = demand_in_path / self.max_demand # d_in_path
        
        # Node features that are Binary flags (7-8):
        node_features[path_indices, 7] = 1.0 # in_path flag
        
        frontier = self.current_path[-1]
        path_set = set(self.current_path)  # O(1) lookup
        valid_neighbors = self.adj.get(frontier, set()) - path_set  # Set difference with safe lookup
        valid_indices = [self.node_to_idx[node] for node in valid_neighbors]
        node_features[valid_indices, 8] = 1.0  # is_valid_next

        # Edge index and edge features dont dynamically change. Already set in __init__.
        # Steps left:
        steps_taken = len(self.current_path) - 1  # -1 because we start with 1 node
        steps_left_norm = 1.0 - (steps_taken / self.MAX_PATH_LENGTH)
        
        # Return the state as a dict
        state: Dict[str, Any] = {
            "node_features": node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "steps_left": np.array([steps_left_norm], dtype=np.float32)
        }
        
        return state
    
    def reset( self, ) -> None:
        """
        TODO: Can I just sample the action space and get a random initial state?
        """
        self.current_path = self._initialize_current_path(use_random=self.random_path_init)
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
            name="",  # Empty name to prevent automatic output folder creation
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

        # Required for adding bus passenger demand via world.adddemand(..., mode="bus_passenger")
        world.set_bus_handler(BusHandler)

        # Size of the demand is required in observation space.
        world = self._allocate_demand_by_service(world, demand_csv, current_path=self.current_path)
        return world
        
    def _apply_action(self, ) -> None:
        """
        Invoked internally by step 
        - Add stops based on STOP_SPACING.
        - Add necessary vehicles to the world.
            - Add buses based on the current path and SERVICE_FREQUENCY.
        # TODO: This only works on a single path.

        """

        # 1. Determine bus stops based on STOP_SPACING
        # For simplicity, make all nodes in path as stops (can be refined later)
        bus_stops = self.current_path[::self.STOP_SPACING]

        # 2. Right now, only setup to work with single bus route. 
        # Create a single bus and let set_bus_route handle the SERVICE_FREQUENCY
        # TODO: Add more buses if more than 1 route
        bus_name = "bus_route_0"
        departure_time = 0  # First bus starts at time 0
        
        bus = self.world.addVehicle(
            orig=self.current_path[0], 
            dest=self.current_path[1],  # Next node in path
            departure_time=int(departure_time),
            name=bus_name, # unique name for each bus
            mode="bus"
        )

        # 3. Set the bus route - this will create SERVICE_FREQUENCY number of buses:
        buses = bus.set_bus_route(
            path=self.current_path,
            stops=bus_stops,
            is_circular=False, # Do not make routes circular 
            capacity=self.BUS_CAPACITY,
            stop_duration=self.STOP_DURATION,
            service_frequency=self.SERVICE_FREQUENCY, # This creates multiple buses for the frequency
            sim_horizon=self.horizon # Pass simulation horizon for proper bus spacing
        )

        print(f"Original bus '{bus_name}': set_bus_route created additional {len(buses) - 1} buses")
                
        # Print bus summary AFTER all buses are created
        all_buses = [v for v in self.world.VEHICLES.values() if hasattr(v, 'mode') and v.mode == 'bus']
        print(f"\nTotal buses in simulation: {len(all_buses)} - {[b.name for b in all_buses]}")

    def _get_final_metrics(self, handler: BusHandler, current_path_str: list) -> int:
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
            # else:
            #     # Skip buses that haven't started yet (negative operating time) or haven't moved
            #     print(f"  -> Skipping bus {bus.name}: {'not started yet' if total_operating_time <= 0 else 'no distance traveled'}")

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
        
        # for bus_name, utilization_list in bus_utilizations.items():
        #     print(f"\nUtilization for bus: {bus_name}")
        #     print(f"Utilization: {utilization_list}")

        # 3. Travel time distribution (wait time + in-vehicle time)
        # 4. Waiting time distribution (wait_time is only calculated for completed passengers upon alighting)
        wait_time_dstr = []
        movement_time_dstr = []
        travel_time_dstr = [] # wait time + in-vehicle time
        
        total_wait_time = 0.0
        # Handle COMPLETED trips (from passenger_stats, not buses)
        for p_stats in handler.passenger_stats:
            wait_time = p_stats['wait_time']
            total_wait_time += wait_time
            wait_time_dstr.append(wait_time) 
            movement_time_dstr.append(p_stats['in_vehicle_time'])
            travel_time_dstr.append(p_stats['total_travel_time'])
            
        # Handle PARTIAL/ONGOING trips. This loop only looks at passengers who are currently a part of any bus.
        # When a passenger alights at their destination (completing the trip), they are explicitly removed from the bus's passengers list
        on_going_count_sim_end = 0
        for bus in all_buses:
            for passenger in bus.passengers:
                # Passengers with partial trips
                # if passenger.is_on_bus: # Safety check, though it should always be True here
                wait_time = passenger.board_time - passenger.wait_start_time
                total_wait_time += wait_time

                movement_time = self.world.TIME - passenger.board_time
                wait_time_dstr.append(wait_time)
                movement_time_dstr.append(movement_time)
                travel_time_dstr.append(wait_time + movement_time)
                on_going_count_sim_end += 1
                

        # Handle STILL-WAITING (not onboarded). Separately add passengers who are still waiting for the bus to arrive 
        # They only have wait time and have no in-vehicle time; no point in adding travel time)
        total_waiting_time_sim_end = 0.0
        for waiting_list in handler.waiting_passengers.values():
            for passenger in waiting_list:
                if passenger.wait_start_time is not None:
                    wait_time = self.world.TIME - passenger.wait_start_time
                    wait_time_dstr.append(wait_time)
                    total_waiting_time_sim_end += wait_time

        # 5. From above, also waiting passengers at the end of the sim
        waiting_count_sim_end = sum(len(passengers) for passengers in handler.waiting_passengers.values()) # At the end of the sim, how many passengers are still waiting at stops.

        # 6. Completed trips
        completed_count = len(handler.passenger_stats) # This works because the stats are only added when passengers alight the bus.
        
        # 7. Onboarded passengers
        total_onboarded_count = completed_count + on_going_count_sim_end

        return {'avg_bus_speed_mps': avg_bus_speed_mps, 
                'avg_bus_utilization_pct': avg_bus_utilization_pct, 
                'travel_time_dstr': travel_time_dstr,
                'waiting_time_dstr': wait_time_dstr,
                'movement_time_dstr': movement_time_dstr, # movement time
                'waiting_count_sim_end': waiting_count_sim_end, 
                'total_waiting_time_sim_end': total_waiting_time_sim_end,
                'completed_count': completed_count,
                'total_onboarded_count': total_onboarded_count,
                'total_wait_time': total_wait_time}

    def _get_initial_metrics(self, handler: BusHandler, current_path_str: list) -> Dict[str, float]:
        """
        Compute metrics (which can be computed before the sim starts):
        - Route length (meters). Since we use heuristic baselines like shortest path, this should be part of the metrics .
        - Wanting to onboard count (count of passengers whose O-D pairs are served by current route)
        average bus speed, and average bus utilization throughout the journey.
        """

        # 1. Route length 
        route_length = 0.0
        if len(self.current_path) < 2:
            raise ValueError("Current path must have at least 2 nodes")
        
        for i in range(len(self.current_path) - 1):
            route_length += self.link_lengths[(str(self.current_path[i]), str(self.current_path[i+1]))]
        
        # 2. Wanting to onboard 
        wanting_to_onboard = 0
        # Counting all potential passengers whose O-D is served by the route.
        for passenger in handler.pending_passengers:
            if self._is_od_served(passenger.origin_stop.name, passenger.dest_stop.name, current_path_str):
                wanting_to_onboard += 1

        return {'route_length': route_length, 'wanting_to_onboard': wanting_to_onboard}

    def _step_until(self, until_t: int, print_metrics: bool = True) -> Dict[str, int]:
        """
        Run the simulation until the given time.

        Notes: 
        - Pending passengers → passengers that have been created by demand but not yet boarded a bus (start time has not arrived)
        - Waiting passengers → passengers that have been created by demand but not yet boarded a bus (waiting for a bus to arrive)
        - Onboarded passengers → passengers that have been boarded a bus and are currently traveling on it
        - Completed passengers → passengers that have reached their destination and have completed their trip

        - All time metrics are converted to minutes.
        - Data to be collected to plot Distributions: 
            - Waiting time 
                - distribution can be used to determine the quality of waiting time (how equitable is the distribution)
                - passengers who completed trips + currently traveling + still waiting at sim end.
            - Travel time 
                - passengers who completed trips + currently traveling in buses
        - Metrics should relate to performance of the route.
        """

        handler = self.world.bus_handler
        current_path_str = [str(node) for node in self.current_path]

        # Some metrics need to be collected at the begining of the sim? Like wanting to onboard.
        # Collect initial wanting count before simulation moves passengers
        initial_metrics = self._get_initial_metrics(handler, current_path_str)

        self.world.exec_simulation(until_t=until_t)

        final_metrics = self._get_final_metrics(handler, current_path_str)

        # Calculation section 
        total_wait_time_minutes = (1/60) * final_metrics['total_wait_time'] # How much time did the passengers who completed the trips, or who are currently traveling, spent waiting for the bus to arrive.
        wait_time_sim_end_minutes = (1/60) * final_metrics['total_waiting_time_sim_end'] # At the end of the simulation, how much time did the passengers who are still waiting at stops, spent waiting for the bus to arrive.
        movement_time_minutes = (1/60) * sum(final_metrics['movement_time_dstr']) # For trip completion how much time was spent in-vehicle for passengers who completed the trip + still in the bus at the end of sim.
        total_travel_time_minutes = (1/60) * sum(final_metrics['travel_time_dstr']) # Movement time + waiting time for passengers who completed the trip + still in the bus at the end of sim .
        
        # Service rate calculation (used in both metrics dict and print statements)
        service_rate_pct = 100 * (final_metrics['completed_count'] / initial_metrics['wanting_to_onboard']) if initial_metrics['wanting_to_onboard'] > 0 else 0.0 # What % of demand was fulfilled.
        
        # Per-passenger averages (used in print statements)
        total_combined_wait_time = total_wait_time_minutes + wait_time_sim_end_minutes
        avg_wait_time_minutes = total_combined_wait_time / len(final_metrics['waiting_time_dstr']) if len(final_metrics['waiting_time_dstr']) > 0 else 0
        avg_movement_time_minutes = movement_time_minutes / len(final_metrics['movement_time_dstr']) if len(final_metrics['movement_time_dstr']) > 0 else 0
        avg_travel_time_minutes = sum(final_metrics['travel_time_dstr']) / len(final_metrics['travel_time_dstr']) / 60 if len(final_metrics['travel_time_dstr']) > 0 else 0
        
        # Onboarding rate calculation
        onboard_rate_pct = (final_metrics['total_onboarded_count'] / initial_metrics['wanting_to_onboard'] * 100) if initial_metrics['wanting_to_onboard'] > 0 else 0
        
        # Route efficiency calculation (passengers completed per km of route)
        route_efficiency_passengers_per_km = (final_metrics['completed_count'] / initial_metrics['route_length'] * 1000) if initial_metrics['route_length'] > 0 else 0.0

        metrics = {
            # Waiting metrics (all times in seconds for consistency).
            'total_wait_time': final_metrics['total_wait_time'], # Total wait time for completed/traveling passengers (seconds)
            'wait_time_sim_end': final_metrics['total_waiting_time_sim_end'], # Wait time for passengers still waiting at sim end (seconds)
            'sim_end_waiting_passengers_count': final_metrics['waiting_count_sim_end'], # Count of passengers still waiting at sim end
            'avg_wait_time': avg_wait_time_minutes * 60, # Average wait time per passenger (seconds)

            # Travel metrics (all times in seconds for consistency).
            'wanting_to_onboard': initial_metrics['wanting_to_onboard'], # Count of passengers whose O-D is served by route
            'total_onboarded_count': final_metrics['total_onboarded_count'], # Count of passengers who boarded buses
            'completed_trip_passengers_count': final_metrics['completed_count'], # Count of passengers who completed trips
            'movement_time': sum(final_metrics['movement_time_dstr']), # Total in-vehicle time for completed/traveling passengers (seconds)
            'total_travel_time': sum(final_metrics['travel_time_dstr']), # Total travel time for completed/traveling passengers (seconds)

            # Data collected (raw distributions in seconds).
            'waiting_time_dstr': final_metrics['waiting_time_dstr'], # Waiting time distribution for all passengers (seconds)
            'movement_time_dstr': final_metrics['movement_time_dstr'], # In-vehicle time distribution for completed/traveling passengers (seconds)
            'travel_time_dstr': final_metrics['travel_time_dstr'], # Total travel time distribution for completed/traveling passengers (seconds)

            # Route metrics.
            'route_length': initial_metrics['route_length'], # Route length (meters)
            'bus_utilization': final_metrics['avg_bus_utilization_pct'], # Average bus utilization (percentage)
            'average_bus_speed': final_metrics['avg_bus_speed_mps'], # Average bus speed (meters/second)
            'service_rate': service_rate_pct, # Percentage of demand fulfilled (completed/wanting)
            'onboard_rate': onboard_rate_pct, # Percentage of demand that boarded buses (onboarded/wanting)
            'route_efficiency': route_efficiency_passengers_per_km, # Passengers completed per km of route (passengers/km)
        }

        # pending_count = len(handler.pending_passengers)
        # end_of_sim_onboarded = final_metrics['total_onboarded_count'] - final_metrics['completed_count'] # Still on board.
        # total_validation = pending_count + final_metrics['waiting_count_sim_end'] + end_of_sim_onboarded + final_metrics['completed_count']
        # # Validation (wanting to onboard at the start of the sim) must be = (pending + waiting + onboarded + completed) at the end of the sim.
        # print("Validation:")
        # print(f"\tWanting to onboard at the start of the sim: {initial_metrics['wanting_to_onboard']}")
        # print(f"\tPending at end of sim: {pending_count}")
        # print(f"\tWaiting at end of sim: {final_metrics['waiting_count_sim_end']}")
        # print(f"\tOnboarded at end of sim: {end_of_sim_onboarded}")
        # print(f"\tCompleted at end of sim: {final_metrics['completed_count']}")
        # print(f"\tTotal (pending + waiting + onboarded + completed): {total_validation}")

        if print_metrics:
            print("\n" + "="*70)
            print("SIMULATION METRICS")
            print("="*70)
            
            # Route Information
            print("\nROUTE INFORMATION:")
            print(f"   Route Path:               {' → '.join(current_path_str)}")
            print(f"   Route Length:             {metrics['route_length']/1000:.2f} km")
            print(f"   Average Bus Speed:        {metrics['average_bus_speed']:.2f} m/s ({metrics['average_bus_speed']*3.6:.1f} km/h)")
            print(f"   Bus Utilization:          {metrics['bus_utilization']:.1f}%")
            
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

    def compute_reward(self, sim_result: Dict[str, int]) -> float:
        """
        Passenger Travel Efficiency Focused 
    
        reward = β₀ × service_rate - β₁ × (avg_wait_time / max_wait_time) + β₂ × route_efficiency + β₃ × bus_utilization
        
        where, components:
        - service_rate: % of passengers who completed their trips (0-100)
            - Rewards routes that actually complete passenger trips (encourages connecting high-demand O-D pairs)
            - Prevents low-demand route hacking
        - avg_wait_time: Average waiting time per passenger (seconds)
            - Penalizes routes with long passenger wait times (improves passenger experience quality)
            - max_wait_time: Normalization constant for wait time penalty (1800s = 30min)
        - route_efficiency: Passengers served per km of route (passengers/km)
            - Prevents wastefully long routes with few passengers (encourages compact, high-demand route design, prevents arbitrarily long routes)
        - bus_utilization: Average bus capacity utilization (0-100%)
            - Forces agent to choose routes with actual demand, preventing gaming via low-demand routes
        
        - Units are normalized: 
            - service_rate: [0,100] → [0,1] (divide by 100)
            - avg_wait_time: [0,1800s] → [0,1] (normalize by max_wait_time = 30min)
            - route_efficiency: [0,max_route_efficiency pax/km] → [0,1] (normalize by max_route_efficiency, capped at 1.0)  
            - bus_utilization: [0,100%] → [0,1] (divide by 100)
            - All components now balanced on [0,1] scale for proper β coefficient weighting
        
        ---------
        On normalizing the reward: 
        - Applying the Welford Normalization to the returns, not the absolute reward values.
        - Normalizing raw rewards can be a problem, example: 
            - Episode 1: Total travel time = 1000s → reward = -1000
            - Episode 100: Total travel time = 500s → reward = -500
            - Without normalization: Clear improvement! (-500 > -1000)
            - With normalization: Both might map to ~0 (relative to running mean)
            - The agent can't tell it's improving!
        ---------
        Potential pitfalls: 
        1. If the rewards are too small like 0.0001, then gradients are too small to be effective.
           This can happen because: 
            - Route lengths are un-normalized (can be large)
            - Total travel time can be large
            - Since bus capacity is relatively low (~40) it can only serve a small number of passengers.

        2. An improper reward formulation could lead to reward hacking: 
           Example: 
            - Maximize reward by selecting nodes with low demand so the average travel time is low.
           How to prevent: 
            - Add an utilization component
        ---------
        - Since this reward is for each node extended, instead of the passengers served, we can possibly look at only new passengers served?
        
        """
        # Units normalized. 
        
        # Reward coefficients (β parameters) - all work on [0,1] normalized inputs
        beta0 = 50.0     # Service rate importance (primary)
        beta1 = 30.0     # Wait time penalty strength 
        beta2 = 20.0     # Route efficiency bonus 
        beta3 = 25.0     # Bus utilization bonus (prevent reward hacking)
        
        # Normalization constants - TODO: These should be data-driven for the specific network
        max_wait_time = 1800.0  # 30 minutes - based on transit service standards (should analyze actual wait time distribution)
        max_route_efficiency = 30.0  # 20 pax/km - rough estimate (should analyze demand density and bus capacity in the network)
        # 40 passenger bus capacity ÷ ~2 km average route length = 20 pax/km

        # Extract metrics from simulation results  
        service_rate = sim_result['service_rate']  # In percentage [0-100]
        avg_wait_time = sim_result['avg_wait_time']  # In seconds
        route_efficiency = sim_result['route_efficiency']  # Passengers per km
        bus_utilization = sim_result['bus_utilization']  # In percentage [0-100]
        
        # Normalize all components to [0-1] scale for balanced reward components
        service_rate_norm = service_rate / 100.0  
        wait_time_norm = avg_wait_time / max_wait_time  
        route_efficiency_norm = min(route_efficiency / max_route_efficiency, 1.0)
        bus_utilization_norm = bus_utilization / 100.0  # [0-100%] → [0-1]  
        
        # Calculate reward components (all β coefficients now work on [0-1] normalized inputs)
        service_component = beta0 * service_rate_norm
        wait_time_penalty = beta1 * wait_time_norm  
        efficiency_component = beta2 * route_efficiency_norm
        utilization_component = beta3 * bus_utilization_norm  # Prevents low-demand route hacking
        
        # Total reward calculation
        total_reward = service_component - wait_time_penalty + efficiency_component + utilization_component
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"\tService rate component: {service_component:.2f} (β₀={beta0} × {service_rate:.1f}%)")
        print(f"\tWait time penalty: -{wait_time_penalty:.2f} (β₁={beta1} × {avg_wait_time:.0f}s/{max_wait_time:.0f}s)")
        print(f"\tRoute efficiency component: {efficiency_component:.2f} (β₂={beta2} × {route_efficiency:.2f} pax/km)")
        print(f"\tBus utilization component: {utilization_component:.2f} (β₃={beta3} × {bus_utilization:.1f}%)")
        
        return total_reward    

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Return (obs, reward, terminated, info).
        Run the simulation on the current route and get metrics.

        Episode termination conditions: 
            - Max path length reached. Max path length is also the number of steps in the episode.   
        """

        # 1. extend the path first
        action_node = self.idx_to_node[action]
        self.current_path = [str(node) for node in self.current_path] + [action_node]
        print(f"Current path: {self.current_path}")

        # 2. Build_world needs to happen every step.
        # i.e., add the network and the classified demand (bus vs car).
        self.world = self.build_world(self.config.get("network"))
        
        # 3. spawn necessary buses, and set their routes.
        self._apply_action()
        
        # 4. Run the full simulation upto horizon end.
        sim_result = self._step_until(self.horizon)
        
        # 5. Compute reward
        reward = self.compute_reward(sim_result)
        
        # 6. Check termination
        terminated = len(self.current_path) >= self.MAX_PATH_LENGTH

        return self._get_state(), reward, terminated, {'sim_result': sim_result}
    
    def _get_valid_indices(self) -> list:
        """
        For a given frontier node, get the indices of valid next nodes.
        - When this is called, the action has not yet been applied i.e. current_path does not have the action attached.
        """
        frontier = self.current_path[-1]
        path_set = set(self.current_path)  # O(1) lookup

        # 1. Get all neighbors of frontier (with safe lookup)
        valid_neighbors = self.adj.get(frontier, set()) 
        
        # 2. Remove nodes that are already in the path
        valid_neighbors = valid_neighbors - path_set  

        # 3. Get the indices of valid next nodes
        valid_indices = [self.node_to_idx[node] for node in valid_neighbors]
        return valid_indices

    def render(self, save_dir: str, render_name: str) -> None:
        """
        - Visualize network + path.
        - Episode simulation gif.
        """
        output_loc = os.path.join(os.path.join(save_dir, "images"), render_name)
        plot_network_demand_and_path(self.world, self.current_path, output_loc)

