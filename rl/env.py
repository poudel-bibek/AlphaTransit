import os
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from uxsim import World
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from uxsim.BusHandler import BusHandler
from rl.env_utils import plot_network_and_demand, plot_network_demand_and_path
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
        now = datetime.now()
        self.training_save_dir = f"./training_data/{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}"

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
        nodes_csv = self.network_dir / f"{self.config.get('network')}_nodes.csv"
        links_csv = self.network_dir / f"{self.config.get('network')}_links.csv"
        demand_csv = self.network_dir / f"{self.config.get('network')}_demand.csv"

        df_nodes = pd.read_csv(nodes_csv, dtype={"name": str})
        df_links = pd.read_csv(links_csv, dtype={"name": str, "start": str, "end": str})
        df_demand = pd.read_csv(demand_csv, dtype={"orig": str, "dest": str})

        # Sort node names numerically, even though they are strings
        self.node_list = sorted(list(df_nodes["name"].unique()), key=lambda x: int(x))
        self.n_nodes = len(self.node_list)
        self.n_edges = int(len(df_links))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}

        # Compute adjacency matrix (assuming directed graph):
        self.adj = defaultdict(set)
        for _, row in df_links.iterrows():
            # Force to strings to match node identifiers
            x, y = str(row["start"]), str(row["end"])
            self.adj[x].add(y)
        
        # Demand matrix: 
        self.od_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for _, row in df_demand.iterrows():
            if row["orig"] in self.node_to_idx and row["dest"] in self.node_to_idx:
                i = self.node_to_idx[row["orig"]]
                j = self.node_to_idx[row["dest"]]
                self.od_matrix[i, j] += row["q"]

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
        self.min_free_flow_speed = df_links["u"].min() # u = free_flow_speed
        self.max_free_flow_speed = df_links["u"].max()
        
        # Edge index and features:
        edge_index_list = []
        edge_attr_list = []
        for _, row in df_links.iterrows():
            x, y = str(row['start']), str(row['end'])
            if x in self.node_to_idx and y in self.node_to_idx:
                i, j = self.node_to_idx[x], self.node_to_idx[y]
                edge_index_list.append([i, j]) # Add the i, j indices of the edge

                # Now edge attributes: Normalize to [0,1] and add
                length_norm = (row['length'] - self.min_length) / (self.max_length - self.min_length) 
                speed_norm = (row['u'] - self.min_free_flow_speed) / (self.max_free_flow_speed - self.min_free_flow_speed) 
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
        demand_csv = self.network_dir / f"{self.config.get('network')}_demand.csv"
        # Read as strings so node names remain consistent with the World (nodes are named as strings)
        demand_df = pd.read_csv(demand_csv, dtype={"orig": str, "dest": str})

        if use_random:
            choice = random.choice(list(demand_df["orig"].unique()))
            print(f"Initializing route randomly at node: {choice}")
            return [choice]
            
        else: 
            demand_df = demand_df.groupby("orig").sum(numeric_only=True).reset_index()
            highest_demand_node = demand_df.loc[demand_df["q"].idxmax()]
            choice = highest_demand_node["orig"]
            print(f"Initializing route at node: {choice}")
            return [choice]

    def _allocate_demand_by_service(self, world: World, demand_csv: str, current_path: list = None) -> None:
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
        The demand csv expects: 
            - Mandatory: orig, dest, start_t, end_t, mode (its optional in sim; mandatory in the RL env to split with alpha) 
            - Either one: Volume or per-second flow rate (q) vehicles/s
        """
        # IMPORTANT: ensure node identifiers are strings to match World node names.
        demand_df = pd.read_csv(demand_csv, dtype={"orig": str, "dest": str})
        print(f"Loading {len(demand_df)} demand records...")
        
        # current_path will have at least one node (where it starts) i.e., it wont have O-D pair then.
        current_path_str = [str(node) for node in current_path] # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
        total_demand = 0 
        bus_demand = 0 
        car_demand = 0 
        for _, row in demand_df.iterrows():
            
            orig, dest = str(row["orig"]), str(row["dest"])
            start_t, end_t, flow_rate = row["start_t"], row["end_t"], row["q"]
            if self._is_od_served(orig, dest, current_path_str):
                bus_demand += flow_rate * self.alpha
                car_demand += flow_rate * (1 - self.alpha)
                total_demand += flow_rate

                # Add demand to the world:
                world.adddemand(orig=orig, 
                                dest=dest, 
                                t_start=start_t, 
                                t_end=end_t, 
                                flow=flow_rate * (1 - self.alpha), 
                                mode="vehicle")

                world.adddemand(orig=orig, 
                                dest=dest, 
                                t_start=start_t, 
                                t_end=end_t, 
                                flow=flow_rate * self.alpha, 
                                mode="bus_passenger")
            else: 
                car_demand += flow_rate
                total_demand += flow_rate

                # Add demand to the world:
                world.adddemand(orig=orig, 
                                dest=dest, 
                                t_start=start_t, 
                                t_end=end_t, 
                                flow=flow_rate, 
                                mode="vehicle")

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
        valid_neighbors = self.adj[frontier] - path_set  # Set difference
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
        
        img_dir = os.path.join(self.training_save_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Optional: Build temp world for initial visualization only (not persistent)
        temp_world = self.build_world(self.config.get("network"))
        output_path = os.path.join(img_dir, f"00_{self.config.get('network')}_demand_network.png")

        plot_network_and_demand(temp_world, output_path) # Visualize only after building world

        # initial state
        state = self._get_state()

        return state, {}
    
    def build_world(self, network: str) -> World:
        """
        generate nodes from csv: 
        - expects: name, x, y
        generate links from csv: 
        - expects: link_name, start, end, length, free_flow_speed (u), jam_density (kappa), merge_priority
        """
        if not network:
            raise ValueError("Network name must be provided in config['network']")

        nodes_csv = self.network_dir / f"{network}_nodes.csv"
        links_csv = self.network_dir / f"{network}_links.csv"
        demand_csv = self.network_dir / f"{network}_demand.csv"

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
        world.generate_Links_from_csv(str(links_csv))

        # Required for adding bus passenger demand via world.adddemand(..., mode="bus_passenger")
        world.set_bus_handler(BusHandler)

        # Size of the demand is required in observation space.
        world = self._allocate_demand_by_service(world, demand_csv, current_path=self.current_path)
        return world
        
    def _apply_action(self, action: str) -> None:
        """
        Invoked internally by step prior to state transition.
        - Based on the action (new node selected), increase the route path by one hop.
        - Add stops based on STOP_SPACING.

        Add necessary vehicles to the world.
        - Add buses based on the current path and SERVICE_FREQUENCY.
        # TODO: This only works on a single path.

        """

        # 1. Create bus route based on current path
        self.current_path = [str(node) for node in self.current_path] + [action]
        
        # 2. Determine bus stops based on STOP_SPACING
        # For simplicity, make all nodes in path as stops (can be refined later)
        bus_stops = self.current_path[::self.STOP_SPACING]

        # Calculate headway (time between buses) based on SERVICE_FREQUENCY
        # SERVICE_FREQUENCY is buses per hour, so:
        # - If SERVICE_FREQUENCY = 1: headway = 3600s (one bus for the whole hour)
        # - If SERVICE_FREQUENCY = 2: headway = 1800s (buses at 0s and 1800s)  
        # - If SERVICE_FREQUENCY = 6: headway = 600s (buses every 10 minutes)
        headway_seconds = 3600 / self.SERVICE_FREQUENCY
        
        # Calculate how many buses we need to spawn during the simulation
        # For a 1-hour simulation with SERVICE_FREQUENCY=6, this gives us 6 buses
        # For a 2-hour simulation with SERVICE_FREQUENCY=6, this gives us 12 buses
        num_buses_to_spawn = int(self.horizon / headway_seconds)
        print(f"Spawning {num_buses_to_spawn} buses")
        
        # Spawn buses at regular intervals throughout the simulation
        for bus_index in range(num_buses_to_spawn):
            # Calculate when this bus should start
            departure_time = bus_index * headway_seconds
            
            bus_name = f"bus_route_{bus_index}"
            # Only spawn if within simulation horizon
            if departure_time < self.horizon:
                # Create bus with staggered departure time
                bus = self.world.addVehicle(
                    orig=self.current_path[0], 
                    dest=self.current_path[1],  # Next node in path
                    departure_time=int(departure_time),
                    name=bus_name, # unique name for each bus
                    mode="bus"
                )

                # Set the bus route:
                bus.set_bus_route(
                    path=self.current_path,
                    stops=bus_stops,
                    is_circular=False, # Do not make routes circular 
                    capacity=self.BUS_CAPACITY,
                    stop_duration=self.STOP_DURATION,
                    service_frequency=self.SERVICE_FREQUENCY, # Still pass this for record-keeping
                )
        
        # Print bus summary AFTER all buses are created
        buses = [v for v in self.world.VEHICLES.values() if hasattr(v, 'mode') and v.mode == 'bus']
        print(f"\nBuses in simulation: {len(buses)} - {[b.name for b in buses]}")

    def _step_until(self, until_t: int, print_metrics: bool = True) -> Dict[str, int]:
        """
        Run the simulation until the given time.
        Total travel time consists of passengers currently in bus still traveling as well as the ones who completed.
        """

        self.world.exec_simulation(until_t=until_t)
        # self.world.analyzer.get_metrics()

        # Only gather passenger specific metrics here:
        metrics = {
            'total_passengers_wanting_to_onboard': 0,
            'total_passengers_onboarded': 0, 
            'total_passengers_waiting_at_stops': 0,
            'total_passengers_completed_trip': 0,
            'total_wait_time': 0.0,
            'total_movement_time': 0.0, 
            'total_travel_time': 0.0,
        }

        handler = self.world.bus_handler
            
        # 1. Count passengers by current state
        pending_count = len(handler.pending_passengers)
        waiting_count = sum(len(passengers) for passengers in handler.waiting_passengers.values())
        
        # Count passengers currently on buses
        onboard_count = 0
        current_onboard_travel_time = 0.0  # Time spent by current passengers
        
        for vehicle in self.world.VEHICLES.values():
            if hasattr(vehicle, 'mode') and vehicle.mode == 'bus':
                if hasattr(vehicle, 'passengers'):
                    onboard_count += len(vehicle.passengers)
                    # Add partial travel time for current passengers
                    for passenger in vehicle.passengers:
                        if hasattr(passenger, 'board_time') and passenger.board_time is not None:
                            current_onboard_travel_time += (self.world.TIME - passenger.board_time)
        
        completed_count = len(handler.passenger_stats)
        
        # 2. Calculate time metrics from completed passengers
        completed_wait_time = sum(p['wait_time'] for p in handler.passenger_stats)
        completed_movement_time = sum(p['in_vehicle_time'] for p in handler.passenger_stats) 
        completed_total_time = sum(p['total_travel_time'] for p in handler.passenger_stats)
        
        # 3. Fill metrics
        metrics.update({
            'total_passengers_wanting_to_onboard': pending_count + waiting_count,
            'total_passengers_onboarded': onboard_count,
            'total_passengers_waiting_at_stops': waiting_count,
            'total_passengers_completed_trip': completed_count,
            'total_wait_time': completed_wait_time,
            'total_movement_time': completed_movement_time + current_onboard_travel_time,
            'total_travel_time': completed_total_time + current_onboard_travel_time,
            'time_s': int(self.world.TIME)
        })
        
        if print_metrics:
            print("\n" + "="*50)
            print("SIMULATION METRICS")
            print("="*50)
            print(f"Simulation Time:          {metrics['time_s']:,}s")
            print(f"Passengers Wanting Bus:   {metrics['total_passengers_wanting_to_onboard']:,}")
            print(f"Passengers Onboarded:     {metrics['total_passengers_onboarded']:,}")
            print(f"Passengers Waiting:       {metrics['total_passengers_waiting_at_stops']:,}")
            print(f"Passengers Completed:     {metrics['total_passengers_completed_trip']:,}")
            print(f"Total Wait Time:          {metrics['total_wait_time']:,.1f}s")
            print(f"Total Movement Time:      {metrics['total_movement_time']:,.1f}s")
            print(f"Total Travel Time:        {metrics['total_travel_time']:,.1f}s")
            print("="*50)

        # self.world.bus_handler.print_bus_activity_history()
        return metrics

    def compute_reward(self, sim_result: Dict[str, int]) -> float:
        """
        Components: 
        1.Travel efficiency: 
        - First priority on what we are optimizing for. 
        - Minimize total travel time across all passengers. 
            - Do not use average travel time. This could lead to reward hacking i.e., agent may connect nodes with low demand.
            - Using total travel time also accounts for ``demand-weighted efficiency''.

        2. Further reward hacking prevention: 
        - Add a utilization component to the reward
        - Intuition: good routes cover high-demand O-D pairs 
        - Set a minimum total demand that must be served by the route?

        """
 
        # 1. Extract travel time data and compute component 1: travel efficiency
        # The negative of total travel time (minimize travel time = maximize negative travel time)
        reward_component_1 = -sim_result['total_travel_time']
        
        total_reward = reward_component_1

        print(f"Total reward: {total_reward}: \n\tComponent 1: {reward_component_1}")
        return total_reward

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Return (obs, reward, terminated, truncated, info).
        Run the simulation on the current route and get metrics.

        Episode termination conditions: 
            - Max path length reached. Max path length is also the number of steps in the episode.
        Episode truncation conditions: 
            - Gets stuck in a dead-end i.e., had only 1 neighbor, (which is already in the path)      
        """
        
        # 1. Check truncation (before action is applied)
        truncated = False
        valid_indices = self._get_valid_indices()
        if len(valid_indices) == 0:
            truncated = True
            print(f" ❌ Invalid: Action {action} is a dead-end. No valid next nodes.")
            # Immediately return with truncated=True, reward=0, terminated=False
            truncation_penalty = -100.0
            return self._get_state(), truncation_penalty, False, truncated, {}

        # 2. Build_world needs to happen every step.
        # i.e., add the network and the classified demand (bus vs car).
        self.world = self.build_world(self.config.get("network"))
        
        # 3. Add action to current_path, spawn necessary buses, and set their routes.
        action = self.idx_to_node[action]
        self._apply_action(action)
        print(f"Current path: {self.current_path}")
        
        # 4. Run the full simulation upto horizon end.
        sim_result = self._step_until(self.horizon)
        
        # 5. Compute reward
        reward = self.compute_reward(sim_result)
        
        # 6. Check termination
        terminated = len(self.current_path) >= self.MAX_PATH_LENGTH

        return self._get_state(), reward, terminated, truncated, {}
    
    def _get_valid_indices(self) -> list:
        """
        For a given frontier node, get the indices of valid next nodes.
        - When this is called, the action has not yet been applied i.e. current_path does not have the action attached.
        """
        frontier = self.current_path[-1]
        path_set = set(self.current_path)  # O(1) lookup

        # 1. Get all neighbors of frontier
        valid_neighbors = self.adj[frontier] 
        
        # 2. Remove nodes that are already in the path
        valid_neighbors = valid_neighbors - path_set  

        # 3. Get the indices of valid next nodes
        valid_indices = [self.node_to_idx[node] for node in valid_neighbors]
        return valid_indices

    def render(self, render_name: str) -> None:
        """
        - Visualize network + path.
        - Episode simulation gif.
        """
        img_dir = os.path.join(self.training_save_dir, "images")
        output_loc = os.path.join(img_dir, render_name)
        plot_network_demand_and_path(self.world, self.current_path, output_loc)

