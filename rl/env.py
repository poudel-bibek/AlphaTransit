import random
import numpy as np
import pandas as pd
import gymnasium as gym
from uxsim import World
from pathlib import Path
from collections import defaultdict
from uxsim.BusHandler import BusHandler
from rl.env_utils import plot_network_and_demand
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
        self.alpha = self.config.get("alpha")
        self.radius = self.config.get("radius")
        self.random_path_init = self.config.get("random_path_init")
        
        # Constraints:
        self.MAX_PATH_LENGTH = self.config.get("max_path_length")
        self.MIN_PATH_LENGTH = self.config.get("min_path_length")
        self.world, self.current_path = None, []

        # Lookup dicts for efficient queries:
        nodes_csv = self.network_dir / f"{self.config.get('network')}_nodes.csv"
        links_csv = self.network_dir / f"{self.config.get('network')}_links.csv"
        demand_csv = self.network_dir / f"{self.config.get('network')}_demand.csv"

        df_nodes = pd.read_csv(nodes_csv, dtype={"name": str})
        df_links = pd.read_csv(links_csv, dtype={"name": str})
        df_demand = pd.read_csv(demand_csv, dtype={"orig": str, "dest": str})

        # Sort node names numerically, even though they are strings
        self.node_list = sorted(list(df_nodes["name"].unique()), key=lambda x: int(x))
        self.n_nodes = len(self.node_list)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}

        print(f"Number of nodes: {self.n_nodes}")
        print(f"Node list: {self.node_list}")
        print(f"Node to idx: {self.node_to_idx}")
        print(f"Idx to node: {self.idx_to_node}")

        # Compute adjacency matrix (assuming directed graph):
        self.adj = defaultdict(set)
        for _, row in df_links.iterrows():
            x, y = row["start"], row["end"]
            self.adj[x].add(y)
        print(f"\nAdjacency matrix: {self.adj}")

        # Demand matrix: 
        self.od_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for _, row in df_demand.iterrows():
            if row["orig"] in self.node_to_idx and row["dest"] in self.node_to_idx:
                i = self.node_to_idx[row["orig"]]
                j = self.node_to_idx[row["dest"]]
                self.od_matrix[i, j] += row["q"]
        print(f"\nOD matrix: {self.od_matrix}")

        # Demand vectors:
        demand_out = self.od_matrix.sum(axis=1)
        demand_in = self.od_matrix.sum(axis=0)
        print(f"\nDemand out: {demand_out}")
        print(f"Demand in: {demand_in}")

        # Normalization constants: 
        self.max_demand = max(demand_out.max(), demand_in.max()) if demand_out.size > 0 else 1.0
        self.min_demand = min(demand_out.min(), demand_in.min()) if demand_out.size > 0 else 0.0
        print(f"Max demand: {self.max_demand}, Min demand: {self.min_demand}")

        self.max_degree = max(len(neighbors) for neighbors in self.adj.values()) if self.adj else 0
        self.min_degree = min(len(neighbors) for neighbors in self.adj.values()) if self.adj else 0
        print(f"Max degree: {self.max_degree}, Min degree: {self.min_degree}")

        self.min_x = df_nodes["x"].min()
        self.max_x = df_nodes["x"].max()
        self.min_y = df_nodes["y"].min()
        self.max_y = df_nodes["y"].max()
        self.min_length = df_links["length"].min()
        self.max_length = df_links["length"].max()
        self.min_free_flow_speed = df_links["u"].min() # u = free_flow_speed
        self.max_free_flow_speed = df_links["u"].max()
        self.min_jam_density = df_links["kappa"].min() # kappa = jam_density
        self.max_jam_density = df_links["kappa"].max()
        
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
        1. For each node, in the network, following features: 
            - coordinates (x, y) - min-max normalized
            - degree - divide by max degree in the network
            - in_path - binary
            - d_out: sum of all O-D flows emanating from node i  - divide by max demand in the network
            - d_in: sum of all O-D flows arriving at node i - divide by max demand in the network
            - d_out_path: sum of all O-D flows emanating from node i to nodes within the path - divide by max demand in the network   
                - "If I connect i to the path, how many of i's trips could be served by the current path?"
            - d_in_path: sum of all O-D flows arriving at node i from nodes within the path - divide by max demand in the network
                - "How much demand would ride from existing path nodes to i if i becomes served?"
        2. Action mask (to inform the policy about the feasible set):
            - For each node, binary i.e. 0 = allowed action, 1 = not allowed action.
            - Not allowed actions:
                - If node is already in the path.
            - This is not an intrinsic property of the node, it's a constraint.
        3. Edge index: 
            - Directed edges (start -> end) as indices into nodes.
        4. Steps left until max_route_length is reached.
        
        Notes: 
        - d_out_path and d_in_path are "path-aware" demand service vectors
        - How to enforce the action mask?
            - I could enforece it simulation level i.e., by just disallowing the action (env rejection) but this is not a good idea for several reasons.
                1. Hurts learning stability: 
                - PPO computes log_probs from policy's distribution, if samplles are rejected (and env substitutes the action for another) then the log_probs are not valid.

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

            - So the approriate way to enforce is: 
                - Keep action space fixed: Discrete(N) where N is the number of nodes in the network.
                - Use action mask in the observation to inform the policy about the feasible set.
                - When constructing a categorical distribution from the policy output, mask out the logits related to invalid actions i.e., logits[invalid] = -inf (or a large negative)
                    - Lets call this a masked distribution.
                - So that invalid actions are not sampled
                - Compute log probabilities from the masked distribution. Use the masked distribution in the act, evaluate as necessary.
                - Print a warning if invalid action slips through (In practice, with proper masking, this shouldn't happen and I dont expect this, but due to incorrect implementation, it could happen).
        """

        return gym.spaces.Box(low=0, high=1, shape=(1,))

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

    def _allocate_demand_by_service(self, world: World, demand_csv: str, current_path: list = None, alpha: float = 0.3, radius: float = 0.5) -> None:
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
            # Coerce to strings explicitly to align with node names in World
            orig, dest = str(row["orig"]), str(row["dest"])
            start_t, end_t, flow_rate = row["start_t"], row["end_t"], row["q"]
            if self._is_od_served(orig, dest, current_path_str):
                bus_demand += flow_rate * alpha
                car_demand += flow_rate * (1 - alpha)
                total_demand += flow_rate

                # Add demand to the world:
                world.adddemand(orig=orig, 
                                dest=dest, 
                                t_start=start_t, 
                                t_end=end_t, 
                                flow=flow_rate * (1 - alpha), 
                                mode="vehicle")

                world.adddemand(orig=orig, 
                                dest=dest, 
                                t_start=start_t, 
                                t_end=end_t, 
                                flow=flow_rate * alpha, 
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
        # Print demand split stats:

        return world

    def _is_od_served(self, orig: str, dest: str, current_path_str: list) -> bool:
        """
        Check if both items in the OD pair are served by the route.
        TODO: implement node radius within the route. Current version: nodes lies in the route.
        """
        if len(current_path_str) < 2: 
            return False
        return orig in current_path_str and dest in current_path_str

    def _get_state(self) -> Dict[str, Any]:
        """
        Build observation state as a dict with:

        """
        pass
    

    def reset( self, ) -> None:
        """
        """

        self.current_path = self._initialize_current_path(use_random=self.random_path_init)
        self.world = self.build_world(self.config.get("network"))

        self.n_nodes, self.n_links = len(self.world.NODES), len(self.world.LINKS)
        plot_network_and_demand(self.world, f"{self.world.name}_demand_network.png") # Visualize after building world

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
        world = self._allocate_demand_by_service(world, demand_csv, current_path=self.current_path, alpha=self.alpha, radius=self.radius)
        return world
        
    def apply_action(self, action: Any) -> None:
        """
        Invoked internally by step prior to state transition.
        Based on the node selected, increase the route path by one hop.
        """
        pass
    
    def _apply_action_mask(self,) -> None:
        """
        """

        MASK_VALUE = -1e10 # large negative 
        pass
    
    def compute_reward(self) -> float:
        """
        """
        pass

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Return (obs, reward, terminated, truncated, info).
        """
        pass

    def close(self) -> None:
        """
        Terminal conditions: 
        - Max path length reached.
        """
        pass

    def render(self) -> None:
        """
        """
        pass

