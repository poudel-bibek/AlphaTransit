from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from uxsim import World
from uxsim.BusHandler import BusHandler
from rl.env_utils import plot_network_and_demand

class TransitEnv(gym.Env):
    """
    Transit environment for RL.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        The order of operations performed in the constructor matters. 
        Be careful while modifying.
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

        
    @property
    def observation_space(self) -> gym.Space:
        """
        State (going to be network dependent): 
        - The network graph: 
            - Nodes with node features (x, y)
            - Links with link features (length, free_flow_speed)
            - Demand with demand features (orig, dest, start_t, end_t, q)
        - Current path (partial path so far)
            - Path nodes
            - Could this be encoded somehow? How to represent this?

        """

        return gym.spaces.Box(low=0, high=1, shape=(1,))

    @property
    def action_space(self) -> gym.Space:
        """
        Select the next node to add to the path.
        """
        return gym.spaces.Discrete(1)

    def reset( self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """
        """
        self.current_path = self._initialize_current_path(use_random=self.random_path_init)
        self.world = self.build_world(self.config.get("network"))

        self.n_nodes, self.n_links = len(self.world.NODES), len(self.world.LINKS)
        plot_network_and_demand(self.world, f"{self.world.name}_demand_network.png") # Visualize after building world
    
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

    def build_world(self, network: str) -> World:
        """
        generate nodes from csv: 
        - expects: name, x, y
        generate links from csv: 
        - expects: link_name, start_node, end_node, length, free_flow_speed (u), jam_density (kappa), merge_priority
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
        

    def apply_action(self, action: Any) -> None:
        """
        Invoked internally by step prior to state transition.
        Based on the node selected, increase the route path by one hop.
        """
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
        """
        pass

    def render(self) -> None:
        """
        """
        pass

