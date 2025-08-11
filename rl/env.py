from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import gymnasium as gym
from uxsim import World
from rl.env_utils import plot_network_and_demand

class TransitEnv(gym.Env):
    """
    Transit environment for RL.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        """
        self.config = dict(config)
        self.horizon = self.config.get("horizon")
        self.delta_t = self.config.get("delta_t") # deltat = deltan * reaction_time
        self.delta_n = self.config.get("delta_n")

        # Important params: 
        self.SERVICE_FREQUENCY = self.config.get("service_frequency")
        self.STOP_SPACING = self.config.get("stop_spacing")

        # Constraints:
        self.MAX_PATH_LENGTH = self.config.get("max_path_length")
        self.MIN_PATH_LENGTH = self.config.get("min_path_length")

        self.n_demand, self.world = self.build_world(self.config.get("network"))
        self.n_nodes, self.n_links = len(self.world.NODES), len(self.world.LINKS)
        plot_network_and_demand(self.world, f"{self.world.name}_demand_network.png") # Visualize after building world
        
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

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        """
        pass
    
    def build_world(self, network: str) -> World:
        """
        generate nodes from csv: 
        - expects: name, x, y
        generate links from csv: 
        - expects: link_name, start_node, end_node, length, free_flow_speed, jam_density, merge_priority
        generate demand from csv: 
        - expects: orig, dest, start_t, end_t, q
        
        """
        if not network:
            raise ValueError("Network name must be provided in config['network']")

        project_root = Path(__file__).resolve().parents[1]
        network_dir = project_root / "networks" / network
        nodes_csv = network_dir / f"{network}_nodes.csv"
        links_csv = network_dir / f"{network}_links.csv"
        demand_csv = network_dir / f"{network}_demand.csv"

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
        world.generate_demand_from_csv(str(demand_csv))

        n_demand = 10 # TODO: get from demand csv
        return n_demand, world
    
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

