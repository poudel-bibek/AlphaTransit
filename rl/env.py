from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from uxsim import World

class TransitEnv(gym.Env):
    """
    Transit environment for RL.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        """
        self.config = dict(config)
        self.world = self.build_world(self.config.get("network"))
        self.horizon = self.config.get("horizon")
        self.delta_t = self.config.get("delta_t") # deltat = deltan * reaction_time
        self.delta_n = self.config.get("delta_n")

        # Important params: 
        self.SERVICE_FREQUENCY = self.config.get("service_frequency")
        self.STOP_SPACING = self.config.get("stop_spacing")

    @property
    def observation_space(self) -> gym.Space:
        """
        """
        return gym.spaces.Box(low=0, high=1, shape=(1,))

    @property
    def action_space(self) -> gym.Space:
        """
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
        """
        W = World(
            name=network,
            deltan=1,
            tmax=self.horizon,
            print_mode=0, 
            save_mode=0, 
            show_mode=0,
            random_seed=self.config.get("random_seed"),
        )

        return W
    
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


## Example usage
if __name__ == "__main__":
    env = TransitEnv()
    env.reset()
    env.step(0)
    env.close()