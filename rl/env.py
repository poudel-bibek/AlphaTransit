from typing import Any, Dict, Optional, Tuple

import gymnasium as gym


class TransitEnv(gym.Env):
    """
    Gym-compatible environment scaffold for transit simulations.
    Provides reset/step plus helper methods for clarity.
    No environment logic is implemented in this skeleton.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration and placeholder spaces.
        Action/observation spaces are defined in implementation.
        No simulation state is created at this stage.
        """
        self.config = dict(config or {})
        self.action_space = None
        self.observation_space = None

    def build_observation_space(self) -> gym.Space:
        """
        Define and return the observation space object.
        Replace with a concrete Box/Dict/Tuple space later.
        """
        pass

    def build_action_space(self) -> gym.Space:
        """
        Define and return the action space object.
        Replace with Discrete/Box/MultiDiscrete as needed.
        """
        pass

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        Return the first observation and an info dictionary.
        Implementation details are deferred.
        """
        pass

    def apply_action(self, action: Any) -> None:
        """
        Apply the provided action to the environment state.
        Invoked internally by step prior to state transition.
        No side effects in the skeleton.
        """
        pass

    def compute_reward(self) -> float:
        """
        Compute reward for the most recent transition.
        Encapsulates task objectives and potential shaping.
        Returns a placeholder value when implemented.
        """
        pass

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Advance one timestep given an action.
        Return (obs, reward, terminated, truncated, info) per gym.
        Transition logic will be added later.
        """
        pass


