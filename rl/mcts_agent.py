from typing import Any, Dict

class MCTSAgent:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, env: Any, config: Dict[str, Any]) -> None:
        self.env = env
        self.config = dict(config)

    def train(self) -> None:
        """
        """
        pass

    def evaluate(self, policy_path: str, save_dir: str) -> Dict[str, Any]:
        """
        """
        pass

