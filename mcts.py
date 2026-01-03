"""
MCTS (Monte Carlo Tree Search) training and evaluation.
Placeholder module - implementation yet to come.
"""
from typing import Any, Dict
from rl.env import TransitEnv
from rl.mcts_agent import MCTSAgent


# =============================================================================
# MCTS entry points for train and eval modes.
# Called from main.py when algorithm == "mcts".
# =============================================================================

def mcts_train(config: Dict[str, Any]) -> None:
    """
    Entry point for MCTS training mode.
    TODO: Implement MCTS training pipeline in rl/mcts_agent.py
    """
    env = TransitEnv(config)
    mcts_agent = MCTSAgent(env, config)
    # TODO: Implement MCTS training pipeline in rl/mcts_agent.py
    pass

def mcts_eval(config: Dict[str, Any]) -> None:
    """
    Entry point for MCTS evaluation mode.
    TODO: Implement MCTS evaluation pipeline in rl/mcts_agent.py
    """
    env = TransitEnv(config)
    mcts_agent = MCTSAgent(env, config)
    # TODO: Implement MCTS evaluation pipeline in rl/mcts_agent.py
    pass
