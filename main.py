import torch
from rl.env import TransitEnv
from rl.baselines import RandomWalk, DemandCoverage, ShortestPath, RewardMaximization, RealWorld, GeneticAlgorithm
from rl.parallel_env import _cap_worker_threads
from config import get_config, set_global_seeds
from ppo import train as ppo_train, ppo_eval
from mcts import train as mcts_train, mcts_eval

def main():

    config = get_config()
    algorithm = config.get("algorithm")
    mode = config["mode"]

    # Cap CPU threads globally to prevent oversubscription with multiprocessing
    _cap_worker_threads()

    # Baselines can be executed regardless of algorithm selection.
    if mode == "baseline":
        config["wandb_off"] = True
        # means mode is baseline
        env = TransitEnv(config)
        
        baseline_classes = {
            "random_walk": RandomWalk,
            "demand_cover": DemandCoverage,
            "shortest_path": ShortestPath,
            "reward_max": RewardMaximization,
            "real_world": RealWorld,
            "genetic": GeneticAlgorithm,
        }
        
        BaselineClass = baseline_classes[config["baseline_type"]]
        baseline = BaselineClass(env, config, num_runs=config["num_eval_runs"], base_seed=config["seed"])
        baseline.run()
        return

    if mode in {"train", "eval"}:
        if algorithm is None:
            raise ValueError("--algorithm is required for train/eval modes. Use --algorithm ppo or --algorithm mcts")
        set_global_seeds(config["seed"])

    device = torch.device("cuda" if (config["gpu"] and torch.cuda.is_available()) else "cpu")
    config["device"] = device

    # Dispatch to the appropriate algorithm module
    if algorithm == "ppo":
        if mode == "train":
            ppo_train(config)
            return

        if mode == "eval":
            ppo_eval(config)
            return

    if algorithm == "mcts":
        if mode == "train":
            mcts_train(config)
            return

        if mode == "eval":
            mcts_eval(config)
            return

if __name__ == "__main__":
    main()

"""
Scripts: 
python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=transit_center --alpha=0.3
python main.py --mode=baseline --baseline_type=real_world --save_animations --route_init=transit_center --alpha=0.3

python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=demand_cover --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=shortest_path --save_animations --route_init=random --alpha=0.3
python main.py --mode=baseline --baseline_type=reward_max --save_animations --route_init=random --alpha=0.3

python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=transit_center --alpha=1.0
python main.py --mode=baseline --baseline_type=real_world --save_animations --route_init=transit_center --alpha=1.0

python main.py --mode=baseline --baseline_type=random_walk --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=demand_cover --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=shortest_path --save_animations --route_init=random --alpha=1.0
python main.py --mode=baseline --baseline_type=reward_max --save_animations --route_init=random --alpha=1.0

python main.py --algorithm ppo --gpu --anneal_lr
python main.py --algorithm mcts --gpu
"""
