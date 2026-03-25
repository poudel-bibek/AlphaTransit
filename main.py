import torch
from rl.env import TransitEnv
from baselines import RandomWalk, DemandCoverage, ShortestPath, RewardMaximization, RealWorld, GeneticAlgorithm, NeuralEvolutionary, EvolutionaryAlgorithm, PureMCTS
from rl.parallel_env import _cap_worker_threads
from config import get_config, set_global_seeds
from ppo import train as ppo_train, ppo_eval
from alpha import train as alpha_train, alpha_eval

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
            "neural_evolutionary": NeuralEvolutionary,
            "evolutionary": EvolutionaryAlgorithm,
            "mcts": PureMCTS,
        }
        
        BaselineClass = baseline_classes[config["baseline_type"]]
        baseline = BaselineClass(env, config, num_runs=config["num_eval_runs"], base_seed=config["seed"])
        baseline.run()
        return

    if mode in {"train", "eval"}:
        if algorithm is None:
            raise ValueError("--algorithm is required for train/eval modes. Use --algorithm ppo or --algorithm alphatransit")
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

    if algorithm == "alphatransit":
        if mode == "train":
            alpha_train(config)
            return

        if mode == "eval":
            alpha_eval(config)
            return

if __name__ == "__main__":
    main()
