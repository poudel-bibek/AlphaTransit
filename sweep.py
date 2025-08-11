from typing import Any, Dict, Optional, Sequence

# Import the train entrypoint to demonstrate config flow
from main import train


class SweepRunner:
    """
    Orchestrates hyperparameter sweeps compatible with W&B.
    Wraps training entrypoints with a config-driven interface.
    Only the interfaces are provided here.
    """

    def __init__(self, base_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Accept a base configuration to be merged with sweep values.
        No external services are initialized in the skeleton.
        Prepares for later integration with logging backends.
        """
        self.base_config = dict(base_config or {})

    def run(self, sampled_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Merge base config with sampled parameters and run training.
        Returns the metrics dict produced by the training routine.
        This is the per-trial entrypoint for sweep agents.
        """
        final_config = merge_configs(self.base_config, sampled_params)
        return sweep_main(final_config)


def build_sweep_config() -> Dict[str, Any]:
    """
    Define the sweep search space and default settings.
    Return a config dict suitable for W&B or custom runners.
    Implementation will populate parameters later.
    """
    # Example sweep config with a few hyperparameters.
    return {
        "method": "bayes",
        "metric": {"name": "evals/avg_ped_arrival", "goal": "minimize"},
        "parameters": {
            "lr": {"values": [1e-4, 3e-4, 1e-3]},
            "clip_coef": {"values": [0.1, 0.2, 0.3]},
            "gae_lambda": {"values": [0.9, 0.95]},
            "num_envs": {"values": [4, 8]},
            "seed": {"values": [0, 1, 2]},
        },
    }


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shallow-merge two configuration dictionaries.
    Values in override take precedence over values in base.
    Returns a new dictionary without mutating inputs.
    """
    merged = dict(base)
    merged.update(dict(override))
    return merged


def sweep_main(config: Dict[str, Any]) -> None:
    """
    Entry point used by an agent to run a single trial.
    Delegates to the training routine with the provided config.
    Logging and checkpointing will be added later.
    """
    return train(config)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Optionally launch a local sweep controller or agent.
    Keeps sweep orchestration separate from CLI main.
    Implement the actual sweep setup in subsequent steps.
    """
    # Demonstrate local, dependency-free sweep execution using 3 trials.
    base_config = {
        "total_timesteps": 5000,
        "exp_name": "dummy_sweep",
    }
    runner = SweepRunner(base_config)

    # Pick a few representative configurations as if sampled by a sweep service
    trial_params = [
        {"lr": 3e-4, "clip_coef": 0.2, "gae_lambda": 0.95, "num_envs": 4, "seed": 0},
        {"lr": 1e-4, "clip_coef": 0.1, "gae_lambda": 0.90, "num_envs": 8, "seed": 1},
        {"lr": 1e-3, "clip_coef": 0.3, "gae_lambda": 0.95, "num_envs": 4, "seed": 2},
    ]

    print("[sweep] Running", len(trial_params), "dummy trials...")
    for idx, params in enumerate(trial_params, start=1):
        print(f"\n[sweep] Trial {idx} with params: {params}")
        metrics = runner.run(params)
        print(f"[sweep] Trial {idx} result: {metrics}")


if __name__ == "__main__":
    main()


