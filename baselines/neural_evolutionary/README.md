# Neural Evolutionary Algorithm Baseline

This code is adapted from the [transit_learning](https://github.com/AHolliday/transit_learning) repository by Andrew Holliday et al. It is used to produce baseline results for comparison with AlphaTransit.

The code corresponds to the following papers:

1. A. Holliday, A. El-Geneidy, and G. Dudek, "Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning," *Transportmetrica B: Transport Dynamics*, 13(1), 2025. [arXiv:2404.05894](https://arxiv.org/abs/2404.05894)

2. A. Holliday and G. Dudek, "A Neural-Evolutionary Algorithm for Autonomous Transit Network Design," in *2024 IEEE International Conference on Robotics and Automation (ICRA)*, IEEE, 2024. [arXiv:2403.07917](https://arxiv.org/abs/2403.07917)

Files required to run Bee Colony and Neural Evolutionary on the Bloomington instance are included here, along with additional configs from the original repository for reference. For the full codebase, see the original repository linked above.

## Constraint Matching

For a fair comparison, all AlphaTransit route constraints must be enforced on Holliday's methods. The table below lists each constraint, whether it is natively supported, and where code changes are made to enforce it.

| # | Constraint | Value | Natively Supported? | Code Change |
|---|-----------|-------|---------------------|-------------|
| C1 | Number of routes ($K$) | 16 | Yes | Set in `cfg/eval/bloomington.yaml` |
| C2 | Max route length ($L_{\max}$) | 14 | Yes | Set in `cfg/eval/bloomington.yaml` |
| C3 | Min route length ($L_{\min}$) | 2 | Yes | Set in `cfg/eval/bloomington.yaml` |
| C4 | Simple paths (no repeated nodes) | — | Yes | Enforced natively by route construction |
| C5 | Connected routes (consecutive nodes share an edge) | — | Yes | Enforced natively by neighbor-based construction |
| C6 | Bidirectional edges | — | Yes | Mumford format assumes symmetric adjacency |
| C7 | Hub-start (all routes begin at transit center) | Node 96 (Mumford idx 95) | **No** | Modified below |

**C7 is the only mismatch.** It is enforced differently for each baseline:

**Bee Colony:** Modified initialization and mutation operators (3 code locations):

| File | Function | Change |
|------|----------|--------|
| `learning/initialization.py` | `nikolic_init()` | Force all route start nodes to Mumford index 95 |
| `learning/bee_colony.py` | `get_bee_1_variants()` | Force replacement route starts to index 95 |
| `learning/bee_colony.py` | `get_bee_2_variants()` | Post-mutation check: revert if hub-start lost |

**Neural Evolutionary:** Hub-start enforced at the model level during training and inference. The GNN is self-trained on synthetic cities where the highest-demand node is designated as hub. During route construction, non-hub start nodes are masked to $-\infty$ in `PathCombiningRouteGenerator.step()` (models.py). The published pretrained weights do not support hub-start (see below).

**Why pretrained weights do not work:** The published weights from Holliday et al. were trained on synthetic cities without any hub-start constraint. When used in the Neural Evolutionary Algorithm, the GNN proposes routes starting at arbitrary nodes. Since all such routes violate hub-start, every neural mutation is rejected, and the algorithm degrades to plain Bee Colony. We therefore self-train the GNN with hub-start enforced during training.

## Two-Environment Workflow

Holliday's code requires Python 3.9 and PyTorch 1.12, which are incompatible with AlphaTransit's runtime (Python 3.14, PyTorch 2.11). The workflow has two stages:

**Step 1: Generate routes** (in `holliday` conda env)
```bash
conda activate holliday
cd baselines/neural_evolutionary

# Run Bee Colony
PYTHONPATH=. BCO_LOG_CSV=training_data/ea_log.csv \
  python learning/bee_colony.py eval.dataset.path=datasets/bloomington \
  +eval=bloomington init.method=nikolic +run_name=bloomington_ea

# Self-train GNN with hub constraint (one-time, ~2-4 hrs GPU)
PYTHONPATH=. python simulation/citygraph_dataset.py datasets/synthetic_20_hub mixed \
  -n 32768 --min 20 --max 20 --ovaldemand
PYTHONPATH=. python learning/inductive_route_learning.py \
  dataset.kwargs.path=datasets/synthetic_20_hub +run_name=self_trained_hub

# Run LC-100 with self-trained weights
HUB_NODE=95 PYTHONPATH=. python learning/eval_route_generator.py \
  +model.weights=weights/self_trained/inductive_self_trained_hub.pt \
  eval.dataset.path=datasets/bloomington +eval=bloomington \
  +run_name=bloomington_lc100_hub

# Run Neural Evolutionary with self-trained weights
HUB_NODE=95 PYTHONPATH=. BCO_LOG_CSV=training_data/nea_log.csv \
  python learning/bee_colony.py --config-name neural_bco_mumford \
  +model.weights=weights/self_trained/inductive_self_trained_hub.pt \
  eval.dataset.path=datasets/bloomington +eval=bloomington \
  init.path=output_routes/nn_construction_bloomington_lc100_hub_routes.pkl \
  +run_name=bloomington_nea
```
This produces route pickle files in `output_routes/`.

**Step 2: Evaluate routes in AlphaTransit simulator** (in `alphatransit` conda env)
```bash
conda activate alphatransit
cd /path/to/AlphaTransit

# Evaluate EA routes
python main.py --mode baseline --baseline_type evolutionary --alpha 0.3
python main.py --mode baseline --baseline_type evolutionary --alpha 1.0

# Evaluate NEA routes
python main.py --mode baseline --baseline_type neural_evolutionary --alpha 0.3
python main.py --mode baseline --baseline_type neural_evolutionary --alpha 1.0
```
The `_HollidayBaseline` class in `__init__.py` loads the pre-computed route pickles and evaluates them through UXsim.

---

*Original README from transit_learning follows below.*

---

# License

This work is released as free software under the GNU Public License.  All constituent source code files are covered by this license.  See the file COPYING for the full legal details of the license.

# Usage

To use this software, first set up a python environment with its dependencies.  The environment.yml file describes these dependencies, and if you're using the conda environment management tool or one of its offshoots, can be used to set up the environment appropriately.

For all scripts, run with `-h` or `--help` for some information on usage and arguments.  Most scripts are configured using the hydra library [https://hydra.cc/], and so the standard hydra CLI allows you to modify their configuration with command-line arguments.

## Training

If you're not using the pre-trained model weights (information on how to get them in the "Model Weights" section), you'll need to train your own model.  To generate a training dataset, use the `simulation/citygraph_dataset.py` script.  Note that right now, there's a bug for running the algorithm on batches of graphs with different numbers of nodes, so you should pass the same value to `--min` and `--max` to make sure all graphs in the dataset have the same size.  The dataset will be output to the directory you specify.

To train a model, use the script `learning/inductive_route_training.py`.  You will need to specify the path to your generated training dataset directory as follows:

```python learning/inductive_route_training.py dataset.kwargs.path=/path/to/your/dataset```

By default, the model will be trained over a range of cost weights from 0 to 1.  To train just on an operator perspective setting, add the argument `experiment/cost_function=op`,
or to train on a passenger perspective setting, add `experiment/cost_function=pp`.

Training should take around 3-6 hours on a modern commercial GPU.

You can optionally add the argument `+run_name=my_run_name` to name the training run, which will affect the name of the tensorboard logs (stored by default in a directory called `training_logs`) and the name of the output weight file.  If this is not provided, the current date and time will be used as the name of the run.

When training is complete, the trained weights will be stored in the directory `output` in a file named `inductive_[run-name].pt`.

## Evaluation

We mainly evaluate our methods on the Mandl and Mumford datasets, which can be downloaded as a single archive from [Christine Mumford's website](https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/Outline.html).  Download the archive and extract it to a directory on your system.

Each script described in this section prints a line of comma-separated statistics about the best transit network it finds, with the header format:
,cost,C_p (minutes),C_o (minutes),d_0,d_1,d_2,d_{un},# disconnected node pairs,# stops out of bounds,running time (seconds),number of iterations

Each also saves the best transit network as a pickled torch tensor which can be read by other scripts, in a directory called `output_routes`.  The filename will contain the run name that can be provided to each script with `+run_name=my_run_name`.  If no run name is provided, the date and time when the script was launched will be used instead.

To evaluate a model on a Mumford city, use the script `learning/eval_route_generator.py`.  You must provide a `.pt` file with model weights, the path to the `Instances` sub-directory of the mumford dataset, and the name of the city on which to evaluate (`mandl` or `mumford0` - `mumford3`), as follows:
```
python learning/eval_route_generator.py +model.weights=path_to_weights.pt eval.dataset.path=/path/to/mumford/Instances +eval=mandl +run_name=my_mandl_lc100
```

To run the evolutionary algorithm (EA) on a city using the network generated by the above LC-100 run, the signature is similar, but without model weights:
```
python learning/bee_colony.py eval.dataset.path=/path/to/mumford/Instances +eval=mandl init.path=output_routes/nn_construction_my_mandl_lc100_routes.pkl
```

And to run the neural evolutionary algorithm (NEA), use the same script but specify the `neural_bco_mumford` config file, and provide model weights and the path to the transit network from LC-100 to be used as the starting network:
```
python learning/bee_colony.py --config-name neural_bco_mumford +model.weights=path_to_weights.pt eval.dataset.path=/path/to/mumford/Instances +eval=mandl init.path=output_routes/nn_construction_my_mandl_lc100_routes.pkl
```

Note that "bee colony" is a holdover from an earlier stage in this research project, where we were using a "bee colony optimization" algorithm.

# Model weights

Model weights used for the ITSC experiments can be downloaded from the following link:
https://www.cim.mcgill.ca/~mrl/projs/transit_learning/itsc_2023

Those used for the most up-to-date PPO experiments (forthcoming) can be downloaded from:
https://www.cim.mcgill.ca/~mrl/projs/transit_learning/ppo_2025

# Citation

If you make use of this code for academic work, please cite our associated conference paper, "Augmenting Transit Network Design Algorithms with Deep Learning":

```
@inproceedings{holliday2024autonomous,
    author = {Holliday, Andrew and Dudek, Gregory},
    title = {A Neural-Evolutionary Algorithm for Autonomous Transit Network Design},
    year = {2024},
    booktitle = {presented at 2024 IEEE International Conference on Robotics and Automation (ICRA)},
    organization = {IEEE}
}
```
