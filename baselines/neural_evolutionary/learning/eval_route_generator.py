# Copyright 2023 Andrew Holliday
# 
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
# 
# Transit Learning is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.

import logging as log

import torch
from torch_geometric.data import Batch
from omegaconf import DictConfig, OmegaConf
import hydra

from torch_geometric.loader import DataLoader
from simulation.citygraph_dataset import CityGraphData, \
    get_dataset_from_config, STOP_KEY
from simulation.transit_time_estimator import RouteGenBatchState
import learning.utils as lrnu
from torch_utils import get_batch_tensor_from_routes, dump_routes


def sample_from_model(model, state, cost_obj, n_samples=20, 
                      sample_batch_size=None):
    model.eval()
    # duplicate the state across the samples
    graph_data = state.graph_data
    data_list = graph_data.to_data_list()
    flat_sample_inputs = data_list * n_samples

    if sample_batch_size is None:
        if state.batch_size > 1:
            sample_batch_size = state.batch_size
        else:
            sample_batch_size = n_samples

    all_plans = []
    all_costs = []
    for ii in range(0, len(flat_sample_inputs), sample_batch_size):
        chunk = flat_sample_inputs[ii:ii+sample_batch_size]
        n_routes = state.n_routes_to_plan
        if state.batch_size == 1:
            min_route_len = state.min_route_len.squeeze()
            max_route_len = state.max_route_len.squeeze()
        else:
            # TODO this doesn't actually work right now, the indexing will go
             # past the end of the tensor because it's only as long as one 
             # batch.
            min_route_len = state.min_route_len[ii:ii+sample_batch_size]
            max_route_len = state.max_route_len[ii:ii+sample_batch_size]
        batch_state = RouteGenBatchState(chunk, cost_obj, 
                                         state.n_routes_to_plan,
                                         min_route_len, max_route_len)
        with torch.no_grad():
            plan_out = model(batch_state, greedy=False)
            batch_costs = cost_obj(plan_out.state).cost
        all_plans += plan_out.state.routes
        all_costs.append(batch_costs)

    all_costs = torch.cat(all_costs, dim=0).reshape(n_samples, -1)

    # # plot a histogram of the costs, with 1/10th as many bins as samples
    # import matplotlib.pyplot as plt
    # plt.hist(all_costs.cpu().numpy().flatten(), bins= n_samples // 1)
    # plt.savefig("cost_hist.png")
    # print(f"Average cost of samples: {all_costs.mean()}")

    _, min_indices = all_costs.min(0)
    batch_size = len(flat_sample_inputs) // n_samples
    best_plans = [all_plans[mi * batch_size + ii] \
                  for ii, mi in enumerate(min_indices)]
    best_plans_tensor = get_batch_tensor_from_routes(best_plans)
    state.add_new_routes(best_plans_tensor)
    return state


def eval_model(model, eval_dataloader, eval_cfg, cost_obj, sum_writer=None, 
               iter_num=0, n_samples=None, silent=False, 
               sample_batch_size=None, return_routes=False, device=None):
    log.debug("evaluating our model on test set")
    model.eval()
    if n_samples is None:
        method_fn = lambda state, *args, **kwargs: \
            (model(state, greedy=True).state, None)
    else:
        method_fn = lambda state, cost_obj, *args, **kwargs: \
            (sample_from_model(model, state, cost_obj, n_samples=n_samples, 
                               sample_batch_size=sample_batch_size), 
             None)
    cost, _, metrics, routes = \
        lrnu.test_method(method_fn, eval_dataloader, eval_cfg, None, cost_obj, 
                         sum_writer, device=device, silent=silent, 
                         iter_num=iter_num, return_routes=True)
    if return_routes:
        routes = get_batch_tensor_from_routes(routes)
        return cost, metrics, routes
    else:
        return cost, metrics


@hydra.main(version_base=None, config_path="../cfg", 
            config_name="eval_model_mumford")
def main(cfg: DictConfig):
    global DEVICE
    assert 'model' in cfg, "Must provide config for model!"
    DEVICE, run_name, _, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(cfg, 'nn_construction_', 
                                             weights_required=True)

    # load the data
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    # evaluate the model on the dataset
    n_samples = cfg.get('n_samples', None)
    sbs = cfg.get('sample_batch_size', cfg.batch_size)
    _, _, routes = eval_model(model, test_dl, cfg.eval, cost_obj, 
        n_samples=n_samples, sample_batch_size=sbs, return_routes=True, 
        device=DEVICE)
    
    dump_routes(run_name, routes.cpu())


if __name__ == "__main__":
    main()
