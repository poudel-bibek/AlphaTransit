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

import copy
import math
import random
import logging as log
from pathlib import Path
from itertools import product

from tqdm import tqdm
import torch
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from omegaconf import DictConfig, OmegaConf
import hydra

# need to import CityGraphData here to allow unpickling of new datasets
from simulation.citygraph_dataset import CityGraphData, CityGraphDataset, \
    get_default_train_and_eval_split, get_dynamic_training_set, STOP_KEY, \
    DEMAND_KEY
from simulation.transit_time_estimator import RouteGenBatchState
import learning.utils as lrnu
from learning.models import FeatureNorm, get_mlp
from learning.eval_route_generator import eval_model


BLMODE_NONE = "none"
BLMODE_GREEDY = "greedy"
BLMODE_ROLL = "rolling average"
BLMODE_NN = "neural"


class Baseline:
    def update(self, mean_cost):
        raise NotImplementedError

    def get_baseline(self, graph_data, n_routes, cost_weights, sumwriter=None, 
                     n_eps=None):
        raise NotImplementedError
    

class FixedBaseline:
    def __init__(self, baseline=0, *args, **kwargs):
        self.baseline = baseline

    def update(self, costs):
        self.baseline = costs.mean()

    def get_baseline(self, graph_data, n_routes, cost_weights, sumwriter=None,
                     n_eps=None):
        if sumwriter is not None:
            sumwriter.add_scalar("baseline", self.baseline, n_eps)
        batch_size = graph_data.num_graphs
        return self.baseline[None, None].expand(batch_size, n_routes)
    

class RollingBaseline:
    def __init__(self, alpha):
        assert 0 < alpha <= 1
        self.alpha = alpha
        self.baseline = None

    def update(self, costs):
        mean_cost = costs.mean()
        if self.baseline is None:
            self.baseline = mean_cost
        else:
            self.baseline = self.alpha * mean_cost + (1 - self.alpha) * \
                self.baseline
    
    def get_baseline(self, graph_data, cost_weights, sumwriter=None, 
                     n_eps=None):
        if sumwriter is not None:
            if self.baseline is None:
                sumwriter.add_scalar("baseline", 0, n_eps)
            else:
                sumwriter.add_scalar("baseline", self.baseline, n_eps)

        if self.baseline is None:
            return 0
        else:
            return self.baseline


class NNBaseline:
    def __init__(self, learning_rate=0.0005, decay=0.01):
        self.learning_rate = learning_rate
        self.decay = decay
        self.optim = None
        self.model = None
        self._curr_estimate = None
        self.loss_fn = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            FeatureNorm(self.input_dim, 0.001),
            get_mlp(3, self.input_dim*2, in_dim=self.input_dim, out_dim=1, 
                    dropout=0.0)
        ).to(DEVICE)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate, 
                                        weight_decay=self.decay)

    @property
    def input_dim(self):
        # total demand, num demand edges, mean and std edge demand,
         # mean and std edge time weighted by demand on edge, 
         # max num of mean stop features, # routes so far and # routes left,
         # weights of the two cost components
        return 1 + 1 + 2 + 2 + 4 + 11

    def update(self, costs):
        # do backprop
        self.optim.zero_grad()
        costs = costs.to(self._curr_estimate.dtype)
        loss = self.loss_fn(self._curr_estimate, costs)
        loss.backward()
        self.optim.step()
        self._curr_estimate = None
        # update the feature normalization statistics
        self.model[0].update()

    def inputs_from_data(self, graph_data, cost_weights):
        dev = graph_data[STOP_KEY].x.device
        input_data = torch.zeros(graph_data.num_graphs, self.input_dim, 
                                 dtype=torch.float, device=dev)
        input_data[:, 0] = graph_data.demand.sum(dim=(1,2))

        for bi, dl in enumerate(graph_data.to_data_list()):
            dmd_graph = dl[DEMAND_KEY]
            input_data[bi, 1] = dmd_graph.num_edges
            input_data[bi, 2] = dmd_graph.edge_attr[:, 0].mean()
            if dmd_graph.num_edges > 1:
                input_data[bi, 3] = dmd_graph.edge_attr[:, 0].std()
            else:
                input_data[bi, 3] = 0

            dmd_weighted_times = dmd_graph.edge_attr[:, 0] * \
                dmd_graph.edge_attr[:, 1]
            input_data[bi, 4] = dmd_weighted_times.mean()
            input_data[bi, 5] = dmd_weighted_times.mean()
            x_dim = dl[STOP_KEY].x.shape[1]
            input_data[bi, 6:6+x_dim] = dl[STOP_KEY].x.mean(dim=0)

        for ii, cw in enumerate(cost_weights.values()):
            data_idx = -(1 + ii)
            input_data[:, data_idx] = cw

        return input_data

    def set_input_norm(self, all_graph_data):
        if not isinstance(all_graph_data, Batch):
            all_graph_data = Batch.from_data_list(all_graph_data)
        input_data = self.inputs_from_data(all_graph_data, {})
        self.model[0].running_mean[...] = input_data.mean(dim=0)
        self.model[0].running_var[...] = input_data.var(dim=0)
 
        self.model[0].running_mean[-15:] = 0
        self.model[0].running_var[-15:] = 1
    
    def from_state(self, state):
        # set up the input data        
        input_data = self.inputs_from_data(state.graph_data, 
                                           state.cost_weights)
        glob_feats = state.get_global_state_features()
        input_data[..., -glob_feats.shape[-1]:] = glob_feats
        
        baseline = self.model(input_data).squeeze(-1)
        self._curr_estimate = baseline

        assert baseline.isfinite().all()

        return baseline.detach()
    
    def get_baseline(self, graph_data, n_routes, cost_weights, state_vecs,
                     sumwriter=None, n_eps=None):
        # set up the input data
        assert self._curr_estimate is None, "Baseline estimate already exists!"
        
        input_data = self.inputs_from_data(graph_data, cost_weights)
        # add extra features and batch dimension for the numbers of routes
        input_data = input_data[:, None].repeat(1, state_vecs.shape[-2], 1)
        input_data[..., -state_vecs.shape[-1]:] = state_vecs
        
        baseline = self.model(input_data).squeeze(-1)
        self._curr_estimate = baseline

        sumwriter.add_scalar("baseline", baseline.mean(), n_eps)
        sumwriter.add_scalar("baseline std", baseline.std(), n_eps)

        assert baseline.isfinite().all()

        return baseline.detach()
    

def eval_model_over_nroutes(model, eval_dl, n_routes_range, eval_cfg, cost_obj, 
                            sumwriter=None, n_eps=0, n_samples=None, 
                            silent=False):
    # evaluate the model over a variety of n_routes, and take the mean
    eval_costs = []
    all_metrics = None
    cached_eval_n_routes = eval_cfg.get('n_routes', None)
    for n_routes in n_routes_range:
        eval_cfg.n_routes = n_routes
        mean_cost, metrics = \
            eval_model(model, eval_dl, eval_cfg, cost_obj, silent=silent, 
                       n_samples=n_samples, device=DEVICE)
        eval_costs.append(mean_cost)

        if all_metrics is None:
            all_metrics = metrics
        else:
            for key, val in metrics.items():
                all_metrics[key] = torch.cat((all_metrics[key], val))

    if cached_eval_n_routes is not None:
        # restore the cached routes, just in case
        eval_cfg.n_routes = cached_eval_n_routes

    mean_eval_cost = sum(eval_costs) / len(eval_costs)
    if sumwriter is not None:
        sumwriter.add_scalar("val cost", mean_eval_cost, n_eps)
        for metric_name in ['ATT', 'RTT', '$d_{un}$', 
                            '# disconnected node pairs']:
            mean_value = all_metrics[metric_name].mean()
            sumwriter.add_scalar(f"val {metric_name}", mean_value, n_eps)

    return mean_eval_cost


def render_scenario(graph, routes, show_demand=False):
    # draw the routes on top of the city
    num_colours = max(2, int(np.ceil((2 * len(routes)) ** (1/3))))
    dim_points = np.linspace(0.1, 0.9, num_colours)
    # filter out grayscale colours
    colours = [cc for cc in product(dim_points, dim_points, dim_points)
                if len(np.unique(cc)) > 1]
    colours = iter(colours)

    nx_graph = nx.DiGraph()
    for ii in range(graph.num_nodes):
        nx_graph.add_node(ii)
    posdict = {ii: pos.cpu().numpy() 
               for ii, pos in enumerate(graph[STOP_KEY].pos)}
    nx.draw_networkx_nodes(nx_graph, pos=posdict, node_size=10)
    widths = torch.linspace(10, 2, len(routes))
    for width, colour, route in zip(widths, colours, routes):
        route_graph = nx.DiGraph()
        if type(route) is torch.Tensor:
            route = [ss.item() for ss in route]
        for jj in range(len(route) - 1):
            route_graph.add_edge(route[jj], route[jj+1])
        nx.draw_networkx_edges(route_graph, pos=posdict, width=width, 
                               edge_color=colour)
    
    if show_demand:
        dmd_graph = nx.DiGraph()
        for ii, jj in graph[DEMAND_KEY].edge_index.T:
            dmd_graph.add_edge(ii.item(), jj.item())
        demands = graph.demand[graph[DEMAND_KEY].edge_index[0], 
                               graph[DEMAND_KEY].edge_index[1]]
        widths = demands.cpu().numpy() * 5                               
        nx.draw_networkx_edges(dmd_graph, pos=posdict, width=widths,
                               edge_color='red', style="dotted")

    plt.show()


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass


def train_ppo(model, min_n_routes, max_n_routes, cfg, optimizer, 
              train_dataloader, val_dataloader, cost_obj, eval_fn, 
              return_scale=1, sumwriter=None, optuna_trial=None, device=None, 
              checkpoint_dir=None):
    batch_size = train_dataloader.batch_size
    assert batch_size == cfg.ppo.minibatch_size, \
        "For now, minibatch size must be the same as the batch size, due to " \
        "an unresolved issue with concatenating states."
    # assert cfg.ppo.minibatch_size % batch_size == 0, \
    #     "minibatch size must be a multiple of the batch size"
    assert (cfg.ppo.horizon * batch_size) % cfg.ppo.minibatch_size == 0, \
        "minibatch size must wholly divide the number of samples per epoch"

    if sumwriter is None:
        # instantiate a dummy summary writer so that the calls to it still work
        sumwriter = DummySummaryWriter()
    
    # set up neural network baseline
    val_module = NNBaseline(learning_rate=cfg.baseline_lr)
    # val_module.set_input_norm(train_dataloader.dataset)

    best_val_cost = float('inf')
    n_routes = max_n_routes
    model.eval()
    n_states_per_minibatch = cfg.ppo.minibatch_size // batch_size
    gamma = cfg.discount_rate
    epsilon = cfg.ppo.epsilon
    horizon = cfg.ppo.horizon

    # set up progress bar
    pbar = tqdm(total=cfg.ppo.n_iterations)
    
    train_gen = iter(train_dataloader)
    n_routes = None
    ep_finished = None
    for iteration in range(cfg.ppo.n_iterations):
        if iteration % cfg.ppo.val_period == 0:
            # do a pass on the validation set
            avg_val_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
            if iteration == 0:
                model.update_and_freeze_feature_norms()

            if optuna_trial is not None:
                optuna_trial.report(avg_val_cost, iteration)
                if iteration > 0 and optuna_trial.should_prune():
                    # don't prune just because of a bad initialization
                    raise optuna.exceptions.TrialPruned()

            # Kool et al. does this with a paired t-test, we're keeping it simple
            # for the moment.
            if avg_val_cost < best_val_cost:
                # update the best model and the baseline
                best_model = copy.deepcopy(model)
                best_val_cost = avg_val_cost

        if n_routes is None:
            # use the max n routes first, so that if we're going to run out
             # of memory, we do so right away
            n_routes = max_n_routes
        else:
            # pick a random number of routes in the evaluation range
            n_routes = random.randint(min_n_routes, max_n_routes)

        if ep_finished is None or ep_finished.all():
            # all eps in this batch have finished, so get the next batch
            try:
                data = next(train_gen)
            except StopIteration:
                # we ran out of data, so iterate over it again
                train_gen = iter(train_dataloader)
                data = next(train_gen)
            data = data.to(device)

            # sample cost weights
            cost_weights = cost_obj.sample_variable_weights(data.num_graphs, 
                                                            device)
            state = RouteGenBatchState(data, cost_obj, n_routes, 
                                       cfg.eval.min_route_len,
                                       cfg.eval.max_route_len,
                                       cost_weights=cost_weights)
            state = model.setup_planning(state)
            ep_finished = state.is_done()

            # compute the cost of the scenarios with no transit
            if cfg.diff_reward:
                with torch.no_grad():
                    base_result = cost_obj(state)
                    base_cost = base_result.cost
                    prev_cost = base_cost
            
        # set up the buffers to hold the rollout
        buf_rewards = torch.zeros((horizon, state.batch_size), 
                                  device=device)
        buf_val_ests = buf_rewards.clone()
        buf_logits = buf_rewards.clone()
        done_at_step_mask = buf_rewards.clone().bool()
        buf_actions = torch.zeros((horizon, state.batch_size, 2),
                                  device=device, dtype=torch.long)

        buf_actions = torch.full((horizon, state.batch_size, 2), -1,
                                 device=device, dtype=torch.long)
        buf_states = []

        # roll out the specified number of steps
        with torch.no_grad():    
            for tt in range(horizon):
                done_at_step_mask[tt] = state.is_done()
                val_ests = val_module.from_state(state)
                buf_val_ests[tt] = val_ests
                buf_states.append(state.clone().to_device('cpu'))
                actions, logits, _ = model.step(state)
                buf_actions[tt] = actions
                # update the states with the actions
                state.shortest_path_action(actions)
    
                buf_logits[tt] = logits
                if cfg.diff_reward:
                    # reward is decrease in cost
                    result = cost_obj(state)
                    rewards = (prev_cost - result.cost)  * return_scale
                    # update the cost used to compute the differential. This 
                     # should be the previous cost if the episode is not done,
                     # and the cost with no transit otherwise.
                    prev_cost = result.cost * ~state.is_done() + \
                        base_cost * state.is_done()
                elif state.is_done().any():                    
                    # reward is the decrease in cost at last step, 0 otherwise
                    result = cost_obj(state)
                    rewards = -result.cost * return_scale * state.is_done()
                else:
                    # rewards are all 0 since no episode is done
                    rewards = torch.zeros_like(buf_rewards[0])

                buf_rewards[tt] = rewards
                
                # mark which episodes have reached the end at least once
                ep_finished |= state.is_done()

                state.reset_dones()
            
            # compute the returns and advantages
            final_val_ests = val_module.from_state(state)
            if cfg.ppo.use_gae:
                # use generalized advantage estimation
                buf_advantages = torch.zeros_like(buf_rewards)
                lastgaelam = 0
                for tt in reversed(range(horizon)):
                    if tt == horizon - 1:
                        nextnonterminal = ~state.is_done()
                        nextvalues = final_val_ests
                    else:
                        nextnonterminal = ~done_at_step_mask[tt + 1]
                        nextvalues = buf_val_ests[tt + 1]
                    delta = buf_rewards[tt] + \
                        gamma * nextvalues * nextnonterminal - \
                        buf_val_ests[tt]
                    lambda_term = gamma * cfg.ppo.gae_lambda * \
                        nextnonterminal * lastgaelam
                    buf_advantages[tt] = lastgaelam = delta + lambda_term
                buf_returns = buf_advantages + buf_val_ests
            else:
                # compute the returns and the advantages simply
                buf_returns = torch.zeros_like(buf_rewards).to(device)
                for tt in reversed(range(horizon)):
                    if tt == horizon - 1:
                        nextnonterminal = ~state.is_done()
                        next_return = final_val_ests
                    else:
                        nextnonterminal = ~done_at_step_mask[tt + 1]
                        next_return = buf_returns[tt + 1]
                    nextnonterminal = nextnonterminal.to(torch.float32)
                    buf_returns[tt] = buf_rewards[tt] + \
                        gamma * nextnonterminal * next_return
                buf_advantages = buf_returns - buf_val_ests
            
        # log the average return and baseline on the rolled-out episodes
        sumwriter.add_scalar("avg return", buf_rewards.mean(), pbar.n)
        sumwriter.add_scalar("baseline", buf_val_ests.mean(), pbar.n)

        # train on the rolled-out episodes
        train_orders = [torch.randperm(len(buf_states), device=device)
                        for _ in range(cfg.ppo.n_epochs)]
        train_order = torch.cat(train_orders)
        for idxs in torch.split(train_order, n_states_per_minibatch):
            mb_states = [buf_states[ii] for ii in idxs]
            if n_states_per_minibatch == 1:
                mb_states = mb_states[0]
            else:
                mb_states = RouteGenBatchState.batch_from_list(mb_states)
            # move states to chosen device
            mb_states = mb_states.to_device(device)
            mb_acts = buf_actions[idxs].flatten(0, 1)
            mb_old_logits = buf_logits[idxs].flatten(0, 1)
            mb_returns = buf_returns[idxs].flatten(0, 1)
            mb_advs = buf_advantages[idxs].flatten(0, 1)

            # compute V(s) loss and update the value estimator
            val_module.from_state(mb_states)                
            val_module.update(mb_returns)

            # step model forward
            _, logits, entropy = model.step(mb_states, actions=mb_acts)
            
            # normalize advantages by the minibatch statistics
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

            # compute clip loss
            ratios = (logits - mb_old_logits).exp()
            if (ratios == 0).any():
                log.warning(f"Some ratios are zero: {ratios}")
            if ratios.isinf().any():
                log.warning(f"Some ratios are very extreme: {ratios}")

            assert ratios.isfinite().all()
                
            clipped_ratios = ratios.clamp(1 - epsilon, 1 + epsilon)
            clip_obj = torch.minimum(ratios * mb_advs,
                                     clipped_ratios * mb_advs)
            objective = clip_obj.mean()
            objective += entropy.mean() * cfg.entropy_weight

            # do backprop and weight updates
            optimizer.zero_grad()
            objective.backward()
            # based on Andrychowicz et al. 2021, we clip the gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5,
                                           error_if_nonfinite=True)
            optimizer.step()

        # save a checkpoint every val period
        if checkpoint_dir is not None and iteration % cfg.ppo.val_period == 0:
            ckpt_filename = checkpoint_dir / f"iter{iteration}.pt"
            torch.save(model.state_dict(), ckpt_filename)

        pbar.update(1)

    # final eval
    avg_val_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
    if avg_val_cost < best_val_cost:
        # update the best model and the baseline
        best_model = copy.deepcopy(model)
    pbar.close()

    return best_model, best_val_cost


def train(model, min_n_routes, max_n_routes, cfg, optimizer, train_dataloader, 
          val_dataloader, cost_obj, eval_fn, sumwriter=None, optuna_trial=None,
          device=None, checkpoint_dir=None):
    """Train a model on a dataset.
    
    Args:
    eval_model -- a function that takes:
        - a model, 
        - a dataloader, 
        - a summary writer,
        - and the number of episodes so far
        as input, and returns a scalar cost.
    """
    if sumwriter is None:
        # instantiate a dummy summary writer so that the calls to it still work
        sumwriter = DummySummaryWriter()

    best_model = copy.deepcopy(model)
    best_cost = float('inf')
    try:
        graphs_per_epoch = len(train_dataloader.dataset)
    except TypeError:
        graphs_per_epoch = 9 * len(val_dataloader.dataset)
    batches_per_epoch = \
        math.ceil(graphs_per_epoch / train_dataloader.batch_size)
    pbar = tqdm(total=graphs_per_epoch * cfg.reinforce.n_epochs)

    bl_module = None
    if cfg.baseline_mode == BLMODE_NONE:
        baseline = 0
    elif cfg.baseline_mode == BLMODE_ROLL:
        bl_module = RollingBaseline(cfg.bl_alpha)
    elif cfg.baseline_mode == BLMODE_NN:
        bl_module = NNBaseline()
        bl_module.set_input_norm(train_dataloader.dataset)
    elif cfg.baseline_mode == BLMODE_GREEDY:
        bl_module = FixedBaseline()
    
    if device is None:
        device = DEVICE

    log.info(f"discount rate is {cfg.discount_rate}")

    n_routes = None
    for epoch in range(cfg.reinforce.n_epochs):
        avg_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
        if epoch == 0:
            model.update_and_freeze_feature_norms()

        if optuna_trial is not None:
            optuna_trial.report(avg_cost, epoch)
            if epoch > 0 and optuna_trial.should_prune():
                # don't prune just because of a bad initialization
                raise optuna.exceptions.TrialPruned()

        # Kool et al. does this with a paired t-test, we're keeping it simple
         # for the moment.
        if avg_cost < best_cost:
            # update the best model and the baseline
            best_model = copy.deepcopy(model)
            best_cost = avg_cost
            if cfg.baseline_mode == BLMODE_GREEDY:
                bl_module.update(-avg_cost * cfg.reward_scale)

        dl_iter = iter(train_dataloader)
        for _ in range(batches_per_epoch):
            if n_routes is None:
                # use the max n routes first, so that if we're going to run out
                 # of memory, we do so right away
                n_routes = max_n_routes
            else:
                # pick a random number of routes in the evaluation range
                n_routes = random.randint(min_n_routes, max_n_routes)
            data = next(dl_iter)
            if device.type != 'cpu':
                data = data.cuda()

            # sample cost weights
            cost_weights = \
                cost_obj.sample_variable_weights(data.num_graphs, device)
            state = RouteGenBatchState(data, cost_obj, n_routes, cfg.eval.min_route_len,
                                       cfg.eval.max_route_len, cost_weights=cost_weights)
            state = model.setup_planning(state)
            
            # run those graphs though the net
            all_logits = []
            all_entropy = 0
            n_routes_so_far = []
            state_vecs = []

            if cfg.diff_reward:
                base_result = cost_obj(state)
                base_cost = base_result.cost
                prev_cost = base_cost
                rewards = []

            while not state.is_done().all():
                n_routes_so_far.append(state.n_finished_routes)
                state_vecs.append(state.get_global_state_features())
                actions, logits, entropy = model.step(state)
                state.shortest_path_action(actions)
                if cfg.diff_reward:
                    result = cost_obj(state)
                    reward = (prev_cost - result.cost) * cfg.reward_scale
                    rewards.append(reward)
                    prev_cost = result.cost * ~state.is_done()
             
                all_logits.append(logits)
                all_entropy += entropy

            logits = torch.stack(all_logits, dim=1)

            if cfg.diff_reward:
                rewards = torch.stack(rewards, dim=1)
                # apply the discount factor, gamma, to get the returns
                rev_rwds = rewards.flip(-1)
                prev_rr = None
                rev_rtrns = []
                for rr in rev_rwds.transpose(1, 0):
                    if prev_rr is not None:
                        rr = rr + prev_rr * cfg.discount_rate
                    rev_rtrns.append(rr)
                    prev_rr = rr
                returns = torch.stack(rev_rtrns, dim=1).flip(-1)
            else:
                # all returns are just the final return
                result = cost_obj(state)
                costs = result.cost
                costs = costs[:, None].expand(-1, logits.shape[1])
                returns = -costs * cfg.reward_scale
                
            n_routes_so_far = torch.stack(n_routes_so_far, dim=1)
            state_vecs = torch.stack(state_vecs, dim=1)

            route_lens = [len(rr) for br in state.routes for rr in br]
            sumwriter.add_scalar("avg # stops on a route", 
                                 np.mean(route_lens), pbar.n)

            sumwriter.add_scalar("avg train cost", costs[:, -1].mean(), pbar.n)
            
            # Log some stats with the summary writer
            avg_rtrn = returns.mean().item()
            sumwriter.add_scalar("avg return", avg_rtrn, pbar.n)

            for metric_name, metric in result.get_metrics().items():
                mean_value = metric.mean()
                sumwriter.add_scalar(f"avg {metric_name}", mean_value, pbar.n)

            # Log some stats with the summary writer
            # if plan_out.route_est_vals is not None:
            #     baseline = plan_out.route_est_vals
            #     sumwriter.add_scalar("baseline", baseline.mean().item(), n_eps)
            #     returns -= baseline.detach()
            #     sumwriter.add_scalar("avg return vs baseline", returns.mean(),
            #                           n_eps)
            #     sumwriter.add_scalar("std return vs baseline", returns.std(), 
            #                           n_eps)
            if bl_module is not None:
                baseline = bl_module.get_baseline(data, n_routes, 
                                                  state.cost_weights,
                                                  state_vecs, sumwriter, 
                                                  pbar.n)
                no_action_mask = n_routes_so_far == n_routes
                # don't make any updates on these entries
                returns[no_action_mask] = baseline[no_action_mask]
                if cfg.baseline_mode != BLMODE_GREEDY:
                    bl_module.update(returns)
                returns -= baseline

                sumwriter.add_scalar("avg return vs baseline", returns.mean(),
                                     pbar.n)
                sumwriter.add_scalar("std return vs baseline", returns.std(), 
                                     pbar.n)

            objective = 0
            # compute the training signal on the logits
            route_signal = returns
            route_obj = (route_signal * logits).mean()
            sumwriter.add_scalar("route obj", route_obj.item(), pbar.n)
            objective += route_obj
            # add entropy to the signal
            objective += cfg.entropy_weight * entropy.mean()
            sumwriter.add_scalar("entropy", entropy.mean().item(), pbar.n)

            sumwriter.add_scalar("objective", objective.item(), pbar.n)

            # backprop and weight update
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.update(data.num_graphs)

        if checkpoint_dir is not None:
            ckpt_filename = checkpoint_dir / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_filename)

    # final eval
    avg_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
    if avg_cost < best_cost:
        # update the best model and the baseline
        best_model = copy.deepcopy(model)

    pbar.close()

    return best_model, best_cost


def setup_and_train(cfg: DictConfig, trial: optuna.trial.Trial = None):
    global DEVICE
    DEVICE, run_name, sum_writer, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(cfg, 'inductive_')
    # model always has legal route lengths by construction, so we don't need
     # to penalize it for illegal routes
    cost_obj.ignore_stops_oob = True

    optimizer_type = getattr(torch.optim, cfg.optimizer)
    optimizer = optimizer_type(model.parameters(), lr=cfg.lr, 
                               weight_decay=cfg.decay, 
                               maximize=True)
    val_batch_size = cfg.batch_size * 4
    
    if cfg.dataset.type == 'pickle':
        train_ds, val_ds = get_default_train_and_eval_split(
            **cfg.dataset.kwargs)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, 
                              shuffle=True)
    elif cfg.dataset.type == 'dynamic':
        train_ds = get_dynamic_training_set(**cfg.dataset.kwargs)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)
        val_ds = CityGraphDataset(cfg.dataset.val_path)
    elif cfg.dataset.type == 'mumford':
        do_scaling = cfg.dataset.get('scale_dynamically', True)
        data = CityGraphData.from_mumford_data(cfg.dataset.path, 
                                               cfg.dataset.city,
                                               scale_dynamically=do_scaling)
        train_ds = [data] * cfg.batch_size
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)
        val_ds = [data] * val_batch_size

    val_dl = DataLoader(val_ds, batch_size=val_batch_size)

    # run training
    model.to(DEVICE)
    # set a range of random cost parameters to use during validation
    val_cost_obj = copy.deepcopy(cost_obj)
    val_weights = val_cost_obj.sample_variable_weights(val_batch_size, DEVICE)
    val_cost_obj.set_weights(**val_weights)
    
    eval_n_routes = cfg.eval.n_routes
    if type(eval_n_routes) is int:
        eval_n_routes = [eval_n_routes]
        
    # define evaluation function
    eval_fn = lambda mm, dd, sw, ne: \
        eval_model_over_nroutes(mm, dd, eval_n_routes, cfg.eval, val_cost_obj, 
                                sw, ne, n_samples=1, silent=True)

    if 'outdir' in cfg:
        output_dir = Path(cfg.outdir)
    else:
        output_dir = Path('output')

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    min_n_routes = min(eval_n_routes)
    max_n_routes = max(eval_n_routes)
    checkpoint_dir = output_dir / f'{run_name}_checkpoints'
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    train_fn = train_ppo if 'ppo' in cfg else train
    best_model, best_cost = train_fn(
        model, min_n_routes, max_n_routes, cfg, optimizer, train_dl, 
        val_dl, cost_obj, eval_fn, sumwriter=sum_writer, device=DEVICE, 
        optuna_trial=trial, checkpoint_dir=checkpoint_dir)

    # save the new trained model
    torch.save(best_model.state_dict(), output_dir / (run_name + '.pt'))
    
    return best_cost

@hydra.main(version_base=None, config_path="../cfg", config_name="ppo_20nodes")
def main(cfg: DictConfig):
    return setup_and_train(cfg)

if __name__ == "__main__":
    main()
