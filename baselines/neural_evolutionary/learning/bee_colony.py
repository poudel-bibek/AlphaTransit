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
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
import hydra

import os
from models import check_extensions_add_connections
import torch_utils as tu
from simulation.citygraph_dataset import get_dataset_from_config
from simulation.transit_time_estimator import RouteGenBatchState
import learning.utils as lrnu
from learning.initialization import get_direct_sat_dmd

# [C7 hub-start] Mumford index 95 = AlphaTransit node 96 (Bloomington transit center)
HUB_NODE = int(os.environ.get("HUB_NODE", 95))


def bee_colony(state, cost_obj, init_network, n_bees=10, passes_per_it=5, 
               mod_steps_per_pass=2, shorten_prob=0.2, n_iterations=400, 
               n_type1_bees=None, n_type2_bees=None, silent=False, 
               force_linking_unlinked=False, bee_model=None, 
               sum_writer=None):
    """Implementation of the method of  Nikolic and Teodorovic (2013).
    
    state -- A RouteGenBatchState object representing the initial state.
    cost_obj -- A function that determines the cost (badness) of a network.  In
        the paper, this is wieghted total travel time.
    n_routes -- The number of routes in each candidate network, called NBL in
        the paper.
    n_bees -- The number of worker bees, called B in the paper.
    passes_per_it -- The number of forward and backward passes to perform each
        iteration, called NP in the paper.
    mod_steps_per_pass -- The number of modifications each bee considers in the
        forward pass, called NC in the paper.
    shorten_prob -- The probability that type-2 bees will shorten a route,
        called P in the paper.  In their experiments, they use 0.2.
    n_iters -- The number of iterations to perform, called IT in the paper.
    n_type1_bees -- There are 2 types of bees used in the algorithm, which
        modify the solution in different ways.  This parameter determines the
        balance between them.  The paper isn't clear how many of each they use,
        so by default we make it half-and-half.
    silent -- if true, no tqdm output or printing
    bee_model -- if a torch model is provided, use it as the only bee type.
    """
    if n_type1_bees is None:
        n_type1_bees = n_bees // 2
    if n_type2_bees is None:
        # assume no type-3 bees if not specified
        n_type2_bees = n_bees - n_type1_bees
    n_type3_bees = n_bees - n_type1_bees - n_type2_bees

    if n_type3_bees > 0:
        # instantiate a random path-combining model
        rpc_model = lrnu.get_random_path_combiner()
    else:
        rpc_model = None

    batch_size = state.batch_size

    dev = state.device
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = state.max_n_nodes

    # get all shortest paths
    shortest_paths, _ = tu.reconstruct_all_paths(state.nexts)

    demand = torch.nn.functional.pad(state.demand, (0, 1, 0, 1))
    n_routes = state.n_routes_to_plan
    best_networks = torch.full((batch_size, n_routes, max_n_nodes), -1,
                                device=dev)
    best_networks[:, :, :init_network.shape[-1]] = init_network

    # expand state to networks

    # compute can-be-directly-satisfied demand matrix
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths, 
                                        cost_obj.symmetric_routes)

    # set up required matrices
    street_node_neighbours = (state.street_adj.isfinite() &
                              (state.street_adj > 0))
    bee_idxs = torch.arange(n_bees, device=dev)                      

    log.debug("starting BCO")

    # set up the cost function to work with batches of bees
    # "multiply" the batch for the bees
    exp_states = sum([[substate] * n_bees 
                      for substate in state.batch_to_list()], [])
    bee_states = RouteGenBatchState.batch_from_list(exp_states)
    if bee_model is not None:
        bee_states = bee_model.setup_planning(bee_states)

    metric_names = cost_obj.get_metric_names()
    
    def batched_cost_fn(bee_networks):
        bee_states.replace_routes(bee_networks.flatten(0,1))

        result = cost_obj(bee_states)

        costs = result.cost.reshape(batch_size, n_bees)
        other_metrics = result.get_metrics_tensor()
        other_metrics = other_metrics.reshape(batch_size, n_bees, -1)
        return costs, other_metrics

    # initialize bees' networks
    # batch size x n_bees x n_routes x max_n_nodes
    bee_networks = best_networks[:, None].repeat(1, n_bees, 1, 1)
    # evaluate and record the reward of the initial network
    bee_costs, bee_metrics = batched_cost_fn(bee_networks)
    best_costs, best_idxs = bee_costs.min(1)
    best_metrics = bee_metrics[batch_idxs, best_idxs]

    if sum_writer is not None:
        # log the initial values of various metrics
        for name, vals in zip(metric_names, best_metrics.unbind(-1)):
            sum_writer.add_scalar(f'best {name}', vals.mean(), 0)

    cost_history = torch.zeros((batch_size, n_iterations + 1), device=dev)
    cost_history[:, 0] = best_costs

    # --- CSV logging for training curves ---
    import csv as _csv
    import os as _os
    _log_path = _os.environ.get("BCO_LOG_CSV", "")
    _log_file = None
    _log_writer = None
    if _log_path:
        _log_file = open(_log_path, "w", newline="")
        _csv_fields = ["iteration", "best_cost", "pop_mean_cost", "pop_min_cost", "pop_max_cost"] + list(metric_names)
        _log_writer = _csv.DictWriter(_log_file, fieldnames=_csv_fields)
        _log_writer.writeheader()
        # Log initial state
        _init_row = {"iteration": 0, "best_cost": best_costs.mean().item(),
                     "pop_mean_cost": bee_costs.mean().item(),
                     "pop_min_cost": bee_costs.min().item(),
                     "pop_max_cost": bee_costs.max().item()}
        for name, vals in zip(metric_names, best_metrics.unbind(-1)):
            _init_row[name] = vals.mean().item()
        _log_writer.writerow(_init_row)
        _log_file.flush()

    for iteration in tqdm(range(n_iterations), disable=silent):
        for pi in range(passes_per_it):
            for mi in range(mod_steps_per_pass):

                # # flatten batch and bee dimensions
                # expanded_demand = demand[:, None].expand(-1, n_bees, -1, -1)
                # flat_exp_demand = expanded_demand.flatten(0, 1)
                # flat_bee_scens = bee_networks.flatten(0, 1)
                # route_dsds = aggr_edges_over_sequences(flat_bee_scens, 
                #                                     flat_exp_demand[..., None])
                # route_dsds.squeeze_(-1)
                # # choose routes to modify
                # route_scores = 1 / route_dsds
                # route_scores[route_scores.isinf()] = 10**10
                # flat_chosen_route_idxs = route_scores.multinomial(1).squeeze(1)
                # chosen_route_idxs = flat_chosen_route_idxs.reshape(batch_size, 
                #                                                    n_bees)                

                # unlike the original paper, choose routes uniformly at random
                chosen_route_idxs = torch.randint(high=n_routes.item(), 
                                                  size=(batch_size, n_bees), 
                                                  device=dev)                
                new_bee_networks = \
                    get_mutants(bee_networks, chosen_route_idxs, n_type1_bees, 
                                n_type2_bees, direct_sat_dmd, shorten_prob, 
                                street_node_neighbours, shortest_paths, 
                                force_linking_unlinked, bee_model, rpc_model, 
                                bee_states)

                new_bee_costs, new_bee_metrics = \
                    batched_cost_fn(new_bee_networks)

                better_idxs = new_bee_costs < bee_costs
                bee_networks[better_idxs] = new_bee_networks[better_idxs]
                bee_costs[better_idxs] = new_bee_costs[better_idxs]
                bee_metrics[better_idxs] = new_bee_metrics[better_idxs]        

            # do "backward pass"

            # update the best solution found so far
            current_best_cost, current_best_idx = bee_costs.min(1)
            is_improvement = current_best_cost < best_costs
            improvement_idx = current_best_idx[is_improvement]
            best_costs[is_improvement] = current_best_cost[is_improvement]
            best_networks[is_improvement] = \
                bee_networks[is_improvement, improvement_idx]
            cost_history[:, iteration + 1] = best_costs
            new_best = bee_metrics[is_improvement, improvement_idx]
            best_metrics[is_improvement] = new_best     

            # decide whether each bee is a recruiter or follower
            max_bee_costs, _ = bee_costs.max(dim=1)
            min_bee_costs, _ = bee_costs.min(dim=1)
            spread = max_bee_costs - min_bee_costs
            # avoid division by 0
            spread[spread == 0] = 1

            qualities = (max_bee_costs[:, None] - bee_costs) / spread[:, None]
            min_quality, _ = qualities.min(1)
            follow_probs = (-qualities + min_quality[:, None]).exp()
            are_recruiters = follow_probs < torch.rand(n_bees, device=dev)

            if not are_recruiters.any():
                # no recruiters, so make everyone keep their own network.
                continue

            # decide which recruiter each follower will follow
            denom = (qualities * are_recruiters).sum(1)
            # avoid division by 0
            denom[denom == 0] = 1
            recruit_probs = qualities / denom[:, None]
            recruit_probs[~are_recruiters] = 0
            # set probs where there is no probability to 1, so multinomial
             # doesn't complain.
            no_valid_recruiters = recruit_probs.sum(-1) == 0
            recruit_probs[no_valid_recruiters] = 1
            recruiters = recruit_probs.multinomial(n_bees, replacement=True)
            recruiters[no_valid_recruiters] = bee_idxs
            recruiters[are_recruiters] = \
                bee_idxs[None].expand(batch_size, -1)[are_recruiters]

            # update the networks and costs of the followers
            bee_networks = bee_networks[batch_idxs[:, None], recruiters]
            bee_costs = bee_costs[batch_idxs[:, None], recruiters]
            bee_metrics = bee_metrics[batch_idxs[:, None], recruiters]

        if sum_writer is not None:
            # log the various metrics
            for name, vals in zip(metric_names, best_metrics.unbind(-1)):
                sum_writer.add_scalar(f'best {name}', vals.mean(), iteration+1)

        # --- CSV logging per iteration ---
        if _log_writer is not None:
            _row = {"iteration": iteration + 1,
                    "best_cost": best_costs.mean().item(),
                    "pop_mean_cost": bee_costs.mean().item(),
                    "pop_min_cost": bee_costs.min().item(),
                    "pop_max_cost": bee_costs.max().item()}
            for name, vals in zip(metric_names, best_metrics.unbind(-1)):
                _row[name] = vals.mean().item()
            _log_writer.writerow(_row)
            _log_file.flush()

        # --- Verbose console logging every 10 iterations ---
        if not silent and (iteration + 1) % 10 == 0:
            _metrics_str = ", ".join(
                f"{name}={vals.mean().item():.4f}"
                for name, vals in zip(metric_names, best_metrics.unbind(-1)))
            print(f"  [iter {iteration+1}/{n_iterations}] "
                  f"best_cost={best_costs.mean().item():.4f}, "
                  f"pop_mean={bee_costs.mean().item():.4f}, "
                  f"{_metrics_str}")

    if _log_file is not None:
        _log_file.flush()
        _log_file.close()

    # return the best solution
    state.replace_routes(best_networks)
    return state, cost_history


def get_mutants(bee_networks, chosen_route_idxs, n_type1, n_type2,
                direct_sat_dmd, shorten_prob, street_node_neighbours,
                shortest_paths, force_linking_unlinked, bee_model=None, 
                rpc_model=None, env_state=None):
    bee_networks = bee_networks.clone()

    # flatten batch and bee dimensions
    gather_idx = chosen_route_idxs[..., None, None]
    max_n_nodes = bee_networks.shape[3]
    gather_idx = gather_idx.expand(-1, -1, -1, max_n_nodes)
    modified_routes = bee_networks.gather(2, gather_idx).squeeze(2)

    n_bees = bee_networks.shape[1]
    scen_idxs = torch.randperm(n_bees, device=bee_networks.device)
    type1_idxs = scen_idxs[:n_type1]
    type2_idxs = scen_idxs[n_type1:n_type1 + n_type2]
    type3_idxs = scen_idxs[n_type1 + n_type2:]

    # modify type 1 routes
    if bee_model is not None:
        # run it on all bees...
        new_type1_networks = get_neural_variants(bee_model, env_state, 
                                                 bee_networks,
                                                 chosen_route_idxs)
        # ...and keep only the type 1 bee routes
        new_type1_routes = new_type1_networks[:, type1_idxs, -1]
    else:
        unsel_routes = tu.get_unselected_routes(bee_networks, chosen_route_idxs)
        remaining_state = env_state.clone()
        remaining_state.replace_routes(unsel_routes.flatten(0,1))
        new_type1_routes = get_bee_1_variants(remaining_state, modified_routes,
                                              direct_sat_dmd, shortest_paths,
                                              force_linking_unlinked)
        # take on the type-1 routes
        new_type1_routes = new_type1_routes[:, type1_idxs]

    # modify type 2 routes
    new_type2_routes = get_bee_2_variants(modified_routes[:, type2_idxs], 
                                          shorten_prob, street_node_neighbours)
    assert ((new_type2_routes > -1).sum(dim=-1) > 0).all()

    # insert the modified routes in the new network
    new_routes = torch.cat((new_type1_routes, new_type2_routes), 
                            dim=1)
    if rpc_model is not None:
        # modify type 3 routes
        new_type3_networks = get_neural_variants(rpc_model, env_state,
                                                  bee_networks,
                                                  chosen_route_idxs)
        new_type3_routes = new_type3_networks[:, type3_idxs, -1]
        new_routes = torch.cat((new_routes, new_type3_routes), dim=1)

    bee_networks.scatter_(2, gather_idx, new_routes[..., None, :])
    return bee_networks


# [C7 hub-start] Post-mutation check: revert if hub-start lost
def get_neural_variants(model, env_state, bee_networks, drop_route_idxs,
                        greedy=False):
    bee_dim = bee_networks.ndim == 4
    if not bee_dim:
        # there is no bee dimension, so add one
        bee_networks = bee_networks.unsqueeze(1)
        drop_route_idxs = drop_route_idxs.unsqueeze(1)

    # flatten batch and bee dimensions
    batch_size = bee_networks.shape[0]
    n_bees = bee_networks.shape[1]
    n_routes = bee_networks.shape[2]
    keep_mask = torch.ones(bee_networks.shape[:3], dtype=bool, 
                           device=bee_networks.device)
    keep_mask.scatter_(2, drop_route_idxs[..., None], False)
    flat_kept_routes = bee_networks[keep_mask]
    # this works because we remove the same # of routes from each network
    max_n_nodes = bee_networks.shape[3]
    flatbee_kept_routes = flat_kept_routes.reshape(
        batch_size * n_bees, n_routes - 1, max_n_nodes)

    # plan a new route with the model
    env_state.replace_routes(flatbee_kept_routes)
    result = model(env_state, greedy=False)
    env_state = result.state
    routes = tu.get_batch_tensor_from_routes(env_state.routes, 
                                             bee_networks.device)
    if bee_dim:
        routes = routes.reshape(batch_size, n_bees, n_routes, -1)
    pad_size = max_n_nodes - routes.shape[-1]
    routes = torch.nn.functional.pad(routes, (0, pad_size), value=-1)

    # [C7] Revert any network where the new route lost the hub-start
    new_route_starts = routes[..., -1, 0]  # start of the newly added route
    hub_lost = new_route_starts != HUB_NODE
    if hub_lost.any():
        # Pad original bee_networks to match routes shape if needed
        orig = bee_networks
        if orig.shape[-1] < routes.shape[-1]:
            orig = torch.nn.functional.pad(
                orig, (0, routes.shape[-1] - orig.shape[-1]), value=-1)
        routes[hub_lost] = orig[hub_lost]

    return routes


# [C7 hub-start] Force replacement route starts to Mumford index 95 (= AlphaTransit node 96)
def get_bee_1_variants(remaining_state, batch_bee_routes, direct_sat_dmd_mat,
                       shortest_paths, force_linking_unlinked=False):
    """
    batch_bee_routes: a batch_size x n_bees x n_nodes tensor of routes
    direct_sat_dmd_mat: a batch_size x n_nodes x n_nodes tensor of 
        directly-satisfied demand by the shortest-path route between each pair
        of nodes.
    shortest_paths: a batch_size x n_nodes x n_nodes tensor of the shortest
        paths between each pair of nodes.
    """
    # choose which terminal to keep
    dev = batch_bee_routes.device
    keep_start_term = torch.rand(batch_bee_routes.shape[:2], device=dev) > 0.5

    # choose the new terminal
    # first, compute the demand that would be satisfied by new terminals
    route_lens = (batch_bee_routes > -1).sum(-1)
    batch_idxs = torch.arange(batch_bee_routes.shape[0]).unsqueeze(1)
    route_starts = batch_bee_routes[:, :, 0]
    dsd_from_starts = direct_sat_dmd_mat[batch_idxs, route_starts]
    route_ends = batch_bee_routes.gather(2, route_lens[..., None] - 1)
    route_ends.squeeze_(-1)
    dsd_to_ends = direct_sat_dmd_mat[batch_idxs, :, route_ends]
    # batch_size x n_bees x n_nodes
    new_route_dsds = keep_start_term[..., None] * dsd_from_starts + \
                     ~keep_start_term[..., None] * dsd_to_ends
        
    # then, choose the new terminal proportional to the demand
    # if no demand is satisfiable, set all probs to non-zero for multinomial()
    no_demand_satisfied = new_route_dsds.sum(-1) == 0
    new_route_dsds[no_demand_satisfied] = 1
    # set the terminal that is already in the route to 0, so it can't be chosen
    kept_terms = keep_start_term * route_starts + ~keep_start_term * route_ends
    new_route_dsds.scatter_(2, kept_terms[..., None], 0)

    if force_linking_unlinked:
        # if there are any unlinked node pairs, consider only new routes that 
         # link at least one of them
        sps_from_starts = shortest_paths[batch_idxs, route_starts]
        sps_to_ends = shortest_paths[batch_idxs, :, route_ends]
        candidate_routes = keep_start_term[..., None, None] * sps_from_starts + \
                           ~keep_start_term[..., None, None] * sps_to_ends
        candidate_routes = torch.nn.functional.pad(candidate_routes, 
                                                   (0, 0, 0, 1),)
        extends_if_needed = check_extensions_add_connections(
            remaining_state.has_path, candidate_routes.flatten(0,1))
        # fold the output batch dimension back into (batch, bees)
        shape = tuple(batch_bee_routes.shape[:2]) + (-1,)
        extends_if_needed = extends_if_needed.reshape(shape)
        # only consider routes that extend coverage if not full coverage
        zero_mask = ~extends_if_needed
        nonzero_will_remain = \
            ((new_route_dsds > 0) & extends_if_needed).any(2)
        zero_mask[~nonzero_will_remain] = False        
        new_route_dsds[zero_mask] = 0.0
        # new_route_dsds[~extends_if_needed] = 0.0
     
    # sample the terminals
    flat_new_terms = new_route_dsds.flatten(0,1).multinomial(1).squeeze(-1)
    new_terms = flat_new_terms.reshape(batch_bee_routes.shape[:2])

    # if all demand is zero, dummy might get chosen.  If so, leave route alone.
    dummy_node = direct_sat_dmd_mat.shape[-1] - 1
    chose_dummy = new_terms == dummy_node
    new_terms[chose_dummy] = (~keep_start_term * route_starts + \
                              keep_start_term * route_ends)[chose_dummy]

    new_starts = keep_start_term * route_starts + ~keep_start_term * new_terms
    new_ends = ~keep_start_term * route_ends + keep_start_term * new_terms

    # [C7] Force all replacement routes to start at hub node
    new_starts[:] = HUB_NODE
    new_routes = shortest_paths[batch_idxs, new_starts, new_ends]

    # pad the end of the new routes to match the existing ones
    n_pad_stops = batch_bee_routes.shape[-1] - new_routes.shape[-1]
    new_routes = torch.nn.functional.pad(new_routes, (0, n_pad_stops), 
                                         value=-1)
    assert ((new_routes > -1).sum(dim=-1) > 0).all()

    return new_routes


# [C7 hub-start] Post-mutation check: revert if hub-start lost
def get_bee_2_variants(batch_bee_routes, shorten_prob, are_neighbours):
    """
    batch_bee_routes: a batch_size x n_bees x n_nodes tensor of routes
    shorten_prob: a scalar probability of shortening each route
    are_neighbours: a batch_size x n_nodes x n_nodes boolean tensor of whether
        each node is a neighbour of each other node
        
    """
    bee_dim = batch_bee_routes.ndim == 3
    if bee_dim:
        # flatten the batch and bee dimensions to ease what follows
        flat_routes = batch_bee_routes.flatten(0,1)
        n_bees = batch_bee_routes.shape[1]
    else:
        flat_routes = batch_bee_routes
        n_bees = 1

    # expand and reshape are_neighbours to match flat_routes
    are_neighbours = are_neighbours[:, None].expand(-1, n_bees, -1, -1)
    are_neighbours = are_neighbours.flatten(0,1)
    # convert from boolean to float to allow use of torch.multinomial()
    neighbour_probs = are_neighbours.to(dtype=torch.float32)
    # add a padding column for scattering
    neighbour_probs = torch.nn.functional.pad(neighbour_probs, (0, 1, 0, 1))

    dev = batch_bee_routes.device
    keep_start_term = torch.rand(flat_routes.shape[0], device=dev) > 0.5
    keep_start_term.unsqueeze_(-1)

    route_lens = (flat_routes > -1).sum(-1, keepdim=True)

    # shorten chosen routes at chosen end
    shortened_at_start = flat_routes.roll(shifts=-1, dims=-1)
    shortened_at_start[:, -1] = -1
    shortened_at_end = flat_routes.scatter(1, route_lens - 1, -1)
    shortened = keep_start_term * shortened_at_start + \
        ~keep_start_term * shortened_at_end  

    # extend chosen routes at chosen ends
    n_nodes = are_neighbours.shape[-1]
    # choose new extended start nodes
    route_starts = flat_routes[:, 0]
    rs_gatherer = route_starts[:, None, None].expand(-1, n_nodes+1, 1)
    start_nbr_probs = neighbour_probs.gather(2, rs_gatherer).squeeze(-1)
    # set the probabilities of nodes already on the route to 0
    bbr_scatterer = tu.get_update_at_mask(flat_routes, flat_routes==-1, n_nodes)
    start_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_start_options = start_nbr_probs.sum(-1) == 0
    start_nbr_probs[no_start_options] = 1
    chosen_start_exts = start_nbr_probs.multinomial(1).squeeze(-1)
    chosen_start_exts[no_start_options] = -1
    extended_starts = flat_routes.roll(shifts=1, dims=-1)
    extended_starts[:, 0] = chosen_start_exts

    # choose new extended end nodes
    route_ends = flat_routes.gather(1, route_lens - 1)
    re_gatherer = route_ends[:, None].repeat(1, 1, n_nodes+1)
    re_gatherer[re_gatherer == -1] = n_nodes
    end_nbr_probs = neighbour_probs.gather(1, re_gatherer).squeeze(-2)
    end_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_end_options = end_nbr_probs.sum(-1) == 0
    end_nbr_probs[no_end_options] = 1
    chosen_end_exts = end_nbr_probs.multinomial(1)
    chosen_end_exts[no_end_options] = -1
    # pad the routes before scattering, so that full-length routes don't cause
     # an index-out-of-bounds error.
    extended_ends = torch.nn.functional.pad(flat_routes, (0, 1))
    extended_ends = extended_ends.scatter(1, route_lens, chosen_end_exts)
    extended_ends = extended_ends[..., :-1]

    extended_routes = extended_ends * keep_start_term + \
        extended_starts * ~keep_start_term

    # assemble the shortened routes
    shorten = shorten_prob > torch.rand(flat_routes.shape[0], device=dev)
    # remove the last dimension, since we don't need it for gathering anymore
    route_lens = route_lens.squeeze(-1)
    shorten &= route_lens > 2
    shorten.unsqueeze_(-1)
    shortened_part = shortened * shorten
    # assemble the extended routes
    extend_at_start = ~no_start_options[..., None] & ~keep_start_term
    extend_at_end = ~no_end_options[..., None] & keep_start_term
    extend = (extend_at_start | extend_at_end) & ~shorten
    extended_part = extended_routes * extend
    # assemble the unmodified routes (ones with no valid extension)
    keep_same = ~(extend | shorten)
    same_part = flat_routes * keep_same
    # combine the three
    out_routes = shortened_part + extended_part + same_part
    
    out_lens = (out_routes > -1).sum(-1)
    assert ((out_lens - route_lens).abs() <= 1).all()

    # [C7] Revert any route whose start is no longer the hub node
    hub_lost = out_routes[:, 0] != HUB_NODE
    out_routes[hub_lost] = flat_routes[hub_lost]

    # fold back into batch x bees
    if bee_dim:
        out_routes = out_routes.reshape(batch_bee_routes.shape)
    return out_routes


@hydra.main(version_base=None, config_path="../cfg", config_name="bco_mumford")
def main(cfg: DictConfig):
    global DEVICE
    use_neural_bees = cfg.get('neural_bees', False)
    if use_neural_bees:
        prefix = 'neural_bco_'
    else:
        prefix = 'bco_'

    DEVICE, run_name, sum_writer, cost_obj, bee_model = \
        lrnu.process_standard_experiment_cfg(cfg, prefix, 
                                             weights_required=True)

    # read in the dataset
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    force_linking_unlinked = cfg.get('force_linking_unlinked', False)

    if not use_neural_bees:
        bee_model = None
    elif bee_model is not None:
        bee_model.force_linking_unlinked = force_linking_unlinked
        bee_model.eval()

    nt1b = cfg.get('n_type1_bees', None)
    nt2b = cfg.get('n_type2_bees', None)
    routes = \
        lrnu.test_method(bee_colony, test_dl, cfg.eval, cfg.init, cost_obj, 
            sum_writer=sum_writer, silent=False, n_bees=cfg.n_bees,
            n_iterations=cfg.n_iterations, n_type1_bees=nt1b, n_type2_bees=nt2b,  
            device=DEVICE, bee_model=bee_model, return_routes=True,
            force_linking_unlinked=force_linking_unlinked)[-1]
    
    # save the final routes that were produced
    tu.dump_routes(run_name, routes)
    

if __name__ == "__main__":
    main()
