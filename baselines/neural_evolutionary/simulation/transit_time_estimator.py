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
import copy
from collections import deque
from typing import Union
from collections.abc import Sequence

from numpy import ndarray
import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData
import networkx as nx
from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig

from simulation.citygraph_dataset import STOP_KEY
import torch_utils as tu


MEAN_STOP_TIME_S = 0
AVG_TRANSFER_WAIT_TIME_S = 300
UNSAT_PENALTY_EXTRA_S = 3000
EPSILON = 1e-6


def enforce_correct_batch(matrix, batch_size):
    if matrix.ndim == 2:
        matrix = matrix[None]
    if matrix.shape[0] > 1:
        assert batch_size == matrix.shape[0]
    elif batch_size > 1:
        shape = (batch_size,) + (-1,) * (matrix.ndim - 1)
        matrix = matrix.expand(*shape)
    return matrix


class ExtraStateData(HeteroData):
    """A class for holding data, some of it computed, that is specific to one
    scenario in a state."""
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['base_valid_terms_mat', 
                   'valid_terms_mat',
                   'mean_stop_time',
                   'fixed_routes_file',
                   'transfer_time_s',
                   'total_route_time', 
                   'n_routes_to_plan', 
                   'min_route_len',
                   'max_route_len', 
                   'n_nodes_in_scenario',
                   'directly_connected', 
                   'route_mat',
                   'transit_times',
                   'has_path', 
                   'current_routes',
                   'current_route_time',
                   'current_route_times_from_start',
                   'shortest_path_sequences',
                   'route_nexts',
                   'n_transfers',
                   'fixed_routes']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class RouteGenBatchState:
    def __init__(self, graph_data, cost_obj, n_routes_to_plan, min_route_len=2,
                 max_route_len=None, valid_terms_mat=None, cost_weights=None):
        # do initialization needed to make properties work
        if not isinstance(graph_data, Batch):
            if not isinstance(graph_data, list):
                graph_data = [graph_data]
            graph_data = Batch.from_data_list(graph_data)

        # set members that live directly on this object
        self.graph_data = graph_data
        # right now this must have the same value for all scenarios
        self.symmetric_routes = cost_obj.symmetric_routes
        self._finished_routes = [[] for _ in range(graph_data.num_graphs)]

        # the object isn't ready to give this property yet, so find it here
        dev = graph_data[STOP_KEY].x.device
        max_n_nodes = max([dd.num_nodes for dd in graph_data.to_data_list()])
        if valid_terms_mat is None:
            # all terminal pairs (i,j) are valid except if i = j
            valid_terms_mat = ~torch.eye(max_n_nodes, device=dev, dtype=bool)
            valid_terms_mat = valid_terms_mat.repeat(self.batch_size, 1, 1)

        if cost_weights is None:
            # get the cost weights
            cost_weights = cost_obj.get_weights(device=dev)
        for key, val in cost_weights.items():
            # expand the cost weights to match the batch
            if type(val) is not Tensor:
                val = torch.tensor([val], device=dev)
            if val.numel() == 1:
                val = val.expand(graph_data.num_graphs)
            cost_weights[key] = val

        # make a tensor of n_routes_to_plan, expanded to the right shape
        if type(n_routes_to_plan) is not Tensor:
            n_routes_to_plan = torch.tensor(n_routes_to_plan, device=dev)
        if n_routes_to_plan.numel() == 1:
            n_routes_to_plan = n_routes_to_plan.expand(graph_data.num_graphs)

        extra_datas = []
        transit_times = (1 - torch.eye(max_n_nodes, device=self.device))
        transit_times[transit_times > 0] = float('inf')
        for ii, dd in enumerate(graph_data.to_data_list()):
            extra_data = ExtraStateData()
            extra_data.transit_times = transit_times
            extra_data.route_mat = transit_times.clone()
            dircon = torch.eye(max_n_nodes, device=self.device, dtype=bool)

            extra_data.directly_connected = dircon
            extra_data.has_path = extra_data.directly_connected.clone()
            extra_data.base_valid_terms_mat = valid_terms_mat[ii]
            extra_data.valid_terms_mat = valid_terms_mat[ii]
            extra_data.mean_stop_time = \
                torch.tensor(cost_obj.mean_stop_time_s, device=dev)
            extra_data.transfer_time_s = \
                torch.tensor(cost_obj.avg_transfer_wait_time_s, device=dev)
            extra_data.total_route_time = torch.zeros((), device=dev)
            # make this a tensor so it's stackable
            extra_data.n_routes_to_plan = n_routes_to_plan[ii]

            if isinstance(min_route_len, Tensor):
                if min_route_len.numel() == 1:
                    extra_data.min_route_len = min_route_len
                else:
                    extra_data.min_route_len = min_route_len[ii]
            else:
                extra_data.min_route_len = torch.tensor(min_route_len,
                                                        device=dev)
            extra_data.min_route_len.squeeze_()

            if max_route_len is None:
                max_route_len = dd.num_nodes
            if isinstance(max_route_len, Tensor):
                if max_route_len.numel() == 1:
                    extra_data.max_route_len = max_route_len
                else:
                    extra_data.max_route_len = max_route_len[ii]
            else:
                extra_data.max_route_len = torch.tensor(max_route_len,
                                                        device=dev)
            extra_data.max_route_len.squeeze_()

            extra_data.n_nodes_in_scenario = torch.tensor(dd.num_nodes,
                                                          device=dev)
            extra_data.current_routes = \
                torch.full((max_n_nodes,), -1, device=dev)
            extra_data.current_route_time = torch.zeros((), device=dev)
            extra_data.current_route_times_from_start = \
                torch.zeros((max_n_nodes,), device=dev)
            extra_data.shortest_path_sequences = torch.zeros(
                (0, 0, 0), device=dev)
            extra_data.route_nexts = \
                torch.zeros((max_n_nodes, max_n_nodes), device=dev, 
                            dtype=torch.long)            
            extra_data.n_transfers = extra_data.route_nexts.clone()

            extra_data.norm_node_features = torch.zeros(
                (dd.num_nodes, 0,), device=dev)

            extra_data.cost_weights = {}
            for key, val in cost_weights.items():
                # expand the cost weights to match the batch
                extra_data.cost_weights[key] = val[ii]

            extra_datas.append(extra_data)
            if hasattr(graph_data, 'fixed_routes') and \
               graph_data.fixed_routes.shape[0] > 0:
                # fixed_routes must be the same for all instances in the batch
                extra_data.fixed_routes = graph_data.fixed_routes[0]
            else:
                extra_data.fixed_routes = torch.zeros(0)

        self.extra_data = Batch.from_data_list(extra_datas)

        # this initializes route data
        self.clear_routes()

    def shortest_path_action(self, path_indices):
        routes_are_done = path_indices[:, 0] == -1
        path_seqs = self.get_shortest_path_sequences()
        new_parts = path_seqs[self.batch_indices, path_indices[:, 0], 
                              path_indices[:, 1]]
        new_len = self.current_routes.shape[-1]
        n_pad = max(new_len - new_parts.shape[-1], 0)
        new_parts = torch.nn.functional.pad(new_parts, (0, n_pad), value=-1)

        starting = (self.current_routes[:, 0] == -1) & ~routes_are_done
        extending = ~starting & ~routes_are_done
        ends_at_cur_start = path_indices[:, 1] == self.current_routes[:, 0]
        last_nodes = self.current_routes[self.batch_indices, 
                                         self.current_route_n_stops - 1]
        starts_at_cur_end = path_indices[:, 0] == last_nodes
        valid_action = starting | ends_at_cur_start | starts_at_cur_end | \
            routes_are_done
        assert valid_action.all(), "invalid action!"

        chose_prev = extending & ends_at_cur_start
        chose_next = extending & ~ends_at_cur_start

        first_parts = new_parts * (starting | chose_prev)[:, None] + \
            self.current_routes * (chose_next | routes_are_done)[:, None]
        second_parts = new_parts * chose_next[:, None] + \
            self.current_routes * chose_prev[:, None] + \
            -1 * ~(chose_next | chose_prev)[:, None]
        # cut off the first node of the second parts, since they overlap
        second_parts = second_parts[..., 1:]
        # combine the two parts without dummy nodes in between
        updated_routes = torch.full((self.batch_size, self.max_n_nodes), -1, 
                                    device=self.device)
        updated_routes[..., :first_parts.shape[-1]] = first_parts
        # insert the second parts at the appropriate masks
        first_part_lens = (first_parts > -1).sum(dim=-1)
        scnd_part_lens = (second_parts > -1).sum(dim=-1)
        new_route_lens = first_part_lens + scnd_part_lens
        scnd_part_mask = tu.get_variable_slice_mask(
            updated_routes, dim=1, froms=first_part_lens, tos=new_route_lens)
        scnd_part_len_mask = tu.get_variable_slice_mask(
            second_parts, dim=1, tos=scnd_part_lens)
        updated_routes[scnd_part_mask] = second_parts[scnd_part_len_mask]
        assert updated_routes.max() < self.max_n_nodes

        updated_routes = updated_routes.clamp(min=-1)

        updated_routes[self.is_done()] = -1

        # then, update the lists of routes that are still being planned
        planning_already_done = self.is_done()
        for bi in range(self.batch_size):
            if not planning_already_done[bi] and routes_are_done[bi]:
                route = updated_routes[bi]
                route = route.clone()[route > -1]
                if len(route) > 0:
                    # the route is valid, so add it to the finished set
                     # otherwise, corresponds to a "no-op" route
                    self._finished_routes[bi].append(route)
                    self.extra_data.total_route_time[bi] += \
                        self.current_route_time[bi]
            if planning_already_done[bi] or routes_are_done[bi]:
                updated_routes[bi] = -1

        self.extra_data.current_routes = updated_routes

        # finally, update all the internal stuff based on the current routes.
        # transforms it to a tensor if necessary
        self.extra_data.current_route_time = \
            self.get_total_route_time(updated_routes)

        ncr_times_from_start = torch.zeros_like(self.current_routes, 
                                                dtype=torch.float32)
        ncr_leg_times = tu.get_route_leg_times(self.current_routes,
                                               self.drive_times,
                                               self.mean_stop_time)
        # keep zeros at the start for the first stop
        ncr_times_from_start[:, 1:] = ncr_leg_times.cumsum(dim=1)
        self.extra_data.current_route_times_from_start = \
            ncr_times_from_start

        self._add_routes_to_tensors(updated_routes[:, None])

    def _clear_routes_helper(self, batch_index=None):
        if batch_index is None:
            batch_index = self.batch_indices
        elif type(batch_index) is not Tensor:
            batch_index = torch.tensor(batch_index, device=self.device)

        for bi in batch_index:
            self._finished_routes[bi] = []

        self.extra_data.valid_terms_mat[batch_index] = \
            self.extra_data.base_valid_terms_mat[batch_index].clone()
        self.extra_data.total_route_time[batch_index] = 0

        directly_connected = torch.eye(self.max_n_nodes, device=self.device, 
                                       dtype=bool)
        self.extra_data.directly_connected[batch_index] = \
            directly_connected.expand(len(batch_index), -1, -1)
        transit_times = (1 - torch.eye(self.max_n_nodes, device=self.device))
        transit_times[transit_times > 0] = float('inf')
        self.extra_data.route_mat[batch_index] = transit_times
        self.extra_data.transit_times[batch_index] = transit_times
        self.extra_data.has_path[batch_index] = \
            self.directly_connected[batch_index]
        self.extra_data.current_routes[batch_index] = -1
        self.extra_data.current_route_time[batch_index] = 0
        self.extra_data.current_route_times_from_start[batch_index] = 0
        self.extra_data.route_nexts[batch_index] = 0
        self.extra_data.n_transfers[batch_index] = 0

    def _add_routes_to_tensors(self, batch_new_routes,
                               only_routes_with_demand_are_valid=False, 
                               invalid_directly_connected=False):
        """Takes a tensor of new routes. The first dimension is the batch"""
        # incorporate new routes into the route graphs
        if type(batch_new_routes) is list:
            batch_new_routes = tu.get_batch_tensor_from_routes(batch_new_routes,
                                                               self.device)
        if batch_new_routes.device != self.device:
            batch_new_routes = batch_new_routes.to(self.device)
        # add new routes to the route matrix.
        new_route_mat = tu.get_route_edge_matrix(batch_new_routes, 
            self.drive_times, self.mean_stop_time, self.symmetric_routes)
        self.extra_data.route_mat = \
            torch.minimum(self.route_mat, new_route_mat)

        self.directly_connected[self.route_mat < float('inf')] = True

        # allow connection to any node 'upstream' of a demand dest, or
         # 'downstream' of a demand src.
        if only_routes_with_demand_are_valid:
            float_is_demand = (self.demand > 0).to(torch.float32)
            connected_T = self.nodes_are_connected(2).transpose(1, 2)
            connected_T = connected_T.to(torch.float32)
            valid_upstream = float_is_demand.bmm(connected_T)
            self.extra_data.valid_terms_mat[valid_upstream.to(bool)] = True
            valid_downstream = connected_T.bmm(float_is_demand)
            self.extra_data.valid_terms_mat[valid_downstream.to(bool)] = True

        if invalid_directly_connected:
            self.extra_data.valid_terms_mat[self.directly_connected] = False

        if self.symmetric_routes:
            self.extra_data.valid_terms_mat = self.valid_terms_mat & \
                self.valid_terms_mat.transpose(1, 2)

        self._update_route_data()
    
    def replace_routes(self, batch_new_routes, 
                only_routes_with_demand_are_valid=False, 
                invalid_directly_connected=False):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
        self.add_new_routes(batch_new_routes, 
                            only_routes_with_demand_are_valid,
                            invalid_directly_connected)

    def clear_routes(self):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
        self._update_route_data()

    def reset_dones(self):
        batch_index = torch.where(self.is_done())[0]
        if len(batch_index) > 0:
            self._clear_routes_helper(batch_index)

    def add_new_routes(self, batch_new_routes,
                       only_routes_with_demand_are_valid=False, 
                       invalid_directly_connected=False):
        if type(batch_new_routes) is list:
            batch_new_routes = tu.get_batch_tensor_from_routes(batch_new_routes,
                                                               self.device)

        self._add_routes_to_tensors(batch_new_routes, 
                                    only_routes_with_demand_are_valid, 
                                    invalid_directly_connected)
        self._add_routes_to_list(batch_new_routes)

        # finally, update the total route times with the new routes
        leg_times = tu.get_route_leg_times(batch_new_routes, 
                                           self.graph_data.drive_times,
                                           self.mean_stop_time)
        total_new_time = leg_times.sum(dim=(1,2))

        if self.symmetric_routes:                                           
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = tu.get_route_leg_times(batch_new_routes, 
                                                      transpose_dtm,
                                                      self.mean_stop_time)
            total_new_time += return_leg_times.sum(dim=(1,2))

        self.extra_data.total_route_time += total_new_time        

    def _add_routes_to_list(self, batch_routes):
        for bi in range(self.batch_size):
            for route in batch_routes[bi]:
                if type(route) is list:
                    route = torch.tensor(route, device=self.device)
                length = (route > -1).sum()
                if length < 2:
                    # this is an invalid route
                    log.warn('invalid route!')
                    continue
                self._finished_routes[bi].append(route[:length])
    
    def _update_route_data(self):
        # do things that have to be done whether we added or removed routes
        fw_mat = self.route_mat + self.transfer_time_s[:, None, None]
        nexts, transit_times = tu.floyd_warshall(fw_mat)
        _, path_edge_counts = tu.reconstruct_all_paths(nexts)
        # subtract one transfer time from each transit time to avoid counting
         # a transfer for the first edge of a journey
        transit_times -= self.transfer_time_s[:, None, None]

        self.extra_data.route_nexts = nexts
        self.extra_data.transit_times = transit_times
        self.extra_data.has_path = transit_times < float('inf')
        # number of transfers is number of nodes except start and end
        n_transfers = (path_edge_counts - 2).clamp(min=0)
        # set number of transfers where there is no path to 0
        n_transfers[~self.has_path] = 0
        self.extra_data.n_transfers = n_transfers

    def set_normalized_features(self, norm_stop_features):
        self.extra_data.norm_node_features = norm_stop_features

    def clone(self):
        """return a deep copy of this state."""
        return copy.deepcopy(self)
    
    def to_device(self, device):
        dev_state = self.clone()
        dev_state.graph_data = dev_state.graph_data.to(device)
        dev_state.extra_data = dev_state.extra_data.to(device)
        for ii in range(self.batch_size):
            for jj, route in enumerate(dev_state._finished_routes[ii]):
                dev_state._finished_routes[ii][jj] = route.to(device)
         
        return dev_state
    
    @staticmethod
    def batch_from_list(state_list):
        """return a batch state from a list of states."""
        if len(state_list) == 1:
            return state_list[0]
        
        graph_datas = sum([ss.graph_data.to_data_list() for ss in state_list],
                          [])
        extra_datas = sum([ss.extra_data.to_data_list() for ss in state_list], 
                          [])
        batch_graph_data = Batch.from_data_list(graph_datas)
        batch_extra_data = Batch.from_data_list(extra_datas)
        batch_state = copy.copy(state_list[0])
        batch_state.graph_data = batch_graph_data
        batch_state.extra_data = batch_extra_data
        copied_routes = [[copy.copy(bfr) for bfr in ss._finished_routes]
                         for ss in state_list]
        batch_state._finished_routes = sum(copied_routes, [])
        return batch_state
    
    def batch_to_list(self):
        """return a list of RouteGenBatchState objects, one for each element in
        this batch."""
        if self.batch_size == 1:
            return [self]

        graph_datas = self.graph_data.to_data_list()
        extra_datas = self.extra_data.to_data_list()
        state_list = []
        for gd, ed, routes in zip(graph_datas, extra_datas, 
                                  self._finished_routes):
            state = copy.copy(self)
            state.graph_data = Batch.from_data_list([gd])
            state.extra_data = Batch.from_data_list([ed])
            state._finished_routes = [copy.copy(routes)]
            state_list.append(state)

        return state_list
    
    def index_select(self, idx: Union[slice, Tensor, ndarray, Sequence]):
        """return a new state with only the given indices."""
        state = copy.copy(self)
        state.graph_data = self.graph_data.index_select(idx)
        state.extra_data = self.extra_data.index_select(idx)
        if isinstance(idx, slice):
            state._finished_routes = self._finished_routes[idx]
        else:
            state._finished_routes = [self._finished_routes[ii] for ii in idx]
        return state

    def is_done(self):
        return self.n_routes_left_to_plan == 0

    def get_total_route_time(self, batch_routes):
        if batch_routes.ndim == 2:
            # add a routes dimension
            batch_routes = batch_routes[:, None]

        leg_times = tu.get_route_leg_times(batch_routes, 
                                           self.graph_data.drive_times,
                                           self.mean_stop_time)
        route_time = leg_times.sum(dim=(1,2))

        if self.symmetric_routes:
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = tu.get_route_leg_times(batch_routes, 
                                                      transpose_dtm,
                                                      self.mean_stop_time)
            route_time += return_leg_times.sum(dim=(1,2))

        return route_time
    
    def get_global_state_features(self):
        cost_weights = self.cost_weights_tensor
        diameter = self.drive_times.flatten(1,2).max(1).values
        mean_route_time = self.total_route_time / (
            self.n_routes_to_plan * diameter)

        so_far = self.n_finished_routes
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        n_routes_log_feats = (both + 1).log()
        # use fractions so it's independent of n_routes_to_plan
        n_routes_frac_feats = both / (so_far + left)[:, None]

        n_disconnected_demand_edges = self.get_n_disconnected_demand_edges()
        # as with n_routes feats, use both log and fractional
        log_uncovered = (n_disconnected_demand_edges + 1).log()
        frac_uncovered = n_disconnected_demand_edges / self.n_demand_edges
        uncovered_feats = torch.stack((log_uncovered, 
                                       frac_uncovered
                                     ), dim=-1)
        curr_route_n_stops = self.current_route_n_stops[:, None]

        served_demand = (self.has_path * self.demand).sum(dim=(1, 2))
        tt = self.transit_times.clone()
        tt[~self.has_path] = 0
        total_demand_time = (self.demand * tt).sum(dim=(1,2))
        mean_demand_time = total_demand_time / (served_demand + EPSILON)
        mean_demand_time_frac = mean_demand_time / diameter

        global_features = torch.cat((
            cost_weights, mean_route_time[:, None], n_routes_log_feats, 
            n_routes_frac_feats, uncovered_feats, curr_route_n_stops,
            mean_demand_time_frac[:, None],
        ), dim=-1)
        return global_features

    def get_n_disconnected_demand_edges(self):
        # count the number of demand edges that are disconnected
        nopath = ~self.has_path
        needed_path_missing = nopath & (self.demand > 0)
        n_disconnected_demand_edges = needed_path_missing.sum(dim=(1, 2))
        if self.symmetric_routes:
            # connecting one of the two connects both, so count each 2 as 
             # just 1.
            n_disconnected_demand_edges = n_disconnected_demand_edges / 2
        return n_disconnected_demand_edges

    def get_shortest_path_sequences(self):
        if self.extra_data.shortest_path_sequences.numel() == 0:
            path_seqs, _ = tu.reconstruct_all_paths(self.nexts)
            self.extra_data.shortest_path_sequences = path_seqs
        else:
            path_seqs = self.extra_data.shortest_path_sequences
        return path_seqs
    
    def set_cost_weights(self, new_cost_weights):
        for key, val in new_cost_weights.items():
            # expand the cost weights to match the batch
            self.extra_data.cost_weights[key][...] = val

    @property
    def routes(self):
        """Returns the collection of all routes, including the one currently
            being planned if it has any stops."""
        # copy the lists of routes for each batch element.
        routes = [copy.copy(fr) for fr in self._finished_routes]
        if self.current_routes is not None:
            for batch_idx, current_route in enumerate(self.current_routes):
                if (current_route > -1).sum() > 1:
                    # the route being planned has at least 2 stops, so it is
                     # functional.  Add it.
                    current_route = current_route[current_route > -1]
                    routes[batch_idx].append(current_route)

        return routes
    
    @property
    def batch_indices(self):
        return torch.arange(self.batch_size, device=self.device)
    
    @property
    def n_demand_edges(self):
        n_demand_edges = (self.demand > 0).sum(dim=(1, 2))
        if self.symmetric_routes:
            n_demand_edges = (n_demand_edges / 2).ceil()
        return n_demand_edges

    @property
    def cost_weights_tensor(self):
        cost_weights_list = []
        for key in sorted(self.cost_weights.keys()):
            if type(self.cost_weights[key]) is Tensor:
                cw = self.cost_weights[key].to(self.device)
                if cw.ndim == 0:
                    cw = cw[None]
            else:
                cw = torch.tensor(self.cost_weights[key], 
                                  device=self.device)[None]

            cost_weights_list.append(cw)
        cost_weights = torch.stack(cost_weights_list, dim=1)
        if cost_weights.shape[0] == 1:
            cost_weights = cost_weights.expand(self.batch_size, -1)
        if cost_weights.shape[0] > self.batch_size:
            cost_weights = cost_weights[:self.batch_size]
        return cost_weights
    
    @property
    def node_covered_mask(self):
        have_out_paths = self.directly_connected.any(dim=1)
        if self.symmetric_routes:
            are_covered = have_out_paths
        else:
            are_covered = have_out_paths & self.directly_connected.any(dim=2)
        return are_covered

    # don't expose the extra data directly, just provide this interface.
    @property
    def norm_node_features(self):
        if hasattr(self.extra_data, 'norm_node_features'):
            return self.extra_data.norm_node_features
        else:
            return None
        
    @property
    def norm_cost_weights(self):
        if hasattr(self.extra_data, 'norm_cost_weights'):
            return self.extra_data.norm_cost_weights
        else:
            return None
    
    @property
    def current_routes(self):
        return self.extra_data.current_routes
                
    @property
    def current_route_times_from_start(self):
        return self.extra_data.current_route_times_from_start
    
    @property
    def current_route_time(self):
        return self.extra_data.current_route_time
    
    @property
    def current_route_n_stops(self):
        return (self.current_routes > -1).sum(dim=-1)
    
    @property
    def has_current_route(self):
        return self.current_route_n_stops > 0

    @property
    def n_routes_to_plan(self):
        return self.extra_data.n_routes_to_plan
    
    @property
    def valid_terms_mat(self):
        return self.extra_data.valid_terms_mat
    
    @property
    def mean_stop_time(self):
        return self.extra_data.mean_stop_time
    
    @property
    def transfer_time_s(self):
        return self.extra_data.transfer_time_s

    @property
    def total_route_time(self):
        # time of finished routes plus time of in-progress routes
        return self.extra_data.total_route_time + self.current_route_time    

    @property
    def min_route_len(self):
        return self.extra_data.min_route_len
    
    @property
    def max_route_len(self):
        return self.extra_data.max_route_len
    
    @property
    def n_nodes(self):
        return self.extra_data.n_nodes_in_scenario
    
    @property
    def alpha(self):
        return self.extra_data.cost_weights['demand_time_weight']

    @property
    def cost_weights(self):
        return self.extra_data.cost_weights
    
    @property
    def directly_connected(self):
        return self.extra_data.directly_connected

    @property
    def nexts(self):
        return self.graph_data.nexts

    @property
    def route_mat(self):
        return self.extra_data.route_mat
    
    @property
    def route_nexts(self):
        return self.extra_data.route_nexts
    
    @property
    def transit_times(self):
        return self.extra_data.transit_times
    
    @property
    def has_path(self):
        return self.extra_data.has_path
    
    @property
    def n_transfers(self):
        return self.extra_data.n_transfers

    @property
    def street_adj(self):
        return self.graph_data.street_adj

    @property
    def demand(self):
        return self.graph_data.demand
    
    @property
    def drive_times(self):
        return self.graph_data.drive_times

    @property
    def n_finished_routes(self):
        nrsf = [len(rrs) for rrs in self._finished_routes]
        return torch.tensor(nrsf, dtype=torch.float32,
                            device=self.device)
    
    @property
    def batch_size(self):
        return self.graph_data.num_graphs

    @property
    def n_routes_left_to_plan(self):
        return self.n_routes_to_plan - self.n_finished_routes
    
    def get_n_routes_features(self):
        so_far = self.n_finished_routes
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        return (both + 1).log()

    def nodes_are_connected(self, n_transfers=2):
        dircon_float = self.directly_connected.to(torch.float32)
        connected = dircon_float
        for _ in range(n_transfers):
            # connected by 2 or fewer transfers
            connected = connected.bmm(dircon_float)
        return connected.bool()

    @property
    def device(self):
        return self.graph_data[STOP_KEY].x.device
    
    @property
    def max_n_nodes(self):
        return max(self.n_nodes)
    
    @property
    def route_n_stops(self):
        route_n_stops = [[len(rr) for rr in br] for br in self.routes]
        return torch.tensor(route_n_stops, device=self.device)


@dataclass
class CostHelperOutput:
    total_demand_time: Tensor
    total_route_time: Tensor
    trips_at_transfers: Tensor
    total_demand: Tensor
    unserved_demand: Tensor
    total_transfers: Tensor
    trip_times: Tensor
    n_disconnected_demand_edges: Tensor
    n_stops_oob: Tensor
    n_duplicate_stops: Tensor
    batch_routes: Tensor
    per_route_riders: Optional[Tensor] = None
    cost: Optional[Tensor] = None

    @property
    def mean_demand_time(self):
        served_demand = self.total_demand - self.unserved_demand
        # avoid division by 0
        return self.total_demand_time / (served_demand + EPSILON)

    def get_metrics(self):
        """return a dictionary with the metrics we usually report."""
        frac_tat = self.trips_at_transfers / self.total_demand[:, None]
        percent_tat = frac_tat * 100
        metrics = {
            'cost': self.cost,
            'ATT': self.mean_demand_time / 60,
            'RTT': self.total_route_time / 60,
            '$d_0$': percent_tat[:, 0],
            '$d_1$': percent_tat[:, 1],
            '$d_2$': percent_tat[:, 2],
            '$d_{un}$': percent_tat[:, 3],
            '# disconnected node pairs': 
                self.n_disconnected_demand_edges.float(),
            '# stops out of bounds': self.n_stops_oob.float(),
        }
        return metrics

    def get_metrics_tensor(self):
        """return a tensor with the metrics we usually report."""
        metrics = self.get_metrics()
        metrics = torch.stack([metrics[k] for k in metrics], dim=-1)
        return metrics
    
    def are_constraints_violated(self):
        return (self.n_stops_oob > 0) | \
               (self.n_disconnected_demand_edges > 0) | \
               (self.n_duplicate_stops > 0) # | \
               # (self.n_skipped_stops > 0)
    

class CostModule(torch.nn.Module):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False):
        super().__init__()
        self.mean_stop_time_s = mean_stop_time_s
        self.avg_transfer_wait_time_s = avg_transfer_wait_time_s
        self.symmetric_routes = symmetric_routes
        self.low_memory_mode = low_memory_mode

    def get_metric_names(self):
        dummy_obj = CostHelperOutput(
            torch.zeros(1), torch.zeros(1), torch.zeros(1, 4), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        return dummy_obj.get_metrics().keys()

    def _cost_helper(self, state, return_per_route_riders=False):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        drive_times_matrix = state.drive_times
        demand_matrix = state.demand
        dev = drive_times_matrix.device

        # assemble route graph
        batch_routes = tu.get_batch_tensor_from_routes(state.routes, dev)
        route_lens = (batch_routes > -1).sum(-1)

        log.debug("summing cost components")

        zero = torch.zeros_like(route_lens)
        route_len_delta = (state.min_route_len[:, None] - route_lens)
        route_len_delta = route_len_delta.maximum(zero)
        # don't penalize placeholer "dummy" routes in the tensor
        route_len_delta[route_lens == 0] = 0
        if state.max_route_len is not None:
            route_len_over = (route_lens - state.max_route_len[:, None])
            route_len_delta = route_len_delta + route_len_over.maximum(zero)
        
        # if there is a current route, it's already included in route_len_delta
        n_unstarted_routes = \
            state.n_routes_left_to_plan - \
                (state.has_current_route).to(torch.float32)
        n_stops_oob = route_len_delta.sum(-1) + \
            n_unstarted_routes * state.min_route_len

        assert (n_stops_oob >= 0).all(), "negative stops-out-of-bounds!"

        # calculate the amount of demand at each number of transfers
        trips_at_transfers = torch.zeros(state.batch_size, 4, device=dev)
        # trips with no path get '3' transfers so they'll be included in d_un,
         # not d_0
        n_transfers = state.n_transfers.clone()
        nopath = ~state.has_path
        n_transfers[nopath] = 3
        for ii in range(3):
            d_i = (demand_matrix * (n_transfers == ii)).sum(dim=(1, 2))
            trips_at_transfers[:, ii] = d_i
        
        d_un = (demand_matrix * (n_transfers > 2)).sum(dim=(1, 2))
        trips_at_transfers[:, 3] = d_un

        # calculate some more quantities of interest
        trip_times = state.transit_times.clone()
        trip_times[nopath] = 0
        demand_time = demand_matrix * trip_times
        total_dmd_time = demand_time.sum(dim=(1, 2))
        demand_transfers = demand_matrix * state.n_transfers
        total_transfers = demand_transfers.sum(dim=(1, 2))
        unserved_demand = (demand_matrix * nopath).sum(dim=(1, 2))
        total_demand = demand_matrix.sum(dim=(1,2))

        n_duplicate_stops = count_duplicate_stops(state.max_n_nodes, 
                                                  batch_routes)
        # n_skipped_stops = count_skipped_stops(batch_routes)

        output = CostHelperOutput(
            total_dmd_time, state.total_route_time, trips_at_transfers, 
            total_demand, unserved_demand, total_transfers, trip_times,
            state.get_n_disconnected_demand_edges(), n_stops_oob, 
            n_duplicate_stops, batch_routes
        )

        if return_per_route_riders:
            _, used_routes = \
                tu.get_route_edge_matrix(batch_routes, drive_times_matrix,
                                         self.mean_stop_time_s, 
                                         self.symmetric_routes, 
                                         self.low_memory_mode, 
                                         return_used_routes=True)

            used_routes.unsqueeze_(-1)
            route_seqs = tu.aggregate_edge_features(state.route_nexts, 
                                                    used_routes, 'concat')
            route_seqs.squeeze_(-1)
            per_route_riders = torch.zeros(batch_routes.shape[:2], device=dev)
            for bi in range(state.batch_size):
                for ri in range(batch_routes.shape[1]):
                    srcs, dsts, _ = torch.where(route_seqs[bi] == ri)
                    ri_demand = demand_matrix[bi, srcs, dsts].sum()
                    per_route_riders[bi, ri] = ri_demand

            output.per_route_riders = per_route_riders
        
        return output


class MyCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False,
                 demand_time_weight=0.5, route_time_weight=0.5, 
                 constraint_violation_weight=5, variable_weights=False,
                 ignore_stops_oob=False, pp_fraction=0.33, 
                 op_fraction=0.33):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes, low_memory_mode)
        self.demand_time_weight = demand_time_weight
        self.route_time_weight = route_time_weight
        self.constraint_violation_weight = constraint_violation_weight
        self.variable_weights = variable_weights
        if self.variable_weights:
            # the fraction of variable weights sampled that are PP and OP
            self.pp_fraction = pp_fraction
            self.op_fraction = op_fraction
            assert pp_fraction + op_fraction <= 1, \
                "fractions of extreme samples must sum to <= 1"
        self.ignore_stops_oob = ignore_stops_oob

    def sample_variable_weights(self, batch_size, device=None):
        if not self.variable_weights:
            dtw = torch.full((batch_size,), self.demand_time_weight, 
                             device=device)
            rtw = torch.full((batch_size,), self.route_time_weight, 
                              device=device)
        else:
            random_number = torch.rand(batch_size, device=device)
            # Initialized to zero, which is the right value for OP
            dtw = torch.zeros(batch_size, device=device)
            # Set demand time weight to 1 where we're using PP
            is_pp = random_number < self.pp_fraction
            dtw[is_pp] = 1.0
            # Set demand time weight to a random value in [0,1] where it's
             # neither OP nor PP
            extremes_fraction = self.pp_fraction + self.op_fraction
            is_intermediate = random_number >= extremes_fraction
            n_intermediate = is_intermediate.sum()
            dtw[is_intermediate] = torch.rand(n_intermediate, device=device)

            # reward time weight is 
            rtw = 1 - dtw
 
        return {
            'demand_time_weight': dtw,
            'route_time_weight': rtw
        }
    
    def get_weights(self, device=None):
        dtm = self.demand_time_weight
        if type(dtm) is not Tensor:
            dtm = torch.tensor([dtm], device=device)
        rtm = self.route_time_weight
        if type(rtm) is not Tensor:
            rtm = torch.tensor([rtm], device=device)
        
        return {
            'demand_time_weight': dtm,
            'route_time_weight': rtm
        }
    
    def set_weights(self, demand_time_weight=None, route_time_weight=None, 
                    constraint_violation_weight=None):
        if demand_time_weight is not None:
            self.demand_time_weight = demand_time_weight
        if route_time_weight is not None:
            self.route_time_weight = route_time_weight
        if constraint_violation_weight is not None:
            self.constraint_violation_weight = constraint_violation_weight

    def forward(self, state, constraint_weight=None, no_norm=False, 
                return_per_route_riders=False):
        cho = self._cost_helper(state, return_per_route_riders)
        cost_weights = state.cost_weights
        if 'demand_time_weight' in cost_weights:
            demand_time_weight = cost_weights['demand_time_weight']
        else:
            demand_time_weight = self.demand_time_weight
        if 'route_time_weight' in cost_weights:
            route_time_weight = cost_weights['route_time_weight']
        else:
            route_time_weight = self.route_time_weight
            
        if constraint_weight is None:
            constraint_weight = self.constraint_violation_weight

        # if we have more weights than routes, truncate the weights
        if type(demand_time_weight) is Tensor and \
           demand_time_weight.shape[0] > state.batch_size:
            demand_time_weight = demand_time_weight[:state.batch_size]
        if type(route_time_weight) is Tensor and \
           route_time_weight.shape[0] > state.batch_size:
            route_time_weight = route_time_weight[:state.batch_size]
        if type(constraint_weight) is Tensor and \
           constraint_weight.shape[0] > state.batch_size:
            constraint_weight = constraint_weight[:state.batch_size]

        # normalize all time values by the maximum drive time in the graph
        time_normalizer = state.drive_times.flatten(1,2).max(1).values

        n_routes = state.n_routes_to_plan

        # fraction of demand not covered by routes, and fraction of routes
        frac_uncovered = cho.n_disconnected_demand_edges / state.n_demand_edges
        if state.max_route_len is None:
            denom = n_routes * state.max_route_len
        else:
            denom = n_routes * state.min_route_len
        # avoid division by 0
        denom[denom == 0] = 1
        # unserved demand is treated as taking twice the diameter of the graph
         # to get where it's going
        served_demand = cho.total_demand - cho.unserved_demand
        demand_cost = cho.mean_demand_time * served_demand + \
            cho.unserved_demand * time_normalizer * 2
        demand_cost /= cho.total_demand
        # demand_cost = cho.mean_demand_time
        route_cost = cho.total_route_time

        # average trip time, total route time, and trips-at-n-transfers
        if not no_norm:
            # normalize cost components
            demand_cost = demand_cost / time_normalizer
            route_cost = route_cost / (time_normalizer * n_routes + 1e-6)

        cost = demand_cost * demand_time_weight + \
            route_cost * route_time_weight

        # compute the weight for the violated-constraint penalty, as an
         # upper bound on how bad the demand and route cost components may be
        # demand_constraint_weight = 2
        # edge_times = state.street_adj.isfinite() * state.drive_times
        # max_edge_time = edge_times.flatten(1,2).max(-1)[0]
        # route_constraint_weight = 2 * max_edge_time * state.n_nodes
        # route_constraint_weight /= time_normalizer
        # dynamic_cv_weight = demand_constraint_weight * demand_time_weight + \
        #     route_constraint_weight * route_time_weight
        # constraint_weight *= dynamic_cv_weight 

        const_viol_cost = frac_uncovered + 0.1 * (frac_uncovered > 0)
        if not self.ignore_stops_oob:
            frac_stops_oob = cho.n_stops_oob / denom
            const_viol_cost += frac_stops_oob + 0.1 * (frac_stops_oob > 0)

        cost += const_viol_cost * constraint_weight
        cho.cost = cost

        assert cost.isfinite().all(), "invalid cost was computed!"
        assert (cost >= 0).all(), "cost is negative!"

        return cho
    

class MultiObjectiveCostModule(MyCostModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variable_weights = True
        # don't sample any edge cases
        self.pp_fraction = 0.33
        self.op_fraction = 0.33

    def sample_weights(self, batch_size, device=None):
        weights = self.sample_variable_weights(batch_size, device)
        # introduce one of each type of edge case
        weights['demand_time_weight'][0] = 1.0
        weights['demand_time_weight'][1] = 0.0
        weights['route_time_weight'][0] = 0.0
        weights['route_time_weight'][1] = 1.0
        return weights

    def forward(self, state, return_per_route_riders=False):
        cho = self._cost_helper(state, return_per_route_riders)
        costs = torch.stack((cho.mean_demand_time, cho.total_route_time), 
                            dim=-1)
        any_violations = cho.are_constraints_violated()
        return costs, any_violations


class NikolicCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False,
                 unsatisfied_penalty_extra_s=UNSAT_PENALTY_EXTRA_S, 
                 ):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes, low_memory_mode)
        self.unsatisfied_penalty_extra_s = unsatisfied_penalty_extra_s

    def forward(self, state):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        cho = self._cost_helper(state)
        # Note that unlike Nikolic, we count trips that take >2 transfers as 
         # satisfied.
        tot_sat_demand = cho.total_demand - cho.unserved_demand
        w_2 = cho.total_demand_time / tot_sat_demand
        no_sat_dmd = torch.isclose(tot_sat_demand, 
                                   torch.zeros_like(tot_sat_demand))
        # if no demand is satisfied, set w_2 to the average time of all trips 
         # plus the penalty
        w_2[no_sat_dmd] = cho.trip_times[no_sat_dmd].mean(dim=(-2,-1))
        w_2 += self.unsatisfied_penalty_extra_s

        cost = cho.total_demand_time + w_2 * cho.unserved_demand

        assert not ((cost == 0) & (cho.total_demand > 0)).any()

        log.debug("finished nikolic")
        assert cost.isfinite().all(), "invalid cost was computed!"

        cho.cost = cost
        
        return cho

    def get_weights(self, device=None):
        return {}
    

def get_cost_module_from_cfg(cost_cfg: DictConfig, low_memory_mode=False,
                             symmetric_routes=True):
    # setup the cost function
    if cost_cfg.type == 'nikolic':
        cost_obj = NikolicCostModule(low_memory_mode=low_memory_mode, 
                                     **cost_cfg.kwargs)
    elif cost_cfg.type == 'mine':
        cost_obj = MyCostModule(low_memory_mode=low_memory_mode, 
                                **cost_cfg.kwargs)
    elif cost_cfg.type == 'multi':
        cost_obj = MultiObjectiveCostModule(
            low_memory_mode=low_memory_mode,
            **cost_cfg.kwargs)                                            
    return cost_obj


def check_for_duplicate_routes(routes_tensor):
    """check if any routes are duplicates of each other.

    In theory we want to avoid duplicate routes.  But no other work seems to 
    care.  Mumford doesn't check for them.b

    routes_tensor: a tensor of shape (batch_size, n_routes, route_len)"""
    routes_tensor = routes_tensor[:, None]
    same_stops = routes_tensor == routes_tensor.transpose(1, 2)
    routes_are_identical = same_stops.all(dim=-1)
    any_routes_are_identical = routes_are_identical.any(-1).any(-1)
    return any_routes_are_identical


def network_is_connected(routes_tensor, n_nodes):
    if routes_tensor.ndim == 2:
        # add a batch dimension
        routes_tensor = routes_tensor[None]

    are_connected = torch.zeros(routes_tensor.shape[0], dtype=bool)
    for bi in range(routes_tensor.shape[0]):
        elem_routes = routes_tensor[bi]
        adj_matrix = torch.eye(n_nodes+1, dtype=bool)
        for route in elem_routes:
            for src, dst in zip(route[:-1], route[1:]):
                adj_matrix[src, dst] = True
        # make symmetrical
        adj_matrix = adj_matrix | adj_matrix.t()
        # remove the dummy node entries of adj_matrix
        adj_matrix = adj_matrix[:-1, :-1]

        visited = [False] * n_nodes
        start_node = 0  # Start from an arbitrary node
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            if not visited[node]:
                visited[node] = True
                for neighbour, is_connected in enumerate(adj_matrix[node]):
                    if is_connected and not visited[neighbour]:
                        queue.append(neighbour)
        
        are_connected[bi] = all(visited)
    
    return are_connected


def count_duplicate_stops(n_nodes, networks):
    if networks.ndim == 2:
        batch_size = 1
        networks = networks[None]
    else:
        batch_size = networks.shape[0]
    # count duplicate stops in routes
    route_lens = (networks > -1).sum(dim=-1)
    nodes_on_routes = tu.get_nodes_on_routes_mask(n_nodes, networks)
    n_nodes_covered = nodes_on_routes[..., :-1].sum(-1)
    n_ntwk_duplicates = (route_lens - n_nodes_covered).sum(-1)

    return n_ntwk_duplicates


def count_skipped_stops(network, shortest_path_lens):
    path_n_skips = shortest_path_lens - 2
    path_n_skips.clamp_(min=0)
    leg_n_skips = tu.get_route_leg_times(network, path_n_skips)
    total_n_skips = leg_n_skips.sum(dim=(1, 2))
    return total_n_skips
