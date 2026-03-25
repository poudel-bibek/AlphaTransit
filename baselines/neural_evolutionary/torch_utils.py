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

from pathlib import Path
import pickle
from typing import Tuple

import torch

# currently unused, but might be useful in the future
def batch_2combinations(sequences):
    dev = sequences.device
    max_seq_len = sequences.shape[-1]
    triu_idxs = torch.triu_indices(max_seq_len, max_seq_len, 1, device=dev)
    # flatten all the non_sequence dimensions into one
    flat_seqs = sequences.reshape(-1, max_seq_len)
    from_grid = flat_seqs[..., None].expand(-1, -1, max_seq_len)
    from_idxs = from_grid[..., triu_idxs[0], triu_idxs[1]]
    to_grid = flat_seqs[..., None, :].expand(-1, max_seq_len, -1)
    to_idxs = to_grid[..., triu_idxs[0], triu_idxs[1]]
    flat_combinations = torch.stack([from_idxs, to_idxs], dim=-1)
    combinations = flat_combinations.reshape(*sequences.shape[:-1], -1, 2)
    return combinations


def get_update_at_mask(tensor, mask, new_value=0):
    """Pytorch's backprop doesn't like in-place updates to tensors in the 
        computation graph, so we often need to get a new tensor with the 
        same values at unmasked locations and a new value at masked locations.
        The way of doing it is a little counter-intuitive, so we've defined 
        a helper function to hide the details.
        
        new_value: a scalar or a tensor matching the shape of tensor and mask.
            Defaults to 0.
        """
    src_elem_is_inf = tensor.isinf()
    src_has_inf = src_elem_is_inf.any()
    if src_has_inf:
        # multiplying infinity by 0 gives nan, so clone and assign instead
        tensor = tensor.clone()
        tensor[mask] = 0

    if type(new_value) is torch.Tensor:
        new_has_inf = new_value.isinf().any()
        if new_has_inf:
            new_value = new_value.clone()
    else:
        # new value is a scalar
        new_has_inf = abs(new_value) == float('inf')
        if new_has_inf:
            new_value = torch.full_like(tensor, new_value)
    
    if new_has_inf:
        # multiplying infinity by 0 gives nan, so clone and assign instead
        new_value = new_value.clone()
        new_value[~mask] = 0

    # add singleton dimensions to the mask as needed for the below arithmetic
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    updated = tensor * ~mask + new_value * mask

    return updated


def square_pdist(vectors):
    flat_dists = torch.pdist(vectors)
    num_vecs = vectors.shape[0]
    square = torch.zeros((num_vecs, num_vecs))
    end_i = 0
    for ii in range(num_vecs - 1):
        start_i = end_i
        end_i = start_i + num_vecs - (ii + 1)
        square[ii, ii + 1:] = flat_dists[start_i:end_i]
    # make it diagonally symmetrical
    square += square.T
    return square


class AllShortestPaths:
    def __init__(self, nexts, dists, precompute=False):
        self.dists = dists
        self.nexts = nexts
        self._cached_paths = {}
        if precompute:
            seq_tensor, route_lens = reconstruct_all_paths(nexts)
            for from_i in range(nexts.shape[-1]):
                tos = {}
                for to_i in range(nexts.shape[-1]):
                    if to_i == from_i:
                        continue
                    path = seq_tensor[from_i, to_i, :route_lens[from_i, to_i]]
                    tos[to_i] = path
                self._cached_paths[from_i] = tos

    def get_length(self, source, dest):
        return self.dists[source, dest]

    def get_path(self, source, dest):
        if (source, dest) in self._cached_paths:
            return self._cached_paths[(source, dest)]
        else:
            path = reconstruct_path(source, dest, self.nexts)
            self._cached_paths[(source, dest)] = path
            return path


def floyd_warshall(edge_cost_tensor, return_raw_tensors=True):
    """A pytorch-vectorized implementation of the Floyd-Warshall algorithm.
    
    Input tensor should be a matrix of edge costs, where element (u,v) is the
    cost of traversing the edge from u to v if there is one, and is otherwise
    infinity.  It may have a third batch dimension.

    We vectorize the two inner loops of the algorithm.  The outermost loop must
    be sequential: the calculation at i assumes results from iteration i-1 are
    present.

    For the pseudocode of the unvectorized algorithm, see:
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction
    """
    has_batch_dim = edge_cost_tensor.ndim == 3
    if not has_batch_dim:
        # add a batch dimension
        edge_cost_tensor = edge_cost_tensor[None]

    # put the batch dimension last
    edge_cost_tensor = edge_cost_tensor.permute(1, 2, 0)
    batch_size = edge_cost_tensor.shape[2]

    device = edge_cost_tensor.device
    dists = edge_cost_tensor.clone().detach()

    # set up next-node array
    num_nodes = dists.shape[0]
    nexts = -torch.ones((num_nodes, num_nodes, batch_size), 
                        dtype=int, device=device)
    nexts[torch.eye(num_nodes, dtype=bool)] = \
        torch.arange(num_nodes, device=device)[:, None]
    
    has_edge_mat = dists < float("inf")
    for vv in range(num_nodes):
        has_edge_to_v = has_edge_mat[:, vv]
        nexts[has_edge_to_v, vv] = vv

    # the core loop
    for kk in range(num_nodes):
        # vectorize two levels
        from_k_dists = dists[kk]
        to_k_dists = dists[:, kk]
        through_k_dists = to_k_dists[:, None] + from_k_dists[None]
        update_mask = dists > through_k_dists
        dists[update_mask] = through_k_dists[update_mask]

        # tile the k-th column into N x N
        new_nexts = nexts[:, kk:kk+1].repeat(1, num_nodes, 1)
        nexts[update_mask] = new_nexts[update_mask]

    # handle different batch-size cases for the returned value
    nexts = nexts.permute(2, 0, 1)
    dists = dists.permute(2, 0, 1)

    if return_raw_tensors:
        return nexts, dists
    else:
        if not has_batch_dim:
            retval = AllShortestPaths(nexts.squeeze(0), dists.squeeze(0))
        else:
            retval = [AllShortestPaths(nn, dd) for nn, dd in zip(nexts, dists)]
        return retval


def reconstruct_path(source, dest, nexts):
    """Given a source, dest, and the nexts matrix returned by floyd-warshall,
       returns the shortest path from source to dest."""
    if nexts[source, dest] < 0:
        return None
    path = [source]
    curr_node = source
    while curr_node != dest:
        curr_node = nexts[curr_node, dest]
        path.append(curr_node)
    # return a list of integers, not tensors
    path = [ee.item() if type(ee) is torch.Tensor else ee for ee in path]
    return path


def reconstruct_all_paths(nexts):
    n_nodes = nexts.shape[-1]
    node_idxs = torch.arange(n_nodes, device=nexts.device)[:, None]
    if nexts.ndim == 3:
        batch_size = nexts.shape[0]
        node_idxs = node_idxs.expand(batch_size, -1, -1)
    path_tensor, path_lens = \
        aggregate_node_features(nexts, node_idxs, agg_mode="concat",
                                return_node_counts=True)
    if nexts.ndim == 3 or nexts.shape[-1] != 1:
        path_tensor.squeeze_(-1)

    # set all padding elements to -1
    invalid_mask = get_variable_slice_mask(path_tensor, -1, froms=path_lens)
    path_tensor[invalid_mask] = -1

    return path_tensor, path_lens


def aggregate_node_features(nexts, features, agg_mode="sum", 
                            return_node_counts=False):
    """
    features is a tensor of node features that will be aggregated along the 
    shortest path.

    agg_mode: either "sum", "mean", or "concat".
    """
    # translate node features to edge features: each "edge" feature is the
     # feature of the edge's target node.
    has_batch_dim = features.ndim == 3
    if not has_batch_dim:
        # no batch dimension, so add one
        features = features[None]

    n_nodes = features.shape[1]

    # repeat the new dimension n_nodes times, giving ef_ij = nf_j
    edge_feats = features[:, None].expand(-1, n_nodes, -1, -1).clone()
    node_idxs = torch.arange(n_nodes)
    # fill the diagonal with 0s
    edge_feats[:, node_idxs, node_idxs] = 0

    edge_agg_mode = "concat" if agg_mode == "concat" else "sum"
    aggs, edge_counts = \
        aggregate_edge_features(nexts, edge_feats, edge_agg_mode, 
                                return_edge_counts=True)
    # the final aggregates are missing the first node's feature, so add it
    if agg_mode == "concat":
        # transpose the (to, from) axes of edge features 
        first_feats = edge_feats.permute(0, 2, 1, 3)
        # add a sequence dimension
        first_feats = first_feats[:, :, :, None]
        # pre-append the first node's feature along the sequence dimension
        aggs = torch.cat((first_feats, aggs), dim=-2)
    else:
        # add the first node's feature dimension
        aggs = aggs + features[:, :, None]

    if agg_mode == "mean" or return_node_counts:
        node_counts = edge_counts + 1
        if not has_batch_dim:
            nexts = nexts[None]
        if agg_mode == "mean":
            aggs = aggs / node_counts[..., None]
        # force node counts to be 0 for invalid paths and self-loops.
        node_counts[nexts == -1] = 0
        node_counts[:, node_idxs, node_idxs] = 0

    if not has_batch_dim:
        # remove the batch dim we added above
        aggs.squeeze_(0)
        if agg_mode == "mean" or return_node_counts:
            node_counts.squeeze_(0)

    if return_node_counts:
        return aggs, node_counts
    else:
        return aggs


def aggregate_edge_features(nexts, features, agg_mode="sum", 
                            return_edge_counts=False):
    """
    features is a tensor of edge features that will be aggregated along the 
        shortest path.  Must be 3-dimensional (n_nodes x n_nodes x feature dim)
        or 4-dimensional (batch_dim x n_nodes x n_nodes x feature dim)

    agg_mode: "sum", "mean", or "concat".
    """
    nexts_has_batch_dim = nexts.ndim == 3
    if nexts_has_batch_dim:
        # make batch dimension come last
        nexts = nexts.permute(1, 2, 0)
        batch_size = nexts.shape[2]
    else:
        nexts = nexts[..., None]
        batch_size = 1

    features_has_batch_dim = features.ndim == 4
    if features_has_batch_dim:
        # move batch dim to just before feature dim
        features = features.permute(1, 2, 0, 3)
    else:
        # add a batch dimension
        features = features[..., None]

    dev = features.device
    n_nodes = nexts.shape[0]
    node_idxs = torch.arange(n_nodes, device=dev)
    batch_idxs = torch.arange(batch_size, device=dev).expand(n_nodes, n_nodes, 
                                                             -1)
    cur_node, dst_node = torch.meshgrid(node_idxs, node_idxs, indexing='ij')
    if nexts.ndim == 3:
        dst_node = dst_node[..., None].expand(-1, -1, batch_size)
        cur_node = cur_node[..., None].expand(-1, -1, batch_size)
    if agg_mode == "concat":
        aggs = []
    else:
        aggs = torch.zeros_like(features)
    if agg_mode == "mean" or return_edge_counts:
        edge_counts = torch.zeros_like(nexts)

    for ni in range(n_nodes):
        next_node = nexts[cur_node, dst_node, batch_idxs]
        next_feats = features[cur_node, next_node, batch_idxs]
        if agg_mode == "concat":
            # the final edge feature is always a self-loop, since no 
             # non-looping path has more than n_nodes - 1 edges, so skip it.
            if ni < n_nodes - 1:
                if ((cur_node != dst_node) & (nexts != -1)).any():
                    # also, to save space, only append features if any are 
                     # valid next nodes
                    aggs.append(next_feats)
        else:
            aggs = aggs + next_feats
        if agg_mode == "mean" or return_edge_counts:
            edge_counts += (cur_node != dst_node) & (nexts != -1)
        
        cur_node = next_node

    if agg_mode == "mean":
        # make 0s into 1s to avoid division by 0
        denom = edge_counts + (edge_counts == 0)
        retval = aggs / denom[..., None]
    elif agg_mode == "concat":
        if n_nodes == 1:
            # the "sequence" is just the single self-loop feature with a seq dim
            retval = features[None]
        elif len(aggs) == 0:
            retval = torch.zeros((0, n_nodes, n_nodes, batch_size, 
                                  features.shape[-1]), device=dev)
        else:
            retval = torch.stack(aggs)
    else:
        retval = aggs
    
    if features_has_batch_dim:
        if agg_mode == "concat":
            # starts as sequence, from, to, batch, feature
            # make it batch, from, to, sequence, feature
            retval = retval.permute(3, 1, 2, 0, 4)
        else:
            retval = retval.permute(2, 0, 1, 3)
        if return_edge_counts:
            edge_counts = edge_counts.permute(2, 0, 1)
    else:
        retval = retval.squeeze(-1)
        if return_edge_counts:
            edge_counts = edge_counts.squeeze(-1)
    
    if return_edge_counts:
        return retval, edge_counts
    else:
        return retval


def get_path_edge_index(path, get_dense_edges=True, device=None):
    # assemble the edge indices
    if device is None and type(path) is torch.Tensor:
        device = path.device
    if type(path) is list:
        path = torch.tensor(path, device=device)
    if get_dense_edges:
        return torch.combinations(path).T
    else:
        # each stop has an edge only to the next stop on the route
        from_indices = path[:-1]
        to_indices = path[1:]
        return torch.tensor([from_indices, to_indices], dtype=int)


def aggr_edges_over_sequences(seqs_tensor, edge_features, agg_mode="sum", 
                              sparse=False, low_memory_mode=False):
    """
    Assumes a batch dim exists.  Note that in "concat" mode, a mask is returned
        along with the feature sequence.  The mask is True where the sequence
        is invalid.  In general, the valid elements of the sequence will *not*
        be contiguous!

    sparse: if True, aggregate edges sparsely.  Otherwise, do dense 
        aggregation.
    low_memory_mode: if True, loop over batch elements in some cases instead of
        dealing with them all at once.
        
    """
    dev = edge_features.device
    max_seq_len = seqs_tensor.shape[-1]
    # augment the feature matrix with a fringe of zeros so indexing to -1 won't
     # affect the sum.
    # aug_features = torch.nn.functional.pad(edge_features, (0, 0, 0, 1, 0, 1,))
    aug_shape = list(edge_features.shape)
    aug_shape[-3] += 1
    aug_shape[-2] += 1
    aug_features = torch.zeros(aug_shape, device=dev)
    aug_features[:, :-1, :-1] = edge_features
    
    # It's possible to do this without a loop, but it takes huge amounts of 
     # memory, so don't.
    aggs_shape = seqs_tensor.shape[:-1] + (edge_features.shape[-1],)
    if agg_mode == "concat":
        # add a sequence dimension
        aggs_shape = aggs_shape[:-1] + (0, aggs_shape[-1])

    aggs = torch.zeros(aggs_shape, dtype=torch.float32, device=dev)
    if agg_mode == "mean":
        edge_counts = torch.zeros(aggs.shape[:3], dtype=int, device=dev)
    if agg_mode == "concat":
        mask = torch.zeros(aggs_shape[:-1], dtype=bool, device=dev)

    batch_size = seqs_tensor.shape[0]
    batch_idxs = torch.arange(batch_size, device=dev)
    # expand the batch indexes to match any extra "batch" dims in seqs_tensor
    n_extra_batch_dims = seqs_tensor.ndim - 2
    for _ in range(n_extra_batch_dims):
        # add a new dimension after the main batch dimension
        batch_idxs = batch_idxs[:, None]
    batch_idxs = batch_idxs.expand(seqs_tensor.shape[:-1])

    batch_idxs = batch_idxs[..., None]
    for ii in range(max_seq_len - 1):
        from_idxs = seqs_tensor[..., ii][..., None]
        if sparse:
            to_idxs = seqs_tensor[..., ii+1][..., None]
        else:
            to_idxs = seqs_tensor[..., ii+1:]

        if low_memory_mode and agg_mode in ["mean", "sum"]:
            for bi in range(aug_features.shape[0]):
                out_edges = aug_features[bi, from_idxs[bi], to_idxs[bi]]
                aggs[bi] += out_edges.sum(dim=-2)

        else:
            out_edges = aug_features[batch_idxs, from_idxs, to_idxs]
            if agg_mode == "concat":
                aggs = torch.cat((aggs, out_edges), dim=-2)
                mask = torch.cat((mask, to_idxs == -1), dim=-1)
            else:
                aggs += out_edges.sum(dim=-2)

        if agg_mode == "mean":
            edge_counts += (to_idxs > -1).sum(dim=-1)

    if agg_mode == "mean":
        # avoid division by 0 by making 0s into 1s
        denoms = edge_counts + (edge_counts == 0)
        aggs /= denoms[..., None]

    if agg_mode == "concat":
        return aggs, mask
    
    return aggs


def get_route_leg_times(batch_routes, drive_times_matrix, mean_stop_time_s=0):
    add_route_dim = batch_routes.ndim == 2
    if add_route_dim:
        # add a route dimension
        batch_routes = batch_routes[:, None]

    batch_idxs = torch.arange(batch_routes.shape[0], 
                              device=batch_routes.device)
    # batch_size x n_routes x route_len - 1
    leg_times = drive_times_matrix[batch_idxs[:, None, None], 
                                   batch_routes[..., :-1], 
                                   batch_routes[..., 1:]]
    if type(mean_stop_time_s) is torch.Tensor and mean_stop_time_s.ndim == 1:
        # make number of dimensions match leg_times
        mean_stop_time_s = mean_stop_time_s[..., None, None]

    leg_times += mean_stop_time_s
    invalid_leg_mask = (batch_routes == -1)[..., 1:]
    leg_times = get_update_at_mask(leg_times, invalid_leg_mask)
    if add_route_dim:
        # remove the extra dimension we added from the output
        leg_times.squeeze_(1)
    return leg_times


def get_route_edge_matrix(batch_routes, drive_times_matrix, mean_stop_time_s,
                          symmetric_routes=True, low_mem_mode=False,
                          return_used_routes=False):
    """
    batch_routes: (batch_size, n_routes, max_route_len) tensor of route indices
    drive_times_matrix: (batch_size, n_nodes, n_nodes) tensor of shortest-path
        drive times between nodes
    mean_stop_time_s: mean time spent at each stop in seconds
    symmetric_routes: if True, treat routes as going both ways along their
        stops.

    Returns a (batch_size, max_route_len, max_route_len) tensor of 
        the shortest time on a route between each pair of stops.
    """
    batch_size = batch_routes.shape[0]
    max_route_len = batch_routes.shape[-1]
    n_routes = batch_routes.shape[1]
    n_street_nodes = drive_times_matrix.shape[-1]
    dev = batch_routes.device
    # add a dummy row and column
    route_edge_mat = torch.full((n_street_nodes+1, n_street_nodes+1), 
                                float('inf'), dtype=drive_times_matrix.dtype, 
                                device=dev)
    route_edge_mat.fill_diagonal_(0)
    if n_routes == 0:
        outval = route_edge_mat.repeat(batch_size, 1, 1)[:, :-1, :-1]
        if return_used_routes:
            return outval, torch.zeros_like(outval, dtype=int)[:, 0]
        else:
            return outval

    route_edge_mat = route_edge_mat.repeat(batch_size, n_routes, 1, 1)

    batch_idxs = torch.arange(batch_size, device=dev)
    # batch_size x n_routes x route_len - 1
    leg_times = get_route_leg_times(batch_routes, drive_times_matrix, 
                                    mean_stop_time_s)
    # batch_size x n_routes x route_len x route_len
    interstop_times = get_route_interstop_times(leg_times)

    invalid_stop = batch_routes == -1
    invalid_dst = invalid_stop[..., None].expand(-1, -1, -1, max_route_len)
    interstop_times[invalid_dst] = float('inf')
    invalid_src = invalid_stop[..., None, :].expand(-1, -1, max_route_len, -1)
    interstop_times[invalid_src] = float('inf')
    route_idxs = torch.arange(n_routes, device=dev)

    if low_mem_mode:
        # Avoid OOM errors for large tensors.  Doing it by trying to count the
         # actual memory used doesn't work, so pass a flag instead.
        for bi in batch_idxs:
            route_edge_mat[bi, route_idxs[:, None, None], 
                        batch_routes[bi, ..., None], 
                        batch_routes[bi, ..., None, :]] = interstop_times[bi]
    else:
        route_edge_mat[batch_idxs[:, None, None, None], 
                       route_idxs[None, :, None, None],
                       batch_routes[..., None], 
                       batch_routes[..., None, :]] = interstop_times
    
    route_edge_mat, min_idxs = route_edge_mat[..., :-1, :-1].min(dim=1)

    if symmetric_routes:
        # add routes going back the other way
        transpose_dtm = drive_times_matrix.permute(0, 2, 1)
        reverse_edges, reverse_used_routes = \
            get_route_edge_matrix(batch_routes, transpose_dtm,
                                  mean_stop_time_s, symmetric_routes=False,
                                  return_used_routes=True)
        reverse_edges = reverse_edges.permute(0, 2, 1)
        route_edge_mat = torch.minimum(route_edge_mat, reverse_edges)

    if return_used_routes:
        # used routes are just the min idxs, but with -1 where there's no route
        used_routes = min_idxs
        used_routes[route_edge_mat == float('inf')] = -1
        # set diagonal elements of each batch matrix to -1
        node_idxs = torch.arange(n_street_nodes, device=dev)
        used_routes[:, node_idxs, node_idxs] = -1

        if symmetric_routes:
            use_reverse = reverse_edges == route_edge_mat
            reverse_used_routes = reverse_used_routes.permute(0, 2, 1)
            used_routes[use_reverse] = reverse_used_routes[use_reverse]
        
        return route_edge_mat, used_routes

    return route_edge_mat


def get_route_interstop_times(leg_times):
    """leg_times: An n_routes x route_length - 1 tensor.
    
    Returns an n_routes x route_length x route_length tensor of the time it 
    takes to get from one stop to another.
    """
    cum_dists = leg_times.cumsum(-1)
    # preappend some zeros so the next operation will give time 
     # from node i to other node j
    cum_dists = torch.nn.functional.pad(cum_dists, (1, 0))
    inter_stop_dists = \
        cum_dists[..., None, :] - cum_dists[..., :, None]
    inter_stop_dists[inter_stop_dists < 0] = float('inf')
    return inter_stop_dists


def get_batch_tensor_from_routes(routes, device=None, max_route_len=None):
    """
    Given a list or list of lists of routes, return an equivalent batched route
    tensor. Dummy elements will have value -1.

    Routes must be either tensors or lists of ints.
    """
    # check if the input has a batch dimension; add it if not
    if len(routes[0]) == 0:
        # empty list of routes
        batch_size = len(routes)
        return torch.empty((batch_size, 0, 0), dtype=int, device=device)
    if isinstance(routes[0], torch.Tensor) or type(routes[0][0]) is int:
        # elements are either tensors or lists of ints, so there's no batch
        batch_routes = [routes]
    else:
        batch_routes = routes

    if device is None and isinstance(routes[0][0], torch.Tensor):
        device = routes[0][0].device
        
    scen_tensors = [get_tensor_from_varlen_lists(sr, device)[0] 
                    for sr in batch_routes]
    batch_size = len(scen_tensors)
    max_n_routes = max([st.shape[0] for st in scen_tensors])
    if not max_route_len:
        max_route_len = max([st.shape[1] for st in scen_tensors])
    batch_tensor = torch.full((batch_size, max_n_routes, max_route_len), -1)
    for bi, st in enumerate(scen_tensors):
        batch_tensor[bi, :st.shape[0], :st.shape[1]] = st
    return batch_tensor.to(device=device)


def get_tensor_from_varlen_lists(lists, device=None, add_dummy=False):
    """Converts a list of lists or 1D tensors with possibly-variable lengths
        into a single 2D tensor and a mask indicating which entries of the
        tensor are padding values."""
    tensor_len = len(lists)
    if len(lists) > 0:
        max_list_len = max([len(rr) for rr in lists])
    else:
        max_list_len = 0
    if device is None and len(lists) > 0 and type(lists[0]) is torch.Tensor:
        device = lists[0].device
    if add_dummy:
        tensor_len += 1
    out_tensor = \
        torch.full((tensor_len, max_list_len), -1, dtype=int, device=device)
    for ri, llist in enumerate(lists):
        if type(llist) is not torch.Tensor:
            llist = torch.tensor(llist, device=device)
        out_tensor[ri, :len(llist)] = llist
    padding_mask = out_tensor < 0
    return out_tensor, padding_mask


def get_unselected_routes(networks, selected_idxs):
    # create a mask that is 0 at all nodes of selected routes, 1 elsewhere
    # add route and node dimensions to the selected indices
    scatter_idxs = selected_idxs[..., None, None]
    # expand the node dimension to the size of the tensor
    exp_target = [-1] * (scatter_idxs.ndim - 1) + [networks.shape[-1]]
    scatter_idxs = scatter_idxs.expand(exp_target)
    mask = torch.ones_like(networks).scatter(-2, scatter_idxs, 0)
    # assemble a tensor of indices to gather the unselected routes
    unselected_shape = list(networks.shape)
    unselected_shape[-2] -= 1
    unselected_idxs = mask.nonzero()[:, -2].reshape(*unselected_shape)
    # get the unselected routes and return them
    unselected_routes = networks.gather(-2, unselected_idxs)
    return unselected_routes


def get_nodes_on_routes_mask(n_nodes, networks):
    """Returns an n_networks x n_routes x n_nodes+1 tensor of booleans,
       where T_ijk = True where node i is on route j in network k, False 
       otherwise."""
    if networks.ndim == 2:
        networks = networks[None]
    n_networks = networks.shape[0]
    n_routes = networks.shape[1]

    are_nodes_on_route = torch.zeros((n_networks, n_routes, n_nodes+1),
                                     dtype=bool, device=networks.device)
    # replace -1 dummy elements with n_nodes, so they work with scatter
    scatter_networks = networks.clone()
    scatter_networks[scatter_networks == -1] = n_nodes
    # set the nodes on the routes to True
    are_nodes_on_route.scatter_(2, scatter_networks, True)
    # set the dummy node to False, it's not "on" any node
    are_nodes_on_route[..., -1] = False
    return are_nodes_on_route


def dump_routes(run_name, all_route_sets, out_dir='output_routes'):
    """
    run_name: a string to identify the run
    all_route_sets: a list of tensors, the routes used for each city
    """
    if type(out_dir) is str:
        out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    out_path = out_dir / (run_name + '_routes.pkl')
    if type(all_route_sets) is not torch.Tensor:
        all_route_sets = get_batch_tensor_from_routes(all_route_sets)
        
    all_route_sets = all_route_sets.cpu()
    # convert to a list of route sets for each city
    if all_route_sets.ndim == 2:
        all_route_sets = [all_route_sets]
    elif all_route_sets.ndim == 3:
        all_route_sets = [rs for rs in all_route_sets]

    with out_path.open('wb') as ff:
        pickle.dump(all_route_sets, ff)


def load_routes_tensor(path, device=None):
    if type(path) is str:
        path = Path(path)
    with path.open('rb') as ff:
        route_sets = pickle.load(ff)
    return torch.stack(route_sets, 0).to(device=device)


def cat_var_size_tensors(tensors, dim=0):
    """tensors: a list of tensors with the same ndim but variable sizes."""
    shapes = torch.tensor([tt.shape for tt in tensors])
    max_shape, _ = shapes.max(dim=0)
    padded_tensors = []
    for tensor in tensors:
        # wierdly, padding is a tuple of 2 * ndim ints, where element 0 is the
         # number of elements to pad at the start of the *last* dim, element 1
         # is the number to pad at the end of the last dim, elements 2 and 3 
         # are for the start and end of the 2nd-to-last dim, and so on.
        padding = [(0, (ps - ss).item()) 
                   for ps, ss in zip(max_shape, tensor.shape)]
        padding[dim] = (0, 0)
        padding = sum(reversed(padding), ())
        padded = torch.nn.functional.pad(tensor, padding)
        padded_tensors.append(padded)

    return torch.cat(padded_tensors, dim)


def get_indices_from_mask(mask, dim=0, pad_value=-1):
    """Given a boolean mask, return the a tensor of indices of the True 
       elements."""
    n_indices = mask.sum(dim=dim)
    shape = list(mask.shape)
    shape[dim] = n_indices.max().item()
    dev = mask.device
    indices = torch.full(shape, pad_value, dtype=int, device=dev)

    dims = [1] * len(shape)
    dims[dim] = -1
    # add a 1-length dim for each other dimension of the mask
    indices_source = torch.arange(mask.shape[dim], device=dev).reshape(*dims)
    indices_source = indices_source.expand_as(mask)

    var_slice_mask = get_variable_slice_mask(indices, dim, tos=n_indices)
    indices[var_slice_mask] = indices_source[mask]

    return indices


def get_variable_slice_mask(tensor, dim=0, froms=None, tos=None):
    """Suppose we want to slice along some dimension, but want different slices
       along that dimension at each location.  This function produces a mask
       that is True for all the included elements."""
    assert froms is not None or tos is not None, "Must provide indices"
    # 'while' just in case dim < -ndim, but that's doubtful
    assert tensor.ndim > 0, "Can't slice a scalar"
    while dim < 0:
        dim += tensor.ndim
    
    froms = _make_all_indices_positive(tensor.shape[dim], froms)
    tos = _make_all_indices_positive(tensor.shape[dim], tos)

    rng = torch.arange(tensor.shape[dim], device=tensor.device)
    # expand range to have the same shape as the input tensor
    dims = [1] * tensor.ndim
    dims[dim] = -1
    rng = rng.reshape(*dims).expand_as(tensor)
    
    mask = torch.ones_like(tensor, dtype=bool)
    if froms is not None:
        mask &= _get_forward_slice_mask(dim, rng, froms)
    if tos is not None:
        mask &= ~_get_forward_slice_mask(dim, rng, tos)
    return mask

# functions only used in get_variable_slice_mask().

def _make_all_indices_positive(dim_size, indices):
    """Assumes no -ve index is < -dim_size."""
    if indices is not None:
        is_neg = indices < 0
        indices = indices + is_neg * dim_size
    return indices


def _get_forward_slice_mask(dim, rng, idxs):
    is_start_idx = rng == idxs.unsqueeze(dim)
    return is_start_idx.cumsum(dim).to(bool)


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        # indices = indices // dim
        # use this form instead to silence the warning about deprecation
        indices = torch.div(indices, dim, rounding_mode='floor')

    coord = torch.stack(coord[::-1], dim=-1)

    return coord