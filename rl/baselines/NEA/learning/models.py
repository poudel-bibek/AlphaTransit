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
import logging as log
import math
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (GATv2Conv, GCNConv, MessagePassing, SGConv,
                                BatchNorm, GraphNorm)
import torch_utils as tu
from trunc_normal import TruncatedNormal

from simulation.citygraph_dataset import \
    DEMAND_KEY, STOP_KEY, STREET_KEY, ROUTE_KEY, CityGraphData
from simulation.transit_time_estimator import RouteGenBatchState

Q_FUNC_MODE = "q function"
PLCY_MODE = "policy"
GFLOW_MODE = "gflownet"

# FEAT_NORM_MOMENTUM = 0.001
FEAT_NORM_MOMENTUM = 0.0

# minimum and maximum log probs to use when "forcing" an action selection
 # use this in place of -float('inf') to avoid nans in some badly-behaved
 # backprops
TORCH_FMAX = 10**10
TORCH_FMIN = -TORCH_FMAX


GREEDY = False

MLP_DEFAULT_DROPOUT = 0.0
ATTN_DEFAULT_DROPOUT = 0.0
DEFAULT_NONLIN = nn.ReLU


PlanResults = namedtuple(
    "PlanResults", 
    ["state", "stop_logits", "route_logits", "freq_logits", 
    "entropy", "stops_tensor", "routes_tensor", "freqs_tensor", "stop_est_vals", 
    "route_est_vals", "freq_est_vals"]
    )

RouteGenResults = namedtuple(
    "GenRouteResults", ["routes", "route_descs", "logits", "selections", 
                        "est_vals"]
    )

RouteChoiceResults = namedtuple(
    "RouteChoiceResults", ["logits", "selections", "est_vals"]
    )

FreqChoiceResults = namedtuple(
    "FreqChoiceResults", ["freqs", "logits", "selections", "est_vals"]
    )


class SymmetricLogUnit(nn.Module):
    """For positive X, it's the positive component of log(X + 1).  For 
    negative X, it's the negative of the positive component of log(-X + 1)."""
    def forward(self, xx):
        return symmetric_log_unit(xx)


def symmetric_log_unit(xx):
    """For positive X, it's the positive component of log(X + 1).  For 
    negative X, it's the negative of the positive component of log(-X + 1)."""
    log_abs = torch.log(xx.abs() + 1)
    return log_abs * (xx > 0) - log_abs * (xx <= 0)


# Backbone encoder modules
class GraphEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim


class KoolGraphEncoder(GraphEncoder):
    def __init__(self, in_dim, enc_dim, n_heads=8, n_layers=3, ff_dim=512):
        super().__init__(enc_dim)
        self.linear_embedding = nn.Linear(in_dim, enc_dim, bias=True)
        enc_layer = nn.TransformerEncoderLayer(enc_dim, n_heads, ff_dim)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, node_features, adjacency_matrix=None, *args, **kwargs):
        """Adjacency matrix is treated as binary!"""
        node_features = self.linear_embedding(node_features)
        if adjacency_matrix is not None:
            if type(adjacency_matrix) is not torch.Tensor:
                adjacency_matrix = torch.tensor(adjacency_matrix, 
                    dtype=torch.bool)
        node_features = node_features[:, None, :]
        encoded_feats = self.transformer(node_features, adjacency_matrix > 0)
        return encoded_feats.squeeze(dim=1)


# Node scorer modules


class KoolNextNodeScorer(nn.Module):
    # This follows Kool et al 2019
    def __init__(self, embed_dim, clip_size=10):
        """
        embed_dim: the dimension of the embeddings.
        softmax_temp: the temperature of the softmax distribution over nodes.
            default is 1 (no change to energies).
        clip_size: maximum absolute value of scores that can be returned.
        """
        super().__init__()
        self.feature_dim = embed_dim
        self.clip_size = clip_size

        self.node_embedding_projector = nn.Linear(embed_dim, embed_dim*3,
                                                  bias=False)
        self.context_layer2_projector = nn.Linear(embed_dim, embed_dim)

        self.attn_embeddings = None

    def precompute(self, node_vecs):
        embeddings = self.node_embedding_projector(node_vecs)
        self.attn_embeddings = embeddings.chunk(3, dim=-1)
    
    def forward(self, context, mask=None):
        """
        context: the context vector specific to this step.
        mask: a mask that is True for invalid choices.
        """
        # context = context.reshape(1, -1)
        k_layer1, v_layer1, k_layer2 = self.attn_embeddings
        # perform the attention operations

        if k_layer1.ndim == 3: # and context.ndim == 2:
            # there's a batch dimension in the embeddings, so add an 
            # appropriate dimension to the context tensor
            context = context[:, None, :]

        layer1_attn = torch.matmul(context, k_layer1.transpose(-2, -1))
        layer1_attn = layer1_attn / np.sqrt(k_layer1.size(-1))
        layer1_scores = torch.softmax(layer1_attn, dim=-1)
        context_layer2 = torch.matmul(layer1_scores, v_layer1)
        context_layer2 = self.context_layer2_projector(context_layer2)
        layer2_attn = torch.matmul(context_layer2, k_layer2.transpose(-2, -1))
        scores = layer2_attn / np.sqrt(k_layer2.size(-1))
        
        if self.clip_size:
            # limit the range of magnitudes of the energies with a tanh.
            scores = self.clip_size * torch.tanh(scores)

        if scores.dim() == 3:
            scores = scores.squeeze(-2)
    
        if mask is not None:
            # keep masked elements as -inf, so they'll have probability 0.
            scores[mask] = TORCH_FMIN
        
        return scores

    def reset(self):
        self.attn_embeddings = None


# not used right now but might want to use later


class RoutesEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode_nodepairs(self, nodepair_descs, routes_tensor, **encode_args):
        """
        nodepair_descs: batch_size x n_nodepairs x embed_dim tensor
        """
        route_feats, pad_mask = \
            tu.aggr_edges_over_sequences(routes_tensor, nodepair_descs, 'concat')
        
        # flatten multiple batch dimensions
        n_batch_dims = routes_tensor.ndim - 1
        if n_batch_dims > 1:
            old_batch_dim = pad_mask.shape[:-1]
            route_feats = route_feats.flatten(0, n_batch_dims - 1)
            pad_mask = pad_mask.flatten(0, n_batch_dims - 1)
        
        # remove sequences that are all padding
        is_valid_seq = ~pad_mask.all(dim=-1)
        route_feats = route_feats[is_valid_seq]
        pad_mask = pad_mask[is_valid_seq]        
        enc = self.encode(route_feats, pad_mask, embed_pos=False, 
                          **encode_args)

        # add back locations for the all-padding sequences
        feat_dim = enc.shape[-1]
        full_enc_shape = (is_valid_seq.shape[0], feat_dim)
        full_enc = torch.zeros(full_enc_shape, dtype=enc.dtype, 
                               device=enc.device)
        full_enc[is_valid_seq] = enc
        # add back multiple batch dimensions
        if n_batch_dims > 1:
            full_enc = full_enc.reshape(old_batch_dim + (feat_dim,))
        return full_enc
        
    def forward(self, node_descs, routes_tensor, padding_mask, **encode_args):
        # put everything on the right device
        dev = node_descs.device
        routes_tensor = routes_tensor.to(device=dev)
        padding_mask = padding_mask.to(device=dev)

        # build route tensor and mask
        old_shape = None
        nodes_have_batch = node_descs.ndim > 2
        if nodes_have_batch:
            batch_size = node_descs.shape[0]
            nodes_per_batch = node_descs.shape[-2]
            node_descs = node_descs.reshape(-1, node_descs.shape[-1])

            if routes_tensor.ndim == 2:
                # give the routes tensor a batch dim
                sizes = (batch_size,) + (-1,) * routes_tensor.ndim
                routes_tensor = routes_tensor[None]
                routes_tensor = routes_tensor.expand(sizes)

            if padding_mask.ndim == 2:
                # give the pad mask a batch dim
                sizes = (batch_size,) + (-1,) * padding_mask.ndim
                padding_mask = padding_mask[None]
                padding_mask = padding_mask.expand(sizes)

        if routes_tensor.ndim > 2:
            # we're dealing with multiple routes per batch element
            old_shape = routes_tensor.shape
            batch_size = old_shape[0]
            if nodes_have_batch:
                # add offsets to each batch element to the right set of nodes
                 # in the flattened node_descs
                offsets = torch.arange(0, batch_size * nodes_per_batch, 
                                       nodes_per_batch, 
                                       device=node_descs.device)
                reshape_dims = (-1,) + (1,) * (len(old_shape) - 1)
                routes_tensor = routes_tensor + offsets.reshape(reshape_dims)

            # flatten multiple batch dimensions 
            routes_tensor = routes_tensor.reshape(-1, routes_tensor.shape[-1])
            padding_mask = padding_mask.reshape(-1, routes_tensor.shape[-1])

        # cut out invalid rows (padding mask is true everywhere), then add them
         # back
        enc_shape = (routes_tensor.shape[0], node_descs.shape[-1])
        valid_seqs = ~padding_mask.all(dim=1)
        routes_tensor = routes_tensor[valid_seqs]
        padding_mask = padding_mask[valid_seqs]

        # get node features for routes
        route_node_feats = node_descs[routes_tensor]
        encode_out = self.encode(route_node_feats, padding_mask, **encode_args)
        if type(encode_out) is tuple:
            valid_enc = encode_out[0]
        else:
            valid_enc = encode_out
        enc = torch.zeros(enc_shape, device=dev)
        enc[valid_seqs] = valid_enc
        # remove the sequence dimension in the seq-len-1 encoding
        enc = enc.squeeze(1)
        if old_shape is not None:
            enc = enc.reshape(old_shape[:-1] + (enc.shape[-1],))
        if type(encode_out) is tuple:
            return (enc,) + encode_out[1:]
        else:
            return enc

    def encode(self, route_seqs, padding_mask):
        raise NotImplementedError


class SumRouteEncoder(RoutesEncoder):
    def encode_nodepairs(self, nodepair_descs, routes_tensor, **encode_args):
        """
        nodepair_descs: batch_size x n_nodepairs x embed_dim tensor
        """
        route_feats = \
            tu.aggr_edges_over_sequences(routes_tensor, nodepair_descs, 'sum')

        return route_feats


class MeanRouteEncoder(RoutesEncoder):
    def encode(self, route_seqs, padding_mask, *args, **kwargs):
        enc = mean_pool_sequence(route_seqs, padding_mask)
        return enc


class MaxRouteEncoder(RoutesEncoder):
    def encode(self, route_seqs, padding_mask, *args, **kwargs):
        unsq_pad = padding_mask.unsqueeze(-1)
        enc = route_seqs * ~unsq_pad + TORCH_FMIN * unsq_pad
        enc, _ = route_seqs.max(dim=1)
        return enc


class LatentAttnRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_heads=8, n_layers=2, 
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        self.encoder = LatentAttentionEncoder(embed_dim, 1, n_heads, n_layers,
                                              dropout)
        self.end_token = nn.Parameter(torch.randn(embed_dim))

    def encode(self, route_seqs, padding_mask, embed_pos=True):
        # TODO somehow also encode the inter-stop times
        enc = self.encoder(route_seqs, padding_mask=padding_mask, 
                           embed_pos=embed_pos).squeeze(1)
        return enc


class TransformerRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_heads, n_layers, 
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, 4*embed_dim,
                                               dropout=dropout, 
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.end_token = nn.Parameter(torch.randn(embed_dim))
        # self.final_encoder = LatentAttentionEncoder(embed_dim, 1, n_heads, 1,
        #                                             dropout)

    def encode(self, route_seqs, padding_mask, embed_pos=True, 
               return_nodes=False):
        # append the end token to each route sequence
        seq_lens = (~padding_mask).sum(dim=1)
        max_seq_len = seq_lens.max()
        end_token = self.end_token.expand(route_seqs.shape[0], -1)
        if max_seq_len + 1 > route_seqs.shape[1]:
            # pad the end of route seqs and padding mask
            placeholder = torch.zeros_like(end_token)[:, None]
            route_seqs = torch.cat((route_seqs, placeholder), dim=1)
            padding_mask = torch.cat((padding_mask, 
                                      torch.ones((route_seqs.shape[0], 1),
                                                 dtype=bool)), 
                                     dim=1)

        # insert end tokens
        batch_idxs = torch.arange(route_seqs.shape[0])
        route_seqs[batch_idxs, seq_lens] = end_token
        # set locations corresponding to new end token to False
        padding_mask[batch_idxs, seq_lens] = False

        if embed_pos:
            sinseq = get_sinusoid_pos_embeddings(route_seqs.shape[1], 
                                                route_seqs.shape[2])
            sinseq = sinseq[None]
            route_seqs = route_seqs + sinseq.to(device=route_seqs.device)
            
        tfed = self.transformer(route_seqs, src_key_padding_mask=padding_mask)
        # take the embedding at the placeholder token as the descriptor
        route_descs = tfed[batch_idxs, seq_lens]
        if return_nodes:
            return route_descs, tfed
        else:
            return route_descs


class ConvRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_layers):
        super().__init__()
        layers = sum([[nn.Conv1d(embed_dim, embed_dim, 3), 
                       DEFAULT_NONLIN(),
                       nn.AvgPool1d(2)] for _ in range(n_layers)], start=[])
        # applied to a boolean, this should act like "or" over each pooled 
         # block
        self.mask_pooler = nn.MaxPool1d(2**n_layers)
        self.encoder = nn.Sequential(*layers)

    def encode(self, route_seqs, padding_mask):
        # swap sequence and channel dims, since that's what conv1d expects
        route_seqs = route_seqs.permute(0, 2, 1)
        conved = self.encoder(route_seqs)
        # swap sequence and channel dims back
        conved = conved.permute(0, 2, 1)
        
        # compute new padding mask
        valid_mask = (~padding_mask[:, None]).to(dtype=float)
        pooled_valid_mask = self.mask_pooler(valid_mask)
        pooled_valid_mask = pooled_valid_mask[:, 0].to(dtype=bool)
        # due to padding, the shapes might not match perfectly
        pooled_valid_mask = pooled_valid_mask[:, :conved.shape[1]]
        pooled_pad_mask = ~pooled_valid_mask

        # do mean-pooling over pooled sequences
        pooled = mean_pool_sequence(conved, pooled_pad_mask)
        return pooled
    

class ContinuousGaussianActor(nn.Module):
    def __init__(self, n_layers, embed_dim, in_dim=None,
                 min_action=None, max_action=None, min_logstd=None, 
                 max_logstd=None, bias=True):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.mean_and_std_net = get_mlp(n_layers, embed_dim, in_dim=in_dim,
                                        out_dim=2, bias=bias)

    def forward(self, inpt, pick_max=False):
        means_and_stds = self.mean_and_std_net(inpt)
        means = means_and_stds[..., 0]
        if self.min_action is not None or self.max_action is not None:
            means = means.clamp(self.min_action, self.max_action)

        # stds must be positive, so exponentiate them
        logstds = means_and_stds[..., 1]
        if self.min_logstd is not None or self.max_logstd is not None:
            logstds = logstds.clamp(self.min_logstd, self.max_logstd)
        stds = torch.exp(logstds)

        if self.min_action is not None or self.max_action is not None:
            dstrb = TruncatedNormal(means, stds, self.min_action, 
                                    self.max_action)
        else:
            dstrb = torch.distributions.Normal(means, stds)

        if pick_max:
            # take the action at the mean of the distribution
            actions = means
        else:
            # sample from the distribution and limit to valid range
            actions = dstrb.sample()

        log_probs = dstrb.log_prob(actions)
        return actions, log_probs


class ContinuousGaussFixedStdActor(nn.Module):
    def __init__(self, n_layers, embed_dim, in_dim=None,
                 min_action=None, max_action=None, fixed_std=None, bias=True):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        if fixed_std is None:
            if min_action is not None and max_action is not None:
                self.fixed_std = (max_action - min_action) / 5
            else:
                raise ValueError(
                    "Must provide fixed_std or min_action and max_action")
        else:
            self.fixed_std = fixed_std
            
        self.mean_net = get_mlp(n_layers, embed_dim, in_dim=in_dim, out_dim=1, 
                                bias=bias)

    def forward(self, inpt, greedy=False):
        means = self.mean_net(inpt).squeeze(-1)
        means = means.clamp(self.min_action, self.max_action)
        stds = torch.full_like(means, self.fixed_std)
    
        dstrb = TruncatedNormal(means, stds, self.min_action, self.max_action)
        if greedy:
            # take the action at the mean of the distribution
            actions = means
        else:
            # sample from the distribution and limit to valid range
            actions = dstrb.sample()

        log_probs = dstrb.log_prob(actions)
        return actions, log_probs, stds


class FrequencySelector(nn.Module):
    def __init__(self, embed_dim, min_frequency_Hz=1/7200, 
                 max_frequency_Hz=1/60):
        """default min_frequency_Hz is one bus every two hours.
           default max_frequency_Hz is one bus every minute."""
        super().__init__()
        self.mean_and_std_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, embed_dim),
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, 2)
        )

        self.freq_embedder = nn.Sequential(
            nn.Linear(1, embed_dim),
            DEFAULT_NONLIN()
        )
        assert min_frequency_Hz < max_frequency_Hz
        # approximately 1.44 Hz
        self.min_logstd = -1
        self.min_action = self._freq_Hz_to_action(min_frequency_Hz)
        self.max_action = self._freq_Hz_to_action(max_frequency_Hz)
        # wide enough to give a uniform distribution
        self.max_logstd = np.log((self.max_action - self.min_action) * 10000)


    def forward(self, route_descriptor, greedy=False, rollout_freqs=None,
                min_frequency_Hz=None, max_frequency_Hz=None):
        assert not greedy or rollout_freqs is None, \
            "greedy frequency selection is incompatible with rollout!"
        # compute the parameters of the gaussians from which to sample freqs
        normal_param_tensor = self.mean_and_std_net(route_descriptor)
        # stop the mean from being lower than the minimum allowed action
        if min_frequency_Hz:
            min_action = self._freq_Hz_to_action(min_frequency_Hz)
        else:
            min_action = self.min_action
        if max_frequency_Hz:
            max_action = self._freq_Hz_to_action(max_frequency_Hz)
        else:
            max_action = self.max_action
        assert min_action < max_action

        means = normal_param_tensor[..., 0].clamp(min_action, max_action)
        # clamp the std devs in a sensible range for numerical stability
        logstds = normal_param_tensor[..., 1].clamp(self.min_logstd, 
                                                    self.max_logstd)
        # stds must be positive, so exponentiate them
        stds = torch.exp(logstds)
        dstrb = torch.distributions.Normal(means, stds)
        if greedy:
            # greedily take the action at the center of the distribution
            actions = means
        elif rollout_freqs is None:
            # sample the action from the distribution
            actions = dstrb.sample()
        else:
            # take the dictated action
            actions = self._freq_Hz_to_action(rollout_freqs)

        # enforce the minimum allowable frequency for a route that's included        
        actions.clamp_(self.min_action, self.max_action)
        log_probs = dstrb.log_prob(actions)

        freqs_Hz = self._action_to_freq_Hz(actions)
        descriptors = self.freq_embedder(actions[:, None])
        return freqs_Hz, log_probs, descriptors

    def _freq_Hz_to_action(self, freq_Hz):
        # the "action" is the log of the per-hour frequency.
        freq_hourly = freq_Hz * 3600
        if type(freq_hourly) is torch.Tensor:
            return torch.log(freq_hourly)
        else:
            return np.log(freq_hourly)

    def _action_to_freq_Hz(self, action):
        if type(action) is torch.Tensor:
            freq_hourly = torch.exp(action)
        else:
            freq_hourly = np.exp(action)
        return freq_hourly / 3600


class FeatureNorm(nn.Module):
    def __init__(self, dim, momentum=FEAT_NORM_MOMENTUM):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.zeros(dim))
        self.register_buffer('avg_count', torch.tensor(1))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('frozen', torch.tensor(False))
        self.new_mean = None
        self.new_var = None
        self.new_count = 0

    def freeze(self):
        self.frozen[...] = True

    def extra_repr(self) -> str:
        return f'mean={self.running_mean}, var={self.running_var}, ' \
            f'initialized={self.initialized}, count={self.avg_count}'

    def forward(self, xx):
        assert xx.shape[-1] == self.running_mean.shape[-1], \
            "input tensor has the wrong dimensionality!"
        
        x_size = xx.shape[0]
        if x_size == 0:
            # the tensor is empty, so we don't need to do anything.
            return xx
        
        if not self.initialized or not self.frozen:
            # just use x's own mean and variance
            old_shape = xx.shape
            xx = xx.reshape(-1, xx.shape[-1])
            x_mean = xx.mean(0).detach()
            x_var = xx.var(0).detach()
            # we're done with the need for flattened x, so reshape it
            xx = xx.reshape(old_shape)

        if not self.frozen:
            if self.new_mean is None:
                # initialize the new-samples mean
                self.new_mean = x_mean
                self.new_var = x_var
                self.new_count = x_size
            else:
                # update new-samples mean
                nmrtr = (self.new_mean * self.new_count + x_mean * x_size)
                updated_count = self.new_count + x_size
                updated_mean = nmrtr / updated_count
                
                # update new-samples variance using the variance-combining formula:
                # v_c = n_1(v_1 + (m_1 - m_c)^2) + n_2 * (v_2 + (m_1 - m_2)^2) 
                #        / (n_1 + n_2)
                mean_diff_1 = (self.new_mean - updated_mean) ** 2
                mean_diff_2 = (x_mean - updated_mean) ** 2
                prev_part = self.new_count * (self.new_var + mean_diff_1)
                sample_part = x_size * (x_var + mean_diff_2)
                self.new_var = (prev_part + sample_part) / updated_count

                # update count of new samples
                self.new_count = updated_count

        if not self.initialized:
            # we don't have any running statistics yet, so just normalize by
             # minibatch statistics.
            shift = x_mean
            denom = x_var.sqrt()
        
        else:
            # avoid division by 0
            shift = self.running_mean
            denom = self.running_var.sqrt()

        # avoid division by nan or 0. nans will occur in variance if batch 
         # size is 1.
        denom[(denom == 0) | denom.isnan()] = 1
        out = (xx - shift) / denom
        return out

    def update(self):
        if self.frozen:
            # do nothing
            return
        
        if self.new_mean is not None:
            if not self.initialized:
                # just set the initial statistics
                self.running_mean[...] = self.new_mean
                self.running_var[...] = self.new_var
                self.avg_count = torch.tensor(self.new_count)
                self.initialized[...] = True
            else:
                # update running statistics
                # scale update size in proportion to how big the sample is
                alpha = self.momentum * self.new_count / self.avg_count
                # if new count is *much* bigger than old, don't let alpha 
                # be greater than 1...that would make things wierd
                alpha = min(alpha, 1.0)
                updated_mean = alpha * self.new_mean + \
                    (1 - alpha) * self.running_mean
                
                # implements the variance-combing formula, assuming
                 # n_1 / (n_1 + n_2) = alpha, n_2 / (n_1 + n_2) = 1 - alpha
                t1 = self.new_var + (self.new_mean - updated_mean) ** 2
                t2 = self.running_var + (self.running_mean - updated_mean) ** 2
                self.running_var[...] = alpha * t1 + (1 - alpha) * t2

                # update the running mean *after* using it to compute the new
                 # running variance
                self.running_mean[...] = updated_mean
                self.avg_count = self.momentum * self.new_count + \
                    (1 - self.momentum) * self.avg_count

            self.new_mean = None
            self.new_var = None
            self.new_count = 0

        else:
            log.warning("FeatureNorm was updated without any samples!")


class NodepairDotScorer(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_transform = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_transform = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, flat_node_vecs, batch_n_nodes=None):
        queries = self.query_transform(flat_node_vecs)
        queries /= math.sqrt(self.embed_dim)
        keys = self.key_transform(flat_node_vecs)
        dev = flat_node_vecs.device

        if batch_n_nodes is not None:
            # fold the queries and keys into a batch dimension
            max_n_nodes = max(batch_n_nodes)
            batch_size = len(batch_n_nodes)
            folded_queries = torch.zeros((batch_size, max_n_nodes, 
                                          self.embed_dim), device=dev)
            folded_keys = folded_queries.clone()
            for bi, num_nodes in enumerate(batch_n_nodes):
                folded_queries[bi, :num_nodes, :queries.shape[-1]] = \
                    queries[:num_nodes]
                folded_keys[bi, :num_nodes, :keys.shape[-1]] = \
                    keys[:num_nodes]
        
        else:
            folded_queries = queries[None].contiguous()
            folded_keys = keys[None].contiguous()
        
        # because the folded tensors are zero everywhere there is no real
         # input, we don't need to mask anything.
        dot_prod = torch.bmm(folded_queries, folded_keys.transpose(-2, -1))
        return dot_prod


class RouteScorer(nn.Module):
    def __init__(self, embed_dim, nonlin_type, dropout, n_mlp_layers=2,
                 mlp_width=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_extra_feats = 13
        self.dropout = dropout
        if mlp_width is None:
            mlp_width = self.in_dim * 2
        self.out_mlp = get_mlp(n_mlp_layers, mlp_width, nonlin_type, 
                               dropout, in_dim=self.in_dim, out_dim=1)
        self.extras_norm = FeatureNorm(self.n_extra_feats)

    @property
    def in_dim(self):
        return self.embed_dim * 2 + self.n_extra_feats

    def forward(self, state, node_descs, route_idxs, route_time, 
                node_padding_mask=None):
        # assemble route sequences
        route_gather_idxs = route_idxs * (route_idxs > -1)
        route_gather_idxs = \
            route_gather_idxs.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        route_seqs = node_descs.gather(1, route_gather_idxs)

        route_len = (route_idxs > -1).sum(dim=1, dtype=torch.float)
        route_feats = torch.stack((route_time, route_len), dim=1)
        global_feats = state.get_global_state_features()
        extra_feats = torch.cat((route_feats, global_feats), dim=1)

        # pass sequences through encoder
        route_desc = self.encode(node_descs, route_seqs, node_padding_mask,
                                 route_idxs == -1)
        
        global_node_desc = node_descs.mean(dim=1)

        # normalize extra features
        extra_feats = self.extras_norm(extra_feats)
        # pass encoding and features through MLP
        in_vec = torch.cat((route_desc, 
                            global_node_desc, 
                            extra_feats), dim=1)
        scores = self.out_mlp(in_vec)
        return scores 

    def encode(self, node_descs, route_seqs, node_pad_mask=None, 
               route_pad_mask=None, embed_pos=False):
        raise NotImplementedError


class GlobalMeanNodesRouteScorer(RouteScorer):
    @property
    def in_dim(self):
        return self.n_extra_feats + self.embed_dim

    def encode(self, node_descs, route_seqs, node_pad_mask=None, 
               route_pad_mask=None, embed_pos=False):
        return torch.zeros((node_descs.shape[0], 0), device=node_descs.device)


class ValueNetScorer(GlobalMeanNodesRouteScorer):
    def forward(self, state, node_descs, node_pad_mask):
        time_placeholder = torch.zeros((state.batch_size), device=state.device)
        route_placeholder = torch.full((state.batch_size, state.max_n_nodes), 
                                       -1, dtype=int, device=state.device)
        return super().forward(state, node_descs, route_placeholder, 
                               time_placeholder, node_pad_mask)


class RouteUniformScorer(RouteScorer):
    def __init__(self, value=0.0, *args, **kwargs):
        super(RouteScorer, self).__init__()
        self.value = value

    def forward(self, state, node_descs, *args, **kwargs):
        batch_size = node_descs.shape[0]
        return torch.full((batch_size, 1), self.value, 
                          device=node_descs.device)


class RouteAlphaScorer(RouteScorer):
    def __init__(self, *args, **kwargs):
        super(RouteScorer, self).__init__()

    def forward(self, state, node_descs, *args, **kwargs):
        route_time_weight = state.cost_weights['route_time_weight']
        # this score gives halting probability = route_time_weight
        halt_score = ((1 - route_time_weight) / route_time_weight).log() / -2
        # clamp these to our torch fmin and fmax
        halt_score.clamp_(TORCH_FMIN, TORCH_FMAX)
        return halt_score[:, None]


class RouteEndpointsScorer(RouteScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        placeholder = torch.zeros((1, 1, self.embed_dim * 2), 
                                  dtype=torch.float32)
        self.register_buffer('placeholder', placeholder)

    @property
    def in_dim(self):
        return self.n_extra_feats + self.embed_dim * 3

    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        # get first and last node in each route sequence
        first_nodes = route_seqs[:, 0]
        idxs = (~route_pad_mask).sum(dim=1, keepdim=True) - 1
        route_is_empty = route_pad_mask[:, 0]
        # negative indices will occur if the route is empty
        idxs[route_is_empty] = 0
        idxs = idxs[..., None].expand(-1, -1, route_seqs.shape[-1])
        last_nodes = route_seqs.gather(1, idxs).squeeze(1)
        descs = torch.cat((first_nodes, last_nodes), dim=-1)
        # where the route is empty, use a placeholder
        descs[route_is_empty] = self.placeholder
        # append the mean descriptor of the nodes
        # mean_descs = mean_pool_sequence(node_descs, node_pad_mask)
        # descs = torch.cat((descs, mean_descs), dim=-1)
        return descs
    

class RouteMeanScorer(RouteScorer):
    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        return mean_pool_sequence(route_seqs, route_pad_mask)


class RouteTransformerScorer(RouteScorer):
    def __init__(self, n_heads, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = TransformerRouteEncoder(self.embed_dim, n_heads, 
                                               n_layers, self.dropout)

    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask,
               embed_pos=True):
        route_descs = self.encoder.encode(route_seqs, route_pad_mask, 
                                          embed_pos)
        return route_descs


class RouteLatentScorer(RouteScorer):
    def __init__(self, n_heads, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LatentAttentionEncoder(self.embed_dim, n_heads=n_heads,
                                              n_layers=n_layers, 
                                              dropout=self.dropout)
        
    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask,
               embed_pos=True):
        query = mean_pool_sequence(node_descs, node_pad_mask)[:, None]
        enc = self.encoder(route_seqs, query, route_pad_mask, embed_pos)
        return enc.squeeze(1)


class DummyMLP(nn.Module):
    def __init__(self, out_dim, out_val=0):
        super().__init__()
        self.out_dim = out_dim
        self.out_val = out_val

    def forward(self, inpt):
        out_shape = tuple(inpt.shape[:-1]) + (self.out_dim,)
        output = torch.full(out_shape, self.out_val, device=inpt.device)
        return output


class RouteGeneratorBase(nn.Module):
    def __init__(self, backbone_net, mean_stop_time_s, 
                 embed_dim, n_nodepair_layers, nonlin_type=DEFAULT_NONLIN, 
                 dropout=MLP_DEFAULT_DROPOUT, symmetric_routes=True, 
                 temperature=1, fixed_frequency_Hz=0.00074,
                 only_routes_with_demand_are_valid=False, 
                 low_memory_mode=False):
        """nonlin_type: a nonlin, or the string name of a nonlin, like 'ReLU' 
            or 'LeakyReLU' or 'Tanh'"""
        super().__init__()
        # make this a parameter of the model so it will be kept in state_dict
        # mst = torch.tensor(mean_stop_time_s, dtype=torch.float32)
        self.mean_stop_time_s = mean_stop_time_s

        self.dropout = dropout
        if type(nonlin_type) is str:
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.embed_dim = embed_dim
        self.n_nodepair_layers = n_nodepair_layers
        self.backbone_net = backbone_net
        self.symmetric_routes = symmetric_routes
        self.temperature = temperature
        self.fixed_freq = fixed_frequency_Hz
        self.only_routes_with_demand_are_valid = \
            only_routes_with_demand_are_valid
        self.low_memory_mode = low_memory_mode

        self.edge_feat_dim = backbone_net.in_edge_dim
        if self.edge_feat_dim is None:
            self.edge_feat_dim = 0
        # + 2 for the cost weights, + 2 for the two n_routes features
        if symmetric_routes:
            self.full_nodepair_dim = self.edge_feat_dim + embed_dim
        else:
            self.full_nodepair_dim = self.edge_feat_dim + embed_dim * 2

        # edge feats: demand, drive time, has route, route time, has street,
         # street time, has 1-transfer path, has 2-transfer path, has any path,
         # shortest transit path time, is same node
        if self.edge_feat_dim > 0:
            self.edge_norm = FeatureNorm(self.edge_feat_dim - 1)
            self.time_norm = FeatureNorm(1)
        self.node_norm = FeatureNorm(backbone_net.in_node_dim)

    @property
    def edge_key_order(self):
        return (DEMAND_KEY, ROUTE_KEY)

    def update_and_freeze_feature_norms(self):
        self.update_feature_norms()
        self.freeze_feature_norms()

    def update_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.update()

    def freeze_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.freeze()

    def plan(self, *args, **kwargs):
        return self.forward_oldenv(*args, **kwargs)        

    def setup_planning(self, state):         
        # apply normalization to input features
        norm_stops_x = self.node_norm(state.graph_data[STOP_KEY].x)
        state.set_normalized_features(norm_stops_x)
        return state

    def step(self, state, greedy):
        log.debug("stepping")
        # apply normalization to input features
        norm_stops_x = state.norm_node_features
        # assemble route data objects
        input_data_list = state.graph_data.to_data_list()
        for dd in input_data_list:
            # create a fully-connected graph with node embeddings
            dd.edge_index = dd[DEMAND_KEY].edge_index
            dd.x = norm_stops_x[:dd.num_nodes]
            norm_stops_x = norm_stops_x[dd.num_nodes:]

        # get edge features and assign them to the batch
        if type(self.backbone_net) in [EdgeGraphNet, GraphAttnNet]:
            edge_features_square = self._get_edge_features(state)
            for ef, dd in zip(edge_features_square, input_data_list):
                dd.edge_attr = ef[dd.edge_index[0], dd.edge_index[1]]
        else:
            edge_features_square = None
            
        input_batch = Batch.from_data_list(input_data_list)

        # run GNN forward
        if self.backbone_net.gives_edge_features:
            route_node_embeds, _ = self.backbone_net(input_batch)
        else:
            route_node_embeds = self.backbone_net(input_batch)
 
        # do routing and choosing (different subclasses do it differently)
        log.debug("get routes")
        batch_new_routes, logits, entropy = self._get_routes(
            1, state, route_node_embeds, edge_features_square, greedy)
        
        # update the state
        state.add_new_routes(
            batch_new_routes, self.only_routes_with_demand_are_valid)
        return state, logits, entropy
    
    def _get_edge_features(self, state):
        # edge feats: demand, drive time, has route, route time, has street,
         # street time, has 1-transfer path, has 2-transfer path, has any path,
         # shortest transit path time, is same node
        edge_feature_parts = [state.demand]
        # has a direct route
        has_direct_route = state.route_mat.isfinite()
        edge_feature_parts.append(has_direct_route)
        # time on direct route 
        finite_direct_times = tu.get_update_at_mask(state.route_mat, 
                                                    ~has_direct_route)
        edge_feature_parts.append(finite_direct_times)
        # has street
        has_street = state.graph_data.street_adj.isfinite()
        edge_feature_parts.append(has_street)
        # street time
        street_times = tu.get_update_at_mask(state.graph_data.street_adj,
                                             ~has_street)
        edge_feature_parts.append(street_times)
        # has 1-transfer path, 2-transfer path, any path, shortest transit
        # compute route shortest path lengths
        edge_feature_parts.append(state.has_path)
        # times of existing paths
        state.transit_times[~state.has_path] = 0
        edge_feature_parts.append(state.transit_times)
        is_direct_path = state.directly_connected
        flt_is_direct_path = is_direct_path.to(torch.float32)
        flt_upto_1trnsfr_path = flt_is_direct_path.bmm(flt_is_direct_path)
        flt_upto_2trnsfr_path = flt_upto_1trnsfr_path.bmm(flt_is_direct_path)
        upto_1trnsfr_path = flt_upto_1trnsfr_path.bool()
        is_1trnsfr_path = upto_1trnsfr_path ^ is_direct_path
        edge_feature_parts.append(is_1trnsfr_path)
        upto_2trnsfr_path = flt_upto_2trnsfr_path.bool()
        is_2trnsfr_path = upto_2trnsfr_path ^ upto_1trnsfr_path
        edge_feature_parts.append(is_2trnsfr_path)

        # is same node
        eye = torch.eye(state.max_n_nodes, device=state.device)
        eye = eye.expand(state.batch_size, -1, -1)
        edge_feature_parts.append(eye)
        edge_features = torch.stack(edge_feature_parts, dim=-1)

        # add planes with cost weights
        cost_weight_planes = state.cost_weights_tensor[:, None, None, :]
        cost_weight_planes = cost_weight_planes.expand(-1, state.max_n_nodes, 
                                                        state.max_n_nodes, -1)
        edge_features = torch.cat((edge_features, cost_weight_planes), dim=-1)
        edge_features = self.edge_norm(edge_features)

        drive_times = self.time_norm(state.drive_times[..., None])
        edge_features = torch.cat((edge_features, drive_times), dim=-1)

        return edge_features

    # def _get_edge_features(self, state):
    #     # edge feats: demand, drive time, has route, route time, has street,
    #      # street time, has 1-transfer path, has 2-transfer path, has any path,
    #      # shortest transit path time, is same node
    #     # has a direct route
    #     bool_feats = []
    #     has_direct_route = state.route_mat.isfinite()
    #     bool_feats.append(has_direct_route)
    #     # has street
    #     has_street = state.street_adj.isfinite()
    #     bool_feats.append(has_street)
    #     bool_feats.append(state.has_path)

    #     # has 1-transfer path, 2-transfer path, any path, shortest transit
    #     is_direct_path = state.directly_connected
    #     flt_is_direct_path = is_direct_path.to(torch.float32)
    #     flt_upto_1trnsfr_path = flt_is_direct_path.bmm(flt_is_direct_path)
    #     flt_upto_2trnsfr_path = flt_upto_1trnsfr_path.bmm(flt_is_direct_path)
    #     upto_1trnsfr_path = flt_upto_1trnsfr_path.bool()
    #     is_1trnsfr_path = upto_1trnsfr_path ^ is_direct_path
    #     bool_feats.append(is_1trnsfr_path)
    #     upto_2trnsfr_path = flt_upto_2trnsfr_path.bool()
    #     is_2trnsfr_path = upto_2trnsfr_path ^ upto_1trnsfr_path
    #     bool_feats.append(is_2trnsfr_path)

    #     # is same node
    #     eye = torch.eye(state.max_n_nodes, device=state.device)
    #     eye = eye.expand(state.batch_size, -1, -1)
    #     bool_feats.append(eye)
    #     bool_feats = torch.stack(bool_feats, dim=-1).to(dtype=torch.float32)

    #     # time on direct route 
    #     numerical_parts = [state.demand]
    #     finite_direct_times = get_update_at_mask(state.route_mat, 
    #                                              ~has_direct_route)
    #     numerical_parts.append(finite_direct_times)

    #     # street time
    #     street_times = get_update_at_mask(state.street_adj, ~has_street)
    #     numerical_parts.append(street_times)

    #     # times of existing paths
    #     transit_times = state.transit_times.clone()
    #     transit_times[~state.has_path] = 0
    #     numerical_parts.append(transit_times)
    #     numerical_feats = torch.stack(numerical_parts, dim=-1)
    #     numerical_feats = self.edge_norm(numerical_feats)

    #     # add planes with cost weights
    #     # we know they range roughly uniformly from 0 to 1, so don't bother 
    #      # scaling
    #     cost_weight_planes = state.cost_weights_tensor[:, None, None, :]
    #     cost_weight_planes = cost_weight_planes.expand(-1, state.max_n_nodes, 
    #                                                     state.max_n_nodes, -1)
    #     edge_features = torch.cat(
    #         (bool_feats, numerical_feats, cost_weight_planes), dim=-1)

    #     drive_times = self.time_norm(state.drive_times[..., None])
    #     edge_features = torch.cat((edge_features, drive_times), dim=-1)

    #     return edge_features

    def forward(self, state, greedy=False):
        # do a full rollout
        state = self.setup_planning(state)

        log.debug("starting route-generation loop")
        all_logits = []
        all_entropy = 0
        for _ in range(state.n_routes_to_plan):
            state, logits, entropy = self.step(state, greedy)
            all_logits.append(logits)
            all_entropy += entropy
    
        routes_tensor = \
            tu.get_batch_tensor_from_routes(state.routes, state.device)
        # TODO make this return a named tuple with names that actually make 
         # sense
        logits = torch.cat(all_logits, dim=1)
        result = PlanResults(
            state=state, stop_logits=None,
            route_logits=logits, freq_logits=None, entropy=entropy, 
            stops_tensor=None, routes_tensor=routes_tensor, freqs_tensor=None, 
            stop_est_vals=None, route_est_vals=None, freq_est_vals=None
        )
        return result
    
    @staticmethod
    def fold_node_descs(node_descs, state):
        # fold the node description vectors so that the batch dimension
         # is separate from the nodes-in-a-batch-elem dimension
        folded_node_descs = torch.zeros((state.batch_size, state.max_n_nodes, 
                                         node_descs.shape[-1]), 
                                         device=node_descs.device)
        node_pad_mask = torch.zeros((state.batch_size, state.max_n_nodes), 
                                    dtype=bool, device=node_descs.device)
        for bi, num_nodes in enumerate(state.n_nodes):
            folded_node_descs[bi, :num_nodes, :node_descs.shape[-1]] = \
                node_descs[:num_nodes]
            node_pad_mask[bi, num_nodes:] = True
        
        return folded_node_descs, node_pad_mask

    def set_invalid_scores_to_fmin(self, scores, valid_terminals_mat):
        scores_holder = torch.full_like(scores, TORCH_FMIN)
        no_valid_options = valid_terminals_mat.sum(dim=(1,2)) == 0
        if no_valid_options.any():
            # no valid options, so make all terminal pairs equally likely
            off_diag = ~torch.eye(scores.shape[-1], device=scores.device, 
                                  dtype=bool)
            where_nvo = torch.where(no_valid_options)[0]
            scores_holder[where_nvo[:, None], off_diag] = 0

        # valid_terminals_mat keeps changing, so clone so pytorch doesn't 
         # complain during backprop
        vtm = valid_terminals_mat.clone()
        scores_holder[vtm] = scores[vtm]

        return scores_holder

    def get_biased_scores(self, scores, bias, are_connected):
        # no bias for demand that's already covered
        # bias = get_update_at_mask(bias, are_connected)
        scores += bias
        # # set scores for invalid connections to 0
        scores = scores * ~are_connected

        return scores


class PathCombiningRouteGenerator(RouteGeneratorBase):
    def __init__(self, *args, n_pathscorer_layers=3, pathscorer_hidden_dim=16,
                 halt_scorer_type='endpoints', n_halt_layers=3, n_halt_heads=4,
                 force_linking_unlinked=False, logit_clip=None, 
                 serial_halting=True, max_act_len=None, **kwargs):
        """Generates routes by combining shortest paths.

        n_pathscorer_layers -- number of layers in the MLP that updates path
            scores
        pathscorer_hidden_dim -- number of hidden units in the MLP that updates
            path scores
	halt_scorer_type -- the type of halt scorer module to use.
        n_halt_layers -- number of layers in the network that computes the 
            halting score.
        n_halt_heads -- number of heads in the network that computes the
            halting score, if it has heads.
        force_linking_unlinked -- if True, then the model is forced to extend
            routes in ways that link unlinked node pairs.
        logit_clip -- if not None, then the logits of path extensions are 
            clipped to this value.
        serial_halting -- if True, the halt score is used to make a binary
            halt-or-continue choice before choosing each extension.  If False,
            the 'halt' action is part of the same set of actions as the 
            extensions.
        """
        super().__init__(*args, **kwargs)
        self.logit_clip = logit_clip
        self.serial_halting = serial_halting
        # only paths shorter than this can be added to a route
        self.max_act_len = max_act_len
        self.nodepair_scorer = get_mlp(self.n_nodepair_layers, 
                                       self.embed_dim,
                                       self.nonlin_type, self.dropout, 
                                       in_dim=self.full_nodepair_dim, 
                                       out_dim=1)

        assert self.only_routes_with_demand_are_valid is False, 'not supported'
        self.force_linking_unlinked = force_linking_unlinked

        path_scorer_indim = 16
        # self.path_input_norm = FeatureNorm(FEAT_NORM_MOMENTUM, 
        #                                    path_scorer_indim - 1)
        self.path_scorer = nn.Sequential(
            FeatureNorm(path_scorer_indim),
            get_mlp(n_pathscorer_layers, pathscorer_hidden_dim,
                    self.nonlin_type, self.dropout, in_dim=path_scorer_indim, 
                    out_dim=1)
        )
        if halt_scorer_type == "endpoints":
            self.halt_scorer = RouteEndpointsScorer(self.embed_dim, 
                                                    self.nonlin_type, 
                                                    self.dropout,
                                                    n_halt_layers)
        elif halt_scorer_type == "mean":
            self.halt_scorer = RouteMeanScorer(self.embed_dim, 
                                               self.nonlin_type, 
                                               self.dropout)
        elif halt_scorer_type == "transformer":
            self.halt_scorer = RouteTransformerScorer(n_halt_heads, 
                                                      n_halt_layers,
                                                      self.embed_dim, 
                                                      self.nonlin_type,
                                                      self.dropout)
        elif halt_scorer_type == "latent":
            self.halt_scorer = RouteLatentScorer(n_halt_heads, n_halt_layers,
                                                 self.embed_dim, 
                                                 self.nonlin_type,
                                                 self.dropout)
        elif halt_scorer_type == "alpha":
            self.halt_scorer = RouteAlphaScorer()

        # self.value_net = ValueNetScorer(self.embed_dim, self.nonlin_type, 
        #                                 self.dropout, n_halt_layers)

    def forward(self, state: RouteGenBatchState, greedy=False):
        # do a full rollout
        state = self.setup_planning(state)

        log.debug("starting route-generation loop")
        all_logits = []
        all_entropy = 0

        while not state.is_done().all():
            action, logits, entropy = self.step(state, greedy)
            # if the action is to add stops, the state must not be done
            assert not state.is_done()[action[:, 0] > -1].any(), \
                "non-null action when state is done!"
            state.shortest_path_action(action)

            all_logits.append(logits)
            all_entropy += entropy

        routes_tensor = tu.get_batch_tensor_from_routes(state.routes, 
                                                        state.device)
        
        logits = torch.stack(all_logits, dim=1)
        result = PlanResults(
            state=state, stop_logits=None,
            route_logits=logits, freq_logits=None, entropy=entropy, 
            stops_tensor=None, routes_tensor=routes_tensor, freqs_tensor=None, 
            stop_est_vals=None, route_est_vals=None, freq_est_vals=None
        )
        return result
    
    def plan_new_route(self, state: RouteGenBatchState, greedy=False, 
                       actions=None):
        # generate the input batch
        encoding = self._encode_graph(state)
        init_n_routes = state.n_finished_routes

        # while not all routes are done,
        ext_actions_given = actions is not None
        if not ext_actions_given:
            actions = []
        ended = torch.zeros((state.batch_size,), dtype=torch.bool,
                            device=state.device)
        all_logits = 0
        all_entropy = 0.0
        while not ended.all():
            # call step, passing in the input batch
            if ext_actions_given:
                # take the next action from the sequence
                action = actions[:, 0]
                actions = actions[:, 1:]
            else:
                action = None
            action, logits, entropy = self.step(state, greedy, action, 
                                                encoding)

            # sum the logits and entropy where a real action was taken
            all_logits += logits * (~ended)
            all_entropy += entropy * (~ended)
            # update endedness
            just_ended = action[:, 0] == -1
            ended = ended | just_ended
            # set the action to -1 if the route is already done
            action[ended] = -1

            # apply the action to the state
            state.shortest_path_action(action)

            if not ext_actions_given:
                # append the action to the collection
                actions.append(action)

        # return the updated state, the logit, and the actions
        if not ext_actions_given:
            actions = torch.stack(actions, dim=1)

        return actions, all_logits, all_entropy

    def step(self, state: RouteGenBatchState, greedy=False, actions=None,
             precalc_data=None):
        """Take an action for the given state.
        actions -- a batch_size x 2 tensor of predetermined actions to take. If
            None, the actions will be chosen by the model.  If the value is 
            (-1, -1), that means to halt.  Otherwise, the first and second 
            columns give the starting and ending nodes of the path segment to
            add.
        """
        log.debug("stepping")

        if precalc_data is None:
            node_descs, node_pad_mask, np_embeds, path_scores = \
                self._encode_graph(state)
        else:
            node_descs, node_pad_mask, np_embeds, path_scores = precalc_data
        
        path_seqs = state.get_shortest_path_sequences()
        batch_size = state.batch_size
            
        # don't choose segments greater than the max route length
        path_lens = (path_seqs > -1).sum(dim=-1)
        current_route_lens = state.current_route_n_stops
        post_ext_lens = current_route_lens[:, None, None] + path_lens
        exts_are_too_long = post_ext_lens > state.max_route_len[:, None, None]
        paths_are_invalid = exts_are_too_long | ~state.valid_terms_mat

        # experimentally, only allow adding one node at a time
        if self.max_act_len is not None:
            path_too_long = path_lens > self.max_act_len
            paths_are_invalid = paths_are_invalid | path_too_long
        
        if self.force_linking_unlinked:
            # don't choose segments that don't extend coverage, if coverage is
             # not complete
            extends_coverage_if_needed = check_extensions_add_connections(
                state.has_path, path_seqs)
            paths_are_invalid |= ~extends_coverage_if_needed

        # set all invalid path scores to FMIN
        path_scores = self.set_invalid_scores_to_fmin(path_scores,
                                                      ~paths_are_invalid)
        no_valid_options = paths_are_invalid.all(-1).all(-1)
        old_route_is_done = state.is_done() | no_valid_options
        
        # determine if we're extending a route or starting a new one
        # for starting, we ignore the halt choice
        starting = ~state.is_done() & (current_route_lens == 0)

        # update scores and get validity for extensions to non-empty routes
        padding = (0,1, 0,1, 0,0)
        drive_times = torch.nn.functional.pad(state.drive_times, padding)
        padding = (0, 0) + padding
        np_embeds = torch.nn.functional.pad(np_embeds, padding)
        are_on_routes = torch.zeros((batch_size, state.max_n_nodes + 1), 
                                    device=state.device).bool()
        batch_idxs = state.batch_indices        
        are_on_routes[batch_idxs[:, None], state.current_routes] = True
        # make padding column False
        are_on_routes[:, -1] = False
        getext_args = (state, current_route_lens, are_on_routes, path_seqs, 
                       np_embeds, drive_times, path_scores, path_lens, 
                       paths_are_invalid)
        prev_scores, prev_valid = self._get_extension_scores(*getext_args, 
            before_or_after='before')
        next_scores, next_valid = self._get_extension_scores(*getext_args,
            before_or_after='after')
        
        # compute the scores for halting
        halt_scores = self.halt_scorer(state, node_descs, state.current_routes, 
                                       state.current_route_time, node_pad_mask)
        if self.force_linking_unlinked:
            # don't halt if the system is not fully linked and we can
             # continue
            not_fully_linked = ~(state.has_path.all(-1).all(-1))
            halt_scores[not_fully_linked] = TORCH_FMIN

        # don't halt if the route is too short
        halt_scores[current_route_lens < state.min_route_len] = TORCH_FMIN
        # but do halt if it's done, including if there are no extensions
        halt_scores[old_route_is_done] = TORCH_FMAX
        ext_valid = prev_valid | next_valid
        no_valid_ext = ~ext_valid.any(-1).any(-1)
        halt_scores[~starting & no_valid_ext] = TORCH_FMAX

        if self.serial_halting:
            # decide whether to halt
            continue_scores = -halt_scores
            cont_or_halt = torch.cat((continue_scores, halt_scores), dim=-1)

            given_halt_actions = None
            if actions is not None:
                # predetermined actions were given, so force them to be chosen
                given_halt_actions = (actions[:, 0] == -1).to(torch.long)

            halt, corh_logit, corh_ent = select(cont_or_halt, not greedy,
                                                self.temperature,
                                                selection=given_halt_actions)
            assert (corh_logit > TORCH_FMIN).all(), "halt score is too low!"
            corh_logit = corh_logit.squeeze(1)
            chose_halt = halt.squeeze(1).bool()
            
            # update doneness
            route_is_done = old_route_is_done | chose_halt

        ext_scores = prev_scores * prev_valid + next_scores * next_valid + \
            TORCH_FMIN * ~ext_valid
        # for extending, we care whether we chose to halt
        extending = ~starting & ~route_is_done
        xtnding_exp = extending[:, None, None]
        start_scores = self._update_path_scores(state, path_scores, path_lens,
                                                state.drive_times)
        ppe_scores = start_scores * ~xtnding_exp + ext_scores * xtnding_exp

        # select an option
        flat_path_scores = ppe_scores.reshape(batch_size, -1)
        if not self.serial_halting:
            flat_path_scores = torch.cat((flat_path_scores, halt_scores), 
                                         dim=-1)
            halt_idx = flat_path_scores.shape[-1] - 1

        given_ext_actions = None
        if actions is not None:
            # predetermined actions were given, so force them to be chosen
            given_ext_actions = actions[:, 0] * state.n_nodes + actions[:, 1]
            if self.serial_halting:
                given_ext_actions[route_is_done] = 0
            else:
                given_ext_actions[route_is_done] = halt_idx
                given_ext_actions[actions[:, 0] < 0] = halt_idx

        flat_idxs, ext_logit, ext_ent = select(flat_path_scores, not greedy,
                                               self.temperature,
                                               selection=given_ext_actions)

        flat_idxs = flat_idxs.squeeze(-1)
        ext_logit = ext_logit.squeeze(-1)
        if self.serial_halting:
            # these don't contribute if we chose to halt
            ext_logit[chose_halt] = 0.0
            ext_ent[chose_halt] = 0.0
        else:
            # mark the halt action as chosen
            chose_halt = flat_idxs == halt_idx
            route_is_done = route_is_done | chose_halt

        assert ((ext_logit > TORCH_FMIN) | route_is_done).all()

        from_idxs = torch.div(flat_idxs, state.n_nodes, rounding_mode='floor')
        to_idxs = flat_idxs % state.n_nodes
        folded_idxs = torch.stack((from_idxs, to_idxs), dim=-1)
        set_to_minus1 = route_is_done[:, None]
        folded_idxs = folded_idxs * ~set_to_minus1 + -1 * set_to_minus1

        # combine the logits and entropies of the decisions
        if self.serial_halting:
            logits = corh_logit + ext_logit
            entropy = corh_ent + ext_ent
        else:
            logits = ext_logit
            entropy = ext_ent

        # is_halt = (folded_idxs == -1).any(-1)
        # badlen = (current_route_lens < state.min_route_len) & \
        #     (current_route_lens > 0)
        # if (is_halt & badlen).any():
        #     log.warning("Halting a route that's too short!")

        return folded_idxs, logits, entropy

    def _encode_graph(self, state: RouteGenBatchState):
        # assemble route data objects
        input_batch = copy.copy(state.graph_data)
        input_batch.x = state.norm_node_features
        input_batch.edge_index = input_batch[DEMAND_KEY].edge_index
        edge_feats_square = self._get_edge_features(state)
        if type(self.backbone_net) in [EdgeGraphNet, GraphAttnNet]:
            isedge_mask = torch.eye(state.max_n_nodes, device=state.device)
            isedge_mask = ~(isedge_mask.bool())
            if state.max_n_nodes * state.batch_size > input_batch.num_nodes:
                # some graphs are smaller than the max size, so mask out the
                 # invalid node locations.
                isedge_mask = isedge_mask.repeat(state.batch_size, 1, 1)
                for bi in range(state.batch_size):
                    isedge_mask[bi, state.n_nodes[bi]:] = False
                    isedge_mask[bi, :, :state.n_nodes[bi]:] = False
            else:
                isedge_mask = isedge_mask.expand(state.batch_size, -1, -1)

            edge_feats_flat = edge_feats_square[isedge_mask]
            input_batch.edge_attr = edge_feats_flat

        path_seqs = state.get_shortest_path_sequences()

        # run GNN forward
        if self.backbone_net.gives_edge_features:
            node_descs, _ = self.backbone_net(input_batch)
        else:
            node_descs = self.backbone_net(input_batch)

        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)
        if self.symmetric_routes:
            # just sum them
            np_embeds = node_descs[:, None] + node_descs[:, :, None]
        else:
            # concatenate them along the last dimension
            np_embeds = get_node_pair_descs(node_descs)
        np_embeds = torch.cat((np_embeds, edge_feats_square), dim=-1)
        np_scores = self.nodepair_scorer(np_embeds).squeeze(-1)

        assert (np_scores.abs() < 10**6).all(), "Nodepair scores are " \
            "blowing up, something wierd is going on!"

        path_scores = tu.aggr_edges_over_sequences(path_seqs, 
                                                   np_scores[..., None], 'sum')
        path_scores.squeeze_(-1)

        return node_descs, node_pad_mask, np_embeds, path_scores

    def _get_extension_scores(self, state: RouteGenBatchState, route_lens, 
                              are_on_routes, all_paths, nodepair_embeds, 
                              drive_times, path_scores, path_lens, 
                              invalid_terms, before_or_after='after'):
        log.debug("get extension scores")
        assert before_or_after in ['before', 'after']
        lens_wo_term = route_lens - 1
        # clamp for cases where the route is empty, so we don't get -1
        lens_wo_term.clamp_(min=0)
        times_from_start = state.current_route_times_from_start
        routes = state.current_routes
        final_times = times_from_start.gather(1, lens_wo_term[:, None])
        times_to_end = (times_from_start - final_times).abs()
        if before_or_after == 'after':
            # terminal is last node
            mask = tu.get_variable_slice_mask(routes, 1, froms=lens_wo_term)
            routes_but_terminal = tu.get_update_at_mask(routes, mask, -1)[:, :-1]
            terminal_nodes = routes.gather(1, lens_wo_term[:, None]).squeeze(-1)
            times_to_end = tu.get_update_at_mask(times_to_end, mask, 0)[:, :-1]
        else:
            # terminal is first node
            routes_but_terminal = routes[:, 1:].clone()
            terminal_nodes = routes[:, 0]
            times_to_end = times_to_end[:, 1:]
            # swap from- and to-node axes so indexing the first node axis
             # by the terminal gives data on edges/paths *to* the terminal,
             # instead of *from* the terminal.
            drive_times = drive_times.transpose(-2, -1)
            nodepair_embeds = nodepair_embeds.transpose(-3, -2)
            path_scores = path_scores.transpose(-2, -1)
            path_lens = path_lens.transpose(-2, -1)
            all_paths = all_paths.transpose(-3, -2)

        # get possible extensions
        batch_idxs = state.batch_indices
        raw_extension_paths = all_paths[batch_idxs, terminal_nodes]
        extension_paths = raw_extension_paths.clone()
        # ignore the terminal itself in the extensions
        extension_paths[extension_paths == terminal_nodes[:, None, None]] = -1

        times_from_end = drive_times[batch_idxs, terminal_nodes]
        # batch_size x route_len - 1 x n_nodes + 1
        times_on_route = times_from_end[:, None] + times_to_end[:, :, None]
        
        # compute scores for all edges from each node on route
        embeds_on_route = nodepair_embeds[batch_idxs[:, None], 
                                          routes_but_terminal]

        # use a neural network to compute scores and lengths
        norm_times = self.time_norm(times_on_route.flatten(0, 2)[:, None])
        norm_times = norm_times.reshape_as(times_on_route)[..., None]
        log.debug("updating edge scores...")
        embeds_w_newtimes = torch.cat((embeds_on_route[..., :-1], norm_times),
                                      dim=-1)
        from_route_edge_scores = self.nodepair_scorer(embeds_w_newtimes)
        from_route_edge_scores = from_route_edge_scores.squeeze(-1)
        log.debug("edge scores updated")

        # zero out scores corresponding to dummy route nodes
        from_route_edge_scores = \
            from_route_edge_scores * (routes_but_terminal > -1)[..., None]
        # sum over nodes on route to get "score" of visiting each node before/
         # after the current route.
        from_route_node_scores = from_route_edge_scores.sum(dim=-2)
        # compute scores for each possible following sequence
        # ignore first node of each sequence, since it's the terminal
        from_route_pathnode_scores = \
            from_route_node_scores[batch_idxs[:, None, None], extension_paths]
        # zero locations corresponding to "dummy" extension path nodes
        from_route_pathnode_scores[extension_paths == -1] = 0
        from_route_path_scores = from_route_pathnode_scores.sum(dim=-1)

        # add score of route from end node to each node not on route, and add
         # current score
        innate_scores = path_scores[batch_idxs, terminal_nodes]
        ext_scores = from_route_path_scores + innate_scores

        # apply final transformer of lengths
        new_path_lens = path_lens[batch_idxs, terminal_nodes]
        new_lens = new_path_lens + route_lens[:, None]
        # max over the stops already on the route, and drop the dummy row
        added_drive_times = times_from_end[..., :-1]

        ext_scores = self._update_path_scores(state, ext_scores, new_path_lens, 
                                              added_drive_times, route_lens,
                                              times_to_end.max(-1)[0])

        assert ext_scores.isfinite().all()

        revisits = are_on_routes[batch_idxs[:, None, None], extension_paths]            
        revisits = revisits.any(dim=-1)
        is_empty_path = path_lens[batch_idxs, terminal_nodes] == 0
        invalid = revisits | is_empty_path | \
            (new_lens > state.max_route_len[:, None])

        if self.force_linking_unlinked:
            extends_coverage_if_needed = check_extensions_add_connections(
                state.has_path, raw_extension_paths)
            invalid |= ~extends_coverage_if_needed

        out_scores = torch.zeros_like(path_scores)
        out_scores[batch_idxs, terminal_nodes] = ext_scores
        out_valids = torch.zeros_like(path_scores, dtype=bool)
        out_valids[batch_idxs, terminal_nodes] = ~invalid

        if before_or_after == 'before':
            # swap from- and to-node axes back
            out_scores = out_scores.transpose(-2, -1)
            out_valids = out_valids.transpose(-2, -1)

        out_valids &= ~invalid_terms
        out_scores = tu.get_update_at_mask(out_scores, ~out_valids, TORCH_FMIN)

        log.debug("extension scores gotten")

        return out_scores, out_valids

    # def _update_path_scores(self, state, base_path_scores, new_path_lens, 
    #                         new_drive_times):
    #     """Update the path scores based on the new path lengths and drive times.
    #     """
    #     if (base_path_scores > 10**5).any():
    #         log.warning("base path scores are blowing up!")

    #     is_valid = base_path_scores > TORCH_FMIN
    #     update_in = torch.stack((base_path_scores, new_drive_times,
    #                              new_path_lens), dim=-1)

    #     global_feat = state.get_global_state_features()
    #     update_in = nn.functional.pad(update_in, (0, global_feat.shape[-1]))
    #     for _ in range(update_in.ndim - global_feat.ndim):
    #         global_feat = global_feat.unsqueeze(1)
    #     update_in[..., -global_feat.shape[-1]:] = global_feat
    #     update_in = update_in[is_valid]

    #     updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
    #     if self.logit_clip is not None:
    #         upswi = torch.tanh(updated_path_scores_wo_infs)
    #         updated_path_scores_wo_infs = torch.clamp(upswi, -self.logit_clip, 
    #                                                   self.logit_clip)
    #     updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
    #     updated_path_scores[is_valid] = updated_path_scores_wo_infs

    #     return updated_path_scores
    
    def _update_path_scores(self, state: RouteGenBatchState, base_path_scores, 
                            new_path_lens, new_drive_times, prev_len=None, 
                            prev_drive_time=None):
        """Update the path scores based on the new path lengths and drive times.
        """
        is_valid = base_path_scores > TORCH_FMIN
        if prev_len is None:
            prev_len = torch.zeros_like(new_path_lens)
        else:
            # add a dimension to match new_path_lens
            prev_len = prev_len[:, None]
            prev_len = prev_len.expand(-1, new_path_lens.shape[-1])
        if prev_drive_time is None:
            prev_drive_time = torch.zeros_like(new_drive_times)
        else:
            # add a dimension to match new_drive_times
            prev_drive_time = prev_drive_time[:, None]
            prev_drive_time = prev_drive_time.expand(-1, 
                                                     new_drive_times.shape[-1])
        
        update_in = torch.stack((base_path_scores, new_drive_times,
                                 new_path_lens, prev_drive_time, prev_len), 
                                 dim=-1)

        global_feat = state.get_global_state_features()
        update_in = nn.functional.pad(update_in, (0, global_feat.shape[-1]))
        for _ in range(update_in.ndim - global_feat.ndim):
            global_feat = global_feat.unsqueeze(1)

        update_in[..., -global_feat.shape[-1]:] = global_feat

        update_in = update_in[is_valid]

        updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
        if self.logit_clip is not None:
            upswi = torch.tanh(updated_path_scores_wo_infs)
            updated_path_scores_wo_infs = torch.clamp(upswi, -self.logit_clip, 
                                                      self.logit_clip)
        updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
        updated_path_scores[is_valid] = updated_path_scores_wo_infs

        return updated_path_scores
    


class RandomPathCombiningRouteGenerator(PathCombiningRouteGenerator):
    """Behaves like PathCombiningRouteGenerator, but all options are equally
        likely."""
    def __init__(self, halt_prob_is_route_time_weight=False, 
                 *args, **kwargs):
        super().__init__(n_nodepair_layers=0, *args, **kwargs)
        # 0 so adding them gives the same result no matter how many are summed
        self.nodepair_scorer = DummyMLP(1, 0.0)
        self.path_scorer = DummyMLP(1, 1.0)
        if halt_prob_is_route_time_weight:
            self.halt_scorer = RouteAlphaScorer()
        else:
            # halting and continuing are equally likely
            self.halt_scorer = RouteUniformScorer()


class UnbiasedPathCombiner(RouteGeneratorBase):
    def __init__(self, *args, n_heads=1, n_encoder_layers=1, 
                 n_selection_attn_layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        n_extra_path_feats = 2
        self.path_extras_norm = FeatureNorm(n_extra_path_feats)
        # self.path_encoder = MeanRouteEncoder()
        # self.path_encoder = MaxRouteEncoder()
        # self.path_encoder = SumRouteEncoder()
        # self.path_encoder = LatentAttnRouteEncoder(self.embed_dim, 
        #                                            n_heads, n_encoder_layers,
        #                                            self.dropout)
        self.path_encoder = TransformerRouteEncoder(self.embed_dim, 
                                                    n_heads, n_encoder_layers,
                                                    self.dropout)
        self.nodepair_embedder = get_mlp(
            self.n_nodepair_layers, self.embed_dim, self.nonlin_type, 
            self.dropout, in_dim=self.full_nodepair_dim)
        self.path_mlp = get_mlp(
            3, self.embed_dim * 2, self.nonlin_type, self.dropout, 
            in_dim=self.embed_dim + n_extra_path_feats, 
            out_dim=self.embed_dim)

        init_ctxt = nn.Parameter(torch.randn(1, self.embed_dim))
        self.register_parameter(name="init_ctxt", param=init_ctxt)

        self.n_selection_layers = n_selection_attn_layers
        self.kv_embedder = nn.Linear(
            self.embed_dim, self.embed_dim * (2 * n_selection_attn_layers + 1)
        )

        n_extra_global_feats = 5
        self.global_extras_norm = FeatureNorm(n_extra_global_feats)
        init_dim = self.embed_dim * 2 + n_extra_global_feats
        query_embedders = [nn.Linear(init_dim, self.embed_dim)]
        query_embedders += [nn.Linear(self.embed_dim, self.embed_dim)
                            for _ in range(n_selection_attn_layers - 1)]
        self.query_embedders = nn.ModuleList(query_embedders)

        self.halt_scorer = get_mlp(2, self.embed_dim * 2, self.nonlin_type, 
                                   self.dropout, in_dim=self.embed_dim,
                                   out_dim=1)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats,
                    greedy):
        """
        n_routes: int
        node_descs: batch_size x n_nodes x node_feat_dim
        state: batch_size x n_nodes x n_nodes x state_dim
        greedy: bool
        returns: batch_size x n_routes x route_len, all_logits, None
        """
        log.debug("get routes")
        # doesn't support chunk sizes greater than 1
        assert n_routes == 1

        dev = state.device

        # fold node descriptors
        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)
        nodepair_descs = get_node_pair_descs(node_descs)

        # concatenate the extra nodepair features if they're there
        if extra_nodepair_feats is not None:
            nodepair_descs = torch.cat((extra_nodepair_feats, nodepair_descs), 
                                       dim=-1)
        nodepair_descs = self.nodepair_embedder(nodepair_descs)

        if not hasattr(state.graph_data, '_seqs'):
            # compute the shortest paths and store them in the graph_data object
             # so we don't need to recompute them for each route.
            seqs, _ = tu.reconstruct_all_paths(state.graph_data.nexts)
            state.graph_data._seqs = seqs
        else:
            seqs = state.graph_data._seqs        

        # encode the paths
        # with torch.no_grad():
        path_encs = self.encode_path(nodepair_descs, seqs, state.drive_times)
        path_attn_embeds = self.kv_embedder(path_encs)
        
        # do all the key and value embeddings at once
        base_ext_mask = ~state.valid_terms_mat.clone()
        ext_mask = base_ext_mask.flatten(1, 2)
        
        # set the initial context
        global_feats = torch.cat(
            ((state.total_route_time / state.n_routes_to_plan)[..., None], 
             state.get_n_routes_features(), state.cost_weights_tensor), dim=-1)
        global_feats = self.global_extras_norm(global_feats)
        # get average node descriptor
        avg_node = mean_pool_sequence(node_descs, node_pad_mask)
        # concatenate the above
        global_feats = torch.cat((global_feats, avg_node), dim=-1)
        init_ctxt = self.init_ctxt.expand(state.batch_size, -1).to(device=dev)
        context = torch.cat((global_feats, init_ctxt), dim=-1)

        # set up loop-tracking variables and other needed values
        route = torch.full((state.batch_size, state.max_n_nodes), -1, 
                            device=dev)
        is_done = torch.zeros(state.batch_size, device=dev).bool()
        logits = None
        candidates = seqs.flatten(1, 2)
        cdt_times = state.drive_times.flatten(1, 2)
        first_iter = True
        batch_idxs = state.batch_indices
        relevant_attn_embeds = path_attn_embeds.flatten(1, 2)
        node_on_route = torch.zeros((state.batch_size, state.max_n_nodes + 1), 
                                    device=dev).bool()
        route_times = torch.zeros(state.batch_size, device=dev)
        route_len = (route > -1).sum(-1)

        # assemble a route!
        while not is_done.all():
            # compute scores vs current context and pick an extension path
            out_context, ext_probs = self._get_extension_scores(
                state, context, relevant_attn_embeds, ext_mask)
            # zero out scores for paths that are too long
            cdt_lens = (candidates != -1).sum(dim=-1) + route_len[:, None]
            cdt_too_long = cdt_lens > state.max_route_len[:, None]
            ext_probs = tu.get_update_at_mask(ext_probs, cdt_too_long)
            no_valid_ext = (ext_probs.sum(dim=-1) == 0)
            is_done |= no_valid_ext
            
            if not first_iter:
                # decide whether to halt using context
                halt_score = self.halt_scorer(out_context)
                halt_score[is_done] = TORCH_FMAX
                halt_score[route_len < state.min_route_len] = TORCH_FMIN
                cont_score = -halt_score
                cont_or_halt = torch.cat((cont_score, halt_score), dim=-1)
                halt, corh_logit = select(cont_or_halt, not greedy)
                # zero out logits where we've already halted
                corh_logit = corh_logit.squeeze(1) * ~is_done
                # update doneness
                is_done = is_done | halt.squeeze(1).bool()
                logits += corh_logit

            if greedy:
                selection = ext_probs.argmax(dim=-1)
            else:
                # avoid crash for all-zero entries
                ext_probs[no_valid_ext] = 1.0
                selection = ext_probs.multinomial(1)
            step_logit = ext_probs.gather(1, selection).log().squeeze(1)
            step_logit = step_logit + ~is_done
            if first_iter:
                logits = step_logit
            else:
                logits += step_logit + corh_logit

            # update the routes
            route_len = (route > -1).sum(-1)
            candidate_lens = (candidates > -1).sum(-1)
            chosen_ext_lens = candidate_lens.gather(1, selection).squeeze(1)
            # enforce no extension if is_done
            chosen_ext_lens[is_done] = 0
            # compute mask for new stops on the routes
            insert_mask = tu.get_variable_slice_mask(
                route, dim=1, froms=route_len, tos=route_len + chosen_ext_lens)
            # select the chosen extensions
            exp_sel = selection[..., None].expand(-1, -1, candidates.shape[-1])
            chosen_seqs = candidates.gather(1, exp_sel).squeeze(1)
            # insert the chosen extensions
            select_mask = tu.get_variable_slice_mask(
                chosen_seqs, dim=1, tos=chosen_ext_lens)
            # route[insert_mask] = chosen_seqs[select_mask]
            chosen_seqs_inplace = torch.zeros_like(route)
            chosen_seqs_inplace[insert_mask] = chosen_seqs[select_mask]
            route = route * ~insert_mask + chosen_seqs_inplace * insert_mask
            # update which nodes are on the route
            node_on_route[batch_idxs[:, None], chosen_seqs] = True
            # make padding column False
            node_on_route[:, -1] = False
            route_times += cdt_times.gather(1, selection).squeeze(1)

            # update the extension mask based on the current route
            # make only segments continuing from this one valid
            route_len += chosen_ext_lens
            end_nodes = route.gather(1, route_len[..., None] - 1).squeeze()
            relevant_attn_embeds = path_attn_embeds[batch_idxs, end_nodes]
            ext_mask = base_ext_mask[batch_idxs, end_nodes]
            # trim the first node of each candidate, since it's the last node
             # on the current route
            candidates = seqs[batch_idxs, end_nodes][..., 1:]
            cdt_times = state.drive_times[batch_idxs, end_nodes]
            # invalidate segments that revisit stops on the route
            revisits = node_on_route[batch_idxs[:, None, None], candidates]
            revisits = revisits.any(-1)
            ext_mask = ext_mask | revisits

            # determine necessary halts
            must_halt = ext_mask.all(-1)
            is_done = is_done | must_halt
            
            # update the context vector and manual descriptors based on the 
             # current route
            route_descs = self.encode_path(nodepair_descs, route, 
                                           route_times)
            context = torch.cat((global_feats, route_descs), dim=-1)            
            first_iter = False

        # prune extra route sequence padding
        max_route_len = (route > -1).sum(dim=-1).max()
        route = route[:, :max_route_len]

        route_len = (route > -1).sum(-1)

        return route[:, None], logits[:, None], None

    def _get_extension_scores(self, state, context_vec, kv_embeds, 
                              key_mask=None):
        # compute scores vs current context and pick an extension path
        # add query "sequence" dimension
        context_vec = context_vec[..., None, :]
        path_chunks = kv_embeds.chunk(2 * self.n_selection_layers + 1, dim=-1)
        keys = path_chunks[::2]
        values = path_chunks[1::2] + (None,)

        # TODO implement multi-head attention
        for query_embedder, lk, lv in zip(self.query_embedders, keys, values):
            query = query_embedder(context_vec)
            scores = torch.bmm(query, lk.transpose(-1, -2)).squeeze(1)
            scores /= self.embed_dim**0.5
            if key_mask is not None:
                scores[key_mask] = TORCH_FMIN
            scores = torch.nn.functional.softmax(scores, dim=-1)
            if lv is not None:
                # the final layer scores are just the selection probabilities,
                 # so don't use them to update the context
                context_vec = torch.bmm(scores[:, None], lv)

        return context_vec.squeeze(1), scores
    
    def encode_path(self, nodepair_descs, path, path_times):
        if path.ndim == 2:
            added_dim = True
            # add a route dimension
            path = path[:, None]
            path_times = path_times[:, None]
        else:
            added_dim = False
        path_mask = path == -1
        enc = self.path_encoder.encode_nodepairs(nodepair_descs, path)

        path_lens = (~path_mask).sum(-1)
        extra_feats = torch.stack((path_times, path_lens), dim=-1)
        extra_feats = self.path_extras_norm(extra_feats)
        mlp_in = torch.cat((enc, extra_feats), dim=-1)
        path_desc = self.path_mlp(mlp_in)
        if added_dim:
            path_desc.squeeze_(1)
        return path_desc


class NodeWalker(RouteGeneratorBase):
    def __init__(self, *args, n_heads=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_node_scorer = get_mlp(3, self.embed_dim * 2, 
                                         in_dim=self.embed_dim, out_dim=1)
        self.next_node_scorer = get_mlp(3, self.embed_dim * 2, out_dim=1)
        self.next_node_attn = nn.MultiheadAttention(
            self.embed_dim, n_heads, self.dropout, batch_first=True)
        halt_desc = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.register_parameter(name="halt_desc", param=halt_desc)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats,
                    greedy):
        """
        n_routes: int
        node_descs: batch_size x n_nodes x node_feat_dim
        state: batch_size x n_nodes x n_nodes x state_dim
        greedy: bool
        returns: batch_size x n_routes x route_len, all_logits, None
        """
        log.debug("get routes")
        # doesn't support chunk sizes greater than 1
        assert n_routes == 1

        # fold node descriptors
        node_descs, _ = self.fold_node_descs(node_descs, state)

        # pick first node
        node_scores = self.start_node_scorer(node_descs).squeeze(-1)
        cur_node, logits = select(node_scores, not greedy)
        route = cur_node
        cur_node = cur_node.squeeze(-1)

        # set up loop-tracking variables and other needed values
        is_done = torch.zeros(state.batch_size, device=state.device).bool()
        batch_idxs = state.batch_indices
        node_on_route = torch.zeros((state.batch_size, state.max_n_nodes + 1), 
                                    device=state.device).bool()
        adj_mat = state.graph_data.street_adj
        is_adj = adj_mat.isfinite() & (adj_mat > 0)
        halt_desc = self.halt_desc.expand(state.batch_size, -1, -1)

        # assemble a route!
        while not is_done.all():
            # compute scores vs current context and pick an extension path
            # assemble query vector: adjacent nodes and the halt option
            is_valid = is_adj[batch_idxs, cur_node] & ~node_on_route[..., :-1]
            is_valid = is_valid & ~is_done[:, None]
            valid_idxs = tu.get_indices_from_mask(is_valid, dim=-1)
            cdt_feats = node_descs[batch_idxs[:, None], valid_idxs]
            if route.shape[-1] > 1:
                # don't enable halt option unless we have at least two stops
                cdt_feats = torch.cat((cdt_feats, halt_desc), dim=-2)
                valid_idxs = nn.functional.pad(valid_idxs, (0, 1), value=True)

            # select a node
            route_feats = node_descs[batch_idxs[:, None], route]
            route_pad_mask = route == -1
            cdt_descs, _ = self.next_node_attn(
                cdt_feats, route_feats, route_feats, 
                key_padding_mask=route_pad_mask, need_weights=False)
            cdt_descs = torch.cat((cdt_descs, cdt_feats), dim=-1)
            scores = self.next_node_scorer(cdt_descs).squeeze(-1)
            scores = tu.get_update_at_mask(scores, valid_idxs == -1, TORCH_FMIN)
            cur_node, next_logit = select(scores, not greedy)
            cur_node = cur_node.squeeze(-1)

            # update route, node_on_route, logits, is_done
            if route.shape[-1] > 1:
                chose_halt = cur_node == cdt_feats.shape[-2] - 1
                is_done = is_done | chose_halt
                cur_node = tu.get_update_at_mask(cur_node, chose_halt, -1)
                next_logit[chose_halt] = 0.0
                
            route = torch.cat((route, cur_node[:, None]), dim=1)
            logits += next_logit
            node_on_route[batch_idxs, cur_node] = True

        return route[:, None], logits, None


# primitive modules


class LatentAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, latent_size=None, n_heads=8, n_layers=2,
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        # the latent embedding.  We don't learn it, but we make it a parameter
         # so that it will be included in the module's state_dict.
        if latent_size is not None:
            latent = nn.Parameter(torch.randn((1, latent_size, embed_dim)))
            self.register_parameter(name="latent", param=latent)
            # self.latent.requires_grad_(False)
        else:
            self.latent = None
        
        self.nonlin = DEFAULT_NONLIN()
        self.attention_layers = nn.ModuleList(
            nn.MultiheadAttention(embed_dim, n_heads, dropout, 
                                  batch_first=True)
            for _ in range(n_layers)
        )

    def forward(self, seq_to_encode, query=None, padding_mask=None, 
                embed_pos=False, seqlen_scale=None, residual=False):
        """
        seq_to_encode: batch_size x sequence length x embed dim tensor of the
            sequence(s) to encode
        padding_mask: batch_size x sequence length binary tensor indicating
            which elements of seq_to_encode are padding elements
        embed_pos: if True, sinusoidal position embeddings will be added to the
            sequence before encoding it.
        seqlen_scale: if provided, scale the encoding proportionally to the
            number of nodes, making the latent encoding a weighted sum rather 
            than a weighted average; the value of seqlen_scale is the 
            proportionality constant.
        residual: if True, the output of each layer is added to the query to
            get the query to the next layer; if False, the output alone is used
            as the query.
        """
        if seq_to_encode is None:
            # can't encode an empty sequence, so the "encoding" is just the
            # latent vector.
            return self.latent

        if seq_to_encode.ndim == 2:
            # there's no batch dimension, so add one
            seq_to_encode = seq_to_encode[None]

        if self.latent is None:
            assert query is not None, \
                "query_base must be provided if there is no latent vector!"
            
        dev = seq_to_encode.device
        batch_size = seq_to_encode.shape[0]
        if self.latent is not None and query is None:
            batch_latent = self.latent.tile(batch_size, 1, 1).to(device=dev)
            if seq_to_encode.shape[1] == 0:
                # can't encode an empty sequence, so the "encoding" is just the
                # latent vector.
                return batch_latent
            encoding = batch_latent
        else:
            encoding = query

        if embed_pos:
            sinseq = get_sinusoid_pos_embeddings(seq_to_encode.shape[1], 
                                                 seq_to_encode.shape[2])
            if seq_to_encode.ndim == 3:
                sinseq = sinseq[None]
            seq_to_encode = seq_to_encode + sinseq.to(device=dev)

        # ignore empty sequences
        if padding_mask is not None:
            seq_is_empty = padding_mask.all(dim=1)
            padding_mask = padding_mask[~seq_is_empty]
            encoding = encoding[~seq_is_empty]
            seq_to_encode = seq_to_encode[~seq_is_empty]
            seq_lens = padding_mask.shape[-1] - padding_mask.sum()
        else:
            seq_lens = torch.ones((batch_size, 1), device=dev)
            seq_lens *= seq_to_encode.shape[-2]

        # encode the input sequences via attention
        for attn_layer in self.attention_layers:
            attn_vec, _ = attn_layer(encoding, seq_to_encode, seq_to_encode,
                                     key_padding_mask=padding_mask)
            if seqlen_scale:
                attn_vec *= seqlen_scale * seq_lens

            if residual:
                encoding = encoding + attn_vec
            else:
                encoding = attn_vec

            if attn_layer is not self.attention_layers[-1]:
                # apply a non-linearity in between attention layers
                encoding = self.nonlin(encoding)
        
        if padding_mask is not None and seq_is_empty.any():
            batch_latent[~seq_is_empty] = encoding
            encoding = batch_latent
        
        return encoding


class EdgeGraphNetLayer(MessagePassing):
    def __init__(self, in_node_dim, in_edge_dim=0, 
                 out_dim=None, hidden_dim=None, bias=True, 
                 nonlin_type=DEFAULT_NONLIN, n_edge_layers=1, 
                 dropout=MLP_DEFAULT_DROPOUT, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if hidden_dim is None:
            hidden_dim = in_node_dim
        if out_dim is None:
            out_dim = in_node_dim
        in_dim = in_node_dim * 2 + in_edge_dim
        layers = [
            get_mlp(n_edge_layers, hidden_dim, nonlin_type, dropout=dropout, 
                    bias=bias, in_dim=in_dim, out_dim=out_dim),
        ]
        if n_edge_layers == 1:
            layers.append(nonlin_type())
        self.edge_embedder = nn.Sequential(*layers)

    def forward(self, node_features, edge_index, edge_features):
        if edge_features is not None and edge_features.ndim == 1:
            # edge features have one channel, so ensure there's a dim for them
            edge_features = edge_features[:, None]
        edge_features = self.edge_updater(edge_index, x=node_features, 
                                          edge_attr=edge_features)
        size = (node_features.shape[0], node_features.shape[0])
        node_features = self.propagate(edge_index, edge_attr=edge_features,
                                       size=size)

        return node_features, edge_features

    def message(self, edge_attr):
        return edge_attr

    def edge_update(self, x_i, x_j, edge_attr):
        in_vec = torch.cat((x_i, x_j, edge_attr), dim=-1)
        embed = self.edge_embedder(in_vec)
        return embed


class GraphNetBase(nn.Module):
    def __init__(self, n_layers, embed_dim, in_node_dim=None, in_edge_dim=None, 
                 out_dim=None, nonlin_type=DEFAULT_NONLIN, 
                 dropout=ATTN_DEFAULT_DROPOUT, recurrent=False, residual=False,
                 dense=False, n_proj_layers=1, use_norm=False, layer_kwargs={}):
        """nonlin type: nonlinearity, or the string name the non-linearity to 
            use."""
        super().__init__()

        # do some input checking
        assert not (residual and dense)
        assert not (recurrent and dense)

        # set up members
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.in_node_dim = embed_dim if in_node_dim is None else in_node_dim
        self.in_edge_dim = in_edge_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.recurrent = recurrent
        self.residual = residual
        self.dense = dense
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        if type(nonlin_type) is str:
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.nonlin = self.nonlin_type()
        self.embed_dim = embed_dim
        self.layer_kwargs = layer_kwargs

        self.node_in_proj = nn.Identity()
        self.edge_in_proj = nn.Identity()
        self.node_out_proj = nn.Identity()
        self.edge_out_proj = nn.Identity()

        if recurrent:
            if self.in_node_dim != self.embed_dim:
                # project nodes to embedding dimension
                self.node_in_proj = nn.Linear(self.in_node_dim, self.embed_dim)
            if self.gives_edge_features and self.in_edge_dim != self.embed_dim:
                # project edges to embedding dimension
                self.edge_in_proj = nn.Linear(self.in_edge_dim, self.embed_dim)

        final_nodedim = self._get_final_nodedim()
        if self.out_dim != final_nodedim:
            # projected embedded nodes to output dimension
            self.node_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                         in_dim=final_nodedim, 
                                         out_dim=self.out_dim)
            if self.gives_edge_features:
                self.edge_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                             in_dim=self._get_final_edgedim(),
                                             out_dim=self.out_dim)

        # regardless of whether the network is recurrent, we still use separate
         # normalization layers at each step
        self.use_norm = use_norm
        if use_norm:
            norm_layers = [GraphNorm(self.embed_dim) 
                        for _ in range(self.n_layers - 1)]
            self.node_norm_layers = nn.ModuleList(norm_layers)
            if self.gives_edge_features:
                norm_layers = [BatchNorm(self.embed_dim) 
                            for _ in range(self.n_layers - 1)]
                self.edge_norm_layers = nn.ModuleList(norm_layers)

    def _get_layer_node_indims(self):
        return _layer_indims_helper(self.in_node_dim, self.embed_dim, 
                                    self.n_layers, self.dense)

    def _get_layer_edge_indims(self):
        return _layer_indims_helper(self.in_edge_dim, self.embed_dim,
                                    self.n_layers, self.dense)

    def _get_final_nodedim(self):
        indims = self._get_layer_node_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    def _get_final_edgedim(self):
        indims = self._get_layer_edge_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    @property
    def gives_edge_features(self):
        raise NotImplementedError

    def forward(self, data):
        data = self.preprocess(data)
        if self.gives_edge_features:
            output = self._forward_helper(data)
        else:
            output = self._forward_helper(data), None
        return self.postprocess(*output)

    def preprocess(self, data):
        data.x = self.node_in_proj(data.x)
        if data.edge_attr is not None:
            data.edge_attr = self.edge_in_proj(data.edge_attr)
        return data
    
    def postprocess(self, node_embeds, edge_embeds):
        node_embeds = self.node_out_proj(node_embeds)
        if edge_embeds is not None:
            edge_embeds = self.edge_out_proj(edge_embeds)
        if self.gives_edge_features:
            return node_embeds, edge_embeds
        else:
            return node_embeds

    def _forward_helper(self, data):
        node_descs = data.x
        edge_descs = data.edge_attr

        for li, layer in enumerate(self.layers):
            # apply the layer and graph normalization
            out = layer(node_descs, data.edge_index, edge_descs)
            if self.gives_edge_features:
                layer_nodes, layer_edges = out
            else:
                layer_nodes, layer_edges = out, None

            if li < len(self.layers) - 1:
                # this is not the last layer, so apply dropout and nonlinearity
                if self.use_norm:
                    layer_nodes = self.node_norm_layers[li](layer_nodes, 
                                                            data.batch)
                layer_nodes = self.nonlin(self.dropout(layer_nodes))
                if layer_edges is not None:
                    if self.use_norm:
                        layer_edges = self.edge_norm_layers[li](layer_edges)
                    layer_edges = self.nonlin(self.dropout(layer_edges))

            # get next layer's inputs based on this layer's outputs
            if self.residual and node_descs.shape[-1] == layer_nodes.shape[-1]:
                # add the input to the output
                node_descs = node_descs + layer_nodes
                if layer_edges is not None:
                    edge_descs = edge_descs + layer_edges
            elif self.dense:
                # concatenate the input to the output
                node_descs = torch.cat((node_descs, layer_nodes), dim=-1)
                if layer_edges is not None:
                    edge_descs = torch.cat((edge_descs, layer_edges), dim=-1)
            else:
                # just replace the input with the output
                node_descs = layer_nodes
                if layer_edges is not None:
                    edge_descs = layer_edges

        assert node_descs.shape[-1] == self._get_final_nodedim()

        # return the outputs
        if not self.gives_edge_features:
            # don't return the same edge features
            return node_descs
        else:
            return node_descs, edge_descs


class NoOpGraphNet(GraphNetBase):
    def __init__(self, return_edges=False, *args, **kwargs):
        self.return_edges = return_edges
        super().__init__(0, 0, **kwargs)

    def forward(self, data):
        if self.gives_edge_features:
            return data.x, data.edge_attr
        else:
            return data.x

    @property
    def gives_edge_features(self):
        return self.return_edges


class WeightedEdgesNetBase(GraphNetBase):
    def __init__(self, in_edge_dim=1, *args, **kwargs):
        super().__init__(in_edge_dim=1, *args, **kwargs)
        if in_edge_dim > 1:
            # project to range 0 to 1
            self.edge_in_proj = nn.Sequential(
                nn.Linear(in_edge_dim, 1),
                nn.Sigmoid()
            )

    @property
    def gives_edge_features(self):
        return False


class SimplifiedGcn(WeightedEdgesNetBase):
    def __init__(self, n_layers, in_node_dim, out_dim, *args, **kwargs):
        super().__init__(n_layers=n_layers, embed_dim=out_dim, 
                         in_node_dim=in_node_dim, out_dim=out_dim, 
                         recurrent=False, residual=False, dense=False, 
                         *args, **kwargs)
        self.net = SGConv(in_node_dim, out_dim, n_layers)

    def _forward_helper(self, data):
        act1 = self.net(data.x, data.edge_index, data.edge_attr)
        act2 = self.nonlin(act1)
        act3 = self.dropout(act2)
        return act3
    

class Gcn(WeightedEdgesNetBase):
    def __init__(self, layer_type=GCNConv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recurrent:
            layers = [layer_type(self.embed_dim, self.embed_dim, 
                                 **self.layer_kwargs)] * self.n_layers
        else:            
            in_dims = self._get_layer_node_indims()
            layers = [layer_type(in_dim, self.embed_dim, **self.layer_kwargs)
                      for in_dim in in_dims]
            
        self.layers = nn.ModuleList(layers)


class GraphAttnNet(GraphNetBase):
    def __init__(self, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        assert self.embed_dim % n_heads == 0, \
            'embed_dim is not divisible by n_heads!'
        head_dim = self.embed_dim // self.n_heads
        make_layer = lambda in_dim: \
            GATv2Conv(in_dim, head_dim, self.n_heads, 
                      edge_dim=self.in_edge_dim, **self.layer_kwargs)
        
        if self.recurrent:
            layers = [make_layer(self.embed_dim)] * self.n_layers
        else:
            in_dims = self._get_layer_node_indims()
            layers = [make_layer(in_dim) for in_dim in in_dims]

        self.layers = nn.ModuleList(layers)

    @property
    def gives_edge_features(self):
        return False


class EdgeGraphNet(GraphNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        make_layer = lambda ind, ied: \
            EdgeGraphNetLayer(ind, ied, out_dim=self.embed_dim, 
                              nonlin_type=self.nonlin_type,
                              dropout=self.dropout_rate, **self.layer_kwargs)
        
        if self.recurrent:
            layers = [make_layer(self.embed_dim, self.embed_dim)] * \
                self.n_layers
        else:
            in_node_dims = self._get_layer_node_indims()
            in_edge_dims = self._get_layer_edge_indims()
            in_edge_dims[0] = self.in_edge_dim
            layers = [make_layer(ind, ied) for ind, ied in 
                      zip(in_node_dims, in_edge_dims)]

        self.layers = nn.ModuleList(layers)


    @property
    def gives_edge_features(self):
        return True


# helper functions

def get_node_pair_descs(node_descs):
    n_nodes = node_descs.shape[-2]
    exp_dst_nodes = node_descs[:, None].expand(-1, n_nodes, -1, -1)
    exp_src_nodes = exp_dst_nodes.permute(0, 2, 1, 3)
    return torch.cat((exp_src_nodes, exp_dst_nodes), dim=-1)


def check_extensions_add_connections(is_linked, extensions):
    # determine if any path on its own makes new connections
    path_makes_new_connection = \
        tu.aggr_edges_over_sequences(extensions, (~is_linked[..., None]))
    path_makes_new_connection = path_makes_new_connection.bool().squeeze(-1)

    # if a graph is not fully connected, then any path that doesn't make a 
     # new connection is invalid.
    is_fully_connected = is_linked.all(dim=-1).all(dim=-1)
    while is_fully_connected.ndim < path_makes_new_connection.ndim:
        is_fully_connected.unsqueeze_(-1)
    has_valid_coverage = is_fully_connected | path_makes_new_connection

    return has_valid_coverage


def update_connections_matrix(is_linked, new_routes, symmetric_routes=False):
    """Assumes the presence of a batch dimension."""
    batch_idxs = torch.arange(len(new_routes))
    # add dummy padding
    is_linked = torch.nn.functional.pad(is_linked, (0, 1, 0, 1))

    # set newly-connected nodes to be connected
    for stop_idx in range(new_routes.shape[-1] - 1):
        cur_stop = new_routes[:, stop_idx]
        next_stop = new_routes[:, stop_idx+1]
        is_linked[batch_idxs, cur_stop, next_stop] = True
        
    # re-establish false along the dummy areas
    is_linked[:, -1, :] = False
    is_linked[:, :, -1] = False

    # # is_linked = is_linked[:, :-1, :-1]
    # if symmetric_routes:
    #     # make the matrix symmetric
    #     is_linked = is_linked | is_linked.transpose(-1, -2)

    # edge_costs = torch.zeros_like(is_linked, dtype=torch.float32)
    # edge_costs[~is_linked] = float('inf')
    # edge_costs[is_linked] = 1
    # # edge_costs = edge_costs[:, :-1, :-1]
    # node_idxs = torch.arange(edge_costs.shape[-1], device=edge_costs.device)
    # edge_costs[:, node_idxs, node_idxs] = 0
    # _, _, dists = floyd_warshall(edge_costs, True)
    # # return dists.isfinite()
    # fw_is_linked = dists.isfinite()

    for stop_idx in range(new_routes.shape[-1] - 1):
        cur_stop = new_routes[:, stop_idx]
        next_stop = new_routes[:, stop_idx+1]
        linked_to_cur = is_linked[batch_idxs, :, cur_stop]
        must_be_updated_idxs = torch.where(linked_to_cur)
        update_values = is_linked[must_be_updated_idxs[0], 
                                  next_stop[must_be_updated_idxs[0]]]
        is_linked[must_be_updated_idxs[0], must_be_updated_idxs[1]] |= \
            update_values

    # remove the dummy padding 
    is_linked = is_linked[:, :-1, :-1]

    if symmetric_routes:
        # make the matrix symmetric
        is_linked = is_linked | is_linked.transpose(-1, -2)

    # finally, return the connections matrix
    return is_linked


def _layer_indims_helper(in_dim, embed_dim, n_layers, is_dense):
    indims = []
    if is_dense:
        curdim = in_dim
        for _ in range(n_layers):
            indims.append(curdim)
            curdim += embed_dim
    else:
        indims = [in_dim] 
        indims += [embed_dim] * (n_layers - 1)
    return indims


def mean_pool_sequence(sequences, padding_mask=None):
    # 2nd-to-last dim is sequence dim, last dim is feature dim
    if padding_mask is not None:
        sequences = sequences * ~padding_mask[..., None]
        seq_lens = (~padding_mask).sum(dim=-1)[..., None]
    else:
        seq_lens = sequences.shape[-2]
    # avoid generating NaNs by dividing by 0
    seq_lens[seq_lens == 0] = 1
    sums = sequences.sum(dim=-2)
    means = sums / seq_lens
    return means


def get_sinusoid_pos_embeddings(seqlen, ndims, posenc_min_rate=1/10000):
    angle_rate_exps = torch.linspace(0, 1, ndims // 2)
    angle_rates = posenc_min_rate ** angle_rate_exps
    positions = torch.arange(seqlen)
    angles_rad = positions[:, None] * angle_rates[None, :]
    sines = torch.sin(angles_rad)
    cosines = torch.cos(angles_rad)
    return torch.cat((sines, cosines), dim=-1)


def assemble_batch_routes_from_tensors(tensors):
    """
    Takes a list of batch_size x n_routes_in_tensor x max_n_stops_in_tensor
        tensors.  Assembles them into a single tensor of shape
        batch_size x total_n_routes x max_n_stops_across_tensors.
    """
    if tensors is None or len(tensors) == 0 or tensors[0] is None:
        # an "empty" input, so return None
        return None
    batch_size = tensors[0].shape[0]
    dev = tensors[0].device
    max_route_len = max([tt.shape[-1] for tt in tensors])
    n_routes = sum([tt.shape[-2] for tt in tensors])
    out = torch.zeros((batch_size, n_routes, max_route_len), device=dev)
    routes_so_far = 0
    for tt in tensors:
        end_idx = routes_so_far + tt.shape[-2]
        out[:, routes_so_far:end_idx, :tt.shape[-1]] = tt
        routes_so_far = end_idx
    return out


def select(scores, scores_as_logits=False, softmax_temp=1,
           n_selections=1, selection=None):
    """scores: an n_options or batch_size x n_options tensor."""
    if scores.ndim == 1:
        # add a batch dimension
        scores = scores[None, :]
    logits = logsoftmax(scores, softmax_temp).to(dtype=torch.float64)

    # do this to avoid nans from 0 * -inf
    ent_logits = tu.get_update_at_mask(logits, logits.isinf(), 0)
    entropy = -(ent_logits.exp() * ent_logits).sum(dim=-1)
    assert not entropy.isnan().any()

    if selection is None:
        if not scores_as_logits:
            _, selection = logits.topk(n_selections)
        else:
            probs = logits.exp()
            selection = probs.multinomial(n_selections)
            # if fewer than n_selections options survive rounding down by exp()
             # then use topk to select the top n_selections
            needs_topk = (probs > 0).sum(dim=1) < n_selections
            if needs_topk.any():
                _, topk_selection = logits[needs_topk].topk(n_selections)
                selection[needs_topk] = topk_selection

        selection.squeeze_(-1)
    
    if n_selections == 1:
        # add a "selection" dimension
        selection = selection[..., None]
        if scores.ndim == 1:
            selected_logits = logits[selection]
        else:
            # there is a batch dimension
            selected_logits = logits.gather(1, selection)

    else:
        # renormalize the log-probabilities after each selection
        idxs = torch.arange(logits.shape[1]).expand(logits.shape[0], -1)
        idxs = idxs.to(scores.device)
        selection_mask = idxs[..., None] == selection[:, None]
        selection_mask = selection_mask.any(dim=-1)
        selection_neutralizer = torch.zeros_like(scores)
        selection_neutralizer[selection_mask] = TORCH_FMIN
        unselected_scores = scores + selection_neutralizer
        if scores.ndim == 1:
            selected_scores = scores[selection]
        else:
            # there is a batch dimension
            selected_scores = scores.gather(1, selection)

        # flip the selected scores so the first selected comes last
        reordered_scores = \
            torch.cat((unselected_scores, selected_scores.flip(-1)), dim=-1)
        # sum them so the last element is all scores, 2nd last is all but
         # first picked, etc.
        logsumexps = torch.logcumsumexp(reordered_scores, -1)
        # flip again so the first selected comes first
        logdenoms = logsumexps[..., -n_selections:].flip(-1)
        selected_logits = selected_scores - logdenoms

    return selection, selected_logits, entropy


def logsoftmax(energies, softmax_temp=1):
    return nn.functional.log_softmax(energies / softmax_temp, dim=-1)    


def expand_batch_dim(tnsr_to_expand, batch_size, dim=0):
    if tnsr_to_expand.shape[dim] != 1:
        # there's no unitary dimension at the start, so add one
        tnsr_to_expand = tnsr_to_expand.unsqueeze(dim)

    repeats = [1] * tnsr_to_expand.dim()
    repeats[dim] = batch_size
    expanded = tnsr_to_expand.repeat(repeats)
    return expanded


def get_mlp(n_layers, embed_dim, nonlin_type=DEFAULT_NONLIN, 
            dropout=MLP_DEFAULT_DROPOUT, in_dim=None, out_dim=None, bias=True):
    """n_layers is the number of linear layers, so the number of 'hidden 
    layers' is n_layers - 1."""
    layers = []
    for li in range(n_layers):
        if li == 0 and in_dim is not None:
            layer_in_dim = in_dim
        else:
            layer_in_dim = embed_dim
        if li == n_layers - 1 and out_dim is not None:
            layer_out_dim = out_dim
        else:
            layer_out_dim = embed_dim
        layers.append(nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
        if li < n_layers - 1:
            layers.append(nn.Dropout(dropout))
            layers.append(nonlin_type())
    return nn.Sequential(*layers)


def format_stat_msg(tensor, name=''):
    tensor = tensor.to(dtype=torch.float32)
    msg = f"{name} mean: {tensor.mean().item():.2f} " \
          f"std: {tensor.std().item():.2f} " \
          f"min: {tensor.min().item():.2f} " \
          f"max: {tensor.max().item():.2f} " \
          f"size: {tensor.numel()}"
    return msg
