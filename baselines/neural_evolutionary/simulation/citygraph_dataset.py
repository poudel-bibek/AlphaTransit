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

import argparse
import pickle
import shutil
import random
from pathlib import Path
from itertools import permutations
import logging as log

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx
import torch_geometric.utils as pygu
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.transforms import KNNGraph, RemoveIsolatedNodes, \
    BaseTransform, RandomRotate, RandomFlip, Compose

import torch_utils as tu

RAW_GRAPH_FILENAME = 'graph_list.pkl'

STOP_KEY = 'stop'
STREET_KEY = (STOP_KEY, 'street', STOP_KEY)
DEMAND_KEY = (STOP_KEY, 'demand', STOP_KEY)
# we don't use this in here, but we do elsewhere
ROUTE_KEY = (STOP_KEY, 'route', STOP_KEY)


DMD_FEAT_IDX = 0
SHORTESTPATH_FEAT_IDX = 1

OUT_DEMAND_FEAT_IDX = 4
IN_DEMAND_FEAT_IDX = 5

MST = 'mst'
OUT_KNN = 'outgoing_knn'
IN_KNN = 'incoming_knn'
VORONOI = 'voronoi'
GRID4 = '4_grid'
GRID8 = '8_grid'
CIRC = 'circulant'
SMALLWORLD = 'small_world'
MIXED = 'mixed'

MAX_DEMAND = 800.0
MIN_DEMAND = 60.0
MAX_POP_STD_M = 15_000
MIN_POP_STD_M = 2000
MAX_SINK_RADIUS_M = 7_500
MIN_SINK_RADIUS_M = 1_500
CENTERNODES_PER_NODE = 1 / 10

SPEED_MPS = 15.0
SIDE_LENGTH_M = 30_000

def get_default_train_and_eval_split(path, split=0.9, space_scale=0.01, 
                                     demand_scale=0.01):
    transforms = [
        SpaceScaleTransform(1-space_scale, 1+space_scale), 
        DemandScaleTransform(1-demand_scale, 1+demand_scale),
        RandomFlipCity(),
        RandomRotate(180)]
    train_ds = CityGraphDataset(path, transforms=transforms)
    train_size = int(len(train_ds) * split)
    train_ds = train_ds[:train_size]
    eval_ds = CityGraphDataset(path)[train_size:]
    return train_ds, eval_ds


def get_dataset_from_config(ds_cfg, center_nodes=True):
    if ds_cfg['type'] == 'pickle':
        return CityGraphDataset(ds_cfg.path, transforms=None)
    elif ds_cfg['type'] == 'mumford':
        do_scaling = ds_cfg.get('scale_dynamically', True)
        extra_node_feats = ds_cfg.get('extra_node_feats', True)

        fixed_routes_file = ds_cfg.get('fixed_routes', None)
        if fixed_routes_file is not None:
            fixed_routes_path = Path(ds_cfg.path) / fixed_routes_file
            # load the fixed routes from the file
            fixed_routes = tu.load_routes_tensor(fixed_routes_path)
        else:
            fixed_routes = None

        data = CityGraphData.from_mumford_data(ds_cfg.path, ds_cfg.city,
            scale_dynamically=do_scaling, extra_node_feats=extra_node_feats,
            fully_connected_demand=True, center_nodes=center_nodes, 
            fixed_routes=fixed_routes)
        return [data]
    else:
        raise ValueError(f"Unknown dataset type {ds_cfg['type']}")


def get_dynamic_training_set(min_nodes, max_nodes, space_scale, demand_scale, 
                             edge_keep_prob=0.7, data_type='mixed', 
                             directed=False, fully_connected_demand=True):
    transforms = [
        SpaceScaleTransform(1-space_scale, 1+space_scale), 
        DemandScaleTransform(1-demand_scale, 1+demand_scale),
    ]
    return DynamicCityGraphDataset(min_nodes, max_nodes, edge_keep_prob, 
        data_type, directed, fully_connected_demand, transforms)


class CityGraphDataset(InMemoryDataset):
    def __init__(self, root=None, transforms=None):
        # append the required transforms to whatever transforms were passed in
        if transforms is None:
            transforms = []
        
        # always insert pos features.
        transforms.append(InsertPosFeatures())
        transform = Compose(transforms)

        super().__init__(root, transform, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['collated.pt']

    def process(self):
        # read in the list from the pickle file
        with open(Path(self.root) / RAW_GRAPH_FILENAME, 'rb') as ff:
            data_list = pickle.load(ff)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DynamicCityGraphDataset(torch.utils.data.IterableDataset):
    def __init__(self, min_nodes, max_nodes, edge_keep_prob=0.7, 
                 data_type='mixed', directed=False, 
                 fully_connected_demand=True, side_length_m=SIDE_LENGTH_M, 
                 transforms=None, mumford_style=True, pos_only=False):
        super().__init__()
        self.data_type = data_type
        self.edge_keep_prob = edge_keep_prob
        self.directed = directed
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.fully_connected_demand = fully_connected_demand
        self.side_length_m = side_length_m
        self.mumford_style = mumford_style
        self.pos_only = pos_only

        # append the required transforms to whatever transforms were passed in
        if transforms is None:
            transforms = []

        # always insert pos features.
        transforms.append(InsertPosFeatures())
        self.transform = Compose(transforms)

        self.graph_fns = {
            MST: build_mst_graph,
            IN_KNN: build_in4nn_graph,
            OUT_KNN: build_out4nn_graph,
            CIRC: build_circulant_graph,
            VORONOI: build_voronoi_graph,
            GRID4: build_4grid,
            GRID8: build_8grid,
        }
        
        if data_type == SMALLWORLD:
            assert not directed, "smallworld graphs are undirected"

        if self.directed:
            self.nx_creator = nx.DiGraph
        else:
            self.nx_creator = nx.Graph

    def __iter__(self):
        return self._generator()
    
    def _generator(self):
        while True:
            graph = self.generate_graph()
            graph = self.transform(graph)
            yield graph

    def generate_graph(self, draw=False, n_nodes=None, max_n_nodes=None):
        if not max_n_nodes:
            max_n_nodes = self.max_nodes
        if not n_nodes:
            n_nodes = random.randint(self.min_nodes, max_n_nodes)
        assert n_nodes <= max_n_nodes

        # build a street graph with nodes in (-1, 1)
        street_graph = self._generate_street_graph(n_nodes)

        # scale the node positions appropriately
        street_graph.pos = street_graph.pos * self.side_length_m / 2

        graph = CityGraphData()
        fixed_routes = torch.zeros((0, max_n_nodes))
        graph.fixed_routes = fixed_routes
        graph[STOP_KEY].pos = street_graph.pos
        graph[STREET_KEY].edge_index = street_graph.edge_index
        
        # assign distances to edges
        n_nodes = street_graph.num_nodes
        euc_dists = get_euclidean_distances(street_graph.pos)
        # edge weights are drive times in seconds
        street_idx = graph[STREET_KEY].edge_index
        edge_times = euc_dists[street_idx[0], street_idx[1]] / SPEED_MPS
        graph[STREET_KEY].edge_attr = edge_times[:, None]

        street_edge_mat = torch.full((max_n_nodes, max_n_nodes), 
                                     float('inf'))
        street_edge_mat[street_idx[0], street_idx[1]] = edge_times
        street_edge_mat.fill_diagonal_(0)
        graph.street_adj = street_edge_mat

        # compute all shortest paths
        nexts, times = tu.floyd_warshall(street_edge_mat)
        graph.drive_times = times.squeeze(0)
        graph.nexts = nexts.squeeze(0)

        # generate demand
        done = False
        while not done:
            if self.mumford_style:
                demand = uniform_demand_on_edges(street_graph.pos, 1.0)
            else:
                demand = inter_loci_demand(street_graph.pos, draw=draw)
                            
            if not self.directed:
                # make the demand matrix symmetric
                demand = demand.triu()
                demand += demand.T.clone()

            scale = demand.max()
            if scale > 0:
                # scale the demands to be 0 to max demand
                demand /= scale
                
                no_internode_demand = demand == 0
                demand_range = MAX_DEMAND - MIN_DEMAND
                demand = demand * demand_range + MIN_DEMAND
                demand[no_internode_demand] = 0
    
            done = demand.sum() > 0
                        
        if draw:
            plt.hist(demand.numpy().flatten(), bins=20)
            plt.title("Inter-node demand counts")
            plt.yscale('log')
            plt.show()

        # pad the demand matrix to the max size
        n_pad = max_n_nodes - n_nodes
        demand = torch.nn.functional.pad(demand, (0, n_pad, 0, n_pad))
        graph.demand = demand
        if self.fully_connected_demand:
            dmd_idx = torch.tensor(list(permutations(range(n_nodes), 2))).T
        else:
            dmd_idx = torch.stack(torch.where(demand))
        graph[DEMAND_KEY].edge_index = dmd_idx

        # set the edge features
        dmd_feat = demand[dmd_idx[0], dmd_idx[1]]
        drive_time_attr = times[0, dmd_idx[0], dmd_idx[1]]
        dmd_edge_attr = torch.zeros((dmd_feat.shape[0], 2))
        dmd_edge_attr[:, DMD_FEAT_IDX] = dmd_feat
        dmd_edge_attr[:, SHORTESTPATH_FEAT_IDX] = drive_time_attr
        graph[DEMAND_KEY].edge_attr = dmd_edge_attr

        # set node features
        if self.pos_only:
            # pos features will be inserted by the InsertPosFeatures transform.
            graph[STOP_KEY].x = torch.zeros((n_nodes, 0))
        else:
            graph[STOP_KEY].x = get_node_features(street_idx, demand)

        if draw:
            # draw the graph
            ax = plt.axes()
            graph.draw(ax, node_size=100, show_demand=False)
            plt.tight_layout()
            plt.show()

        assert graph is not None
        return graph

    def _generate_street_graph(self, n_nodes):
        if self.data_type == SMALLWORLD:
            return build_smallworld_graph(2, 7)
        else:
            if self.data_type == MIXED:
                choices = [MST, IN_KNN, VORONOI, GRID4, GRID8]
                if self.directed:
                    # if the graphs are directed, out_knn differs from in_knn
                    choices.append(OUT_KNN)

                gen_key = random.choice(choices)
                gen_fn = self.graph_fns[gen_key]
            else:
                gen_fn = self.graph_fns[self.data_type]
            return gen_fn(n_nodes, edge_keep_prob=self.edge_keep_prob, 
                          directed=self.directed)
        

class InsertPosFeatures(BaseTransform):
    def __call__(self, data):
        data = data.clone()
        for key in [STOP_KEY]:
            val = data[key]
            if hasattr(val, 'x') and val.x is not None:
                val.x = torch.cat((val.pos, val.x), dim=1)
            else:
                val.x = val.pos.clone()
        return data


class RandomFlipCity(RandomFlip):
    def __init__(self, axis=0) -> None:
        super().__init__(axis)

    def __call__(self, data: Data) -> Data:
        super().__call__(data[STOP_KEY])
        return data


class SpaceScaleTransform(BaseTransform):
    def __init__(self, min_scale=0.75, max_scale=1.25):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data):
        data = data.clone()
        scale_range = self.max_scale - self.min_scale
        scale = torch.rand(1) * scale_range + self.min_scale
        data[STOP_KEY].pos *= scale
        data[STREET_KEY].edge_attr *= scale
        data.drive_times *= scale
        return data


class DemandScaleTransform(BaseTransform):
    def __init__(self, min_scale=0.5, max_scale=1.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data):
        data = data.clone()
        scale_range = self.max_scale - self.min_scale
        scale = torch.rand(1) * scale_range + self.min_scale
        data[DEMAND_KEY].edge_attr[:, DMD_FEAT_IDX] *= scale
        data.demand *= scale
        if data[STOP_KEY].x.shape[1] > 2:
            data[STOP_KEY].x[:, OUT_DEMAND_FEAT_IDX] *= scale
            data[STOP_KEY].x[:, IN_DEMAND_FEAT_IDX] *= scale
        return data


class CityGraphData(HeteroData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['demand', 'drive_times', 'street_adj', 'nexts']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
                
    def draw(self, ax=None, node_size=10, show_demand=True):
        street_graph = Data(edge_index=self[STREET_KEY].edge_index,
                            edge_attr=self[STREET_KEY].edge_attr,
                            pos=self[STOP_KEY].pos)
        transpose_adj = self.street_adj.transpose(-2, -1)
        undirected_streets = (self.street_adj == transpose_adj).all()
        nx_graph = pygu.to_networkx(street_graph, 
                                    to_undirected=undirected_streets)
        locs = self.pos.cpu().numpy()
        nx.draw(nx_graph, pos=locs, node_size=node_size, width=1, arrowsize=20,
                ax=ax)

        if show_demand:
            transposed_demand = self.demand.transpose(-2, -1)
            dmd_is_directed = not (self.demand == transposed_demand).all()
            creator = nx.DiGraph if dmd_is_directed else nx.Graph
            nx_dmd_graph = nx.from_numpy_array(self.demand.cpu().numpy(), 
                                            create_using=creator)
            de_widths = torch.tensor([dd['weight'] for _, _, dd in 
                                    nx_dmd_graph.edges(data=True)])
            de_widths *= 2 / de_widths.max()
            nx.draw_networkx_edges(nx_dmd_graph, edge_color="red", 
                                pos=locs, width=de_widths,
                                style='dashed', ax=ax)
            
    @staticmethod
    def from_tensors(node_locs, street_adj, demand, pos_only=False):
        graph = CityGraphData()
        n_nodes = node_locs.shape[0]
        graph.fixed_routes = torch.zeros((0, n_nodes))
        graph[STOP_KEY].pos = node_locs
        is_edge = (street_adj > 0) & street_adj.isfinite()
        edge_idx = is_edge.nonzero().t()
        graph[STREET_KEY].edge_index = edge_idx
        graph[STREET_KEY].edge_attr = street_adj[is_edge]
        graph.street_adj = street_adj

        # compute all shortest paths
        nexts, times = tu.floyd_warshall(street_adj)
        graph.drive_times = times.squeeze(0)
        graph.nexts = nexts.squeeze(0)

        # compute demand features
        graph.demand = demand
        demand_nonzeros = demand.nonzero()
        graph[DEMAND_KEY].edge_index = demand_nonzeros.t()
        dmd_feats = demand[demand_nonzeros[:, 0], demand_nonzeros[:, 1]]
        drive_times = times[0, demand_nonzeros[:, 0], demand_nonzeros[:, 1]]
        graph[DEMAND_KEY].edge_attr = torch.stack((dmd_feats, drive_times))

        if pos_only:
            graph[STOP_KEY].x = torch.zeros((n_nodes, 0))
        else:
            graph[STOP_KEY].x = get_node_features(edge_idx, demand)
            
        return graph
        
    @staticmethod
    def from_mumford_data(instances_dir, instance_name='', 
                          assumed_speed_mps=SPEED_MPS,
                          scale_dynamically=True, extra_node_feats=True,
                          fully_connected_demand=True, center_nodes=True, 
                          fixed_routes=None):
        """
        instances_dir: the directory where the Mumford data is stored.
        instance_name: for the instances given in the Mumford dataset, this
            should be Mandl, Mumford0, Mumford1, Mumford2, or Mumford3.
        assumed_speed_mps: used to estimate the true size of the city.
        """
        data_dir = Path(instances_dir)
        # load nodes from coords file
        coords_path = data_dir / (instance_name + 'Coords.txt')
        # first row gives just the number of nodes
        node_locs = torch.tensor(np.genfromtxt(coords_path, skip_header=1),
                                 dtype=torch.float32)
        # center locs at 0, which is what the neural network expects
        if center_nodes:
            node_locs = node_locs - node_locs.mean(dim=0)

        # load street edges from travel times file
        tt_path = data_dir / (instance_name + 'TravelTimes.txt')
        # times are in minutes, so convert to seconds
        street_edge_times_s = torch.tensor(np.genfromtxt(tt_path), 
                                           dtype=torch.float32) * 60

        # can we determine scale from drive times?
        euc_dists = get_euclidean_distances(node_locs)
        edge_drive_dists_m = street_edge_times_s * assumed_speed_mps
        has_edge = (edge_drive_dists_m > 0) & edge_drive_dists_m.isfinite()
        # some of the datasets have some nodes with the same positions.  Ignore
         # those to avoid infinities.
        has_valid_edge = has_edge & (euc_dists > 0)
        edge_ratios = edge_drive_dists_m[has_valid_edge] / \
                      euc_dists[has_valid_edge]
        meters_per_unit = edge_ratios.mean()
        assert meters_per_unit < float('inf')
        if scale_dynamically:
            log.info(f"Estimated meters per unit: {meters_per_unit}")
            side_len = max((node_locs.max(dim=0)[0] - node_locs.min(dim=0)[0]))
            side_len *= meters_per_unit
            log.info(f"Environment side length: {side_len} meters")
            node_locs *= meters_per_unit

        # convert this to our graph representation
        data = CityGraphData()
        if fixed_routes is None:
            fixed_routes = torch.zeros((0, node_locs.shape[0]))

        data.fixed_routes = fixed_routes
        # load demands from demands file
        dmd_path = data_dir / (instance_name + 'Demand.txt')
        od = torch.tensor(np.genfromtxt(dmd_path), dtype=torch.float32)
        # compute node features
        street_idx = torch.stack(torch.where(has_edge))
        if extra_node_feats:
            node_features = get_node_features(street_idx, od)
            node_features = torch.cat((node_locs, node_features), dim=1)
        else:
            node_features = node_locs
        data[STOP_KEY].x = node_features
        data[STOP_KEY].pos = node_locs
        # set street edges
        street_attr = street_edge_times_s[has_edge]
        data[STREET_KEY].edge_index = street_idx
        data[STREET_KEY].edge_attr = street_attr
        data.street_adj = street_edge_times_s
        
        # compute all shortest paths
        nexts, drive_times = tu.floyd_warshall(street_edge_times_s)
        drive_times = drive_times.squeeze(0)
        data.drive_times = drive_times
        data.nexts = nexts.squeeze(0)

        # compute demand features
        has_demand = od > 0
 
        if fully_connected_demand:
            n_nodes = od.shape[0]
            dmd_idx = torch.tensor(list(permutations(range(n_nodes), 2))).T
        else:
            dmd_idx = torch.stack(torch.where(has_demand))

        data[DEMAND_KEY].edge_index = dmd_idx
        demand_feat = od[dmd_idx[0], dmd_idx[1]]
        drive_time_feat = drive_times[dmd_idx[0], dmd_idx[1]]
        dmd_edge_feat = torch.stack((demand_feat, drive_time_feat), dim=1)
        data[DEMAND_KEY].edge_attr = dmd_edge_feat
        data.demand = od

        assert data.fixed_routes is not None
        return data

    @property
    def drive_dists(self):
        return self.drive_times * SPEED_MPS

    @property
    def pos(self):
        return self[STOP_KEY].pos
    

def get_node_features(street_index, demand_matrix):
    n_nodes = demand_matrix.shape[0]
    street_edge_exists = torch.zeros((n_nodes, n_nodes))
    street_edge_exists[street_index[0], street_index[1]] = 1
    out_street_deg = street_edge_exists.sum(dim=0)
    in_street_deg = street_edge_exists.sum(dim=1)
    feats = [out_street_deg, in_street_deg]
    return torch.stack(feats, dim=1)


def get_euclidean_distances(locs):
    n_nodes = locs.shape[0]
    euc_dists = torch.zeros((n_nodes, n_nodes), dtype=locs.dtype)
    upper_tri_idxs = torch.triu_indices(n_nodes, n_nodes, offset=1)
    # populate the above-diagonal entries
    euc_dists[upper_tri_idxs[0], upper_tri_idxs[1]] = torch.pdist(locs)
    # and the below-diagonal entries
    euc_dists = euc_dists + euc_dists.T

    return euc_dists


def uniform_demand_on_edges(node_locs, demand_edge_keep_prob):
    # build a demand matrix with uniformly random demands for each node pair
    n_nodes = node_locs.shape[0]
    while True:
        demand = torch.rand((n_nodes, n_nodes))
        demand[n_nodes:] = 0
        demand[:, n_nodes:] = 0
        # zero out approximately demand_frac fraction of demands
        zero_scores = torch.rand(n_nodes, n_nodes)
        demand[zero_scores > demand_edge_keep_prob] = 0
        demand.fill_diagonal_(0)
        if (demand > 0).sum() > demand.numel() * demand_edge_keep_prob / 2:
            break

    return demand


def locus_based_demand(node_locs, max_loci_frac):
    # build a demand matrix with n_loci loci of demand
    loci_frac = max_loci_frac * torch.rand(1)
    n_nodes = node_locs.shape[0]
    n_loci = (n_nodes * loci_frac).ceil().int().item()
    demand = torch.zeros((n_nodes, n_nodes))
    # randomly choose n_loci loci
    loci = torch.randperm(n_nodes)[:n_loci]
    # demand from each node to each of the loci
    demand[:, loci] = torch.rand((n_nodes, n_loci))
    # demand from each of the loci to each node
    demand[loci, :] = torch.rand((n_loci, n_nodes))
    demand.fill_diagonal_(0)
    return demand


def plot_ellipse(center, radii, colour=None):
    tt = np.linspace(0, 2 * np.pi, 100)
    xs = center[0] + radii[0] * np.cos(tt)
    ys = center[1] + radii[1] * np.sin(tt)
    plt.plot(xs, ys, color=colour)


def make_oval_groups(node_locs, n_groups, min_radius, max_radius):
    n_nodes = node_locs.shape[0]
    is_in_group = torch.zeros((n_nodes, n_groups), dtype=torch.bool)
    cluster_range = max_radius - min_radius

    # for each pop cluster
    centers = []
    shapes = []
    for gi in range(n_groups):
        # pick a node to be its center that is not a neighbour of any other 
         # cluster center and is not part of any other cluster
        is_in_no_group = ~is_in_group.any(dim=1)
        if is_in_no_group.sum() == 0:
            n_groups = gi + 1
            is_in_group = is_in_group[:, :n_groups]
            break
        node = is_in_no_group.float().multinomial(1)
        centers.append(node_locs[node][0])
        # randomly generate an oval around the center node
        radii = torch.rand(2) * cluster_range + min_radius
        shapes.append(radii)
        oval_vals = ((node_locs - node_locs[node])**2 / radii**2).sum(dim=1)
        # all nodes in the oval are now part of the cluster
        is_in_group[:, gi] = oval_vals <= 1
    return is_in_group, centers, shapes


def inter_loci_demand(node_locs, draw=False):
    # pick number of population clusters
    n_nodes = node_locs.shape[0]
    max_centers = max(int(CENTERNODES_PER_NODE * n_nodes), 1)
    n_pop_centers = torch.randint(max_centers, (1,)) + 1
    is_in_cluster, cluster_centers, cluster_shapes = \
        make_oval_groups(node_locs, n_pop_centers, MIN_POP_STD_M, MAX_POP_STD_M)
    min_density = 0.3
    density_range = 1 - min_density

    # range of populations on nodes in a cluster, relative to the density
    pop_var = 0.3
    node_pops_by_type = []
    for is_in_this_cluster in is_in_cluster.T:
        # randomly select a "density" for the cluster
        density = density_range * torch.rand(1) + min_density
        # all nodes in the cluster get population randomly sampled near the 
         # density
        rel_node_pops = (1 - pop_var) + pop_var * torch.rand(n_nodes)
        type_node_pops = rel_node_pops * density * is_in_this_cluster
        node_pops_by_type.append(type_node_pops)

    # n_nodes x n_pop_centers
    node_pops_by_type = torch.stack(node_pops_by_type, dim=1)

    # same procedure for sinks
    n_sinks = (torch.randint(max_centers, (1,)) + 1).item()
    is_in_sink, sink_centers, sink_shapes = \
        make_oval_groups(node_locs, n_sinks, MIN_SINK_RADIUS_M, 
                         MAX_SINK_RADIUS_M)

    if draw:
        # draw ovals for clusters and sinks
        for center, radii in zip(cluster_centers, cluster_shapes):
            plot_ellipse(center, radii, colour='green')
        for center, radii in zip(sink_centers, sink_shapes):
            plot_ellipse(center, radii, colour='orange')
        # draw the population graph
        node_pops = node_pops_by_type.sum(dim=1)
        plt.scatter(node_locs[:, 0], node_locs[:, 1], c=node_pops, 
                    cmap='viridis', label='Nodes, coloured by pop.')
        plt.colorbar()
        plt.legend()
        plt.show()

    # combine the two as before
    n_pop_clusters = is_in_cluster.shape[1]
    n_sinks = is_in_sink.shape[1]
    attractiveness = torch.rand((n_pop_clusters, n_sinks))

    demand_to_sinks = node_pops_by_type.mm(attractiveness)
    # spread the demand around the nodes in the sink
    node_sink_fracs = is_in_sink / is_in_sink.sum(dim=0)
    demand = demand_to_sinks.mm(node_sink_fracs.T)

    # symmetrize, normalize and return
    demand.fill_diagonal_(0)

    return demand
    

def build_in4nn_graph(n_nodes, edge_keep_prob, directed):
    return build_knn_graph(n_nodes, 4, edge_keep_prob, 'target_to_source', 
                           directed)


def build_out4nn_graph(n_nodes, edge_keep_prob, directed):
    return build_knn_graph(n_nodes, 4, edge_keep_prob, 'source_to_target', 
                           directed)


def build_knn_graph(n_nodes, knn, edge_keep_prob=1, flow='target_to_source',
                    directed=True):
    knn_graph = KNNGraph(k=knn, flow=flow, force_undirected=not directed)
    rmv_isolated_nodes = RemoveIsolatedNodes()

    while True:
        # generate the node locations
        locs = torch.rand((n_nodes, 2)) * 2 - 1
        # determine the edges
        street_graph = Data(pos=locs)
        pre_rmv = knn_graph(street_graph)
        street_graph = rmv_isolated_nodes(pre_rmv)

        # drop random edges
        drop_edges(street_graph, edge_keep_prob, directed)
        
        # check whether the graph is connected, fix it if not
        nx_graph = pygu.to_networkx(street_graph, to_undirected=not directed)
        if is_strongly_connected(nx_graph):
            return street_graph
        

def build_circulant_graph(n_nodes, offsets=[1,2], edge_keep_prob=1, 
                          directed=True):
    graph = nx.circulant_graph(n_nodes, offsets)
    graph = pygu.from_networkx(graph)

    # spread nodes evenly around a circle of radius 1
    graph.pos = get_locs_around_unit_circle(n_nodes)

    while True:
        # drop random edges
        graph_copy = graph.clone()
        drop_edges(graph_copy, edge_keep_prob, directed)
        
        # check whether the graph is connected, fix it if not
        nx_graph = pygu.to_networkx(graph_copy, to_undirected=not directed)
        if is_strongly_connected(nx_graph):
            return graph_copy


def get_locs_around_unit_circle(n_nodes):
    # spread nodes evenly around a circle of radius 1
    angle_step = 2 * torch.pi / n_nodes
    angles = torch.arange(n_nodes) * angle_step
    xlocs = torch.cos(angles)
    ylocs = torch.sin(angles)
    locs = torch.stack((xlocs, ylocs), dim=1)
    return locs


def build_voronoi_graph(n_nodes, draw_voronoi=False, *args, **kwargs):
    n_points = int(n_nodes)
    # initial step size for adjusting the number of nodes via binary search.
     # this value seems to work well.
    step = max(n_points // 4, 1)
    while True:
        points = torch.rand((n_points, 2)) * 2 - 1
        vor = Voronoi(points)
        vertices = torch.tensor(vor.vertices, dtype=torch.float32)
        # remove vertices that are outside the (-1, 1) square
        in_square_idxs = torch.where((vertices.abs() < 1).all(dim=1))[0]
        if len(in_square_idxs) != n_nodes:
            # these points don't induce the right number of vertices, so 
             # generate new ones with more or less points as appropriate
            if len(in_square_idxs) < n_nodes:
                n_points += step
            else:
                n_points -= step
            step = max(step // 2, 1)
            continue

        # prune excess vertices
        kept_idxs = in_square_idxs  # [:n_nodes]
        node_locs = vertices[kept_idxs]
        # remove edges to pruned vertices and those which extend to infinity
        edges = [ee for ee in vor.ridge_vertices if ee[0] in kept_idxs and
                                                    ee[1] in kept_idxs]
        edges = torch.tensor(edges, dtype=int).t()
        # map edge's vertex indices to the indices of the kept vertices
        for ii in range(n_nodes):
            old_idx = kept_idxs[ii]
            edges[edges == old_idx] = ii

        # Voronoi graphs are undirected, so make all edges in both directions
        edges = torch.cat((edges, edges.flip(0)), dim=1)
        street_graph = Data(pos=node_locs, edge_index=edges)
        nx_graph = pygu.to_networkx(street_graph, to_undirected=True)
        if is_strongly_connected(nx_graph):
            # we've got a valid one, so we're done!
            break

    if draw_voronoi:
        voronoi_plot_2d(vor)
        plt.show()

    return street_graph


def build_mst_graph(n_nodes, directed, *args, **kwargs):
    assert not directed, "MSTs are undirected graphs!"
    locs = torch.rand((n_nodes, 2)) * 2 - 1
    dists = torch.pdist(locs)
    edges = torch.combinations(torch.arange(n_nodes), r=2).t()
    
    street_graph = Data(pos=locs, edge_index=edges, edge_attr=dists)
    nx_graph = pygu.to_networkx(street_graph, edge_attrs=['edge_attr'], 
                                to_undirected=True)

    # run minimum spanning tree on the fully-connected graph
    mst = nx.minimum_spanning_tree(nx_graph, weight='edge_attr', 
                                   algorithm='kruskal')
    # add minimum edges until desired number of edges is reached
    n_edges = int(3.616 * n_nodes - 34.204)
    # sort edges by distance
    all_edges = sorted(nx_graph.edges(data=True), 
                       key=lambda x: x[2]['edge_attr'])
    while len(mst.edges) < n_edges:
        # add the next shortest edge
        ii, jj, attrs = all_edges.pop(0)
        mst.add_edge(ii, jj, **attrs)

    data = pygu.from_networkx(mst)
    data.pos = locs
    return data


def build_4grid(n_nodes, edge_keep_prob, directed):
    return build_grid(n_nodes, '4-connected', edge_keep_prob, directed)


def build_8grid(n_nodes, edge_keep_prob, directed):
    return build_grid(n_nodes, '8-connected', edge_keep_prob, directed)


def build_grid(n_nodes, grid_type='4-connected', edge_keep_prob=1, 
               directed=True):
    """Careful, this is O(n_nodes), though usually much faster than that."""
    assert n_nodes > 3, "n_nodes must be at least 4 in a grid!"

    # fewest # of rows and columns should be more than 2
    min_factor = 3
    factors = [ii for ii in range(min_factor, n_nodes // min_factor + 1) 
               if n_nodes % ii == 0]
    assert len(factors) > 0, "n_nodes must not be prime!"

    x_nodes = factors[random.randrange(len(factors))]
    y_nodes = n_nodes // x_nodes
    
    side_n_nodes = torch.tensor((x_nodes, y_nodes))
    ranges = side_n_nodes / n_nodes**0.5 + (torch.rand(2) - 0.5) * 0.2
    x_range, y_range = ranges
    # determine the node locations
    x_locs = torch.linspace(-x_range, x_range, x_nodes)
    y_locs = torch.linspace(-y_range, y_range, y_nodes)
    locs = torch.stack(torch.meshgrid(x_locs, y_locs, indexing='ij'), dim=2)
    locs = locs.view(-1, 2)
    
    # set the edges
    edge_index = grid_index(x_nodes, y_nodes, grid_type)
    graph = Data(pos=locs, edge_index=edge_index)

    while True:
        # drop random edges
        graph_copy = graph.clone()
        drop_edges(graph_copy, edge_keep_prob, directed)
        
        # check whether the graph is connected, fix it if not
        nx_graph = pygu.to_networkx(graph_copy, to_undirected=not directed)
        if is_strongly_connected(nx_graph):
            return graph_copy


def grid_index(
    height: int,
    width: int,
    grid_type='4-connected',
    device=None,
):
    """Modified from the function of the same name in 
       torch_geometric.utils.grid"""
    ww = width
    if grid_type == '8-connected':
        kernel = [-ww - 1, -1, ww - 1, -ww, ww, -ww + 1, 1, ww + 1]
        n_to_cut = 3
    elif grid_type == '4-connected':
        kernel = [-1, -ww, ww, 1]
        n_to_cut = 1
    kernel = torch.tensor(kernel, device=device)

    row = torch.arange(height * width, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = torch.arange(n_to_cut, row.size(1) - n_to_cut, dtype=torch.long, 
                         device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index = pygu.coalesce(edge_index, None, height * width)
    return edge_index


def build_smallworld_graph(n_levels, n_nodes_per_level=4, p_attach=0.5):
    """Implements the stochastic hierarchical model described in Ravasz and 
        Laslzo 2008."""
    # TODO implement this!
    base_edges = torch.combinations(torch.arange(n_nodes_per_level)).t()
    center_loc = torch.zeros((1, 2))
    periph_locs = get_locs_around_unit_circle(n_nodes_per_level - 1)
    base_locs = torch.cat((center_loc, periph_locs), dim=0)
    base_graph = Data(pos=base_locs, edge_index=base_edges)
    base_graph.edge_index = pygu.to_undirected(base_graph.edge_index)

    base_offsets = get_locs_around_unit_circle(n_nodes_per_level - 1)
    for ii in range(n_levels - 1):
        # make copies of the base graph
        periph_offsets = base_offsets * (ii + 1.25) * 2
        n_new_cluster_nodes = base_graph.num_nodes
        base_degrees = pygu.degree(base_graph.edge_index[0])
        attach_probs = base_degrees / base_degrees.sum()
        new_graph = base_graph.clone()
        i_p_attach = p_attach ** (ii + 1)
        n_to_attach = int(i_p_attach * n_new_cluster_nodes)

        for offset in periph_offsets:
            dup_edges = base_graph.edge_index + new_graph.num_nodes
            dup_locs = base_graph.pos + offset

            # select p_attach fraction new nodes to attach
            new_targets = torch.randperm(n_new_cluster_nodes)[:n_to_attach]
            new_targets += new_graph.num_nodes
            new_sources = attach_probs.multinomial(n_to_attach,
                                                   replacement=True)
            joining_edges = torch.stack((new_sources, new_targets), dim=0)
            # add new nodes and edges to the graph
            new_graph.edge_index = torch.cat((new_graph.edge_index,
                                              dup_edges, joining_edges), dim=1)
            new_graph.pos = torch.cat((new_graph.pos, dup_locs), dim=0)

        base_graph = new_graph

        # make graph undirected
        base_graph.edge_index = pygu.to_undirected(base_graph.edge_index)
    
    # scale node positions to be between -1 and 1.  We can do it this way 
     # because the nodes are centered at 0 by construction.
    maxes = base_graph.pos.max(dim=0)[0]
    base_graph.pos = base_graph.pos / maxes

    # degrees = pygu.degree(base_graph.edge_index[0])
    # plt.hist(degrees.numpy(), bins=20)
    # plt.yscale('log')
    # plt.show()

    return base_graph


def is_strongly_connected(nx_graph):
    if isinstance(nx_graph, nx.DiGraph):
        return nx.is_strongly_connected(nx_graph)
    else:
        return nx.is_connected(nx_graph)


def drop_edges(graph_data, edge_keep_prob, directed=False):
    """Randomly drop edges from the graph with probability p."""
    nx_graph = pygu.to_networkx(graph_data, to_undirected=not directed)
    removal_mask = torch.rand(nx_graph.number_of_edges()) > edge_keep_prob
    to_remove = [edge for edge, remove in zip(nx_graph.edges(), removal_mask)
                 if remove]
    nx_graph.remove_edges_from(to_remove)
    updated_graph_data = pygu.from_networkx(nx_graph)
    graph_data.edge_index = updated_graph_data.edge_index
    return graph_data


def drop_nodes(graph_data, n_nodes_to_drop):
    """Randomly drop nodes from the graph."""
    assert n_nodes_to_drop < graph_data.num_nodes, "Can't drop all the nodes!"
    # randomly select nodes to drop
    nx_graph = pygu.to_networkx(graph_data)
    drop_idxs = torch.randperm(graph_data.num_nodes)[:n_nodes_to_drop]

    # remove the nodes from the graph
    for drop_idx in drop_idxs:
        nx_graph.remove_node(drop_idx.item())

    graph_data = pygu.from_networkx(nx_graph)

    return graph_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path at which to save the data")
    parser.add_argument("graph_type", choices=[MST, OUT_KNN, IN_KNN, VORONOI, 
                                               GRID4, GRID8, CIRC, SMALLWORLD, 
                                               MIXED],
                        help="type of graph to generate")
    parser.add_argument("--edgekeepprob", type=float, default=0.7)
    parser.add_argument("-n", type=int, default=2**15,
                        help="number of graphs to generate")
    parser.add_argument("--min", type=int, default=10,
                        help="minimum graph size in nodes")
    parser.add_argument("--max", type=int, default=100,
                        help="maximum graph size in nodes")
    parser.add_argument("--draw", action="store_true", 
                        help="render each generated graph")
    parser.add_argument("--delete", action="store_true", 
        help="if provided, and a dataset already exists at the given path, " \
             "delete it before creating a new one.")
    parser.add_argument("--sparse_demand", action="store_true",
        help="make the demand graph not fully connected, so edges with 0 " \
             "demand don't exist.")
    parser.add_argument("--sidelen", type=float, default=SIDE_LENGTH_M, 
                        help="in meters")
    parser.add_argument("-d", "--directed", action="store_true", 
        help="make the street and demand graphs directed (undirected by "\
             "default).")
    parser.add_argument("--ovaldemand", action="store_true", 
        help="If provided, use ovoid demand regions.  Otherwise just use "\
             "uniform demand as in the Mumford dataset.")
    parser.add_argument("--pos_only", action="store_true", 
        help="If provided, give nodes only their positions as features.")
    args = parser.parse_args()

    dataset = DynamicCityGraphDataset(args.min, args.max, args.edgekeepprob,
        args.graph_type, args.directed, not args.sparse_demand, 
        side_length_m=args.sidelen, mumford_style=not args.ovaldemand,
        pos_only=args.pos_only)
    
    graphs = [dataset.generate_graph(args.draw) for _ in tqdm(range(args.n))]

    # save the dataset as a pickled list
    path = Path(args.path)
    if args.delete and path.exists():
        shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        path.mkdir(parents=True)
    with open(path / RAW_GRAPH_FILENAME, 'wb') as ff:
        pickle.dump(graphs, ff)


if __name__ == "__main__":
    main()
