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

from itertools import cycle

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch_geometric.utils as pygu


def draw_coalesced_routes(node_locs, routes, ax, symmetric=True):
    """
    Coalesce all routes into a single set of edges and draw the resulting graph
    """
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    for route in routes:
        route = route.cpu()
        route_index = torch.stack((route[:-1], route[1:]), dim=0)
        edge_index = torch.cat((edge_index, route_index), dim=1)
    data = Data(pos=node_locs, edge_index=edge_index)
    data = data.coalesce()

    nx_graph = pygu.to_networkx(data, to_undirected=symmetric)
    nx.draw(nx_graph, pos=node_locs.cpu().numpy(), ax=ax)


def plot_routes_in_groups(node_locs, routes, group_size=10, symmetric=True):
    """
    Plot the routes in groups of group_size, with each group on a separate
    subplot.  Arrange the subplots in a grid with as close to equal rows and
    columns as possible.
    """
    num_groups = len(routes) // group_size
    if len(routes) % group_size > 0:
        num_groups += 1
    num_rows = int(np.sqrt(num_groups))
    num_cols = int(np.ceil(num_groups / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    axes = axes.flatten()
    base_graph = nx.Graph() if symmetric else nx.DiGraph()
    base_graph.add_nodes_from(range(len(node_locs)))

    node_locs = node_locs.cpu().numpy()
    
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    assert len(colours) <= group_size, \
        "not enough colours for this many routes"
    colour_cycle = cycle(colours)

    for ii in range(num_groups):
        ax = axes[ii]
        start = ii * group_size
        end = start + group_size
        nx.draw(base_graph, pos=node_locs, ax=ax, node_color='black')
        for route in routes[start:end]:
            route_graph = base_graph.copy()
            edges = torch.stack((route[:-1], route[1:]), dim=-1)
            edges = [(aa.item(), bb.item()) for aa, bb in edges]
            route_graph.add_edges_from(edges)
            # draw that route in a unique colour
            nx.draw_networkx_edges(route_graph, pos=node_locs, ax=ax, 
                                   edge_color=next(colour_cycle), width=3,
                                   node_size=0)


