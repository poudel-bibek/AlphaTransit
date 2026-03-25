import logging as log

import torch
from tqdm import tqdm
import networkx as nx

from torch_utils import get_batch_tensor_from_routes, load_routes_tensor, \
    aggr_edges_over_sequences, reconstruct_all_paths
from simulation.transit_time_estimator import RouteGenBatchState
import learning.utils as lrnu

def init_from_cfg(state, init_cfg):
    """Initializes the network according to the configuration.
    
    Args:
        state: The initial state of the system, not including the routes.
        init_cfg: The configuration for the initialization method.
    
    Returns:
        The initial network.
    """
    if init_cfg is None:
        # return torch.zeros((0, 0, 0), device=state.device, dtype=torch.long)
        return None
    elif init_cfg.method == 'load':
        return load_routes_tensor(init_cfg.path, state.device)
    elif init_cfg.method == 'john':
        alpha = init_cfg.get('alpha', state.alpha)
        return john_init(state, alpha, init_cfg.prioritize_direct_connections)
    elif init_cfg.method == 'nikolic':
        return nikolic_init(state)
    else:
        raise ValueError(f'Unknown initialization method: {init_cfg.method}')
    # TODO bring in hyperheuristic initialization scheme


def john_init(state: RouteGenBatchState, alpha=None, 
              prioritize_direct_connections: bool=True,
              show_pbar=False):
    """Constructs a network based on the algorithm of John et al. (2014).
    
    Args:
        state: A representation of the city and the problem constraints.
        alpha: Controls trade-off between transit time and demand. alpha=1
            means only demand matters, alpha=0 means only transit time matters.
    """
    assert state.symmetric_routes, \
        'John et al. (2014) requires symmetric routes'
    assert state.batch_size == 1, \
        'John et al. (2014) does not support batched states'

    # compute weights for all of the edges
    have_edges = (state.street_adj > 0) & state.street_adj.isfinite()
    norm_times = torch.full_like(state.street_adj, float('inf'))
    edge_times = state.street_adj[have_edges]
    norm_times[have_edges] = edge_times / edge_times.max()
    norm_demand = state.demand / state.demand.max()
    if alpha is None:
        # use John's spread of alpha and beta values
        values = torch.linspace(0, 1, 21)
        meshes = torch.meshgrid(values, values, indexing='ij')
        alpha, beta = (mm.flatten()[:, None, None] for mm in meshes)
    else:
        # just use the one passed-in value
        beta = 1 - alpha
    # build the range of edge weight sets based on alpha and beta
    all_edge_costs = beta * norm_times + alpha * (1 - norm_demand)
    all_edge_costs[~have_edges] = float('inf')

    # build a spanning sub-graph of routes
    have_edges = have_edges.squeeze(0)
    networks = []
    for edge_costs in tqdm(all_edge_costs, disable=not show_pbar):
        network = []
        n_nodes = state.n_nodes[0].item()
        are_visited = torch.zeros(n_nodes, dtype=torch.bool, 
                                  device=state.device)
        are_directly_connected = torch.eye(n_nodes, dtype=torch.bool, 
                                           device=state.device)
        pair_attempted = torch.eye(n_nodes, dtype=torch.bool, 
                                   device=state.device)
        # While all nodes are not yet visited:
        while not are_visited.all():
            # select a "seed pair" of vertices to be the start of the route
            scores = edge_costs.clone()
            if len(network) > 0:
                # choose an edge between a covered and an uncovered node
                connects_new = are_visited[:, None] & ~are_visited
                # don't pick pairs that don't connect a new node
                scores[~connects_new] = float('inf')
            
            # don't choose any pairs that have already been attempted
            scores[pair_attempted] = float('inf')

            flat_seed_pair = torch.argmin(scores).item()
            seed_pair = (flat_seed_pair // n_nodes,
                         flat_seed_pair % n_nodes)
            route = list(seed_pair)
            are_on_route = torch.zeros(n_nodes, dtype=torch.bool,
                                       device=state.device)
            are_on_route[[seed_pair]] = True

            # expand the seed pair to form a route by adding adjacent vertices
            while len(route) < state.max_route_len:
                # find valid extension nodes
                are_valid_start_exts = have_edges[route[0]] & ~are_on_route
                are_valid_end_exts = have_edges[route[-1]] & ~are_on_route
                
                # check if any of them are not yet visited at all
                are_valid_exts = are_valid_start_exts | are_valid_end_exts
                if not are_valid_exts.any():
                    # no valid extensions, so we're done
                    break
                
                if (are_valid_exts & ~are_visited).any():
                    # we can visit an unvisited node, so we must do so
                    are_valid_start_exts &= ~are_visited
                    are_valid_end_exts &= ~are_visited
                
                # choose the next node to add by least edge cost
                valid_start_exts = torch.nonzero(are_valid_start_exts).squeeze(-1)
                valid_end_exts = torch.nonzero(are_valid_end_exts).squeeze(-1)
                start_edge_costs = edge_costs[route[0], valid_start_exts]
                end_edge_costs = edge_costs[route[-1], valid_end_exts]
                ext_edge_costs = torch.cat((start_edge_costs, end_edge_costs))
                min_edge_cost = ext_edge_costs.min()
                options = ext_edge_costs == min_edge_cost
                # break ties randomly
                chosen_ext_loc = options.float().multinomial(1).item()
                if chosen_ext_loc < len(start_edge_costs):
                    chosen_node = valid_start_exts[chosen_ext_loc].item()
                    route.insert(0, chosen_node)
                else:
                    end_loc = chosen_ext_loc - len(start_edge_costs)
                    chosen_node = valid_end_exts[end_loc].item()
                    route.append(chosen_node)
                
                # mark the node as on the route and visited
                are_on_route[chosen_node] = True

            # route is finished
            if len(route) >= state.min_route_len:
                # route is long enough to add
                route = torch.tensor(route)
                # mark the route's nodes as visited and directly connected
                are_directly_connected[route, route] = True
                are_visited |= are_on_route
                network.append(route)

            pair_attempted[seed_pair] = True

            if len(network) == state.n_routes_to_plan:
                # we have enough routes, so we're done
                break

        if len(network) < state.n_routes_to_plan:
            # Use approach of Shih and Mahmassani to add more routes
            # sort node pairs by descending demand
            if prioritize_direct_connections:
                # only consider node pairs that are not directly connected
                nodepairs = torch.nonzero(~are_directly_connected)
                pair_demands = state.demand[0, ~are_directly_connected]
            else:
                pair_demands = state.demand[0]
                nodepairs = torch.combinations(torch.arange(n_nodes), r=2)

            sorted_indices = pair_demands.flatten().argsort(descending=True)
            sorted_pairs = [(ss.item(), dd.item())
                            for ss, dd in nodepairs[sorted_indices]]

            # build a networkx graph
            edge_costs[state.street_adj[0].isinf()] = 0
            graph = nx.from_numpy_matrix(edge_costs.numpy())

            # add the existing routes to the graph to get the transit times
            state.add_new_routes([network])

            # For each pair in order, until $|\mathcal{R}| = S$:
            for (src, dst) in sorted_pairs:

                # Use Yen's k-shortest path algorithm (see ref. 22) with k=10 
                 # to see if a valid route between the nodes exists that 
                 # doesn't violate any constraints
                paths = lrnu.yen_k_shortest_paths(graph, src, dst, 10)
    
                for path in paths:
                    # check if the path is valid
                    if state.min_route_len <= len(path) <= state.max_route_len \
                        and path not in network:
                        # it is valid and shorter than the shortest 
                         # transit path between the nodes, add it to 
                         # |\mathcal{R}|
                        path_len = nx.path_weight(graph, path, weight='weight')
                        if path_len < state.transit_times[0, src, dst]:
                            # add the path and move on to the next node pair
                            path = torch.tensor(path)
                            network.append(path)
                            are_visited[path] = True

                            state.add_new_routes([[path]])
                            break                

                if len(network) == state.n_routes_to_plan:
                    # we have enough routes, so we're done
                    break

        if len(network) < state.n_routes_to_plan:
            # raise RuntimeError('Failed to find enough routes')
            log.warning('John init failed to find enough routes')
        if not are_visited.all():
            # raise RuntimeError('Failed to cover all nodes')
            log.warning('John init failed to cover all nodes')
        else:
            networks.append(network)

        # make sure changes to state don't affect the original state
        state.clear_routes()

    if len(networks) == 0:
        raise RuntimeError('John init failed to find any valid networks')
        
    networks = get_batch_tensor_from_routes(networks, state.device, 
                                            state.max_route_len[0])
    return networks      
    

def nikolic_init(state: RouteGenBatchState):
    """Constructs a network based on the algorithm of Nikolic and Teodorovic 
        (2013).
    """
    shortest_paths, _ = reconstruct_all_paths(state.nexts)

    dev = shortest_paths.device
    batch_size = shortest_paths.shape[0]
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = state.max_n_nodes
    dm_uncovered = state.demand.clone()
    # add dummy column and row
    dm_uncovered = torch.nn.functional.pad(dm_uncovered, (0, 1, 0, 1))
    n_routes = state.n_routes_to_plan
    best_networks = torch.full((batch_size, n_routes, max_n_nodes), -1, 
                                device=dev)
    log.info('computing initial network')
    # stop-to-itself routes are all invalid
    terms_are_invalid = torch.eye(max_n_nodes + 1, device=dev, dtype=bool)
    # routes to and from the dummy stop are invalid
    terms_are_invalid[max_n_nodes, :] = True
    terms_are_invalid[:, max_n_nodes] = True
    terms_are_invalid = terms_are_invalid[None].repeat(batch_size, 1, 1)
    for ri in range(n_routes):
        # compute directly-satisfied-demand matrix
        direct_sat_dmd = get_direct_sat_dmd(dm_uncovered, shortest_paths,
                                            state.symmetric_routes)
        # set invalid term pairs to -1 so they won't be selected even if there
         # is no uncovered demand.
        direct_sat_dmd[terms_are_invalid] = -1

        # choose maximum DS route and add it to the initial network
        flat_dsd = direct_sat_dmd.flatten(1, 2)
        _, best_flat_idxs = flat_dsd.max(dim=1)
        # add 1 to account for the dummy column and row
        best_i = torch.div(best_flat_idxs, (max_n_nodes + 1), 
                           rounding_mode='floor')
        best_j = best_flat_idxs % (max_n_nodes + 1)
        # batch_size x route_len
        routes = shortest_paths[batch_idxs, best_i, best_j]
        best_networks[:, ri, :routes.shape[-1]] = routes

        # mark new routes as in use
        terms_are_invalid[batch_idxs, best_i, best_j] = True
        if state.symmetric_routes:
            terms_are_invalid[batch_idxs, best_j, best_i] = True

        # remove newly-covered demand from uncovered demand matrix
        for ii in range(routes.shape[-1] - 1):
            cur_stops = routes[:, ii][:, None]
            later_stops = routes[:, ii+1:]
            dm_uncovered[batch_idxs[:, None], cur_stops, later_stops] = 0
            if state.symmetric_routes:
                maxidx = routes.shape[-1] - 1
                cur_stops = routes[:, maxidx - ii]
                later_stops = routes[:, maxidx - ii - 1]
                dm_uncovered[batch_idxs[:, None], later_stops, cur_stops] = 0

    return best_networks


def get_direct_sat_dmd(stop_dmd, shortest_paths, symmetric):
    direct_sat_dmd = torch.zeros_like(stop_dmd)
    summed_demands = aggr_edges_over_sequences(shortest_paths, 
                                               stop_dmd[..., None])
    if symmetric:
        summed_demands += \
            aggr_edges_over_sequences(shortest_paths.transpose(1,2),
                                      stop_dmd.transpose(1,2)[..., None])
    direct_sat_dmd[:, :-1, :-1] = summed_demands.squeeze(-1)

    # assumes there's a dummy column and row
    direct_sat_dmd[:, -1] = 0
    direct_sat_dmd[:, :, -1] = 0
    return direct_sat_dmd
