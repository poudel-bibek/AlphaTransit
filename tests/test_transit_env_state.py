import numpy as np
import pytest

from rl.env import TransitEnv


def sioux_falls_config(route_init="highest_demand"):
    return {
        "network": "sioux_falls",
        "horizon": 3600,
        "delta_t": 1,
        "delta_n": 5,
        "service_frequency_mode": "max_load",
        "stop_spacing": 1,
        "bus_capacity": 40,
        "stop_duration": 60,
        "alpha": 0.3,
        "demand_warmup": 0.15,
        "ignore_unserved": False,
        "comfort_threshold": 1.0,
        "radius": 0.5,
        "route_init": route_init,
        "transit_center_node": None,
        "ppo_reward_mode": "terminal_intermediate_delta_no_early_stop",
        "num_routes": 3,
        "max_route_length": 5,
        "min_route_length": 2,
        "seed": 42,
    }


@pytest.fixture
def sioux_falls_env():
    return TransitEnv(sioux_falls_config())


def assert_observation_contract(env, state):
    assert set(state) == {
        "node_features",
        "edge_index",
        "edge_features",
        "route_progress",
        "frontier_index",
    }
    assert state["node_features"].shape == (env.n_nodes, env.N_NODE_FEATURES)
    assert state["node_features"].dtype == np.float32
    assert state["edge_index"].shape == env.edge_index.shape
    assert state["edge_index"].dtype == np.int64
    assert state["edge_features"].shape == env.edge_features.shape
    assert state["edge_features"].dtype == np.float32
    assert state["route_progress"].shape == (env.NUM_ROUTES,)
    assert state["route_progress"].dtype == np.float32
    assert 0 <= int(state["frontier_index"]) <= env.n_nodes


def test_sioux_falls_init_loads_network_and_spaces(sioux_falls_env):
    env = sioux_falls_env

    assert env.world is None
    assert env.N_NODE_FEATURES == 16
    assert env.N_EDGE_FEATURES == 2
    assert env.n_nodes == 24
    assert env.n_edges == 76
    assert env.node_list[:3] == ["1", "2", "3"]
    assert env.node_list[-3:] == ["22", "23", "24"]
    assert env.node_to_idx[env.idx_to_node[9]] == 9
    assert env.NO_VALID_ACTION == env.n_nodes == 24
    assert env.action_space.n == env.n_nodes + 1 == 25
    assert env.NUM_ROUTES == 3
    assert env.MAX_ROUTE_LENGTH == 5
    assert env.MIN_ROUTE_LENGTH == 2

    assert env.edge_index.shape == (2, 76)
    assert env.edge_index.dtype == np.int64
    assert env.edge_features.shape == (76, 2)
    assert env.edge_features.dtype == np.float32
    assert np.all(env.edge_features >= 0.0)
    assert np.all(env.edge_features <= 1.0)

    assert env.od_matrix.shape == (24, 24)
    assert env.demand_out.shape == (24,)
    assert env.demand_in.shape == (24,)
    assert env.max_demand == pytest.approx(2712.0)
    assert env._static_node_features.shape == (24, 5)
    assert env._static_node_features.dtype == np.float32
    assert len(env.link_lengths) == 76


def test_reset_initializes_highest_demand_route_and_valid_observation(sioux_falls_env):
    env = sioux_falls_env

    state, info = env.reset(seed=7)

    assert info == {}
    assert env.world is None
    assert env.config["seed"] == 7
    assert env.all_routes == []
    assert env.current_route_index == 0
    assert env.current_route == ["10"]
    assert_observation_contract(env, state)
    assert state["route_progress"].tolist() == pytest.approx([0.2, 0.0, 0.0])
    assert int(state["frontier_index"]) == env.node_to_idx["10"] == 9
    assert env.observation_space.contains(state)


def test_initial_state_marks_current_route_and_valid_next_nodes(sioux_falls_env):
    env = sioux_falls_env
    state, _ = env.reset(seed=7)
    node_features = state["node_features"]

    current_route_indices = set(np.flatnonzero(node_features[:, 13]).tolist())
    assert current_route_indices == {env.node_to_idx["10"]}

    valid_node_ids = {env.idx_to_node[idx] for idx in env._get_valid_indices()}
    assert valid_node_ids == {"9", "11", "15", "16", "17"}
    valid_feature_ids = {
        env.idx_to_node[idx]
        for idx in np.flatnonzero(node_features[:, 15]).tolist()
    }
    assert valid_feature_ids == valid_node_ids

    assert node_features[:, 14].sum() == 0.0
    assert np.count_nonzero(node_features[:, 11:13]) == 0
    np.testing.assert_allclose(node_features[:, :5], env._static_node_features)

    idx10 = env.node_to_idx["10"]
    assert node_features[idx10, 3] == pytest.approx(1.0)
    assert node_features[idx10, 4] == pytest.approx(2706.0 / 2712.0)


def test_initial_state_current_route_demand_columns_match_od_matrix(sioux_falls_env):
    env = sioux_falls_env
    state, _ = env.reset(seed=7)
    node_features = state["node_features"]
    idx10 = env.node_to_idx["10"]
    valid_indices = set(env._get_valid_indices())

    local_out_nonzero = set(np.flatnonzero(node_features[:, 5]).tolist())
    local_in_nonzero = set(np.flatnonzero(node_features[:, 6]).tolist())
    assert local_out_nonzero <= valid_indices
    assert local_in_nonzero <= valid_indices

    for idx in valid_indices:
        assert node_features[idx, 5] == pytest.approx(env.od_matrix[idx, idx10] / env.max_demand)
        assert node_features[idx, 6] == pytest.approx(env.od_matrix[idx10, idx] / env.max_demand)

    np.testing.assert_allclose(node_features[:, 9], env.od_matrix[:, idx10] / env.max_demand)
    np.testing.assert_allclose(node_features[:, 10], env.od_matrix[idx10, :] / env.max_demand)
