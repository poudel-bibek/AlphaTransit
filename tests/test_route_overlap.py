import pytest

from rl.env import TransitEnv


def make_env(all_routes, current_route, current_route_index):
    env = TransitEnv.__new__(TransitEnv)
    env.all_routes = all_routes
    env.current_route = current_route
    env.current_route_index = current_route_index
    return env


def test_overlap_does_not_count_lifecycle_duplicate_current_route():
    env = make_env(
        all_routes=[["A", "B", "C"]],
        current_route=["A", "B", "C"],
        current_route_index=0,
    )

    assert env._calculate_route_overlap_ratio() == 0.0


def test_overlap_does_not_count_mcts_terminal_duplicate_last_route():
    env = make_env(
        all_routes=[["A", "B", "C"], ["D", "E", "F"]],
        current_route=["D", "E", "F"],
        current_route_index=1,
    )

    assert env._calculate_route_overlap_ratio() == 0.0


def test_overlap_lifecycle_duplicate_normalizes_node_ids_to_strings():
    env = make_env(
        all_routes=[["1", "2", "3"]],
        current_route=[1, 2, 3],
        current_route_index=0,
    )

    assert env._calculate_route_overlap_ratio() == 0.0


def test_overlap_still_counts_distinct_identical_current_route():
    env = make_env(
        all_routes=[["A", "B", "C"]],
        current_route=["A", "B", "C"],
        current_route_index=1,
    )

    assert env._calculate_route_overlap_ratio() == 1.0


def test_overlap_still_counts_real_shared_segment():
    env = make_env(
        all_routes=[["A", "B", "C"]],
        current_route=["B", "C", "D"],
        current_route_index=1,
    )

    assert env._calculate_route_overlap_ratio() == pytest.approx(1 / 3)
