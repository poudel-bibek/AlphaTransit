import pytest

from rl.env import TransitEnv
from rl.rewards import compute_transit_reward


PARTIAL_RESULT = {
    "demand_coverage_potential": 0.4,
    "route_overlap_ratio": 0.1,
}


FINAL_RESULT = {
    "demand_coverage_potential": 0.5,
    "demand_coverage_actual": 0.3,
    "total_wait_completed": 1000.0,
    "total_wait_ongoing": 800.0,
    "total_movement_completed": 1200.0,
    "total_movement_ongoing": 600.0,
    "completed_passengers": 2,
    "ongoing_passengers": 1,
    "route_overlap_ratio": 0.2,
    "fleet_size": 16,
    "bus_utilization": 50.0,
}


def test_partial_reward_raw_early_stop_keeps_forced_end_penalty():
    reward = compute_transit_reward(
        PARTIAL_RESULT,
        ppo_reward_mode="terminal_intermediate_raw_early_stop",
        is_route_end=False,
        is_forced_end=True,
        prev_coverage=0.25,
        current_route_index=0,
        num_routes=4,
        current_route_length=5,
        max_route_length=10,
    )

    assert reward == pytest.approx(-0.3)


def test_partial_reward_delta_no_early_stop_ignores_forced_end_penalty():
    reward = compute_transit_reward(
        PARTIAL_RESULT,
        ppo_reward_mode="terminal_intermediate_delta_no_early_stop",
        is_route_end=False,
        is_forced_end=True,
        prev_coverage=0.25,
        current_route_index=0,
        num_routes=4,
        current_route_length=5,
        max_route_length=10,
    )

    assert reward == pytest.approx(2.2)


def test_terminal_only_rewards_only_episode_terminal():
    non_terminal_reward = compute_transit_reward(
        FINAL_RESULT,
        ppo_reward_mode="terminal_only",
        is_route_end=True,
        is_forced_end=False,
        prev_coverage=0.0,
        current_route_index=2,
        num_routes=4,
        current_route_length=10,
        max_route_length=10,
    )
    terminal_reward = compute_transit_reward(
        FINAL_RESULT,
        ppo_reward_mode="terminal_only",
        is_route_end=True,
        is_forced_end=False,
        prev_coverage=0.0,
        current_route_index=3,
        num_routes=4,
        current_route_length=10,
        max_route_length=10,
    )

    assert non_terminal_reward == 0.0
    assert terminal_reward == pytest.approx(30.3333333333)


def test_transit_env_compute_reward_delegates_to_pure_function():
    env = TransitEnv.__new__(TransitEnv)
    env.ppo_reward_mode = "terminal_intermediate_delta_early_stop"
    env.current_route_index = 0
    env.NUM_ROUTES = 4
    env.current_route = ["1", "2", "3", "4", "5"]
    env.MAX_ROUTE_LENGTH = 10

    wrapper_reward = env.compute_reward(
        PARTIAL_RESULT,
        is_route_end=False,
        is_forced_end=True,
        prev_coverage=0.25,
    )
    pure_reward = compute_transit_reward(
        PARTIAL_RESULT,
        ppo_reward_mode=env.ppo_reward_mode,
        is_route_end=False,
        is_forced_end=True,
        prev_coverage=0.25,
        current_route_index=env.current_route_index,
        num_routes=env.NUM_ROUTES,
        current_route_length=len(env.current_route),
        max_route_length=env.MAX_ROUTE_LENGTH,
    )

    assert wrapper_reward == pure_reward


def test_unknown_reward_mode_still_raises_value_error():
    with pytest.raises(ValueError, match="Unknown ppo_reward_mode"):
        compute_transit_reward(
            PARTIAL_RESULT,
            ppo_reward_mode="typo",
            is_route_end=False,
            is_forced_end=False,
            prev_coverage=0.0,
            current_route_index=0,
            num_routes=4,
            current_route_length=5,
            max_route_length=10,
        )


def test_transit_env_terminal_only_non_route_end_does_not_need_route_attrs():
    env = TransitEnv.__new__(TransitEnv)
    env.ppo_reward_mode = "terminal_only"

    reward = env.compute_reward(
        PARTIAL_RESULT,
        is_route_end=False,
        is_forced_end=False,
        prev_coverage=0.25,
    )

    assert reward == 0.0


def test_transit_env_unknown_mode_does_not_need_route_attrs():
    env = TransitEnv.__new__(TransitEnv)
    env.ppo_reward_mode = "typo"

    with pytest.raises(ValueError, match="Unknown ppo_reward_mode"):
        env.compute_reward(
            PARTIAL_RESULT,
            is_route_end=False,
            is_forced_end=False,
            prev_coverage=0.25,
        )


def test_transit_env_compute_reward_passes_terminal_context(monkeypatch):
    env = TransitEnv.__new__(TransitEnv)
    env.ppo_reward_mode = "terminal_only"
    env.current_route_index = 4
    env.NUM_ROUTES = 5

    recorded = {}

    def fake_compute_transit_reward(sim_result, **kwargs):
        recorded["sim_result"] = sim_result
        recorded.update(kwargs)
        return 123.0

    monkeypatch.setattr("rl.env.compute_transit_reward", fake_compute_transit_reward)

    reward = env.compute_reward(
        FINAL_RESULT,
        is_route_end=True,
        is_forced_end=False,
        prev_coverage=0.25,
    )

    assert reward == 123.0
    assert recorded["sim_result"] is FINAL_RESULT
    assert recorded["ppo_reward_mode"] == "terminal_only"
    assert recorded["is_route_end"] is True
    assert recorded["is_forced_end"] is False
    assert recorded["prev_coverage"] == 0.25
    assert recorded["current_route_index"] == 4
    assert recorded["num_routes"] == 5


def test_transit_env_compute_reward_passes_forced_penalty_context(monkeypatch):
    env = TransitEnv.__new__(TransitEnv)
    env.ppo_reward_mode = "terminal_intermediate_delta_early_stop"
    env.current_route = ["1", "2", "3"]
    env.MAX_ROUTE_LENGTH = 9

    recorded = {}

    def fake_compute_transit_reward(sim_result, **kwargs):
        recorded["sim_result"] = sim_result
        recorded.update(kwargs)
        return 456.0

    monkeypatch.setattr("rl.env.compute_transit_reward", fake_compute_transit_reward)

    reward = env.compute_reward(
        PARTIAL_RESULT,
        is_route_end=False,
        is_forced_end=True,
        prev_coverage=0.25,
    )

    assert reward == 456.0
    assert recorded["sim_result"] is PARTIAL_RESULT
    assert recorded["current_route_length"] == 3
    assert recorded["max_route_length"] == 9


def test_delta_reward_clamps_negative_coverage_gain():
    reward = compute_transit_reward(
        PARTIAL_RESULT,
        ppo_reward_mode="terminal_intermediate_delta_no_early_stop",
        is_route_end=False,
        is_forced_end=False,
        prev_coverage=0.5,
        current_route_index=0,
        num_routes=4,
        current_route_length=5,
        max_route_length=10,
    )

    assert reward == pytest.approx(-0.8)


def test_final_reward_handles_zero_served_without_wait_penalty():
    sim_result = dict(FINAL_RESULT)
    sim_result.update(
        {
            "completed_passengers": 0,
            "ongoing_passengers": 0,
            "total_wait_completed": 99999.0,
            "total_wait_ongoing": 99999.0,
            "total_movement_completed": 99999.0,
            "total_movement_ongoing": 99999.0,
        }
    )

    reward = compute_transit_reward(
        sim_result,
        ppo_reward_mode="terminal_intermediate_delta_no_early_stop",
        is_route_end=True,
        is_forced_end=False,
        prev_coverage=0.0,
        current_route_index=0,
        num_routes=16,
        current_route_length=10,
        max_route_length=10,
    )

    assert reward == pytest.approx(45.5)


def test_final_reward_caps_wait_and_movement_penalties():
    sim_result = dict(FINAL_RESULT)
    sim_result.update(
        {
            "completed_passengers": 1,
            "ongoing_passengers": 0,
            "total_wait_completed": 99999.0,
            "total_wait_ongoing": 0.0,
            "total_movement_completed": 99999.0,
            "total_movement_ongoing": 0.0,
        }
    )

    reward = compute_transit_reward(
        sim_result,
        ppo_reward_mode="terminal_intermediate_delta_no_early_stop",
        is_route_end=True,
        is_forced_end=False,
        prev_coverage=0.0,
        current_route_index=0,
        num_routes=16,
        current_route_length=10,
        max_route_length=10,
    )

    assert reward == pytest.approx(15.5)

