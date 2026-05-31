import pytest

from rl.env import TransitEnv


def sioux_falls_config(num_routes=3, max_route_length=5):
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
        "route_init": "highest_demand",
        "transit_center_node": None,
        "ppo_reward_mode": "terminal_intermediate_delta_no_early_stop",
        "num_routes": num_routes,
        "max_route_length": max_route_length,
        "min_route_length": 2,
        "seed": 42,
    }


def make_env(num_routes=3, max_route_length=5):
    env = TransitEnv(sioux_falls_config(num_routes, max_route_length))
    env.reset(seed=7)
    return env


def partial_metrics(coverage):
    return {
        "wanting_to_onboard": coverage * 100.0,
        "total_demand": 100.0,
        "demand_coverage_potential": coverage,
        "route_overlap_ratio": coverage / 10.0,
    }


def patch_partial_metrics(monkeypatch, env, *coverages):
    queued_metrics = [partial_metrics(coverage) for coverage in coverages]
    calls = []

    def fake_partial_metrics():
        calls.append(None)
        if not queued_metrics:
            raise AssertionError("_get_partial_route_metrics called too many times")
        return dict(queued_metrics.pop(0))

    monkeypatch.setattr(env, "_get_partial_route_metrics", fake_partial_metrics)
    return calls


def patch_reward_spy(monkeypatch, env, reward=123.0):
    calls = []

    def fake_compute_reward(
        sim_result,
        is_route_end,
        is_forced_end,
        prev_coverage=0.0,
    ):
        calls.append(
            {
                "sim_result": dict(sim_result),
                "is_route_end": is_route_end,
                "is_forced_end": is_forced_end,
                "prev_coverage": prev_coverage,
            }
        )
        return reward

    monkeypatch.setattr(env, "compute_reward", fake_compute_reward)
    return calls


def forbid_uxsim(monkeypatch, env):
    def fail(*args, **kwargs):
        pytest.fail("UXsim simulation should not run for this step")

    monkeypatch.setattr(env, "build_world", fail)
    monkeypatch.setattr(env, "_apply_action", fail)
    monkeypatch.setattr(env, "_step_until", fail)


def patch_uxsim_spy(monkeypatch, env, terminal_coverage=0.9):
    calls = []
    world = object()

    def fake_build_world(network):
        calls.append(("build_world", network))
        return world

    def fake_apply_action():
        calls.append(("_apply_action", None))

    def fake_step_until(horizon):
        calls.append(("_step_until", horizon))
        return partial_metrics(terminal_coverage)

    monkeypatch.setattr(env, "build_world", fake_build_world)
    monkeypatch.setattr(env, "_apply_action", fake_apply_action)
    monkeypatch.setattr(env, "_step_until", fake_step_until)
    return calls, world


def test_step_valid_action_extends_current_route_without_simulation(monkeypatch):
    env = make_env()
    patch_partial_metrics(monkeypatch, env, 0.10, 0.25)
    reward_calls = patch_reward_spy(monkeypatch, env, reward=7.0)
    forbid_uxsim(monkeypatch, env)

    state, reward, terminated, truncated, sim_result = env.step(env.node_to_idx["9"])

    assert reward == 7.0
    assert terminated is False
    assert truncated is None
    assert env.current_route == ["10", "9"]
    assert env.all_routes == []
    assert env.current_route_index == 0
    assert sim_result == {
        "wanting_to_onboard": 25.0,
        "total_demand": 100.0,
        "demand_coverage_potential": 0.25,
        "route_overlap_ratio": 0.025,
        "route_completed": False,
        "route_forced_end": False,
    }
    assert reward_calls == [
        {
            "sim_result": sim_result,
            "is_route_end": False,
            "is_forced_end": False,
            "prev_coverage": 0.10,
        }
    ]
    assert env.observation_space.contains(state)


def test_step_nonterminal_max_length_route_starts_next_route_without_simulation(
    monkeypatch,
):
    env = make_env()
    env.current_route = ["10", "9", "8", "7"]
    patch_partial_metrics(monkeypatch, env, 0.20, 0.40)
    reward_calls = patch_reward_spy(monkeypatch, env, reward=8.0)
    forbid_uxsim(monkeypatch, env)

    state, reward, terminated, truncated, sim_result = env.step(env.node_to_idx["18"])

    assert reward == 8.0
    assert terminated is False
    assert truncated is None
    assert env.all_routes == [["10", "9", "8", "7", "18"]]
    assert env.current_route_index == 1
    assert env.current_route == ["10"]
    assert sim_result["route_completed"] is True
    assert sim_result["route_forced_end"] is False
    assert reward_calls == [
        {
            "sim_result": sim_result,
            "is_route_end": False,
            "is_forced_end": False,
            "prev_coverage": 0.20,
        }
    ]
    assert env.observation_space.contains(state)


def test_step_no_valid_action_forces_nonterminal_route_end_without_simulation(
    monkeypatch,
):
    env = make_env()
    env.current_route = ["10", "9"]
    patch_partial_metrics(monkeypatch, env, 0.30, 0.35)
    reward_calls = patch_reward_spy(monkeypatch, env, reward=9.0)
    forbid_uxsim(monkeypatch, env)

    state, reward, terminated, truncated, sim_result = env.step(env.NO_VALID_ACTION)

    assert reward == 9.0
    assert terminated is False
    assert truncated is None
    assert env.all_routes == [["10", "9"]]
    assert env.current_route_index == 1
    assert env.current_route == ["10"]
    assert sim_result["route_completed"] is False
    assert sim_result["route_forced_end"] is True
    assert reward_calls == [
        {
            "sim_result": sim_result,
            "is_route_end": False,
            "is_forced_end": True,
            "prev_coverage": 0.30,
        }
    ]
    assert env.observation_space.contains(state)


def test_step_terminal_max_length_route_runs_simulation_once(monkeypatch):
    env = make_env(num_routes=1, max_route_length=3)
    env.current_route = ["10", "9"]
    patch_partial_metrics(monkeypatch, env, 0.40, 0.50)
    reward_calls = patch_reward_spy(monkeypatch, env, reward=10.0)
    uxsim_calls, world = patch_uxsim_spy(monkeypatch, env, terminal_coverage=0.75)

    state, reward, terminated, truncated, sim_result = env.step(env.node_to_idx["8"])

    assert reward == 10.0
    assert terminated is True
    assert truncated is None
    assert env.world is world
    assert env.all_routes == [["10", "9", "8"]]
    assert env.current_route_index == 1
    assert uxsim_calls == [
        ("build_world", "sioux_falls"),
        ("_apply_action", None),
        ("_step_until", 3600),
    ]
    assert sim_result["route_completed"] is True
    assert sim_result["route_forced_end"] is False
    assert reward_calls == [
        {
            "sim_result": sim_result,
            "is_route_end": True,
            "is_forced_end": False,
            "prev_coverage": 0.40,
        }
    ]
    assert env.observation_space.contains(state)


def test_step_terminal_no_valid_action_runs_simulation_once(monkeypatch):
    env = make_env(num_routes=1)
    env.current_route = ["10", "9"]
    patch_partial_metrics(monkeypatch, env, 0.45)
    reward_calls = patch_reward_spy(monkeypatch, env, reward=11.0)
    uxsim_calls, world = patch_uxsim_spy(monkeypatch, env, terminal_coverage=0.80)

    state, reward, terminated, truncated, sim_result = env.step(env.NO_VALID_ACTION)

    assert reward == 11.0
    assert terminated is True
    assert truncated is None
    assert env.world is world
    assert env.all_routes == [["10", "9"]]
    assert env.current_route_index == 1
    assert uxsim_calls == [
        ("build_world", "sioux_falls"),
        ("_apply_action", None),
        ("_step_until", 3600),
    ]
    assert sim_result["route_completed"] is False
    assert sim_result["route_forced_end"] is True
    assert reward_calls == [
        {
            "sim_result": sim_result,
            "is_route_end": True,
            "is_forced_end": True,
            "prev_coverage": 0.45,
        }
    ]
    assert env.observation_space.contains(state)
