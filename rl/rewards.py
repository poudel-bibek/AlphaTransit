from typing import Any, Dict


REWARD_MODES = (
    "terminal_only",
    "terminal_intermediate_raw_early_stop",
    "terminal_intermediate_delta_early_stop",
    "terminal_intermediate_delta_no_early_stop",
)
VALID_REWARD_MODES = set(REWARD_MODES)


def validate_reward_mode(ppo_reward_mode: str) -> None:
    if ppo_reward_mode not in VALID_REWARD_MODES:
        raise ValueError(f"Unknown ppo_reward_mode: {ppo_reward_mode}")


# ---------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------
BETA_0 = 60.0   # Final: demand coverage potential (Psi)
BETA_1 = 45.0   # Final: service rate (sigma)
BETA_2 = 20.0   # Final: average wait time penalty
BETA_3 = 10.0   # Final: average movement time penalty
BETA_4 = 10.0   # Final: route overlap penalty (omega)
BETA_5 = 2.0    # Final: fleet size penalty (F/K)
BETA_6 = 12.0   # Final: bus utilization bonus (u)

BETA_7 = 20.0   # Partial: coverage term (raw or delta)
BETA_8 = 8.0    # Partial: overlap term (omega) magnitude
BETA_9 = 15.0   # Partial: forced-end penalty term magnitude

WAIT_TIME_CAP = 1800.0
MOVEMENT_TIME_CAP = 2400.0


def compute_transit_reward(
    sim_result: Dict[str, Any],
    *,
    ppo_reward_mode: str,
    is_route_end: bool,
    is_forced_end: bool,
    prev_coverage: float,
    current_route_index: int,
    num_routes: int,
    current_route_length: int,
    max_route_length: int,
) -> float:
    """
    Reward function with mode-controlled shaping.

    Supported reward modes:
    - terminal_only
    - terminal_intermediate_raw_early_stop
    - terminal_intermediate_delta_early_stop
    - terminal_intermediate_delta_no_early_stop
    """
    validate_reward_mode(ppo_reward_mode)

    # ---------------------------------------------------------------------
    # Reward intent
    # ---------------------------------------------------------------------
    # Encourage:
    # - broader reachable demand coverage
    # - higher passenger service rate
    # - higher bus utilization
    #
    # Discourage:
    # - long passenger wait/movement times
    # - redundant route overlap
    # - excessive fleet size
    # - premature forced route endings
    #
    # Final reward (b0..b6) captures passenger/operator outcomes after simulation.
    # Partial reward (b7..b9) shapes route construction before terminal simulation.
    # ---------------------------------------------------------------------

    # Episode terminal condition: last route is ending right now.
    # Used only for terminal_only mode.
    is_episode_terminal = is_route_end and (current_route_index == num_routes - 1)

    # ---------------------------------------------------------------------
    # Intermediate reward branch (during route construction)
    # ---------------------------------------------------------------------
    if not is_route_end:
        # terminal_only: no shaping signal during construction.
        if ppo_reward_mode == "terminal_only":
            return 0.0

        # Intermediate PPO modes only use partial metrics available from
        # _get_partial_route_metrics(), so non-terminal steps can be scored
        # without running a full UXsim simulation.
        current_coverage = sim_result['demand_coverage_potential']
        overlap_ratio = sim_result['route_overlap_ratio']

        if ppo_reward_mode == "terminal_intermediate_raw_early_stop":
            # Raw shaping: reward absolute coverage at this step.
            coverage_term = current_coverage
            use_early_stop_penalty = True

        elif ppo_reward_mode == "terminal_intermediate_delta_early_stop":
            # Delta shaping: reward only incremental coverage gain.
            coverage_term = max(0.0, current_coverage - prev_coverage)
            use_early_stop_penalty = True

        else:
            # terminal_intermediate_delta_no_early_stop:
            # same delta shaping, but remove forced-end penalty.
            coverage_term = max(0.0, current_coverage - prev_coverage)
            use_early_stop_penalty = False

        # Shared shaping terms for non-terminal route-building steps.
        # Encourages coverage growth and discourages redundant overlap.
        reward = (BETA_7 * coverage_term) - (BETA_8 * overlap_ratio)

        # Apply early-stop penalty only in modes that enable it.
        # Discourages getting stuck before reaching max route length.
        if is_forced_end and use_early_stop_penalty:
            completion_ratio = current_route_length / max_route_length
            forced_penalty = 1.0 - completion_ratio
            reward -= BETA_9 * forced_penalty

        return reward

    # ---------------------------------------------------------------------
    # Final reward branch (route-end simulation metrics)
    # ---------------------------------------------------------------------
    # terminal_only pays out only once at episode terminal.
    if ppo_reward_mode == "terminal_only" and not is_episode_terminal:
        return 0.0

    # All other modes use the same final simulation-based reward.
    # This is the primary objective that balances rider outcomes and operator costs.
    coverage = sim_result['demand_coverage_potential']
    # Reward hack fix: service_rate has variable denominator (wanting_to_onboard),
    # agent exploits by building short routes. Use fixed denominator (total_demand).
    # service_rate = sim_result['service_rate']
    service_rate = sim_result['demand_coverage_actual']

    total_wait = sim_result['total_wait_completed'] + sim_result['total_wait_ongoing']
    total_movement = sim_result['total_movement_completed'] + sim_result['total_movement_ongoing']
    served = sim_result['completed_passengers'] + sim_result['ongoing_passengers']

    avg_wait = total_wait / served if served > 0 else 0.0
    avg_movement = total_movement / served if served > 0 else 0.0
    avg_wait_norm = min(avg_wait / WAIT_TIME_CAP, 1.0)
    avg_movement_norm = min(avg_movement / MOVEMENT_TIME_CAP, 1.0)

    overlap_ratio = sim_result['route_overlap_ratio']
    fleet_per_route = sim_result['fleet_size'] / num_routes
    utilization_norm = sim_result['bus_utilization'] / 100.0

    final_reward = (
        BETA_0 * coverage +
        BETA_1 * service_rate -
        BETA_2 * avg_wait_norm -
        BETA_3 * avg_movement_norm -
        BETA_4 * overlap_ratio -
        BETA_5 * fleet_per_route +
        BETA_6 * utilization_norm
    )
    return final_reward
