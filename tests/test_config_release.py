import pytest

from config import BASELINE_TYPES, build_arg_parser, normalize_config
from rl.env_utils import initialize_route


class DummyEnv:
    def __init__(self):
        import pandas as pd

        self.config = {"network": "sioux_falls"}
        self.demand_df_cached = pd.DataFrame({"orig": ["1"], "volume": [1.0]})
        self.all_routes = []
        self.route_init = "transit_center"
        self.transit_center_node = "96"
        self.node_to_idx = {"1": 0}


def test_baseline_type_choices_reject_unknown_value():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "baseline", "--baseline_type", "typo"])


def test_non_bloomington_default_route_init_becomes_network_safe():
    config = {
        "network": "sioux_falls",
        "route_init": "transit_center",
        "transit_center_node": None,
    }
    normalized = normalize_config(config)
    assert normalized["route_init"] == "highest_demand"


def test_explicit_transit_center_requires_node_for_unknown_network():
    config = {
        "network": "sioux_falls",
        "route_init": "transit_center",
        "transit_center_node": None,
    }
    with pytest.raises(ValueError, match="requires --transit_center_node"):
        normalize_config(config, explicitly_set={"route_init"})


def test_initialize_route_reports_invalid_transit_center():
    with pytest.raises(ValueError, match="not valid for network"):
        initialize_route(DummyEnv())


def test_baseline_type_list_has_expected_release_values():
    assert "real_world" in BASELINE_TYPES
    assert "neural_evolutionary" in BASELINE_TYPES
