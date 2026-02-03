import numpy as np
from adaprune.core.pruning_params import PruningParams
from adaprune.strategies.gap_based import GapBasedStrategy
from adaprune.strategies.data_aware import DataAwareStrategy
from adaprune.strategies.hybrid import HybridStrategy


def test_increase_pruning_on_overfit():
    strat = GapBasedStrategy()
    params = PruningParams(max_depth=10, min_samples_leaf=1)
    state = {"gap": 0.2, "overfit_score": 0.9, "underfit_score": 0.0}
    new_params = strat.adapt(params, state, {}, 10)
    assert new_params.max_depth < 10


def test_decrease_pruning_on_underfit():
    strat = GapBasedStrategy()
    params = PruningParams(max_depth=5, min_samples_leaf=4)
    state = {"gap": 0.0, "overfit_score": 0.0, "underfit_score": 0.9}
    new_params = strat.adapt(params, state, {}, 10)
    assert new_params.max_depth > 5


def test_high_noise_stronger_pruning():
    strat = DataAwareStrategy()
    params = PruningParams(max_depth=10, min_samples_leaf=1)
    state = {"gap": 0.2, "overfit_score": 0.9, "underfit_score": 0.0}
    data_profile = {"estimated_noise_level": 0.8, "n_samples": 200}
    new_params = strat.adapt(params, state, data_profile, 10)
    assert new_params.max_depth <= 9


def test_early_phase_uses_dataaware():
    strat = HybridStrategy(total_iterations=100)
    params = PruningParams()
    state = {"gap": 0.1, "overfit_score": 0.6, "underfit_score": 0.0}
    data_profile = {"estimated_noise_level": 0.5}
    new_params = strat.adapt(params, state, data_profile, 10)
    assert isinstance(new_params, PruningParams)


def test_late_phase_uses_trend():
    strat = HybridStrategy(total_iterations=100)
    params = PruningParams()
    state = {"val_improvement": 0.0, "plateau_score": 1.0}
    new_params = strat.adapt(params, state, {}, 95)
    assert isinstance(new_params, PruningParams)