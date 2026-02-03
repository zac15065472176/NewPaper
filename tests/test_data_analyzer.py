import numpy as np
from adaprune.core.data_analyzer import DataProfileAnalyzer


class TestDataProfileAnalyzer:
    def test_analyze_returns_dict(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, size=100)
        profile = DataProfileAnalyzer().analyze(X, y)
        assert isinstance(profile, dict)

    def test_basic_stats(self):
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, size=50)
        profile = DataProfileAnalyzer().analyze(X, y)
        assert profile["n_samples"] == 50
        assert profile["n_features"] == 3

    def test_imbalance_ratio(self):
        X = np.random.randn(100, 4)
        y = np.array([0] * 90 + [1] * 10)
        profile = DataProfileAnalyzer().analyze(X, y)
        assert profile["imbalance_ratio"] >= 1.0

    def test_noise_estimation(self):
        X = np.random.randn(200, 6)
        y = np.random.randint(0, 2, size=200)
        profile = DataProfileAnalyzer().analyze(X, y)
        assert 0.0 <= profile["estimated_noise_level"] <= 1.0

    def test_recommended_depth(self):
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, size=100)
        profile = DataProfileAnalyzer().analyze(X, y)
        assert 3 <= profile["recommended_initial_depth"] <= 20

    def test_edge_case_small_dataset(self):
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, size=10)
        profile = DataProfileAnalyzer().analyze(X, y)
        assert "recommended_initial_depth" in profile