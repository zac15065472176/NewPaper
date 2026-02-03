import numpy as np
from adaprune.core.state_monitor import StateMonitor


class TestStateMonitor:
    def test_update(self):
        sm = StateMonitor()
        sm.update(1.0, 1.2, 0.5, 0.4)
        assert len(sm.history) == 1

    def test_overfit_detection(self):
        sm = StateMonitor()
        for _ in range(5):
            sm.update(0.2, 0.8, 0.9, 0.6)
        state = sm.get_state()
        assert state["overfit_score"] > 0.5

    def test_underfit_detection(self):
        sm = StateMonitor()
        for _ in range(5):
            sm.update(1.0, 1.1, 0.5, 0.5)
        state = sm.get_state()
        assert state["underfit_score"] > 0.0

    def test_plateau_detection(self):
        sm = StateMonitor()
        for _ in range(5):
            sm.update(0.5, 0.5, 0.7, 0.7)
        state = sm.get_state()
        assert state["plateau_score"] >= 0.0

    def test_reset(self):
        sm = StateMonitor()
        sm.update(1.0, 1.2, 0.5, 0.4)
        sm.reset()
        assert len(sm.history) == 0