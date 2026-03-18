from __future__ import annotations

import time
import pytest

from src.config import Settings
from src.strategy.uptime_tracker import UptimeTracker, MarketUptime


@pytest.fixture
def tracker(settings: Settings) -> UptimeTracker:
    return UptimeTracker(settings)


class TestMarketUptime:
    def test_uptime_pct_no_data(self):
        m = MarketUptime(market_id="test")
        assert m.uptime_pct == 0.0

    def test_uptime_pct_full(self):
        m = MarketUptime(market_id="test", total_tracked_s=100.0, total_live_s=100.0)
        assert m.uptime_pct == 100.0

    def test_uptime_pct_partial(self):
        m = MarketUptime(market_id="test", total_tracked_s=200.0, total_live_s=150.0)
        assert m.uptime_pct == 75.0


class TestUptimeTracker:
    def test_register_market(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        assert "m1" in tracker._markets

    def test_register_idempotent(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].total_live_s = 999
        tracker.register_market("m1")  # should not reset
        assert tracker._markets["m1"].total_live_s == 999

    def test_record_tick_live(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].last_check_time = time.time() - 1.0

        tracker.record_tick("m1", has_live_quotes=True)

        m = tracker._markets["m1"]
        assert m.total_tracked_s > 0
        assert m.total_live_s > 0
        assert m.quotes_live is True
        assert m.consecutive_gaps_s == 0.0

    def test_record_tick_gap(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].last_check_time = time.time() - 2.0

        tracker.record_tick("m1", has_live_quotes=False)

        m = tracker._markets["m1"]
        assert m.total_tracked_s > 0
        assert m.total_live_s == 0.0
        assert m.quotes_live is False
        assert m.consecutive_gaps_s > 0

    def test_record_tick_auto_registers(self, tracker: UptimeTracker):
        tracker.record_tick("new_market", has_live_quotes=True)
        assert "new_market" in tracker._markets

    def test_longest_gap_tracking(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        now = time.time()
        tracker._markets["m1"].last_check_time = now - 5.0

        tracker.record_tick("m1", has_live_quotes=False)
        gap1 = tracker._markets["m1"].longest_gap_s

        tracker._markets["m1"].last_check_time = time.time() - 10.0
        tracker.record_tick("m1", has_live_quotes=False)
        gap2 = tracker._markets["m1"].longest_gap_s

        assert gap2 >= gap1

    def test_should_widen_below_target(self, tracker: UptimeTracker):
        """When uptime is below target, recommend widening."""
        tracker.register_market("m1")
        tracker._markets["m1"].total_tracked_s = 100.0
        tracker._markets["m1"].total_live_s = 80.0  # 80% < 95% target
        assert tracker.should_widen_instead_of_pull("m1") is True

    def test_should_not_widen_above_target(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].total_tracked_s = 100.0
        tracker._markets["m1"].total_live_s = 98.0  # 98% > 95% target
        assert tracker.should_widen_instead_of_pull("m1") is False

    def test_should_widen_unknown_market(self, tracker: UptimeTracker):
        assert tracker.should_widen_instead_of_pull("unknown") is False

    def test_get_uptime(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].total_tracked_s = 200.0
        tracker._markets["m1"].total_live_s = 190.0
        assert tracker.get_uptime("m1") == 95.0
        assert tracker.get_uptime("unknown") == 0.0

    def test_get_all_uptimes(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].total_tracked_s = 100.0
        tracker._markets["m1"].total_live_s = 90.0

        tracker.register_market("m2")
        tracker._markets["m2"].total_tracked_s = 100.0
        tracker._markets["m2"].total_live_s = 100.0

        uptimes = tracker.get_all_uptimes()
        assert uptimes["m1"] == 90.0
        assert uptimes["m2"] == 100.0

    def test_get_stats(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].total_tracked_s = 3600.0
        tracker._markets["m1"].total_live_s = 3500.0
        tracker._markets["m1"].longest_gap_s = 30.0
        tracker._markets["m1"].quotes_live = True

        stats = tracker.get_stats("m1")
        assert stats["uptime_pct"] == pytest.approx(97.22, abs=0.1)
        assert stats["longest_gap_s"] == 30.0
        assert stats["currently_live"] is True

    def test_get_stats_unknown(self, tracker: UptimeTracker):
        assert tracker.get_stats("unknown") == {}

    def test_session_summary(self, tracker: UptimeTracker):
        tracker.register_market("m1")
        tracker._markets["m1"].total_tracked_s = 100.0
        tracker._markets["m1"].total_live_s = 90.0

        tracker.register_market("m2")
        tracker._markets["m2"].total_tracked_s = 100.0
        tracker._markets["m2"].total_live_s = 98.0

        summary = tracker.get_session_summary()
        assert summary["markets_tracked"] == 2
        assert summary["avg_uptime_pct"] == 94.0
        # m1 at 90% is below 95% target
        assert "m1" in summary["below_target"]
        assert "m2" not in summary["below_target"]
