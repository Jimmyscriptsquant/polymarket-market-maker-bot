from __future__ import annotations

import time
import pytest

from src.config import Settings
from src.risk.adverse_selection import AdverseSelectionGuard


@pytest.fixture
def guard(settings: Settings) -> AdverseSelectionGuard:
    g = AdverseSelectionGuard(settings)
    g.max_exposure_per_market = 500.0
    g.momentum_threshold = 0.03
    g.event_widen_days = 7.0
    g.event_max_multiplier = 5.0
    g.max_one_sided_fills = 3
    return g


class TestDeltaNeutralHedging:
    def test_no_hedge_when_balanced(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 100
        state.no_exposure = 100
        result = guard.compute_hedge("m1")
        assert result is None
        assert state.should_hedge is False

    def test_hedge_when_long_yes(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 300
        state.no_exposure = 50
        result = guard.compute_hedge("m1")
        assert result is not None
        assert result["side"] == "NO"
        assert result["size"] == 250

    def test_hedge_when_long_no(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 50
        state.no_exposure = 400
        result = guard.compute_hedge("m1")
        assert result is not None
        assert result["side"] == "YES"
        assert result["size"] == 350

    def test_no_hedge_small_imbalance(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 105
        state.no_exposure = 100
        result = guard.compute_hedge("m1")
        assert result is None  # < $10 imbalance


class TestFillTriggeredRebalancing:
    def test_record_fill_updates_exposure(self, guard: AdverseSelectionGuard):
        guard.record_fill("m1", "YES", 0.60, 100)
        state = guard.get_state("m1")
        assert state.yes_exposure == 60.0  # 100 * 0.60

    def test_record_fill_triggers_hedge(self, guard: AdverseSelectionGuard):
        result = guard.record_fill("m1", "YES", 0.60, 200)
        # 200 * 0.60 = $120 YES exposure, 0 NO = needs hedge
        assert result is not None
        assert result["side"] == "NO"

    def test_one_sided_fills_widen_spread(self, guard: AdverseSelectionGuard):
        for i in range(4):
            guard.record_fill("m1", "YES", 0.50, 50)
        state = guard.get_state("m1")
        assert state.spread_multiplier >= 2.0

    def test_balanced_fills_no_widen(self, guard: AdverseSelectionGuard):
        guard.record_fill("m1", "YES", 0.50, 50)
        guard.record_fill("m1", "NO", 0.50, 50)
        guard.record_fill("m1", "YES", 0.50, 50)
        guard.record_fill("m1", "NO", 0.50, 50)
        state = guard.get_state("m1")
        assert state.spread_multiplier == 1.0  # No one-sided pattern

    def test_fill_log_tracks_recent(self, guard: AdverseSelectionGuard):
        for i in range(10):
            guard.record_fill("m1", "YES", 0.50, 10)
        state = guard.get_state("m1")
        assert len(state.recent_fills) <= 10  # Pruned to 5 min window


class TestEventAwareWidening:
    def test_no_end_date_no_widen(self, guard: AdverseSelectionGuard):
        mult = guard.get_event_spread_multiplier("m1")
        assert mult == 1.0

    def test_far_future_no_widen(self, guard: AdverseSelectionGuard):
        guard.set_end_date("m1", time.time() + 86400 * 30)  # 30 days out
        mult = guard.get_event_spread_multiplier("m1")
        assert mult == 1.0

    def test_widen_near_resolution(self, guard: AdverseSelectionGuard):
        guard.set_end_date("m1", time.time() + 86400 * 3)  # 3 days out
        mult = guard.get_event_spread_multiplier("m1")
        assert mult > 1.0
        assert mult < guard.event_max_multiplier

    def test_max_widen_at_resolution(self, guard: AdverseSelectionGuard):
        guard.set_end_date("m1", time.time() + 3600)  # 1 hour out
        mult = guard.get_event_spread_multiplier("m1")
        assert mult >= guard.event_max_multiplier * 0.9  # Near max

    def test_block_past_resolution(self, guard: AdverseSelectionGuard):
        guard.set_end_date("m1", time.time() - 100)  # Already past
        mult = guard.get_event_spread_multiplier("m1")
        state = guard.get_state("m1")
        assert state.blocked is True
        assert mult == guard.event_max_multiplier

    def test_linear_ramp(self, guard: AdverseSelectionGuard):
        """Multiplier should increase linearly as resolution approaches."""
        guard.set_end_date("m1", time.time() + 86400 * 3.5)  # 3.5 days = 50% through
        mult = guard.get_event_spread_multiplier("m1")
        # At 50% through 7-day window: mult should be ~3.0 (midpoint of 1.0 to 5.0)
        assert 2.5 < mult < 3.5


class TestMomentumDetection:
    def test_no_data_no_momentum(self, guard: AdverseSelectionGuard):
        mult = guard.detect_momentum("m1")
        assert mult == 1.0

    def test_stable_prices_no_momentum(self, guard: AdverseSelectionGuard):
        now = time.time()
        state = guard.get_state("m1")
        for i in range(10):
            state.prices.append((now - 100 + i * 10, 0.50))
        mult = guard.detect_momentum("m1")
        assert mult == 1.0

    def test_big_move_triggers_momentum(self, guard: AdverseSelectionGuard):
        now = time.time()
        state = guard.get_state("m1")
        # Price moves from 0.50 to 0.55 (10% move > 3% threshold)
        # Need enough points on both sides for median to reflect the move
        state.prices.append((now - 120, 0.50))
        state.prices.append((now - 100, 0.50))
        state.prices.append((now - 80, 0.50))
        state.prices.append((now - 20, 0.55))
        state.prices.append((now - 10, 0.55))
        state.prices.append((now, 0.55))
        mult = guard.detect_momentum("m1")
        assert mult > 1.0

    def test_momentum_capped(self, guard: AdverseSelectionGuard):
        now = time.time()
        state = guard.get_state("m1")
        state.prices.append((now - 60, 0.30))
        state.prices.append((now - 50, 0.31))
        state.prices.append((now - 40, 0.35))
        state.prices.append((now - 30, 0.40))
        state.prices.append((now - 20, 0.50))
        state.prices.append((now, 0.70))  # 133% move
        mult = guard.detect_momentum("m1")
        assert mult <= 4.0  # Capped


class TestExposureCaps:
    def test_within_cap(self, guard: AdverseSelectionGuard):
        ok, reason = guard.check_exposure_cap("m1", "YES", 100)
        assert ok is True

    def test_exceed_cap(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 450
        ok, reason = guard.check_exposure_cap("m1", "YES", 100)
        assert ok is False
        assert "exposure_cap" in reason

    def test_negative_exposure_cap(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.no_exposure = 450
        ok, reason = guard.check_exposure_cap("m1", "NO", 100)
        assert ok is False

    def test_balanced_within_cap(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 400
        state.no_exposure = 400
        # Net = 0, adding YES makes net = 100, within 500 cap
        ok, _ = guard.check_exposure_cap("m1", "YES", 100)
        assert ok is True


class TestCombinedGuardCheck:
    def test_normal_quote_allowed(self, guard: AdverseSelectionGuard):
        ok, mult, reason = guard.check_quote("m1", "YES", 0.50, 100, 0.50)
        assert ok is True
        assert mult >= 1.0
        assert reason == "OK"

    def test_blocked_market(self, guard: AdverseSelectionGuard):
        guard.set_end_date("m1", time.time() - 100)
        guard.get_event_spread_multiplier("m1")  # Sets blocked
        ok, mult, reason = guard.check_quote("m1", "YES", 0.50, 100, 0.50)
        assert ok is False
        assert "resolution" in reason

    def test_exposure_cap_blocks(self, guard: AdverseSelectionGuard):
        state = guard.get_state("m1")
        state.yes_exposure = 480
        ok, mult, reason = guard.check_quote("m1", "YES", 0.50, 100, 0.50)
        assert ok is False
        assert "exposure_cap" in reason

    def test_momentum_widens_spread(self, guard: AdverseSelectionGuard):
        now = time.time()
        state = guard.get_state("m1")
        for i in range(5):
            state.prices.append((now - 100 + i * 20, 0.50))
        state.prices.append((now - 5, 0.56))  # 12% move

        ok, mult, reason = guard.check_quote("m1", "YES", 0.50, 50, 0.56)
        assert ok is True
        assert mult > 1.0  # Spread should be widened


class TestGetStatus:
    def test_status_output(self, guard: AdverseSelectionGuard):
        guard.record_fill("m1", "YES", 0.60, 100)
        status = guard.get_status("m1")
        assert status["market_id"] == "m1"
        assert status["yes_exposure"] == 60.0
        assert status["net_exposure"] == 60.0
        assert status["recent_fills_5m"] == 1
        assert "blocked" in status
        assert "momentum_mult" in status

    def test_reset_market(self, guard: AdverseSelectionGuard):
        guard.record_fill("m1", "YES", 0.60, 100)
        guard.reset_market("m1")
        state = guard.get_state("m1")
        assert state.yes_exposure == 0.0
        assert len(state.recent_fills) == 0
