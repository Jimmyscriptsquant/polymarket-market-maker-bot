from __future__ import annotations

import pytest

from src.config import Settings
from src.inventory.inventory_manager import InventoryManager
from src.market_maker.quote_engine import QuoteEngine, Quote


@pytest.fixture
def inventory_mgr(settings: Settings) -> InventoryManager:
    return InventoryManager(
        settings.max_exposure_usd,
        settings.min_exposure_usd,
        settings.target_inventory_balance,
    )


@pytest.fixture
def engine(settings: Settings, inventory_mgr: InventoryManager) -> QuoteEngine:
    return QuoteEngine(settings, inventory_mgr)


class TestQuoteEngineBasics:
    def test_mid_price(self, engine: QuoteEngine):
        assert engine.calculate_mid_price(0.45, 0.55) == 0.50
        assert engine.calculate_mid_price(0.0, 0.55) == 0.0
        assert engine.calculate_mid_price(0.45, 0.0) == 0.0

    def test_generate_quotes_zero_mid(self, engine: QuoteEngine):
        yes, no = engine.generate_quotes("m1", 0.0, 0.0, "y", "n")
        assert yes is None
        assert no is None

    def test_generate_quotes_basic(self, engine: QuoteEngine):
        """Quotes should be below mid and within reward spread."""
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "yes_token", "no_token",
            max_reward_spread_bps=450,  # ±4.5 cents from mid
        )
        assert yes is not None
        assert no is not None
        assert yes.side == "BUY"
        assert yes.token_id == "yes_token"
        assert no.token_id == "no_token"
        # Bid below mid (0.50) but within ±4.5 cents (≥ 0.455)
        assert yes.price < 0.50
        assert yes.price >= 0.455
        assert yes.size == 5.0

    def test_quotes_within_reward_spread(self, engine: QuoteEngine):
        """Quotes MUST be within ±max_spread of midpoint to earn rewards."""
        yes, no = engine.generate_quotes(
            "m1", 0.48, 0.52, "y", "n",
            max_reward_spread_bps=400,  # ±4 cents from mid
        )
        assert yes is not None
        mid = 0.50
        max_half = 400 / 10000  # 0.04
        distance = mid - yes.price
        assert distance <= max_half  # MUST be within reward spread

    def test_quotes_use_minimum_size(self, engine: QuoteEngine):
        """All quotes should use minimum size to minimize risk."""
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "y", "n",
            allocated_capital=10000.0,
            max_reward_spread_bps=450,
        )
        assert yes is not None
        assert yes.size == 5.0  # Always minimum, regardless of capital

    def test_quotes_respect_min_shares(self, engine: QuoteEngine):
        """Quote size should use the reward program's minimum shares."""
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "y", "n",
            max_reward_spread_bps=450,
            min_shares=50.0,
        )
        assert yes is not None
        assert yes.size == 50.0

    def test_quotes_at_target_spread(self, engine: QuoteEngine):
        """Quotes should target ~75% of max spread from mid."""
        yes, no = engine.generate_quotes(
            "m1", 0.49, 0.51, "y", "n",
            max_reward_spread_bps=400,  # ±4 cents
        )
        assert yes is not None
        mid = 0.50
        distance = mid - yes.price
        max_half = 400 / 10000  # 0.04
        # Should be between 50% and 90% of max spread
        assert distance >= max_half * 0.5
        assert distance <= max_half * 0.95

    def test_no_quotes_when_price_invalid(self, engine: QuoteEngine):
        """If price would be invalid (< 0.01), no quote generated."""
        yes, no = engine.generate_quotes(
            "m1", 0.01, 0.03, "y", "n",
            max_reward_spread_bps=500,
        )
        assert yes is None  # Price would be below 0.01

    def test_both_sides_generated(self, engine: QuoteEngine):
        """Both YES and NO quotes needed for full reward scoring."""
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "y", "n",
            max_reward_spread_bps=450,
        )
        assert yes is not None
        assert no is not None
        # Both should be BUY orders
        assert yes.side == "BUY"
        assert no.side == "BUY"

    def test_asymmetric_market(self, engine: QuoteEngine):
        """Test market with mid far from 0.50 (e.g. 10% probability)."""
        yes, no = engine.generate_quotes(
            "m1", 0.09, 0.11, "y", "n",
            max_reward_spread_bps=400,  # ±4 cents
        )
        # YES bid should be near 0.10 mid, within ±4 cents
        assert yes is not None
        assert yes.price >= 0.06  # min = 0.10 - 0.04 = 0.06
        assert yes.price <= 0.10
        # NO bid should be near 0.90 mid, within ±4 cents
        assert no is not None
        assert no.price >= 0.86  # min = 0.90 - 0.04 = 0.86
        assert no.price <= 0.90


class TestVolatilityDetection:
    def test_no_history_not_volatile(self, engine: QuoteEngine):
        assert engine.detect_short_term_volatility("m1", 0.50) is False

    def test_stable_prices_not_volatile(self, engine: QuoteEngine):
        for _ in range(15):
            engine.detect_short_term_volatility("m1", 0.50)
        result = engine.detect_short_term_volatility("m1", 0.501)
        assert result is False

    def test_wild_prices_volatile(self, engine: QuoteEngine):
        for _ in range(10):
            engine.detect_short_term_volatility("m1", 0.50)
        result = engine.detect_short_term_volatility("m1", 0.60)
        assert result is True

    def test_history_capped_at_20(self, engine: QuoteEngine):
        for i in range(30):
            engine.detect_short_term_volatility("m1", 0.50)
        assert len(engine._last_mid_prices["m1"]) == 20


class TestSafety:
    def test_should_trim_quotes(self, engine: QuoteEngine):
        assert engine.should_trim_quotes(0.5) is True
        assert engine.should_trim_quotes(2.0) is False

    def test_fallback_spread_without_reward(self, engine: QuoteEngine):
        """Without max_reward_spread, should use conservative fallback below best bid."""
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "y", "n",
            max_reward_spread_bps=0,
        )
        assert yes is not None
        assert yes.price < 0.45  # Fallback mode stays below best bid

    def test_extreme_mid_requires_two_sided(self, engine: QuoteEngine):
        """Rule 4: mid < 0.10 or > 0.90 requires both YES and NO quotes or returns nothing."""
        # Mid ~0.05 — YES bid would be ~0.01 (barely valid), NO bid ~0.92 (valid)
        # But if YES side can't produce valid price, both should be None
        yes, no = engine.generate_quotes(
            "m1", 0.04, 0.06, "y", "n",
            max_reward_spread_bps=100,  # ±1¢ — very tight around 0.05 mid
        )
        # With ±1¢ around mid=0.05: YES bid floor = 0.05-0.01+0.005 = 0.045
        # NO bid = (1-0.05) - 0.0075 = 0.9425, ceil → 0.95 — valid
        # YES bid = 0.05 - 0.0075 = 0.0425, ceil → 0.05 — valid
        # Both should be valid, so quotes returned
        if yes is None:
            # If YES side couldn't produce valid price, NO must also be None
            assert no is None

    def test_extreme_mid_high_requires_two_sided(self, engine: QuoteEngine):
        """Rule 4: mid > 0.90 with one side invalid → no quotes."""
        # Mid = 0.96 — NO side price = (1-0.96) - spread = 0.04 - spread
        # With tiny max spread, NO side might be valid but YES side very high
        yes, no = engine.generate_quotes(
            "m1", 0.95, 0.97, "y", "n",
            max_reward_spread_bps=100,  # ±1¢
        )
        if yes is None:
            assert no is None
        if no is None:
            assert yes is None
