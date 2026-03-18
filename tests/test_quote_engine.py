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

    def test_bid_ask_prices(self, engine: QuoteEngine):
        bid = engine.calculate_bid_price(0.50, 10)
        ask = engine.calculate_ask_price(0.50, 10)
        assert bid < 0.50
        assert ask > 0.50
        assert bid == pytest.approx(0.4995)
        assert ask == pytest.approx(0.5005)

    def test_generate_quotes_zero_mid(self, engine: QuoteEngine):
        yes, no = engine.generate_quotes("m1", 0.0, 0.0, "y", "n")
        assert yes is None
        assert no is None

    def test_generate_quotes_basic(self, engine: QuoteEngine):
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "yes_token", "no_token"
        )
        assert yes is not None
        assert no is not None
        assert yes.side == "BUY"
        assert yes.token_id == "yes_token"
        assert no.token_id == "no_token"
        assert yes.price < 0.50  # bid below mid
        assert yes.size > 0

    def test_generate_quotes_with_allocated_capital(self, engine: QuoteEngine):
        """Allocated capital should cap quote size to 10% of capital."""
        yes, no = engine.generate_quotes(
            "m1", 0.45, 0.55, "y", "n", allocated_capital=500.0
        )
        assert yes is not None
        # 10% of 500 = 50, less than default_size 100
        assert yes.size <= 50.0


class TestRewardAdjustedSpread:
    def test_no_reward_returns_base(self, engine: QuoteEngine):
        spread = engine.calculate_reward_adjusted_spread("m1", 10, reward_rate_daily=0.0)
        assert spread == 10

    def test_reward_disabled_returns_base(self, engine: QuoteEngine):
        engine.settings.reward_adjusted_spreads = False
        spread = engine.calculate_reward_adjusted_spread("m1", 10, reward_rate_daily=100.0)
        assert spread == 10

    def test_reward_tightens_spread(self, engine: QuoteEngine):
        """High reward rate should tighten the spread."""
        base = 50
        spread = engine.calculate_reward_adjusted_spread(
            "m1", base, reward_rate_daily=500.0, volume_24h=100000.0
        )
        assert spread < base

    def test_reward_respects_floor(self, engine: QuoteEngine):
        """Spread should never go below min_spread_bps."""
        spread = engine.calculate_reward_adjusted_spread(
            "m1", 10, reward_rate_daily=999999.0, volume_24h=100000.0
        )
        assert spread >= engine.settings.min_spread_bps

    def test_reward_respects_breakeven(self, engine: QuoteEngine):
        """Spread should not go below breakeven."""
        spread = engine.calculate_reward_adjusted_spread(
            "m1", 50, reward_rate_daily=500.0, volume_24h=100000.0,
            breakeven_spread_bps=30.0,
        )
        assert spread >= 30

    def test_moderate_reward(self, engine: QuoteEngine):
        """Moderate reward should partially tighten."""
        spread = engine.calculate_reward_adjusted_spread(
            "m1", 100, reward_rate_daily=50.0, volume_24h=50000.0
        )
        assert engine.settings.min_spread_bps <= spread <= 100


class TestVolatilityDetection:
    def test_no_history_not_volatile(self, engine: QuoteEngine):
        assert engine.detect_short_term_volatility("m1", 0.50) is False

    def test_stable_prices_not_volatile(self, engine: QuoteEngine):
        for _ in range(15):
            engine.detect_short_term_volatility("m1", 0.50)
        result = engine.detect_short_term_volatility("m1", 0.501)
        assert result is False

    def test_wild_prices_volatile(self, engine: QuoteEngine):
        # Build history
        for _ in range(10):
            engine.detect_short_term_volatility("m1", 0.50)
        # Big move
        result = engine.detect_short_term_volatility("m1", 0.60)
        assert result is True

    def test_history_capped_at_20(self, engine: QuoteEngine):
        for i in range(30):
            engine.detect_short_term_volatility("m1", 0.50)
        assert len(engine._last_mid_prices["m1"]) == 20


class TestWidenForUptime:
    def test_widen_multiplies_spread(self, engine: QuoteEngine):
        """Widening for uptime should multiply spread."""
        yes_normal, _ = engine.generate_quotes("m1", 0.45, 0.55, "y", "n")
        yes_wide, _ = engine.generate_quotes(
            "m1", 0.45, 0.55, "y", "n", widen_for_uptime=True
        )
        assert yes_normal is not None
        assert yes_wide is not None
        # Wide quote should have lower bid (wider spread)
        assert yes_wide.price < yes_normal.price


class TestInventorySkew:
    def test_adjust_for_inventory_skew_neutral(self, engine: QuoteEngine):
        size = engine.adjust_for_inventory_skew(100.0, 0.5, "BUY")
        assert size == 100.0

    def test_adjust_for_inventory_skew_long(self, engine: QuoteEngine):
        engine.inventory_manager.inventory.net_exposure_usd = 5000
        engine.inventory_manager.inventory.total_value_usd = 6000
        engine.inventory_manager.inventory.yes_position = 100
        size = engine.adjust_for_inventory_skew(100.0, 0.5, "BUY")
        assert size == 50.0  # Reduced because long + buying

    def test_should_trim_quotes(self, engine: QuoteEngine):
        assert engine.should_trim_quotes(0.5) is True
        assert engine.should_trim_quotes(2.0) is False
