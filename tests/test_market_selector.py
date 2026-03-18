from __future__ import annotations

import time
import pytest

from src.config import Settings
from src.strategy.market_selector import MarketSelector, ScoredMarket
from src.strategy.reward_client import MarketRewardInfo, RewardClient
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def reward_client(settings: Settings) -> RewardClient:
    client = RewardClient(settings)
    client.client = MagicMock()  # Don't make real HTTP calls
    return client


@pytest.fixture
def selector(settings: Settings, reward_client: RewardClient) -> MarketSelector:
    return MarketSelector(settings, reward_client)


def _make_market(
    market_id: str,
    rate: float = 100.0,
    competition: float = 1.0,
    volume: float = 10000.0,
) -> MarketRewardInfo:
    return MarketRewardInfo(
        market_id=market_id,
        question=f"Test market {market_id}",
        rate_per_day=rate,
        competition_score=competition,
        volume_24h=volume,
        yes_token_id=f"yes_{market_id}",
        no_token_id=f"no_{market_id}",
    )


class TestMarketSelector:
    def test_should_rotate_initially(self, selector: MarketSelector):
        """First call should always want to rotate."""
        assert selector.should_rotate() is True

    def test_should_not_rotate_after_scan(self, selector: MarketSelector):
        selector._last_rotation = time.time()
        assert selector.should_rotate() is False

    def test_should_rotate_after_interval(self, selector: MarketSelector):
        selector._last_rotation = time.time() - 7200  # 2h ago
        assert selector.should_rotate() is True

    def test_record_price_and_volatility(self, selector: MarketSelector):
        """Record prices and verify volatility estimation."""
        now = time.time()
        # Simulate stable prices -> low vol
        for i in range(20):
            selector._price_history.setdefault("m1", []).append(
                (now - 1200 + i * 60, 0.50 + 0.0001 * (i % 2))
            )

        vol = selector._estimate_volatility("m1")
        assert vol > 0
        assert vol < 2.0  # Should be finite for near-stable prices

    def test_estimate_volatility_insufficient_data(self, selector: MarketSelector):
        """Falls back to default with < 10 observations."""
        selector._price_history["m2"] = [(time.time(), 0.5)]
        vol = selector._estimate_volatility("m2")
        assert vol == 0.05  # default

    def test_estimate_volatility_volatile(self, selector: MarketSelector):
        """High vol prices should produce higher estimate."""
        now = time.time()
        prices = [0.3, 0.6, 0.35, 0.7, 0.25, 0.65, 0.3, 0.55, 0.4, 0.7, 0.35, 0.6]
        for i, p in enumerate(prices):
            selector._price_history.setdefault("m3", []).append(
                (now - 720 + i * 60, p)
            )

        vol = selector._estimate_volatility("m3")
        # Volatile series should have higher vol than default
        assert vol > 0.05

    @pytest.mark.asyncio
    async def test_scan_and_rank_empty(self, selector: MarketSelector):
        """No markets -> returns existing selection."""
        selector.reward_client.fetch_reward_markets = AsyncMock(return_value=[])
        result = await selector.scan_and_rank()
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_and_rank_selects_top_n(self, selector: MarketSelector):
        """Should rank by efficiency and select top N."""
        markets = [
            _make_market("high", rate=500.0, competition=1.0),
            _make_market("med", rate=100.0, competition=2.0),
            _make_market("low", rate=10.0, competition=5.0),
        ]
        selector.reward_client.fetch_reward_markets = AsyncMock(return_value=markets)

        result = await selector.scan_and_rank()

        assert len(result) == 3
        # Highest efficiency first
        assert result[0].market_id == "high"
        assert result[0].reward_efficiency > result[1].reward_efficiency
        assert result[1].reward_efficiency > result[2].reward_efficiency

    @pytest.mark.asyncio
    async def test_capital_allocation_proportional(self, selector: MarketSelector):
        """Capital should be allocated proportional to efficiency."""
        markets = [
            _make_market("a", rate=300.0, competition=1.0),
            _make_market("b", rate=100.0, competition=1.0),
        ]
        selector.reward_client.fetch_reward_markets = AsyncMock(return_value=markets)

        result = await selector.scan_and_rank()

        total_allocated = sum(m.allocated_capital for m in result)
        assert abs(total_allocated - selector.settings.capital_pool_usd) < 1.0

        # Higher efficiency gets more capital
        assert result[0].allocated_capital > result[1].allocated_capital

    @pytest.mark.asyncio
    async def test_volatile_markets_filtered(self, selector: MarketSelector):
        """Markets above volatility threshold should be excluded."""
        markets = [
            _make_market("calm", rate=100.0),
            _make_market("wild", rate=200.0),
        ]
        selector.reward_client.fetch_reward_markets = AsyncMock(return_value=markets)

        # Give "wild" a very volatile price history
        now = time.time()
        for i in range(20):
            p = 0.2 if i % 2 == 0 else 0.9
            selector._price_history.setdefault("wild", []).append(
                (now - 1200 + i * 60, p)
            )

        result = await selector.scan_and_rank()

        market_ids = [m.market_id for m in result]
        assert "calm" in market_ids
        # "wild" should be filtered out due to high volatility
        assert "wild" not in market_ids

    def test_get_market_capital(self, selector: MarketSelector):
        scored = ScoredMarket(
            info=_make_market("test"), reward_efficiency=10.0, allocated_capital=5000.0
        )
        selector._selected_markets = [scored]
        assert selector.get_market_capital("test") == 5000.0
        assert selector.get_market_capital("unknown") == 0.0

    def test_get_breakeven_spread(self, selector: MarketSelector):
        scored = ScoredMarket(
            info=_make_market("test"), breakeven_spread_bps=25.0
        )
        selector._selected_markets = [scored]
        assert selector.get_breakeven_spread("test") == 25.0
        assert selector.get_breakeven_spread("unknown") == selector.settings.min_spread_bps
