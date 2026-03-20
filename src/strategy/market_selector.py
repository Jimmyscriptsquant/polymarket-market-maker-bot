from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import Settings
from src.strategy.reward_client import MarketRewardInfo, RewardClient

logger = structlog.get_logger(__name__)


@dataclass
class ScoredMarket:
    """A market ranked by reward efficiency."""

    info: MarketRewardInfo
    reward_efficiency: float = 0.0
    realized_volatility: float = 0.0
    allocated_capital: float = 0.0
    breakeven_spread_bps: float = 0.0

    @property
    def market_id(self) -> str:
        return self.info.market_id


class MarketSelector:
    """Ranks and selects optimal markets for reward-optimized LP.

    Core formula:
        reward_efficiency = rate_per_day / (competition_score * realized_volatility)

    Higher efficiency = more reward per unit of risk and competition.
    """

    def __init__(self, settings: Settings, reward_client: RewardClient):
        self.settings = settings
        self.reward_client = reward_client
        self._price_history: dict[str, list[tuple[float, float]]] = {}  # market_id -> [(ts, mid)]
        self._last_rotation: float = 0.0
        self._selected_markets: list[ScoredMarket] = []

    @property
    def selected_markets(self) -> list[ScoredMarket]:
        return self._selected_markets

    def should_rotate(self) -> bool:
        elapsed = time.time() - self._last_rotation
        return elapsed >= self.settings.market_rotation_interval_s

    async def scan_and_rank(self) -> list[ScoredMarket]:
        """Fetch all reward markets, score them, and select top N.

        Ranking formula prioritizes:
        1. Low competition (orderbook depth) — we capture a bigger share of the pool
        2. High reward rate — bigger pool to share
        3. Wide max spread — easier to stay in range, less fill risk
        4. Avoid extreme midpoints (<0.10 or >0.90) — two-sided is harder there
        """
        reward_markets = await self.reward_client.fetch_reward_markets()
        if not reward_markets:
            logger.warning("no_reward_markets_found")
            return self._selected_markets

        # Pre-filter: sort by rate and only check book depth for top candidates
        # This avoids hundreds of API calls for markets we'd never select
        reward_markets.sort(key=lambda m: m.rate_per_day, reverse=True)
        candidates = reward_markets[:self.settings.target_markets_count * 8]

        scored: list[ScoredMarket] = []
        for market in candidates:
            vol = self._estimate_volatility(market.market_id)

            # Skip markets above volatility threshold
            if vol > self.settings.max_volatility_threshold:
                logger.debug("market_too_volatile", market_id=market.market_id, vol=vol)
                continue

            # Fetch orderbook to measure actual competition depth
            book_depth = await self._get_book_depth(market.market_id)

            # Competition: use orderbook depth as primary signal (more reliable than API score)
            # Low depth = low competition = we capture bigger share of reward pool
            # Floor at 10 to avoid division by zero
            comp = max(book_depth, 10.0)

            vol_adj = max(vol, 0.01)

            # Core efficiency: reward rate / competition depth
            # Higher = more reward per unit of competition we face
            efficiency = market.rate_per_day / (comp * vol_adj)

            # Bonus for wider max spread (easier to stay in range)
            spread_bonus = market.max_spread_bps / 450.0  # normalize: 450bps = 1.0
            efficiency *= spread_bonus

            # Penalty for extreme midpoints (harder to quote both sides profitably)
            mid = (market.best_bid + market.best_ask) / 2.0 if market.best_bid > 0 and market.best_ask < 1.0 else 0.5
            if mid < 0.10 or mid > 0.90:
                efficiency *= 0.3  # heavy penalty — two-sided is mandatory and risky

            # Estimate breakeven spread accounting for reward subsidy
            expected_daily_fills = max(market.volume_24h / 100.0, 1.0)
            reward_per_fill = market.rate_per_day / expected_daily_fills
            adverse_selection_bps = vol_adj * 10000
            breakeven = max(adverse_selection_bps - (reward_per_fill * 10000), 1.0)

            scored.append(
                ScoredMarket(
                    info=market,
                    reward_efficiency=efficiency,
                    realized_volatility=vol,
                    breakeven_spread_bps=breakeven,
                )
            )

        # Sort by efficiency descending
        scored.sort(key=lambda s: s.reward_efficiency, reverse=True)

        # Select top N
        top_n = scored[: self.settings.target_markets_count]

        # Allocate capital proportional to efficiency
        self._allocate_capital(top_n)

        self._selected_markets = top_n
        self._last_rotation = time.time()

        logger.info(
            "markets_ranked",
            total_scanned=len(reward_markets),
            eligible=len(scored),
            selected=len(top_n),
            top_market=top_n[0].market_id if top_n else "none",
            top_efficiency=round(top_n[0].reward_efficiency, 2) if top_n else 0,
        )

        return top_n

    def _allocate_capital(self, markets: list[ScoredMarket]) -> None:
        """Allocate capital pool proportional to reward efficiency."""
        total_efficiency = sum(m.reward_efficiency for m in markets)
        if total_efficiency <= 0:
            return

        pool = self.settings.capital_pool_usd
        for market in markets:
            share = market.reward_efficiency / total_efficiency
            market.allocated_capital = round(pool * share, 2)

    async def _get_book_depth(self, market_id: str) -> float:
        """Fetch orderbook and return total resting size (bid + ask depth).

        This is the best proxy for competition — markets with thin books
        mean fewer LPs competing for the reward pool.
        """
        try:
            book = await self.reward_client.fetch_orderbook_snapshot(market_id)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            bid_depth = sum(float(b.get("size", 0)) for b in bids)
            ask_depth = sum(float(a.get("size", 0)) for a in asks)
            return bid_depth + ask_depth
        except Exception:
            return 1000.0  # Conservative default if fetch fails

    def record_price(self, market_id: str, mid_price: float) -> None:
        """Record a price observation for volatility estimation."""
        now = time.time()
        history = self._price_history.setdefault(market_id, [])
        history.append((now, mid_price))

        # Keep last 24h of data (1-minute samples = ~1440 points max)
        cutoff = now - 86400
        self._price_history[market_id] = [(t, p) for t, p in history if t >= cutoff]

    def _estimate_volatility(self, market_id: str) -> float:
        """Estimate realized volatility from price history (log returns std dev).

        Returns annualized volatility as a decimal (e.g., 0.05 = 5%).
        Falls back to a default if insufficient data.
        """
        history = self._price_history.get(market_id, [])

        if len(history) < 10:
            # Not enough data — return moderate default
            return 0.05

        prices = [p for _, p in history]
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i] > 0 and prices[i - 1] > 0:
                log_returns.append(math.log(prices[i] / prices[i - 1]))

        if len(log_returns) < 5:
            return 0.05

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        std_dev = math.sqrt(variance)

        # Estimate samples per day from timestamps
        duration_s = history[-1][0] - history[0][0]
        if duration_s <= 0:
            return 0.05
        samples_per_day = (len(history) / duration_s) * 86400

        # Annualize: daily_vol * sqrt(365)
        daily_vol = std_dev * math.sqrt(samples_per_day)
        annual_vol = daily_vol * math.sqrt(365)

        return min(annual_vol, 2.0)  # Cap at 200%

    def get_market_capital(self, market_id: str) -> float:
        """Get allocated capital for a specific market."""
        for m in self._selected_markets:
            if m.market_id == market_id:
                return m.allocated_capital
        return 0.0

    def get_breakeven_spread(self, market_id: str) -> float:
        """Get reward-adjusted breakeven spread for a market."""
        for m in self._selected_markets:
            if m.market_id == market_id:
                return m.breakeven_spread_bps
        return self.settings.min_spread_bps
