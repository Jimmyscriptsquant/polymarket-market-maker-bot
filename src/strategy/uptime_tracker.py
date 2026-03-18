from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class MarketUptime:
    """Tracks quote uptime for a single market."""

    market_id: str
    total_tracked_s: float = 0.0
    total_live_s: float = 0.0
    last_check_time: float = 0.0
    quotes_live: bool = False
    consecutive_gaps_s: float = 0.0
    longest_gap_s: float = 0.0

    @property
    def uptime_pct(self) -> float:
        if self.total_tracked_s <= 0:
            return 0.0
        return (self.total_live_s / self.total_tracked_s) * 100.0


class UptimeTracker:
    """Tracks per-market quote uptime to maximize reward competitiveness.

    Most LP reward programs weight time-in-market heavily. This tracker
    monitors how long quotes are live vs. pulled, enabling:
    - Uptime % reporting per market
    - Alerts when uptime drops below target
    - Decision to widen spreads instead of pulling quotes during volatility
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._markets: dict[str, MarketUptime] = {}
        self._session_start = time.time()

    def register_market(self, market_id: str) -> None:
        if market_id not in self._markets:
            self._markets[market_id] = MarketUptime(
                market_id=market_id,
                last_check_time=time.time(),
            )

    def record_tick(self, market_id: str, has_live_quotes: bool) -> None:
        """Call on every quote refresh cycle to track uptime."""
        now = time.time()
        market = self._markets.get(market_id)
        if not market:
            self.register_market(market_id)
            market = self._markets[market_id]

        elapsed = now - market.last_check_time if market.last_check_time > 0 else 0
        market.last_check_time = now
        market.total_tracked_s += elapsed

        if has_live_quotes:
            market.total_live_s += elapsed
            market.quotes_live = True
            market.consecutive_gaps_s = 0.0
        else:
            market.quotes_live = False
            market.consecutive_gaps_s += elapsed
            market.longest_gap_s = max(market.longest_gap_s, market.consecutive_gaps_s)

    def should_widen_instead_of_pull(self, market_id: str) -> bool:
        """Recommend widening spread instead of pulling quotes to maintain uptime."""
        market = self._markets.get(market_id)
        if not market:
            return False

        # If uptime is below target, prefer widening over pulling
        if market.uptime_pct < self.settings.uptime_target_pct:
            return True

        return False

    def get_uptime(self, market_id: str) -> float:
        """Get current uptime percentage for a market."""
        market = self._markets.get(market_id)
        return market.uptime_pct if market else 0.0

    def get_all_uptimes(self) -> dict[str, float]:
        """Get uptime percentages for all tracked markets."""
        return {mid: m.uptime_pct for mid, m in self._markets.items()}

    def get_stats(self, market_id: str) -> dict[str, Any]:
        """Get detailed uptime stats for a market."""
        market = self._markets.get(market_id)
        if not market:
            return {}
        return {
            "market_id": market_id,
            "uptime_pct": round(market.uptime_pct, 2),
            "total_tracked_s": round(market.total_tracked_s, 1),
            "total_live_s": round(market.total_live_s, 1),
            "longest_gap_s": round(market.longest_gap_s, 1),
            "currently_live": market.quotes_live,
        }

    def get_session_summary(self) -> dict[str, Any]:
        """Summary of all markets for logging/metrics."""
        session_duration = time.time() - self._session_start
        uptimes = self.get_all_uptimes()
        avg_uptime = sum(uptimes.values()) / len(uptimes) if uptimes else 0.0

        return {
            "session_duration_s": round(session_duration, 1),
            "markets_tracked": len(self._markets),
            "avg_uptime_pct": round(avg_uptime, 2),
            "below_target": [
                mid for mid, pct in uptimes.items()
                if pct < self.settings.uptime_target_pct
            ],
        }
