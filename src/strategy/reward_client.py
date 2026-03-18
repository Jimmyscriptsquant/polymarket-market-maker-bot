from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from src.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class MarketRewardInfo:
    """Reward program data for a single market."""

    market_id: str
    question: str = ""
    rate_per_day: float = 0.0
    max_spread_bps: float = 350.0
    min_shares: float = 50.0
    num_lps: int = 0
    competition_score: float = 1.0  # higher = more competitive
    yes_token_id: str = ""
    no_token_id: str = ""
    best_bid: float = 0.0
    best_ask: float = 1.0
    volume_24h: float = 0.0
    active: bool = True


class RewardClient:
    """Fetches reward program data and market metadata from Polymarket APIs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = settings.polymarket_api_url
        self.rewards_url = settings.rewards_api_url

    async def fetch_reward_markets(self) -> list[MarketRewardInfo]:
        """Fetch all markets with active reward programs."""
        try:
            response = await self.client.get(
                f"{self.rewards_url}/markets",
                params={"active": "true"},
            )
            response.raise_for_status()
            data = response.json()

            markets = []
            for item in data if isinstance(data, list) else data.get("markets", []):
                info = self._parse_reward_market(item)
                if info and info.rate_per_day >= self.settings.min_reward_rate_daily:
                    markets.append(info)

            logger.info("reward_markets_fetched", count=len(markets))
            return markets
        except Exception as e:
            logger.error("reward_markets_fetch_failed", error=str(e))
            return []

    async def fetch_market_competition(self, market_id: str) -> dict[str, Any]:
        """Fetch competition/leaderboard data for a specific market."""
        try:
            response = await self.client.get(
                f"{self.rewards_url}/competitiveness",
                params={"market": market_id},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("competition_fetch_failed", market_id=market_id, error=str(e))
            return {}

    async def fetch_orderbook_snapshot(self, market_id: str) -> dict[str, Any]:
        """Fetch current orderbook for volatility/spread estimation."""
        try:
            response = await self.client.get(
                f"{self.base_url}/book", params={"market": market_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("orderbook_snapshot_failed", market_id=market_id, error=str(e))
            return {}

    def _parse_reward_market(self, raw: dict[str, Any]) -> MarketRewardInfo | None:
        """Parse raw API response into MarketRewardInfo."""
        try:
            market_id = raw.get("condition_id") or raw.get("market_id") or raw.get("id", "")
            if not market_id:
                return None

            tokens = raw.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), {})
            no_token = next((t for t in tokens if t.get("outcome") == "No"), {})

            return MarketRewardInfo(
                market_id=market_id,
                question=raw.get("question", ""),
                rate_per_day=float(raw.get("rewards_daily_rate", 0) or raw.get("rate_per_day", 0)),
                max_spread_bps=float(raw.get("max_spread", 350) or 350),
                min_shares=float(raw.get("min_size", 50) or 50),
                num_lps=int(raw.get("num_lps", 0) or 0),
                competition_score=float(raw.get("competition_score", 1.0) or 1.0),
                yes_token_id=yes_token.get("token_id", ""),
                no_token_id=no_token.get("token_id", ""),
                volume_24h=float(raw.get("volume_24h", 0) or 0),
                active=raw.get("active", True),
            )
        except (ValueError, KeyError) as e:
            logger.warning("market_parse_failed", error=str(e), raw_keys=list(raw.keys()))
            return None

    async def close(self):
        await self.client.aclose()
