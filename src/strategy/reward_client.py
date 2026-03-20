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
            all_items: list[dict] = []
            next_cursor = ""

            # Paginate through all reward markets
            while True:
                params: dict[str, str] = {"limit": "500"}
                if next_cursor:
                    params["next_cursor"] = next_cursor

                response = await self.client.get(
                    f"{self.rewards_url}/markets/current",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    all_items.extend(data)
                    break
                elif isinstance(data, dict):
                    items = data.get("data", [])
                    all_items.extend(items)
                    next_cursor = data.get("next_cursor", "")
                    if not next_cursor or next_cursor == "LTE=" or not items:
                        break
                else:
                    break

            markets = []
            # Allow up to 3x the per-market budget for min_shares
            # (we place min_shares regardless — reward eligibility requires it)
            max_affordable_shares = (self.settings.capital_pool_usd / self.settings.target_markets_count / 0.50) * 3
            for item in all_items:
                info = self._parse_reward_market(item)
                if info and info.rate_per_day >= self.settings.min_reward_rate_daily:
                    # Filter out markets we truly can't afford
                    if info.min_shares > max_affordable_shares:
                        continue
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

    async def fetch_orderbook_snapshot(self, token_id: str) -> dict[str, Any]:
        """Fetch current orderbook by token_id (CLOB /book requires token_id, not market)."""
        try:
            response = await self.client.get(
                f"{self.base_url}/book", params={"token_id": token_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("orderbook_snapshot_failed", token_id=token_id[:16], error=str(e))
            return {}

    def _parse_reward_market(self, raw: dict[str, Any]) -> MarketRewardInfo | None:
        """Parse raw API response into MarketRewardInfo.

        Handles both /rewards/markets/current format:
            {condition_id, rewards_config: [{rate_per_day, ...}], rewards_max_spread, rewards_min_size, total_daily_rate}
        And enriched market format with tokens/question.
        """
        try:
            market_id = raw.get("condition_id") or raw.get("market_id") or raw.get("id", "")
            if not market_id:
                return None

            # Parse rate from rewards_config or direct fields
            rate = float(raw.get("total_daily_rate", 0) or raw.get("native_daily_rate", 0))
            if rate == 0:
                configs = raw.get("rewards_config", [])
                if configs:
                    rate = float(configs[0].get("rate_per_day", 0))
            if rate == 0:
                rate = float(raw.get("rewards_daily_rate", 0) or raw.get("rate_per_day", 0))

            # Max spread: rewards_max_spread is in cents (e.g. 4.5 = 4.5 cents = 450 bps)
            max_spread_cents = float(raw.get("rewards_max_spread", 0) or 0)
            max_spread_bps = max_spread_cents * 100 if max_spread_cents > 0 else 350.0

            min_size = float(raw.get("rewards_min_size", 50) or 50)

            tokens = raw.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), {})
            no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), {})

            return MarketRewardInfo(
                market_id=market_id,
                question=raw.get("question", ""),
                rate_per_day=rate,
                max_spread_bps=max_spread_bps,
                min_shares=min_size,
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
