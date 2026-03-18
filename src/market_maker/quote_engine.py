from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.config import Settings
from src.inventory.inventory_manager import InventoryManager

logger = structlog.get_logger(__name__)


@dataclass
class Quote:
    side: str
    price: float
    size: float
    market: str
    token_id: str


class QuoteEngine:
    def __init__(self, settings: Settings, inventory_manager: InventoryManager):
        self.settings = settings
        self.inventory_manager = inventory_manager
        self._last_mid_prices: dict[str, list[float]] = {}  # market -> recent mids for vol detection

    def calculate_bid_price(self, mid_price: float, spread_bps: int) -> float:
        return mid_price * (1 - spread_bps / 10000)

    def calculate_ask_price(self, mid_price: float, spread_bps: int) -> float:
        return mid_price * (1 + spread_bps / 10000)

    def calculate_mid_price(self, best_bid: float, best_ask: float) -> float:
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        return (best_bid + best_ask) / 2.0

    def calculate_reward_adjusted_spread(
        self,
        market_id: str,
        base_spread_bps: int,
        reward_rate_daily: float = 0.0,
        volume_24h: float = 0.0,
        breakeven_spread_bps: float = 0.0,
    ) -> int:
        """Calculate optimal spread accounting for reward subsidies.

        Key insight: if rewards subsidize your spread enough, you can quote
        tighter than fair and still profit.

        adjusted_spread = max(
            breakeven_spread (from market selector),
            adverse_selection - reward_subsidy_per_fill,
            min_spread_bps floor
        )
        """
        if not self.settings.reward_adjusted_spreads or reward_rate_daily <= 0:
            return base_spread_bps

        # Estimate expected fills per day based on our share of volume
        # Conservative: assume we capture 5% of daily volume
        expected_daily_fills = max((volume_24h * 0.05) / self.settings.default_size, 1.0)

        # Reward subsidy per fill in USD
        reward_per_fill = reward_rate_daily / expected_daily_fills

        # Convert subsidy to bps: subsidy / order_size * 10000
        subsidy_bps = (reward_per_fill / self.settings.default_size) * 10000

        # Tighten spread by the subsidy amount
        adjusted = max(
            base_spread_bps - int(subsidy_bps),
            int(breakeven_spread_bps) if breakeven_spread_bps > 0 else 1,
            self.settings.min_spread_bps,
        )

        if adjusted < base_spread_bps:
            logger.debug(
                "spread_tightened_by_rewards",
                market_id=market_id,
                base_bps=base_spread_bps,
                adjusted_bps=adjusted,
                subsidy_bps=round(subsidy_bps, 1),
                reward_per_fill=round(reward_per_fill, 4),
            )

        return adjusted

    def detect_short_term_volatility(self, market_id: str, mid_price: float) -> bool:
        """Detect if recent price action is volatile (for uptime-aware widening)."""
        history = self._last_mid_prices.setdefault(market_id, [])
        history.append(mid_price)

        # Keep last 20 observations
        if len(history) > 20:
            self._last_mid_prices[market_id] = history[-20:]
            history = self._last_mid_prices[market_id]

        if len(history) < 5:
            return False

        # Check max move in recent window
        recent = history[-10:]
        max_price = max(recent)
        min_price = min(recent)

        if min_price <= 0:
            return False

        move_pct = (max_price - min_price) / min_price
        return move_pct > self.settings.volatility_widen_threshold

    def generate_quotes(
        self,
        market_id: str,
        best_bid: float,
        best_ask: float,
        yes_token_id: str,
        no_token_id: str,
        reward_rate_daily: float = 0.0,
        volume_24h: float = 0.0,
        breakeven_spread_bps: float = 0.0,
        allocated_capital: float = 0.0,
        widen_for_uptime: bool = False,
    ) -> tuple[Quote | None, Quote | None]:
        mid_price = self.calculate_mid_price(best_bid, best_ask)

        if mid_price == 0:
            return (None, None)

        # Track price for volatility detection
        is_volatile = self.detect_short_term_volatility(market_id, mid_price)

        # Base spread
        spread_bps = self.settings.min_spread_bps

        # Apply reward adjustment (tighten if rewards subsidize)
        spread_bps = self.calculate_reward_adjusted_spread(
            market_id=market_id,
            base_spread_bps=spread_bps,
            reward_rate_daily=reward_rate_daily,
            volume_24h=volume_24h,
            breakeven_spread_bps=breakeven_spread_bps,
        )

        # Widen spread during volatility instead of pulling quotes (uptime strategy)
        if is_volatile or widen_for_uptime:
            spread_bps = int(spread_bps * self.settings.wide_spread_multiplier)
            logger.info(
                "spread_widened_for_uptime",
                market_id=market_id,
                spread_bps=spread_bps,
                volatile=is_volatile,
                uptime_widen=widen_for_uptime,
            )

        bid_price = self.calculate_bid_price(mid_price, spread_bps)
        ask_price = self.calculate_ask_price(mid_price, spread_bps)

        # Size: use allocated capital if available, else default
        if allocated_capital > 0:
            base_size = min(allocated_capital * 0.1, self.settings.default_size)  # 10% per quote
        else:
            base_size = self.settings.default_size

        yes_size = self.inventory_manager.get_quote_size_yes(base_size, mid_price)
        no_size = self.inventory_manager.get_quote_size_no(base_size, mid_price)

        yes_quote = None
        no_quote = None

        if self.inventory_manager.can_quote_yes(yes_size):
            yes_quote = Quote(
                side="BUY",
                price=bid_price,
                size=yes_size,
                market=market_id,
                token_id=yes_token_id,
            )

        if self.inventory_manager.can_quote_no(no_size):
            no_quote = Quote(
                side="BUY",
                price=1.0 - ask_price,
                size=no_size,
                market=market_id,
                token_id=no_token_id,
            )

        return (yes_quote, no_quote)

    def adjust_for_inventory_skew(self, base_size: float, price: float, side: str) -> float:
        skew = self.inventory_manager.inventory.get_skew()

        if skew > 0.2:
            if side == "BUY" and self.inventory_manager.inventory.net_exposure_usd > 0:
                return base_size * 0.5
            elif side == "SELL" and self.inventory_manager.inventory.net_exposure_usd < 0:
                return base_size * 0.5

        return base_size

    def should_trim_quotes(self, time_to_close_hours: float) -> bool:
        if time_to_close_hours < 1.0:
            return True
        return False
