from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.config import Settings
from src.inventory.inventory_manager import InventoryManager

logger = structlog.get_logger(__name__)

# Polymarket minimum order size
MIN_ORDER_SIZE = 5.0

# Target spread as fraction of max reward spread (±v from mid).
# The scoring formula is S = ((v-s)/v)^2 * b, which is quadratic.
# At 0.75v: score = 0.0625 (6.25% of max) — low fill risk, some reward.
# At 0.50v: score = 0.25 (25% of max) — moderate fill risk, decent reward.
# We target 0.75 as a balance between fill safety and reward earning.
REWARD_SPREAD_FRACTION = 0.75


@dataclass
class Quote:
    side: str
    price: float
    size: float
    market: str
    token_id: str


class QuoteEngine:
    """Reward-optimized quote engine.

    Polymarket reward scoring formula: S(v,s) = ((v-s)/v)^2 * b
    where v = max spread from mid (cents), s = our spread from mid, b = multiplier.

    The ±v spread means orders must be within v cents of the midpoint.
    Closer to mid = quadratically higher score, but also higher fill risk.

    Rules:
    - Orders MUST be within ±max_spread of midpoint to qualify for rewards
    - Both YES and NO sides needed for full rewards (single-side gets 1/3)
    - Min shares threshold must be met per order
    - Sampled every minute; resting orders only
    """

    def __init__(self, settings: Settings, inventory_manager: InventoryManager):
        self.settings = settings
        self.inventory_manager = inventory_manager
        self._last_mid_prices: dict[str, list[float]] = {}

    def calculate_mid_price(self, best_bid: float, best_ask: float) -> float:
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        return (best_bid + best_ask) / 2.0

    def detect_short_term_volatility(self, market_id: str, mid_price: float) -> bool:
        history = self._last_mid_prices.setdefault(market_id, [])
        history.append(mid_price)

        if len(history) > 20:
            self._last_mid_prices[market_id] = history[-20:]
            history = self._last_mid_prices[market_id]

        if len(history) < 5:
            return False

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
        as_spread_multiplier: float = 1.0,
        orderbook: dict[str, Any] | None = None,
        max_reward_spread_bps: float = 0.0,
        min_shares: float = 0.0,
    ) -> tuple[Quote | None, Quote | None]:
        """Generate reward-qualifying quotes.

        max_reward_spread_bps: ±spread from midpoint in bps (e.g. 400 = ±4 cents).
        This is ALREADY the half-spread — orders must be within this distance of mid.
        """
        mid_price = self.calculate_mid_price(best_bid, best_ask)
        if mid_price <= 0:
            return (None, None)

        # max_reward_spread_bps is ±v from midpoint (e.g. 400 = ±4 cents = 0.04)
        # This IS the half-spread directly — NOT the full bid-ask width
        if max_reward_spread_bps > 0:
            max_half_spread = max_reward_spread_bps / 10000  # e.g. 400 → 0.04
            # Target = 75% of max spread from mid (balance fill safety vs score)
            # Score at 75%: ((1-0.75)/1)^2 = 6.25% of max
            target_half_spread = max_half_spread * REWARD_SPREAD_FRACTION
        else:
            max_half_spread = 0.0
            target_half_spread = self.settings.min_spread_bps * 3 / 10000

        # Calculate raw quote prices at target distance from mid
        yes_bid_price = mid_price - target_half_spread
        no_bid_price = (1.0 - mid_price) - target_half_spread

        # REWARD ELIGIBILITY is the hard constraint for reward farming.
        # Cap: never go outside the max reward spread from mid.
        if max_half_spread > 0:
            min_yes_price = mid_price - max_half_spread + 0.005
            min_no_price = (1.0 - mid_price) - max_half_spread + 0.005
            yes_bid_price = max(yes_bid_price, min_yes_price)
            no_bid_price = max(no_bid_price, min_no_price)
        else:
            # Without reward data, stay below best bid for fill safety
            yes_bid_price = min(yes_bid_price, best_bid - 0.01)
            no_best_bid = 1.0 - best_ask
            if no_best_bid > 0:
                no_bid_price = min(no_bid_price, no_best_bid - 0.01)

        # Round to Polymarket tick sizes (0.01 minimum)
        # Use ceil to stay closer to mid (within reward spread)
        yes_bid_price = math.ceil(yes_bid_price * 100) / 100
        no_bid_price = math.ceil(no_bid_price * 100) / 100

        # Validate prices are in [0.01, 0.99] range
        if yes_bid_price < 0.01 or yes_bid_price > 0.99:
            yes_bid_price = 0.0
        if no_bid_price < 0.01 or no_bid_price > 0.99:
            no_bid_price = 0.0

        # Use the reward program's minimum shares requirement
        quote_size = max(min_shares, MIN_ORDER_SIZE) if min_shares > 0 else MIN_ORDER_SIZE

        logger.info(
            "reward_quotes_generated",
            market_id=market_id,
            mid=round(mid_price, 4),
            yes_bid=yes_bid_price,
            no_bid=no_bid_price,
            half_spread=round(target_half_spread, 4),
            max_spread=round(max_half_spread, 4),
            best_bid=best_bid,
            best_ask=best_ask,
        )

        yes_quote = None
        no_quote = None

        if yes_bid_price > 0 and yes_token_id:
            yes_quote = Quote(
                side="BUY",
                price=yes_bid_price,
                size=quote_size,
                market=market_id,
                token_id=yes_token_id,
            )

        if no_bid_price > 0 and no_token_id:
            no_quote = Quote(
                side="BUY",
                price=no_bid_price,
                size=quote_size,
                market=market_id,
                token_id=no_token_id,
            )

        return (yes_quote, no_quote)

    def should_trim_quotes(self, time_to_close_hours: float) -> bool:
        if time_to_close_hours < 1.0:
            return True
        return False
