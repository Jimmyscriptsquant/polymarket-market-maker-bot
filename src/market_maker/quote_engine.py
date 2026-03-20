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
# At 0.75v: score = 0.0625 (6.25% of max) — low fill risk, tiny reward.
# At 0.50v: score = 0.25 (25% of max) — moderate fill risk, 4x more reward.
# At 0.25v: score = 0.5625 (56% of max) — high fill risk, 9x more reward.
# We target 0.50 for the best risk/reward tradeoff with small capital.
REWARD_SPREAD_FRACTION = 0.50


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

        # Adverse selection protection: if multiplier is high, pull quotes entirely.
        # We can't widen beyond the reward spread (that would lose eligibility),
        # so the safe response to informed flow is to stop quoting temporarily.
        # Multiplier > 1.5 = moderate risk (momentum or one-sided fills detected)
        # Multiplier > 2.5 = high risk (strong signal, pull immediately)
        if as_spread_multiplier > 2.5:
            logger.warning(
                "as_guard_pulling_quotes",
                market_id=market_id,
                multiplier=round(as_spread_multiplier, 2),
                reason="high_adverse_selection_risk",
            )
            return (None, None)

        # max_reward_spread_bps is ±v from midpoint (e.g. 400 = ±4 cents = 0.04)
        # This IS the half-spread directly — NOT the full bid-ask width
        if max_reward_spread_bps > 0:
            max_half_spread = max_reward_spread_bps / 10000  # e.g. 400 → 0.04

            # Under moderate AS risk (1.5-2.5x), widen toward edge of reward range.
            # This reduces score but also reduces fill probability.
            # Normal: 50% of max. Under AS stress: scale toward 90% of max.
            if as_spread_multiplier > 1.5:
                # Interpolate: at 1.5x → 50%, at 2.5x → 90%
                stress = min((as_spread_multiplier - 1.5) / 1.0, 1.0)  # 0.0 to 1.0
                spread_frac = REWARD_SPREAD_FRACTION + stress * (0.90 - REWARD_SPREAD_FRACTION)
                logger.info(
                    "as_widening_within_reward",
                    market_id=market_id,
                    multiplier=round(as_spread_multiplier, 2),
                    spread_frac=round(spread_frac, 3),
                )
            else:
                spread_frac = REWARD_SPREAD_FRACTION

            target_half_spread = max_half_spread * spread_frac
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

        # Rule 4: Extreme midpoint two-sided requirement
        # If mid < 0.10 or mid > 0.90, BOTH sides must be quotable or we earn nothing.
        # Single-sided orders score 0 in extreme ranges (not just 1/3).
        is_extreme = mid_price < 0.10 or mid_price > 0.90
        if is_extreme and (yes_bid_price <= 0 or no_bid_price <= 0):
            logger.warning(
                "extreme_mid_requires_two_sided",
                market_id=market_id,
                mid=round(mid_price, 4),
                yes_bid=yes_bid_price,
                no_bid=no_bid_price,
            )
            return (None, None)

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
            two_sided=(yes_bid_price > 0 and no_bid_price > 0),
            extreme_mid=is_extreme,
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
