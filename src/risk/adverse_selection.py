"""
Adverse Selection Guard — protects the market maker from informed flow.

5 layers of defense:
1. Delta-neutral hedging: auto-hedge fills to stay flat
2. Fill-triggered rebalancing: detect fills, hedge within seconds
3. Event-aware spread widening: widen as resolution approaches
4. Price momentum detection: detect informed flow from price moves
5. Per-market exposure caps: hard limits on directional risk
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class Fill:
    """A detected fill on one of our orders."""

    market_id: str
    side: str  # "YES" or "NO"
    price: float
    size: float
    timestamp: float


@dataclass
class MarketState:
    """Tracked state for adverse selection detection per market."""

    market_id: str
    # Price momentum
    prices: list[tuple[float, float]] = field(default_factory=list)  # (ts, mid)
    # Fill tracking
    recent_fills: list[Fill] = field(default_factory=list)
    # Exposure
    yes_exposure: float = 0.0
    no_exposure: float = 0.0
    # Event timing
    end_date_ts: float = 0.0  # Unix timestamp of resolution
    # Guard outputs
    spread_multiplier: float = 1.0
    should_hedge: bool = False
    hedge_side: str = ""
    hedge_size: float = 0.0
    blocked: bool = False
    block_reason: str = ""


class AdverseSelectionGuard:
    """Central guard that combines all 5 defense layers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._states: dict[str, MarketState] = {}

        # Configurable thresholds
        self.max_exposure_per_market: float = 500.0  # USD directional limit
        self.momentum_window_s: float = 300.0  # 5 min window
        self.momentum_threshold: float = 0.03  # 3% move = informed flow
        self.event_widen_days: float = 7.0  # Start widening 7 days before resolution
        self.event_max_multiplier: float = 5.0  # Max spread multiplier near resolution
        self.fill_hedge_delay_s: float = 2.0  # Hedge within 2s of fill
        self.max_one_sided_fills: int = 3  # 3 fills on same side without hedge = pull

    def get_state(self, market_id: str) -> MarketState:
        if market_id not in self._states:
            self._states[market_id] = MarketState(market_id=market_id)
        return self._states[market_id]

    def set_end_date(self, market_id: str, end_date_ts: float) -> None:
        """Set the resolution date for event-aware widening."""
        self.get_state(market_id).end_date_ts = end_date_ts

    # ── Layer 1: Delta-Neutral Hedging ────────────────────────────────

    def compute_hedge(self, market_id: str) -> dict[str, Any] | None:
        """If we have directional exposure, compute the hedge order needed.

        The key insight for binary markets: YES + NO = $1 always.
        If you hold 100 YES at $0.60, buy 100 NO at $0.40 to lock in $0 net.
        Capital is tied up but you're protected from adverse resolution.
        """
        state = self.get_state(market_id)
        net = state.yes_exposure - state.no_exposure

        if abs(net) < 10:  # Less than $10 imbalance, skip
            state.should_hedge = False
            return None

        if net > 0:
            # Long YES, need to buy NO to hedge
            state.should_hedge = True
            state.hedge_side = "NO"
            state.hedge_size = abs(net)
        else:
            # Long NO, need to buy YES to hedge
            state.should_hedge = True
            state.hedge_side = "YES"
            state.hedge_size = abs(net)

        logger.info(
            "hedge_signal",
            market_id=market_id,
            net_exposure=round(net, 2),
            hedge_side=state.hedge_side,
            hedge_size=round(state.hedge_size, 2),
        )

        return {
            "market_id": market_id,
            "side": state.hedge_side,
            "size": state.hedge_size,
        }

    # ── Layer 2: Fill-Triggered Rebalancing ───────────────────────────

    def record_fill(self, market_id: str, side: str, price: float, size: float) -> dict[str, Any] | None:
        """Record a fill and immediately check if hedge is needed.

        Returns hedge order dict if immediate rebalancing is warranted.
        """
        state = self.get_state(market_id)
        now = time.time()

        fill = Fill(
            market_id=market_id,
            side=side,
            price=price,
            size=size,
            timestamp=now,
        )
        state.recent_fills.append(fill)

        # Update exposure
        usd_value = size * price
        if side == "YES":
            state.yes_exposure += usd_value
        else:
            state.no_exposure += usd_value

        # Prune old fills (keep last 5 min)
        cutoff = now - 300
        state.recent_fills = [f for f in state.recent_fills if f.timestamp > cutoff]

        # Check for one-sided fill pattern (informed flow signal)
        recent_yes = sum(1 for f in state.recent_fills if f.side == "YES")
        recent_no = sum(1 for f in state.recent_fills if f.side == "NO")

        if recent_yes >= self.max_one_sided_fills and recent_no == 0:
            logger.warning(
                "one_sided_fills_detected",
                market_id=market_id,
                side="YES",
                count=recent_yes,
            )
            state.spread_multiplier = max(state.spread_multiplier, 2.0)

        if recent_no >= self.max_one_sided_fills and recent_yes == 0:
            logger.warning(
                "one_sided_fills_detected",
                market_id=market_id,
                side="NO",
                count=recent_no,
            )
            state.spread_multiplier = max(state.spread_multiplier, 2.0)

        logger.info(
            "fill_recorded",
            market_id=market_id,
            side=side,
            price=price,
            size=size,
            yes_exposure=round(state.yes_exposure, 2),
            no_exposure=round(state.no_exposure, 2),
        )

        # Trigger immediate hedge check
        return self.compute_hedge(market_id)

    # ── Layer 3: Event-Aware Spread Widening ──────────────────────────

    def get_event_spread_multiplier(self, market_id: str) -> float:
        """Widen spread as resolution date approaches.

        Rationale: closer to resolution = more informed traders = higher
        adverse selection risk. Gradually widen from 1x to max_multiplier.
        """
        state = self.get_state(market_id)
        if state.end_date_ts <= 0:
            return 1.0

        now = time.time()
        days_until = (state.end_date_ts - now) / 86400

        if days_until <= 0:
            # Past resolution — should not be quoting
            state.blocked = True
            state.block_reason = "market_past_resolution"
            return self.event_max_multiplier

        if days_until > self.event_widen_days:
            return 1.0

        # Linear ramp from 1x to max over the final N days
        progress = 1.0 - (days_until / self.event_widen_days)
        multiplier = 1.0 + progress * (self.event_max_multiplier - 1.0)

        if multiplier > 2.0:
            logger.info(
                "event_spread_widening",
                market_id=market_id,
                days_until=round(days_until, 1),
                multiplier=round(multiplier, 2),
            )

        return multiplier

    # ── Layer 4: Price Momentum Detection ─────────────────────────────

    def record_price(self, market_id: str, mid_price: float) -> None:
        """Record a price observation for momentum detection."""
        state = self.get_state(market_id)
        now = time.time()
        state.prices.append((now, mid_price))

        # Keep only momentum window
        cutoff = now - self.momentum_window_s
        state.prices = [(t, p) for t, p in state.prices if t > cutoff]

    def detect_momentum(self, market_id: str) -> float:
        """Detect price momentum that signals informed flow.

        Returns a multiplier: 1.0 = normal, >1.0 = widen spread.

        Only considers price moves over a meaningful time window (>= 60s)
        to avoid false positives from startup or synthetic book transitions.
        """
        state = self.get_state(market_id)
        if len(state.prices) < 5:
            return 1.0

        # Require at least 60s of price history to avoid startup false positives
        time_span = state.prices[-1][0] - state.prices[0][0]
        if time_span < 60:
            return 1.0

        # Use median of first 3 and last 3 prices to smooth out noise
        first_prices = [p for _, p in state.prices[:3]]
        last_prices = [p for _, p in state.prices[-3:]]
        baseline = sorted(first_prices)[len(first_prices) // 2]
        current = sorted(last_prices)[len(last_prices) // 2]

        if baseline <= 0:
            return 1.0

        move = abs(current - baseline) / baseline

        if move > self.momentum_threshold:
            # Strong directional move = informed flow
            multiplier = 1.0 + (move / self.momentum_threshold)
            multiplier = min(multiplier, 4.0)  # Cap at 4x

            logger.warning(
                "momentum_detected",
                market_id=market_id,
                move_pct=round(move * 100, 2),
                multiplier=round(multiplier, 2),
                direction="UP" if current > baseline else "DOWN",
            )
            return multiplier

        return 1.0

    # ── Layer 5: Per-Market Exposure Caps ─────────────────────────────

    def check_exposure_cap(self, market_id: str, side: str, size_usd: float) -> tuple[bool, str]:
        """Check if adding this quote would exceed per-market directional limit.

        Returns (allowed, reason).
        """
        state = self.get_state(market_id)
        net = state.yes_exposure - state.no_exposure

        if side == "YES":
            new_net = net + size_usd
        else:
            new_net = net - size_usd

        if abs(new_net) > self.max_exposure_per_market:
            reason = (
                f"exposure_cap: net would be ${new_net:.0f}, "
                f"limit is ${self.max_exposure_per_market:.0f}"
            )
            logger.warning("exposure_cap_hit", market_id=market_id, side=side, **{
                "new_net": round(new_net, 2),
                "limit": self.max_exposure_per_market,
            })
            return False, reason

        return True, "OK"

    # ── Combined Guard Check ──────────────────────────────────────────

    def check_quote(
        self, market_id: str, side: str, price: float, size: float, mid_price: float
    ) -> tuple[bool, float, str]:
        """Run all guards and return (allowed, spread_multiplier, reason).

        Call this before placing any quote. It will:
        - Block if market is past resolution
        - Apply event-aware spread widening
        - Apply momentum-based spread widening
        - Apply fill-pattern spread widening
        - Check exposure caps
        """
        state = self.get_state(market_id)

        # Record price for momentum
        self.record_price(market_id, mid_price)

        # Check if blocked
        if state.blocked:
            return False, 1.0, state.block_reason

        # Layer 3: Event timing
        event_mult = self.get_event_spread_multiplier(market_id)
        if state.blocked:  # May have been set by event check
            return False, 1.0, state.block_reason

        # Layer 4: Momentum
        momentum_mult = self.detect_momentum(market_id)

        # Layer 2: Fill-pattern multiplier (already set by record_fill)
        fill_mult = state.spread_multiplier

        # Combined multiplier (take the max — most conservative)
        combined_mult = max(event_mult, momentum_mult, fill_mult)

        # Layer 5: Exposure cap
        size_usd = size * price
        allowed, reason = self.check_exposure_cap(market_id, side, size_usd)
        if not allowed:
            return False, combined_mult, reason

        # Decay fill multiplier back toward 1.0 over time
        if fill_mult > 1.0:
            now = time.time()
            last_fill_time = state.recent_fills[-1].timestamp if state.recent_fills else now
            decay_s = now - last_fill_time
            if decay_s > 60:  # After 60s with no fills, start decaying
                state.spread_multiplier = max(1.0, fill_mult - (decay_s - 60) / 300)

        return True, combined_mult, "OK"

    def get_status(self, market_id: str) -> dict[str, Any]:
        """Get full guard status for a market (for logging/metrics)."""
        state = self.get_state(market_id)
        return {
            "market_id": market_id,
            "yes_exposure": round(state.yes_exposure, 2),
            "no_exposure": round(state.no_exposure, 2),
            "net_exposure": round(state.yes_exposure - state.no_exposure, 2),
            "spread_multiplier": round(state.spread_multiplier, 2),
            "should_hedge": state.should_hedge,
            "hedge_side": state.hedge_side,
            "hedge_size": round(state.hedge_size, 2),
            "blocked": state.blocked,
            "recent_fills_5m": len(state.recent_fills),
            "momentum_mult": round(self.detect_momentum(market_id), 2),
            "event_mult": round(self.get_event_spread_multiplier(market_id), 2),
        }

    def reset_market(self, market_id: str) -> None:
        """Reset all state for a market (e.g., after it resolves)."""
        if market_id in self._states:
            del self._states[market_id]
