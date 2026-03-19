"""
Inventory manager with API sync, PnL tracking, and drawdown limits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Inventory:
    yes_position: float = 0.0
    no_position: float = 0.0
    net_exposure_usd: float = 0.0
    total_value_usd: float = 0.0

    def update(self, yes_delta: float, no_delta: float, price: float):
        self.yes_position += yes_delta
        self.no_position += no_delta

        yes_value = self.yes_position * price
        no_value = self.no_position * (1.0 - price)

        self.net_exposure_usd = yes_value - no_value
        self.total_value_usd = yes_value + no_value

    def get_skew(self) -> float:
        total = abs(self.yes_position) + abs(self.no_position)
        if total == 0:
            return 0.0
        return abs(self.net_exposure_usd) / self.total_value_usd if self.total_value_usd > 0 else 0.0

    def is_balanced(self, max_skew: float = 0.3) -> bool:
        return self.get_skew() <= max_skew


@dataclass
class PnLTracker:
    """Tracks realized and unrealized PnL with drawdown limits."""

    # Realized PnL from completed trades
    realized_pnl: float = 0.0
    # Running total of USDC spent buying positions
    total_cost_basis: float = 0.0
    # Fees paid
    total_fees: float = 0.0

    # High-water mark for drawdown calculation
    peak_pnl: float = 0.0
    # Current drawdown from peak
    current_drawdown: float = 0.0

    # Session tracking
    session_start_time: float = field(default_factory=time.time)
    session_start_balance: float = 0.0
    trade_count: int = 0

    # Per-trade log (last N trades)
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    max_recent_trades: int = 100

    def record_trade(
        self,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        size: float,
        fee: float = 0.0,
    ):
        """Record a completed trade for PnL tracking."""
        usd_value = size * price
        self.total_fees += fee
        self.trade_count += 1

        if side == "BUY":
            self.total_cost_basis += usd_value + fee
        else:
            # Selling = realizing PnL
            self.realized_pnl += usd_value - fee

        # Update high-water mark and drawdown
        total_pnl = self.realized_pnl - self.total_fees
        if total_pnl > self.peak_pnl:
            self.peak_pnl = total_pnl
        self.current_drawdown = self.peak_pnl - total_pnl

        # Log trade
        trade = {
            "time": time.time(),
            "market_id": market_id,
            "side": side,
            "outcome": outcome,
            "price": price,
            "size": size,
            "fee": fee,
            "realized_pnl": round(self.realized_pnl, 4),
        }
        self.recent_trades.append(trade)
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades = self.recent_trades[-self.max_recent_trades:]

    def get_unrealized_pnl(self, positions: dict[str, dict[str, float]], prices: dict[str, float]) -> float:
        """Calculate unrealized PnL from current positions and market prices.

        positions: {market_id: {"yes": qty, "no": qty}}
        prices: {market_id: mid_price}
        """
        unrealized = 0.0
        for market_id, pos in positions.items():
            mid = prices.get(market_id, 0.5)
            yes_val = pos.get("yes", 0) * mid
            no_val = pos.get("no", 0) * (1.0 - mid)
            unrealized += yes_val + no_val
        return unrealized - self.total_cost_basis

    def get_total_pnl(self, unrealized: float = 0.0) -> float:
        return self.realized_pnl + unrealized - self.total_fees

    def get_session_pnl(self) -> float:
        return self.realized_pnl - self.total_fees

    def get_summary(self) -> dict[str, Any]:
        session_duration = time.time() - self.session_start_time
        return {
            "realized_pnl": round(self.realized_pnl, 4),
            "total_fees": round(self.total_fees, 4),
            "net_pnl": round(self.realized_pnl - self.total_fees, 4),
            "peak_pnl": round(self.peak_pnl, 4),
            "current_drawdown": round(self.current_drawdown, 4),
            "trade_count": self.trade_count,
            "session_duration_h": round(session_duration / 3600, 2),
        }


class InventoryManager:
    def __init__(
        self,
        max_exposure_usd: float,
        min_exposure_usd: float,
        target_balance: float = 0.0,
        max_drawdown_usd: float = 500.0,
        daily_loss_limit_usd: float = 200.0,
    ):
        self.max_exposure_usd = max_exposure_usd
        self.min_exposure_usd = min_exposure_usd
        self.target_balance = target_balance
        self.max_drawdown_usd = max_drawdown_usd
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.inventory = Inventory()
        self.pnl = PnLTracker()

        # Per-market positions for PnL calc
        self._positions: dict[str, dict[str, float]] = {}
        self._synced = False

    async def sync_from_api(self, rest_client: Any, address: str) -> None:
        """Sync inventory state from API on startup.

        Fetches USDC balance and open positions to avoid starting from zero.
        """
        try:
            # Get USDC balance
            balance_data = await rest_client.get_balance_allowance(asset_type="COLLATERAL")
            usdc_balance = float(balance_data.get("balance", 0)) / 1e6  # 6 decimals
            self.pnl.session_start_balance = usdc_balance

            logger.info("inventory_synced", usdc_balance=round(usdc_balance, 2))
            self._synced = True

        except Exception as e:
            logger.error("inventory_sync_failed", error=str(e))
            # Continue with zero state — safe but not ideal

    def record_fill(
        self,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        size: float,
        fee: float = 0.0,
    ):
        """Record a fill from WebSocket or trade polling."""
        # Update inventory
        if outcome == "YES":
            if side == "BUY":
                self.inventory.update(size, 0, price)
            else:
                self.inventory.update(-size, 0, price)
        else:
            if side == "BUY":
                self.inventory.update(0, size, price)
            else:
                self.inventory.update(0, -size, price)

        # Update per-market positions
        if market_id not in self._positions:
            self._positions[market_id] = {"yes": 0.0, "no": 0.0}
        key = "yes" if outcome == "YES" else "no"
        delta = size if side == "BUY" else -size
        self._positions[market_id][key] += delta

        # Track PnL
        self.pnl.record_trade(market_id, side, outcome, price, size, fee)

        logger.info(
            "fill_recorded",
            market_id=market_id,
            side=side,
            outcome=outcome,
            price=price,
            size=size,
            net_pnl=round(self.pnl.get_session_pnl(), 4),
        )

    def check_drawdown_limit(self) -> tuple[bool, str]:
        """Check if drawdown limits are breached. Returns (ok, reason)."""
        if self.pnl.current_drawdown > self.max_drawdown_usd:
            reason = f"drawdown ${self.pnl.current_drawdown:.2f} exceeds limit ${self.max_drawdown_usd:.2f}"
            logger.warning("drawdown_limit_hit", drawdown=self.pnl.current_drawdown, limit=self.max_drawdown_usd)
            return False, reason

        session_pnl = self.pnl.get_session_pnl()
        if session_pnl < -self.daily_loss_limit_usd:
            reason = f"daily loss ${abs(session_pnl):.2f} exceeds limit ${self.daily_loss_limit_usd:.2f}"
            logger.warning("daily_loss_limit_hit", loss=abs(session_pnl), limit=self.daily_loss_limit_usd)
            return False, reason

        return True, "OK"

    def update_inventory(self, yes_delta: float, no_delta: float, price: float):
        self.inventory.update(yes_delta, no_delta, price)

    def can_quote_yes(self, size_usd: float) -> bool:
        potential_exposure = self.inventory.net_exposure_usd + size_usd
        return potential_exposure <= self.max_exposure_usd

    def can_quote_no(self, size_usd: float) -> bool:
        potential_exposure = self.inventory.net_exposure_usd - size_usd
        return potential_exposure >= self.min_exposure_usd

    def get_quote_size_yes(self, base_size: float, price: float) -> float:
        if not self.can_quote_yes(base_size):
            max_size = max(0, self.max_exposure_usd - self.inventory.net_exposure_usd)
            return min(base_size, max_size / price) if price > 0 else 0

        if self.inventory.net_exposure_usd > self.target_balance:
            return base_size * 0.5

        return base_size

    def get_quote_size_no(self, base_size: float, price: float) -> float:
        if not self.can_quote_no(base_size):
            max_size = max(0, abs(self.min_exposure_usd - self.inventory.net_exposure_usd))
            return min(base_size, max_size / (1.0 - price)) if price < 1.0 else 0

        if self.inventory.net_exposure_usd < self.target_balance:
            return base_size * 0.5

        return base_size

    def should_rebalance(self, skew_limit: float = 0.3) -> bool:
        return not self.inventory.is_balanced(skew_limit)

    def get_rebalance_target(self) -> tuple[float, float]:
        current_skew = self.inventory.get_skew()
        if current_skew < 0.1:
            return (0.0, 0.0)

        rebalance_yes = -self.inventory.yes_position * 0.5
        rebalance_no = -self.inventory.no_position * 0.5

        return (rebalance_yes, rebalance_no)
