from __future__ import annotations

import asyncio
import signal
import time
from typing import Any

import structlog
from dotenv import load_dotenv

from src.config import Settings, get_settings
from src.execution.order_executor import OrderExecutor
from src.inventory.inventory_manager import InventoryManager
from src.logging_config import configure_logging
from src.market_maker.quote_engine import QuoteEngine
from src.polymarket.order_signer import OrderSigner
from src.polymarket.rest_client import PolymarketRestClient
from src.polymarket.websocket_client import PolymarketWebSocketClient
from src.risk.risk_manager import RiskManager
from src.services import AutoRedeem, start_metrics_server
from src.services.metrics import (
    record_active_markets,
    record_allocated_capital,
    record_avg_uptime,
    record_longest_gap,
    record_reward_efficiency,
    record_reward_rate,
    record_reward_spread_adj,
    record_uptime,
)
from src.strategy import MarketSelector, RewardClient, UptimeTracker

logger = structlog.get_logger(__name__)


class MarketMakerBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        self.rest_client = PolymarketRestClient(settings)
        self.ws_client = PolymarketWebSocketClient(settings)
        self.order_signer = OrderSigner(settings.private_key)
        self.order_executor = OrderExecutor(settings, self.order_signer)

        self.inventory_manager = InventoryManager(
            settings.max_exposure_usd,
            settings.min_exposure_usd,
            settings.target_inventory_balance,
        )
        self.risk_manager = RiskManager(settings, self.inventory_manager)
        self.quote_engine = QuoteEngine(settings, self.inventory_manager)

        self.auto_redeem = AutoRedeem(settings)

        # Reward-optimized strategy components
        self.reward_client = RewardClient(settings)
        self.market_selector = MarketSelector(settings, self.reward_client)
        self.uptime_tracker = UptimeTracker(settings)

        # Per-market state
        self.orderbooks: dict[str, dict[str, Any]] = {}
        self.market_infos: dict[str, dict[str, Any]] = {}
        self.open_orders: dict[str, dict[str, Any]] = {}
        self.last_quote_times: dict[str, float] = {}

    # ── Market discovery ──────────────────────────────────────────────

    async def discover_market(self) -> dict[str, Any] | None:
        """Discover a single market (legacy single-market mode)."""
        if not self.settings.market_discovery_enabled:
            return await self.rest_client.get_market_info(self.settings.market_id)

        try:
            markets = await self.rest_client.get_markets(active=True, closed=False)
            for market in markets:
                if market.get("id") == self.settings.market_id:
                    logger.info("market_discovered", market_id=market.get("id"), question=market.get("question"))
                    return market

            logger.warning("market_not_found", market_id=self.settings.market_id)
            return None
        except Exception as e:
            logger.error("market_discovery_failed", error=str(e))
            return None

    async def discover_reward_markets(self) -> list[dict[str, Any]]:
        """Scan, rank, and select markets via reward-optimized selector."""
        selected = await self.market_selector.scan_and_rank()
        if not selected:
            logger.warning("no_reward_markets_selected, falling back to configured market")
            fallback = await self.discover_market()
            return [fallback] if fallback else []

        # Fetch full market info for each selected market
        infos = []
        for scored in selected:
            try:
                info = await self.rest_client.get_market_info(scored.market_id)
                info["_reward"] = {
                    "rate_per_day": scored.info.rate_per_day,
                    "volume_24h": scored.info.volume_24h,
                    "breakeven_spread_bps": scored.breakeven_spread_bps,
                    "allocated_capital": scored.allocated_capital,
                    "reward_efficiency": scored.reward_efficiency,
                }
                self.market_infos[scored.market_id] = info
                self.uptime_tracker.register_market(scored.market_id)
                infos.append(info)

                # Publish metrics
                record_reward_efficiency(scored.market_id, scored.reward_efficiency)
                record_reward_rate(scored.market_id, scored.info.rate_per_day)
                record_allocated_capital(scored.market_id, scored.allocated_capital)
            except Exception as e:
                logger.error("market_info_fetch_failed", market_id=scored.market_id, error=str(e))

        record_active_markets(len(infos))
        logger.info("reward_markets_ready", count=len(infos))
        return infos

    # ── Orderbook ─────────────────────────────────────────────────────

    async def update_orderbook(self, market_id: str | None = None):
        target = market_id or self.settings.market_id
        try:
            orderbook = await self.rest_client.get_orderbook(target)
            self.orderbooks[target] = orderbook

            if self.ws_client.websocket:
                self.ws_client.register_handler("l2_book_update", self._handle_orderbook_update)
        except Exception as e:
            logger.error("orderbook_update_failed", market_id=target, error=str(e))

    def _handle_orderbook_update(self, data: dict[str, Any]):
        market_id = data.get("market", "")
        if market_id:
            self.orderbooks[market_id] = data.get("book", self.orderbooks.get(market_id, {}))

    # ── Quote refresh (per-market) ────────────────────────────────────

    async def refresh_quotes_for_market(self, market_id: str, market_info: dict[str, Any]):
        """Refresh quotes for a single market with reward-aware parameters."""
        current_time = time.time() * 1000
        last_time = self.last_quote_times.get(market_id, 0)

        if (current_time - last_time) < self.settings.quote_refresh_rate_ms:
            return

        self.last_quote_times[market_id] = current_time

        orderbook = self.orderbooks.get(market_id)
        if not orderbook:
            await self.update_orderbook(market_id)
            orderbook = self.orderbooks.get(market_id, {})

        best_bid = float(orderbook.get("best_bid", 0))
        best_ask = float(orderbook.get("best_ask", 1))

        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            logger.warning("invalid_orderbook", market_id=market_id, best_bid=best_bid, best_ask=best_ask)
            self.uptime_tracker.record_tick(market_id, has_live_quotes=False)
            return

        yes_token_id = market_info.get("yes_token_id", "")
        no_token_id = market_info.get("no_token_id", "")

        # Extract reward parameters
        reward_data = market_info.get("_reward", {})
        reward_rate = reward_data.get("rate_per_day", 0.0)
        volume_24h = reward_data.get("volume_24h", 0.0)
        breakeven_bps = reward_data.get("breakeven_spread_bps", 0.0)
        allocated_capital = reward_data.get("allocated_capital", 0.0)

        # Check if we should widen spread for uptime instead of pulling
        widen = self.uptime_tracker.should_widen_instead_of_pull(market_id)

        # Record price for volatility tracking
        mid = (best_bid + best_ask) / 2.0
        self.market_selector.record_price(market_id, mid)

        yes_quote, no_quote = self.quote_engine.generate_quotes(
            market_id=market_id,
            best_bid=best_bid,
            best_ask=best_ask,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            reward_rate_daily=reward_rate,
            volume_24h=volume_24h,
            breakeven_spread_bps=breakeven_bps,
            allocated_capital=allocated_capital,
            widen_for_uptime=widen,
        )

        await self._cancel_stale_orders(market_id)

        quotes_placed = False
        if yes_quote:
            await self._place_quote(yes_quote, "YES")
            quotes_placed = True
        if no_quote:
            await self._place_quote(no_quote, "NO")
            quotes_placed = True

        # Track uptime
        self.uptime_tracker.record_tick(market_id, has_live_quotes=quotes_placed)

    async def _cancel_stale_orders(self, market_id: str | None = None):
        target = market_id or self.settings.market_id
        try:
            open_orders = await self.rest_client.get_open_orders(
                self.order_signer.get_address(), target
            )

            current_time = time.time() * 1000
            order_ids_to_cancel = []

            for order in open_orders:
                order_time = order.get("timestamp", 0)
                age = current_time - order_time

                if age > self.settings.order_lifetime_ms:
                    order_ids_to_cancel.append(order.get("id"))

            if order_ids_to_cancel:
                await self.order_executor.batch_cancel_orders(order_ids_to_cancel)
        except Exception as e:
            logger.error("stale_order_cancellation_failed", market_id=target, error=str(e))

    async def _place_quote(self, quote: Any, outcome: str):
        is_valid, reason = self.risk_manager.validate_order(quote.side, quote.size * quote.price)

        if not is_valid:
            logger.warning("quote_rejected", reason=reason, outcome=outcome, market=quote.market)
            return

        try:
            order = {
                "market": quote.market,
                "side": quote.side,
                "size": str(quote.size),
                "price": str(quote.price),
                "token_id": quote.token_id,
            }

            result = await self.order_executor.place_order(order)
            logger.info(
                "quote_placed",
                outcome=outcome,
                side=quote.side,
                price=quote.price,
                size=quote.size,
                market=quote.market,
                order_id=result.get("id"),
            )
        except Exception as e:
            logger.error("quote_placement_failed", outcome=outcome, market=quote.market, error=str(e))

    # ── Async loops ───────────────────────────────────────────────────

    async def run_cancel_replace_cycle(self, market_infos: list[dict[str, Any]]):
        """Quote refresh loop across all active markets."""
        while self.running:
            try:
                for info in market_infos:
                    mid = info.get("id") or info.get("condition_id", "")
                    if mid:
                        await self.refresh_quotes_for_market(mid, info)

                await asyncio.sleep(self.settings.cancel_replace_interval_ms / 1000.0)
            except Exception as e:
                logger.error("cancel_replace_cycle_error", error=str(e))
                await asyncio.sleep(1)

    async def run_market_rotation(self):
        """Periodically re-rank markets and rotate capital."""
        while self.running:
            try:
                await asyncio.sleep(self.settings.market_rotation_interval_s)

                if not self.running:
                    break

                logger.info("market_rotation_starting")
                new_infos = await self.discover_reward_markets()
                if new_infos:
                    # Update the shared market_infos reference
                    self.market_infos = {
                        (m.get("id") or m.get("condition_id", "")): m for m in new_infos
                    }

            except Exception as e:
                logger.error("market_rotation_failed", error=str(e))
                await asyncio.sleep(60)

    async def run_uptime_reporter(self):
        """Log uptime stats periodically."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                summary = self.uptime_tracker.get_session_summary()
                logger.info("uptime_report", **summary)

                # Publish per-market uptime metrics
                uptimes = self.uptime_tracker.get_all_uptimes()
                for mid, pct in uptimes.items():
                    record_uptime(mid, pct)
                    stats = self.uptime_tracker.get_stats(mid)
                    if stats:
                        record_longest_gap(mid, stats.get("longest_gap_s", 0))

                if uptimes:
                    record_avg_uptime(sum(uptimes.values()) / len(uptimes))

            except Exception as e:
                logger.error("uptime_reporter_error", error=str(e))
                await asyncio.sleep(60)

    async def run_auto_redeem(self):
        while self.running:
            try:
                if self.settings.auto_redeem_enabled:
                    await self.auto_redeem.auto_redeem_all(self.order_signer.get_address())
                await asyncio.sleep(300)
            except Exception as e:
                logger.error("auto_redeem_error", error=str(e))
                await asyncio.sleep(60)

    # ── Main entry ────────────────────────────────────────────────────

    async def run(self):
        self.running = True

        logger.info("market_maker_starting", mode="reward_optimized")

        # Phase 1: Discover and rank reward markets
        market_infos = await self.discover_reward_markets()

        if not market_infos:
            # Fallback to single configured market
            logger.info("falling_back_to_single_market", market_id=self.settings.market_id)
            single = await self.discover_market()
            if not single:
                logger.error("market_not_available")
                return
            market_infos = [single]
            self.uptime_tracker.register_market(self.settings.market_id)

        # Phase 2: Initialize orderbooks
        for info in market_infos:
            mid = info.get("id") or info.get("condition_id", "")
            if mid:
                await self.update_orderbook(mid)

        # Phase 3: WebSocket subscriptions
        if self.settings.market_discovery_enabled:
            await self.ws_client.connect()
            for info in market_infos:
                mid = info.get("id") or info.get("condition_id", "")
                if mid:
                    await self.ws_client.subscribe_orderbook(mid)

        # Phase 4: Launch concurrent loops
        tasks = [
            self.run_cancel_replace_cycle(market_infos),
            self.run_auto_redeem(),
            self.run_market_rotation(),
            self.run_uptime_reporter(),
        ]

        if self.ws_client.running:
            tasks.append(self.ws_client.listen())

        try:
            await asyncio.gather(*tasks)
        finally:
            await self.cleanup()

    async def cleanup(self):
        self.running = False

        # Cancel orders across all active markets
        for mid in list(self.market_infos.keys()):
            try:
                await self.order_executor.cancel_all_orders(mid)
            except Exception as e:
                logger.error("cleanup_cancel_failed", market_id=mid, error=str(e))

        # Also cancel for the configured market (fallback case)
        try:
            await self.order_executor.cancel_all_orders(self.settings.market_id)
        except Exception:
            pass

        # Log final uptime summary
        summary = self.uptime_tracker.get_session_summary()
        logger.info("final_uptime_report", **summary)

        await self.rest_client.close()
        await self.ws_client.close()
        await self.order_executor.close()
        await self.auto_redeem.close()
        await self.reward_client.close()
        logger.info("market_maker_shutdown_complete")


async def bootstrap(settings: Settings):
    load_dotenv()
    configure_logging(settings.log_level)
    start_metrics_server(settings.metrics_host, settings.metrics_port)

    bot = MarketMakerBot(settings)

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_signal():
        logger.info("shutdown_signal_received")
        bot.running = False
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    try:
        await bot.run()
    finally:
        logger.info("bot_shutdown_complete")


def main():
    settings = get_settings()
    asyncio.run(bootstrap(settings))


if __name__ == "__main__":
    main()
