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
from src.risk.adverse_selection import AdverseSelectionGuard
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


def _best_bid_price(bids: list[dict]) -> float:
    """Get highest bid price from orderbook bids (may not be sorted)."""
    if not bids:
        return 0.0
    return max(float(b.get("price", 0)) for b in bids)


def _best_ask_price(asks: list[dict]) -> float:
    """Get lowest ask price from orderbook asks (may not be sorted)."""
    if not asks:
        return 0.0
    return min(float(a.get("price", 0)) for a in asks)


class MarketMakerBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False

        # Core components — proxy_address is the Polymarket wallet, private_key signs for it
        self.order_signer = OrderSigner(settings.private_key, proxy_address=settings.public_address)
        self.rest_client = PolymarketRestClient(settings, self.order_signer)
        self.ws_client = PolymarketWebSocketClient(settings)
        self.order_executor = OrderExecutor(settings, self.order_signer, self.rest_client)

        self.inventory_manager = InventoryManager(
            settings.max_exposure_usd,
            settings.min_exposure_usd,
            settings.target_inventory_balance,
        )
        self.risk_manager = RiskManager(settings, self.inventory_manager)
        self.quote_engine = QuoteEngine(settings, self.inventory_manager)

        self.auto_redeem = AutoRedeem(settings)

        # Adverse selection protection
        self.as_guard = AdverseSelectionGuard(settings)

        # Reward-optimized strategy components
        self.reward_client = RewardClient(settings)
        self.market_selector = MarketSelector(settings, self.reward_client)
        self.uptime_tracker = UptimeTracker(settings)

        # Per-market state
        self.orderbooks: dict[str, dict[str, Any]] = {}
        self.market_infos: dict[str, dict[str, Any]] = {}
        self.open_orders: dict[str, dict[str, Any]] = {}
        self.last_quote_times: dict[str, float] = {}

        # Token ID mapping: market_id -> {yes_token_id, no_token_id, neg_risk}
        self._token_map: dict[str, dict[str, Any]] = {}

        # Track last known fill timestamp per market for polling
        self._last_fill_check: dict[str, float] = {}

    # -- API Key Setup -------------------------------------------------------

    async def setup_api_credentials(self):
        """Derive API keys for L2 auth using the official py-clob-client."""
        try:
            # py-clob-client derive_api_key is synchronous — run in thread
            creds = await asyncio.to_thread(self.order_signer.derive_api_credentials)
            logger.info("api_credentials_derived", api_key=creds["apiKey"][:8] + "...")
        except Exception as e:
            logger.error("api_key_setup_failed", error=str(e))
            raise

    # -- Market discovery ----------------------------------------------------

    async def discover_market(self) -> dict[str, Any] | None:
        """Fallback: fetch the single configured market directly by condition_id."""
        try:
            info = await self.rest_client.get_market_info(self.settings.market_id)
            if info:
                logger.info("market_discovered", market_id=self.settings.market_id)
                return info
            return None
        except Exception as e:
            logger.error("market_discovery_failed", error=str(e))
            return None

    async def discover_reward_markets(self) -> list[dict[str, Any]]:
        # scan_and_rank selects top N from settings.target_markets_count
        selected = await self.market_selector.scan_and_rank()

        if not selected:
            logger.warning("no_reward_markets_selected")
            # Do NOT fall back to configured market — it may not be sponsored.
            # Only trade markets with confirmed reward pools.
            return []

        infos = []
        now_ts = time.time()
        for scored in selected:
            if len(infos) >= self.settings.target_markets_count:
                break
            try:
                info = await self.rest_client.get_market_info(scored.market_id)

                # Skip markets that resolve within 24 hours — high adverse selection risk
                # and reward earnings won't accumulate meaningfully
                end_date = info.get("end_date_iso") or info.get("end_date") or ""
                if end_date:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        hours_until = (dt.timestamp() - now_ts) / 3600
                        if hours_until < 24:
                            logger.info("skipping_near_resolution_market", market_id=scored.market_id, hours_until=round(hours_until, 1))
                            continue
                    except Exception:
                        pass

                info["_reward"] = {
                    "rate_per_day": scored.info.rate_per_day,
                    "volume_24h": scored.info.volume_24h,
                    "breakeven_spread_bps": scored.breakeven_spread_bps,
                    "allocated_capital": scored.allocated_capital,
                    "reward_efficiency": scored.reward_efficiency,
                    "max_spread_bps": scored.info.max_spread_bps,
                    "min_shares": scored.info.min_shares,
                }
                mid = scored.market_id
                self.market_infos[mid] = info
                self.uptime_tracker.register_market(mid)

                # Extract token IDs and end date
                self._extract_market_metadata(mid, info)

                # Pre-fetch orderbook so we have data for quoting
                tokens = self._token_map.get(mid, {})
                yes_token = tokens.get("yes_token_id", "")
                if yes_token:
                    try:
                        ob = await self.rest_client.get_orderbook(yes_token)
                        self.orderbooks[mid] = ob
                    except Exception:
                        pass

                infos.append(info)

                record_reward_efficiency(mid, scored.reward_efficiency)
                record_reward_rate(mid, scored.info.rate_per_day)
                record_allocated_capital(mid, scored.allocated_capital)
            except Exception as e:
                logger.error("market_info_fetch_failed", market_id=scored.market_id, error=str(e))

        record_active_markets(len(infos))
        logger.info("reward_markets_ready", count=len(infos))
        return infos

    def _extract_market_metadata(self, market_id: str, info: dict[str, Any]):
        """Extract token IDs, neg_risk flag, and end date from market info."""
        tokens = info.get("tokens", [])
        token_map: dict[str, Any] = {"neg_risk": info.get("neg_risk", False)}

        # Try matching by outcome name first (Yes/No)
        for token in tokens:
            outcome = token.get("outcome", "").upper()
            if outcome == "YES":
                token_map["yes_token_id"] = token.get("token_id", "")
            elif outcome == "NO":
                token_map["no_token_id"] = token.get("token_id", "")

        # Fallback: for sports/binary markets with team names, use position
        # First token = "YES" equivalent, second = "NO" equivalent
        if "yes_token_id" not in token_map and len(tokens) >= 2:
            token_map["yes_token_id"] = tokens[0].get("token_id", "")
            token_map["no_token_id"] = tokens[1].get("token_id", "")
            logger.debug(
                "tokens_from_position",
                market_id=market_id,
                outcome_0=tokens[0].get("outcome", ""),
                outcome_1=tokens[1].get("outcome", ""),
            )

        self._token_map[market_id] = token_map

        # Set end date for adverse selection event-aware widening
        end_date = info.get("end_date_iso") or info.get("end_date") or ""
        if end_date:
            try:
                from datetime import datetime, timezone
                if isinstance(end_date, str):
                    dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    self.as_guard.set_end_date(market_id, dt.timestamp())
                    logger.debug("end_date_set", market_id=market_id, end_date=end_date)
            except Exception:
                pass

    # -- Orderbook -----------------------------------------------------------

    async def update_orderbook(self, market_id: str):
        """Fetch orderbook for YES token of a market."""
        token_info = self._token_map.get(market_id, {})
        yes_token = token_info.get("yes_token_id", "")

        if not yes_token:
            logger.warning("no_token_id_for_orderbook", market_id=market_id)
            return

        try:
            orderbook = await self.rest_client.get_orderbook(yes_token)
            self.orderbooks[market_id] = orderbook
        except Exception as e:
            logger.error("orderbook_update_failed", market_id=market_id, error=str(e))

    def _handle_orderbook_update(self, data: dict[str, Any]):
        """Handle WebSocket orderbook update."""
        asset_id = data.get("asset_id", "")
        # Reverse lookup: find market by token ID
        for mid, tokens in self._token_map.items():
            if tokens.get("yes_token_id") == asset_id or tokens.get("no_token_id") == asset_id:
                book = data.get("book") or data
                self.orderbooks[mid] = book
                break

    # -- Fill detection ------------------------------------------------------

    def _handle_trade_update(self, data: dict[str, Any]):
        """Handle WebSocket trade event — detect our fills."""
        maker = data.get("maker_address", "")
        our_address = self.order_signer.get_address().lower()

        if maker.lower() != our_address:
            return  # Not our fill

        market_id = data.get("market", "")
        side = data.get("side", "BUY")
        outcome = data.get("outcome", "YES")
        price = float(data.get("price", 0))
        size = float(data.get("size", 0))

        if not market_id or price <= 0 or size <= 0:
            return

        logger.info("fill_detected_ws", market_id=market_id, side=side, outcome=outcome, price=price, size=size)

        # Update inventory
        self.inventory_manager.record_fill(market_id, side, outcome, price, size)

        # Update AS guard
        hedge = self.as_guard.record_fill(market_id, outcome, price, size)

        # Execute hedge if needed
        if hedge:
            asyncio.create_task(self._execute_hedge(market_id, hedge))

        # EMERGENCY REQUOTE: Fill means someone took our order — immediately
        # cancel remaining orders and requote with updated AS state.
        # This prevents stale quotes from getting picked off during fast moves.
        market_info = self.market_infos.get(market_id)
        if market_info:
            # Reset the quote timer so refresh_quotes_for_market runs immediately
            self.last_quote_times[market_id] = 0
            asyncio.create_task(self._emergency_requote(market_id, market_info))

    async def _execute_hedge(self, market_id: str, hedge: dict[str, Any]):
        """Execute a hedge order from the AS guard."""
        hedge_side = hedge["side"]  # "YES" or "NO"
        hedge_size = hedge["size"]
        tokens = self._token_map.get(market_id, {})

        if hedge_side == "YES":
            token_id = tokens.get("yes_token_id", "")
        else:
            token_id = tokens.get("no_token_id", "")

        if not token_id:
            logger.error("no_token_for_hedge", market_id=market_id, side=hedge_side)
            return

        # Get current price for hedge
        orderbook = self.orderbooks.get(market_id, {})
        asks = orderbook.get("asks", [])
        if not asks:
            logger.warning("no_asks_for_hedge", market_id=market_id)
            return

        # Use best ask + small buffer for likely fill
        best_ask = _best_ask_price(asks)
        if hedge_side == "NO":
            best_ask = 1.0 - best_ask  # Convert for NO token

        if best_ask <= 0:
            return

        neg_risk = tokens.get("neg_risk", False)
        result = await self.order_executor.place_hedge_order(
            market_id=market_id,
            token_id=token_id,
            side=hedge_side,
            size=hedge_size,
            price=best_ask,
            neg_risk=neg_risk,
        )

        if result:
            logger.info("hedge_executed", market_id=market_id, side=hedge_side, size=hedge_size)

    async def poll_fills(self):
        """Fallback fill detection via REST polling (if WS misses fills)."""
        while self.running:
            try:
                await asyncio.sleep(10)  # Poll every 10s

                for market_id in list(self.market_infos.keys()):
                    last_check = self._last_fill_check.get(market_id, 0)
                    now = int(time.time())

                    try:
                        trades = await self.rest_client.get_trades(
                            market=market_id,
                            after=int(last_check) if last_check > 0 else None,
                        )

                        for trade in trades:
                            # Only process maker fills
                            if trade.get("trader_side") != "MAKER":
                                continue

                            trade_time = float(trade.get("match_time", 0))
                            if trade_time <= last_check:
                                continue

                            side = trade.get("side", "BUY")
                            outcome = trade.get("outcome", "YES")
                            price = float(trade.get("price", 0))
                            size = float(trade.get("size", 0))

                            if price > 0 and size > 0:
                                self.inventory_manager.record_fill(market_id, side, outcome, price, size)
                                self.as_guard.record_fill(market_id, outcome, price, size)

                        self._last_fill_check[market_id] = now

                    except Exception as e:
                        logger.debug("fill_poll_error", market_id=market_id, error=str(e))

            except Exception as e:
                logger.error("poll_fills_error", error=str(e))
                await asyncio.sleep(30)

    # -- Quote refresh (per-market) ------------------------------------------

    async def refresh_quotes_for_market(self, market_id: str, market_info: dict[str, Any]):
        current_time = time.time() * 1000
        last_time = self.last_quote_times.get(market_id, 0)

        if (current_time - last_time) < self.settings.quote_refresh_rate_ms:
            return

        self.last_quote_times[market_id] = current_time

        # Check drawdown limits before quoting
        dd_ok, dd_reason = self.inventory_manager.check_drawdown_limit()
        if not dd_ok:
            logger.warning("quoting_paused_drawdown", reason=dd_reason, market_id=market_id)
            self.uptime_tracker.record_tick(market_id, has_live_quotes=False)
            return

        orderbook = self.orderbooks.get(market_id)
        if not orderbook:
            await self.update_orderbook(market_id)
            orderbook = self.orderbooks.get(market_id, {})

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        best_bid = _best_bid_price(bids)
        best_ask = _best_ask_price(asks)

        # For empty or extremely wide books (spread > 90 cents), use synthetic mid
        # These markets have rewards precisely because they need liquidity providers
        book_spread = (best_ask - best_bid) if (best_bid > 0 and best_ask > 0) else 1.0
        if book_spread > 0.90 or best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            # Use 0.50 as default mid for binary markets with no real book
            best_bid = 0.49
            best_ask = 0.51
            logger.info("using_synthetic_book", market_id=market_id, reason="empty_or_wide_book")

        tokens = self._token_map.get(market_id, {})
        yes_token_id = tokens.get("yes_token_id", "")
        no_token_id = tokens.get("no_token_id", "")

        if not yes_token_id or not no_token_id:
            yes_token_id = market_info.get("yes_token_id", "")
            no_token_id = market_info.get("no_token_id", "")

        reward_data = market_info.get("_reward", {})
        reward_rate = reward_data.get("rate_per_day", 0.0)

        # Only quote markets with confirmed reward sponsorship
        if reward_rate <= 0:
            logger.warning("skipping_unsponsored_market", market_id=market_id)
            return

        volume_24h = reward_data.get("volume_24h", 0.0)
        breakeven_bps = reward_data.get("breakeven_spread_bps", 0.0)
        allocated_capital = reward_data.get("allocated_capital", 0.0)

        widen = self.uptime_tracker.should_widen_instead_of_pull(market_id)

        mid = (best_bid + best_ask) / 2.0
        self.market_selector.record_price(market_id, mid)
        self.as_guard.record_price(market_id, mid)

        # AS guard: combined spread multiplier
        as_state = self.as_guard.get_state(market_id)
        if as_state.blocked:
            logger.warning("market_blocked_by_as_guard", market_id=market_id, reason=as_state.block_reason)
            self.uptime_tracker.record_tick(market_id, has_live_quotes=False)
            return

        as_mult = max(
            self.as_guard.get_event_spread_multiplier(market_id),
            self.as_guard.detect_momentum(market_id),
            as_state.spread_multiplier,
        )

        max_reward_spread = reward_data.get("max_spread_bps", 0.0)
        min_shares = reward_data.get("min_shares", 50.0)

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
            as_spread_multiplier=as_mult,
            orderbook=orderbook,
            max_reward_spread_bps=max_reward_spread,
            min_shares=min_shares,
        )

        # PLACE-THEN-CANCEL: Place new orders first, then cancel old ones.
        # This minimizes the gap where no orders are on the book (Rule 5).
        # Brief overlap of old+new orders is safer than a gap with zero orders,
        # since reward sampling requires orders to be resting on the book.
        # Capture existing order IDs before placing new ones.
        old_order_ids: list[str] = []
        try:
            open_orders = await self.rest_client.get_open_orders(market=market_id)
            old_order_ids = [o.get("id") for o in (open_orders or []) if o.get("id")]
        except Exception:
            pass

        quotes_placed = False
        if yes_quote:
            await self._place_quote(yes_quote, "YES")
            quotes_placed = True
        if no_quote:
            await self._place_quote(no_quote, "NO")
            quotes_placed = True

        # Now cancel the OLD orders (new ones are already resting)
        if old_order_ids:
            try:
                await self.order_executor.batch_cancel_orders(old_order_ids)
                logger.info("old_orders_cancelled", market_id=market_id, count=len(old_order_ids))
            except Exception as e:
                logger.error("old_order_cancel_failed", market_id=market_id, error=str(e))

        self.uptime_tracker.record_tick(market_id, has_live_quotes=quotes_placed)

    async def _emergency_requote(self, market_id: str, market_info: dict[str, Any]):
        """Emergency requote after a fill — cancel all orders and refresh immediately.

        This runs outside the normal 60s cycle. When we get filled, our remaining
        orders may be stale (wrong price given the new information). Cancel
        everything and requote with fresh AS guard state.
        """
        try:
            logger.info("emergency_requote", market_id=market_id)
            # Cancel all existing orders immediately
            await self._cancel_all_market_orders(market_id)
            # Update orderbook before requoting
            await self.update_orderbook(market_id)
            # Requote with fresh state (AS guard multiplier is now updated)
            await self.refresh_quotes_for_market(market_id, market_info)
        except Exception as e:
            logger.error("emergency_requote_failed", market_id=market_id, error=str(e))

    async def _cancel_all_market_orders(self, market_id: str):
        """Cancel ALL open orders for a market before placing new ones.

        This prevents order stacking — the #1 cause of unintended fills.
        """
        try:
            open_orders = await self.rest_client.get_open_orders(market=market_id)
            if not open_orders:
                return

            order_ids = [o.get("id") for o in open_orders if o.get("id")]
            if order_ids:
                await self.order_executor.batch_cancel_orders(order_ids)
                logger.info("orders_cancelled_before_refresh", market_id=market_id, count=len(order_ids))
        except Exception as e:
            logger.error("cancel_before_refresh_failed", market_id=market_id, error=str(e))

    async def _place_quote(self, quote: Any, outcome: str):
        is_valid, reason = self.risk_manager.validate_order(quote.side, quote.size * quote.price)

        if not is_valid:
            logger.warning("quote_rejected", reason=reason, outcome=outcome, market=quote.market)
            return

        # AS guard check
        mid = quote.price
        as_ok, as_mult, as_reason = self.as_guard.check_quote(
            quote.market, outcome, quote.price, quote.size, mid
        )
        if not as_ok:
            logger.warning("as_guard_blocked", reason=as_reason, outcome=outcome, market=quote.market)
            return

        try:
            tokens = self._token_map.get(quote.market, {})
            neg_risk = tokens.get("neg_risk", False)

            result = await self.order_executor.place_order(
                token_id=quote.token_id,
                price=quote.price,
                size=quote.size,
                side=quote.side,
                neg_risk=neg_risk,
            )

            logger.info(
                "quote_placed",
                outcome=outcome,
                side=quote.side,
                price=quote.price,
                size=quote.size,
                market=quote.market,
                order_id=result.get("orderID"),
            )
        except Exception as e:
            logger.error("quote_placement_failed", outcome=outcome, market=quote.market, error=str(e))

    # -- Async loops ---------------------------------------------------------

    async def run_cancel_replace_cycle(self, market_infos: list[dict[str, Any]]):
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
        while self.running:
            try:
                await asyncio.sleep(self.settings.market_rotation_interval_s)

                if not self.running:
                    break

                logger.info("market_rotation_starting")
                new_infos = await self.discover_reward_markets()
                if new_infos:
                    self.market_infos = {
                        (m.get("id") or m.get("condition_id", "")): m for m in new_infos
                    }

            except Exception as e:
                logger.error("market_rotation_failed", error=str(e))
                await asyncio.sleep(60)

    async def run_uptime_reporter(self):
        while self.running:
            try:
                await asyncio.sleep(300)
                summary = self.uptime_tracker.get_session_summary()
                pnl_summary = self.inventory_manager.pnl.get_summary()
                logger.info("uptime_report", **summary)
                logger.info("pnl_report", **pnl_summary)

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

    # -- Main entry ----------------------------------------------------------

    async def run(self):
        self.running = True

        logger.info("market_maker_starting", mode="reward_optimized")

        # Phase 0: Setup API credentials
        await self.setup_api_credentials()

        # Phase 0.5: Sync inventory from API
        await self.inventory_manager.sync_from_api(
            self.rest_client, self.order_signer.get_address()
        )

        # Phase 1: Discover and rank reward markets
        market_infos = await self.discover_reward_markets()

        if not market_infos:
            logger.info("falling_back_to_single_market", market_id=self.settings.market_id)
            single = await self.discover_market()
            if not single:
                logger.error("market_not_available")
                return
            market_infos = [single]
            mid = self.settings.market_id
            self.uptime_tracker.register_market(mid)
            self._extract_market_metadata(mid, single)

        # Phase 1.5: Cancel ALL stale orders from previous runs
        for info in market_infos:
            mid = info.get("id") or info.get("condition_id", "")
            if mid:
                await self._cancel_all_market_orders(mid)
        logger.info("stale_orders_cancelled", markets=len(market_infos))

        # Phase 2: Initialize orderbooks
        for info in market_infos:
            mid = info.get("id") or info.get("condition_id", "")
            if mid:
                await self.update_orderbook(mid)

        # Phase 3: WebSocket with fill detection
        if self.settings.market_discovery_enabled:
            await self.ws_client.connect()

            # Register handlers
            self.ws_client.register_handler("book", self._handle_orderbook_update)
            self.ws_client.register_handler("trade", self._handle_trade_update)

            for info in market_infos:
                mid = info.get("id") or info.get("condition_id", "")
                tokens = self._token_map.get(mid, {})
                yes_token = tokens.get("yes_token_id", "")
                if mid and yes_token:
                    await self.ws_client.subscribe_orderbook(yes_token)
                    await self.ws_client.subscribe_trades(mid)

        # Phase 4: Launch concurrent loops
        tasks = [
            self.run_cancel_replace_cycle(market_infos),
            self.run_auto_redeem(),
            self.run_market_rotation(),
            self.run_uptime_reporter(),
            self.poll_fills(),  # Fallback fill detection
        ]

        if self.ws_client.running:
            tasks.append(self.ws_client.listen())

        try:
            await asyncio.gather(*tasks)
        finally:
            await self.cleanup()

    async def cleanup(self):
        self.running = False

        for mid in list(self.market_infos.keys()):
            try:
                await self.order_executor.cancel_all_orders(mid)
            except Exception as e:
                logger.error("cleanup_cancel_failed", market_id=mid, error=str(e))

        try:
            await self.order_executor.cancel_all_orders(self.settings.market_id)
        except Exception:
            pass

        # Log final reports
        summary = self.uptime_tracker.get_session_summary()
        pnl_summary = self.inventory_manager.pnl.get_summary()
        logger.info("final_uptime_report", **summary)
        logger.info("final_pnl_report", **pnl_summary)

        await self.rest_client.close()
        await self.ws_client.close()
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
