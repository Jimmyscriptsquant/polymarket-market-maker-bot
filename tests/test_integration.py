from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings
from src.main import MarketMakerBot


@pytest.fixture
def bot(settings: Settings) -> MarketMakerBot:
    """Create a bot with mocked external dependencies."""
    with patch("src.main.PolymarketRestClient") as mock_rest, \
         patch("src.main.PolymarketWebSocketClient") as mock_ws, \
         patch("src.main.OrderSigner") as mock_signer, \
         patch("src.main.OrderExecutor") as mock_executor, \
         patch("src.main.AutoRedeem") as mock_redeem, \
         patch("src.main.RewardClient") as mock_reward:

        mock_signer_inst = mock_signer.return_value
        mock_signer_inst.get_address.return_value = "0xtest"

        mock_rest_inst = mock_rest.return_value
        mock_rest_inst.get_markets = AsyncMock(return_value=[])
        mock_rest_inst.get_market_info = AsyncMock(return_value={
            "id": settings.market_id,
            "question": "Test?",
            "yes_token_id": "yes_tok",
            "no_token_id": "no_tok",
        })
        mock_rest_inst.get_orderbook = AsyncMock(return_value={
            "best_bid": 0.45, "best_ask": 0.55,
        })
        mock_rest_inst.get_open_orders = AsyncMock(return_value=[])
        mock_rest_inst.close = AsyncMock()

        mock_ws_inst = mock_ws.return_value
        mock_ws_inst.websocket = None
        mock_ws_inst.running = False
        mock_ws_inst.connect = AsyncMock()
        mock_ws_inst.close = AsyncMock()
        mock_ws_inst.register_handler = MagicMock()

        mock_executor_inst = mock_executor.return_value
        mock_executor_inst.place_order = AsyncMock(return_value={"id": "order123"})
        mock_executor_inst.batch_cancel_orders = AsyncMock(return_value=0)
        mock_executor_inst.cancel_all_orders = AsyncMock(return_value=0)
        mock_executor_inst.close = AsyncMock()

        mock_redeem_inst = mock_redeem.return_value
        mock_redeem_inst.close = AsyncMock()
        mock_redeem_inst.auto_redeem_all = AsyncMock()

        mock_reward_inst = mock_reward.return_value
        mock_reward_inst.fetch_reward_markets = AsyncMock(return_value=[])
        mock_reward_inst.close = AsyncMock()

        bot = MarketMakerBot(settings)
        # Replace instances with our mocks
        bot.rest_client = mock_rest_inst
        bot.ws_client = mock_ws_inst
        bot.order_signer = mock_signer_inst
        bot.order_executor = mock_executor_inst
        bot.auto_redeem = mock_redeem_inst
        bot.reward_client = mock_reward_inst
        bot.market_selector.reward_client = mock_reward_inst

        return bot


class TestMarketMakerBotIntegration:
    @pytest.mark.asyncio
    async def test_discover_market_single(self, bot: MarketMakerBot):
        """Fallback to single market discovery."""
        bot.settings.market_discovery_enabled = False
        result = await bot.discover_market()
        assert result is not None
        assert result["id"] == bot.settings.market_id

    @pytest.mark.asyncio
    async def test_discover_reward_markets_fallback(self, bot: MarketMakerBot):
        """When no reward markets, falls back to single market."""
        bot.reward_client.fetch_reward_markets = AsyncMock(return_value=[])
        result = await bot.discover_reward_markets()
        # Should return empty since scan_and_rank returns [] and
        # discover_reward_markets logs a warning
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_update_orderbook(self, bot: MarketMakerBot):
        await bot.update_orderbook("test_market")
        assert "test_market" in bot.orderbooks
        assert bot.orderbooks["test_market"]["best_bid"] == 0.45

    @pytest.mark.asyncio
    async def test_refresh_quotes_for_market(self, bot: MarketMakerBot):
        """Full quote refresh cycle for a market."""
        market_id = "test_market"
        market_info = {
            "id": market_id,
            "yes_token_id": "yes_tok",
            "no_token_id": "no_tok",
            "_reward": {
                "rate_per_day": 100.0,
                "volume_24h": 50000.0,
                "breakeven_spread_bps": 5.0,
                "allocated_capital": 5000.0,
            },
        }

        bot.orderbooks[market_id] = {"best_bid": 0.45, "best_ask": 0.55}
        bot.uptime_tracker.register_market(market_id)

        await bot.refresh_quotes_for_market(market_id, market_info)

        # Should have placed orders
        assert bot.order_executor.place_order.call_count >= 1

        # Uptime should be tracked
        assert bot.uptime_tracker.get_uptime(market_id) >= 0

    @pytest.mark.asyncio
    async def test_refresh_quotes_invalid_orderbook(self, bot: MarketMakerBot):
        """Invalid orderbook should not place orders."""
        market_id = "bad_market"
        market_info = {
            "id": market_id,
            "yes_token_id": "y",
            "no_token_id": "n",
            "_reward": {},
        }
        bot.orderbooks[market_id] = {"best_bid": 0, "best_ask": 1}
        bot.uptime_tracker.register_market(market_id)

        await bot.refresh_quotes_for_market(market_id, market_info)

        # Should NOT have placed any orders
        assert bot.order_executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_cancel_stale_orders(self, bot: MarketMakerBot):
        import time
        stale_order = {
            "id": "old_order",
            "timestamp": (time.time() - 10) * 1000,  # 10s old
        }
        bot.rest_client.get_open_orders = AsyncMock(return_value=[stale_order])

        await bot._cancel_stale_orders("test_market")

        bot.order_executor.batch_cancel_orders.assert_called_once()
        args = bot.order_executor.batch_cancel_orders.call_args[0][0]
        assert "old_order" in args

    @pytest.mark.asyncio
    async def test_place_quote_risk_rejection(self, bot: MarketMakerBot):
        """Quotes that fail risk checks should not be placed."""
        from src.market_maker.quote_engine import Quote

        # Max out exposure
        bot.risk_manager.inventory_manager.inventory.net_exposure_usd = 9999

        quote = Quote(side="BUY", price=0.5, size=10000, market="m1", token_id="t1")
        await bot._place_quote(quote, "YES")

        # Should NOT have placed (exposure exceeded)
        assert bot.order_executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_place_quote_success(self, bot: MarketMakerBot):
        from src.market_maker.quote_engine import Quote

        quote = Quote(side="BUY", price=0.45, size=100, market="m1", token_id="t1")
        await bot._place_quote(quote, "YES")

        bot.order_executor.place_order.assert_called_once()
        call_args = bot.order_executor.place_order.call_args[0][0]
        assert call_args["market"] == "m1"
        assert call_args["price"] == "0.45"

    @pytest.mark.asyncio
    async def test_cleanup(self, bot: MarketMakerBot):
        bot.running = True
        bot.market_infos = {"m1": {}, "m2": {}}

        await bot.cleanup()

        assert bot.running is False
        # Should cancel for all markets + fallback
        assert bot.order_executor.cancel_all_orders.call_count >= 2
        bot.rest_client.close.assert_called_once()
        bot.reward_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cancel_replace_cycle_single_iteration(self, bot: MarketMakerBot):
        """Verify the cycle processes multiple markets."""
        market_infos = [
            {"id": "m1", "yes_token_id": "y1", "no_token_id": "n1", "_reward": {}},
            {"id": "m2", "yes_token_id": "y2", "no_token_id": "n2", "_reward": {}},
        ]

        for info in market_infos:
            bot.orderbooks[info["id"]] = {"best_bid": 0.45, "best_ask": 0.55}
            bot.uptime_tracker.register_market(info["id"])

        bot.running = True

        # Run one iteration then stop
        async def stop_after_one():
            await asyncio.sleep(0.1)
            bot.running = False

        await asyncio.gather(
            bot.run_cancel_replace_cycle(market_infos),
            stop_after_one(),
        )

        # Should have placed orders for both markets
        assert bot.order_executor.place_order.call_count >= 2

    @pytest.mark.asyncio
    async def test_uptime_tracker_integration(self, bot: MarketMakerBot):
        """End-to-end uptime tracking across quote cycles."""
        market_id = "uptime_test"
        market_info = {
            "id": market_id,
            "yes_token_id": "y",
            "no_token_id": "n",
            "_reward": {"rate_per_day": 50.0, "volume_24h": 10000.0,
                        "breakeven_spread_bps": 5.0, "allocated_capital": 2000.0},
        }

        bot.orderbooks[market_id] = {"best_bid": 0.45, "best_ask": 0.55}
        bot.uptime_tracker.register_market(market_id)

        # Run 3 quote cycles
        for _ in range(3):
            bot.last_quote_times[market_id] = 0  # Force refresh
            await bot.refresh_quotes_for_market(market_id, market_info)

        stats = bot.uptime_tracker.get_stats(market_id)
        assert stats["currently_live"] is True
        assert stats["uptime_pct"] >= 0

    @pytest.mark.asyncio
    async def test_handleorderbook_update(self, bot: MarketMakerBot):
        bot.orderbooks["m1"] = {"best_bid": 0.40, "best_ask": 0.60}
        bot._handle_orderbook_update({
            "market": "m1",
            "book": {"best_bid": 0.42, "best_ask": 0.58},
        })
        assert bot.orderbooks["m1"]["best_bid"] == 0.42

    @pytest.mark.asyncio
    async def test_handleorderbook_update_wrong_market(self, bot: MarketMakerBot):
        bot.orderbooks["m1"] = {"best_bid": 0.40, "best_ask": 0.60}
        bot._handle_orderbook_update({
            "market": "m2",
            "book": {"best_bid": 0.42, "best_ask": 0.58},
        })
        # m1 should be unchanged
        assert bot.orderbooks["m1"]["best_bid"] == 0.40
