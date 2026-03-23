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
        mock_signer_inst.api_key = "test-key"
        mock_signer_inst.api_secret = "dGVzdC1zZWNyZXQ="
        mock_signer_inst.api_passphrase = "test-pass"
        mock_signer_inst.build_order.return_value = {"order": {}, "owner": "test-key", "orderType": "GTC"}
        mock_signer_inst.create_l2_headers.return_value = {}

        mock_rest_inst = mock_rest.return_value
        mock_rest_inst.get_markets = AsyncMock(return_value=[])
        mock_rest_inst.get_market_info = AsyncMock(return_value={
            "id": settings.market_id,
            "question": "Test?",
            "tokens": [
                {"outcome": "Yes", "token_id": "yes_tok"},
                {"outcome": "No", "token_id": "no_tok"},
            ],
        })
        mock_rest_inst.get_orderbook = AsyncMock(return_value={
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })
        mock_rest_inst.get_open_orders = AsyncMock(return_value=[])
        mock_rest_inst.get_balance_allowance = AsyncMock(return_value={"balance": "1000000000"})
        mock_rest_inst.close = AsyncMock()

        mock_ws_inst = mock_ws.return_value
        mock_ws_inst.websocket = None
        mock_ws_inst.running = False
        mock_ws_inst.connect = AsyncMock()
        mock_ws_inst.close = AsyncMock()
        mock_ws_inst.register_handler = MagicMock()

        mock_executor_inst = mock_executor.return_value
        mock_executor_inst.place_order = AsyncMock(return_value={"orderID": "order123", "status": "live"})
        mock_executor_inst.batch_cancel_orders = AsyncMock(return_value=0)
        mock_executor_inst.cancel_all_orders = AsyncMock(return_value=0)

        mock_redeem_inst = mock_redeem.return_value
        mock_redeem_inst.close = AsyncMock()
        mock_redeem_inst.auto_redeem_all = AsyncMock()

        mock_reward_inst = mock_reward.return_value
        mock_reward_inst.fetch_reward_markets = AsyncMock(return_value=[])
        mock_reward_inst.close = AsyncMock()

        bot = MarketMakerBot(settings)
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
        bot.settings.market_discovery_enabled = False
        result = await bot.discover_market()
        assert result is not None
        assert result["id"] == bot.settings.market_id

    @pytest.mark.asyncio
    async def test_discover_reward_markets_fallback(self, bot: MarketMakerBot):
        bot.reward_client.fetch_reward_markets = AsyncMock(return_value=[])
        result = await bot.discover_reward_markets()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_update_orderbook(self, bot: MarketMakerBot):
        # Need token map for update_orderbook to work
        bot._token_map["test_market"] = {"yes_token_id": "yes_tok", "no_token_id": "no_tok"}
        await bot.update_orderbook("test_market")
        assert "test_market" in bot.orderbooks
        assert bot.orderbooks["test_market"]["bids"][0]["price"] == "0.45"

    @pytest.mark.asyncio
    async def test_refresh_quotes_for_market(self, bot: MarketMakerBot):
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

        bot.orderbooks[market_id] = {
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        }
        bot._token_map[market_id] = {"yes_token_id": "yes_tok", "no_token_id": "no_tok", "neg_risk": False}
        bot.uptime_tracker.register_market(market_id)

        await bot.refresh_quotes_for_market(market_id, market_info)

        # Should have placed orders
        assert bot.order_executor.place_order.call_count >= 1
        assert bot.uptime_tracker.get_uptime(market_id) >= 0

    @pytest.mark.asyncio
    async def test_refresh_quotes_invalid_orderbook(self, bot: MarketMakerBot):
        market_id = "bad_market"
        market_info = {"id": market_id, "_reward": {}}
        bot.orderbooks[market_id] = {"bids": [], "asks": []}
        bot.uptime_tracker.register_market(market_id)

        await bot.refresh_quotes_for_market(market_id, market_info)
        assert bot.order_executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_cancel_all_market_orders(self, bot: MarketMakerBot):
        open_order = {"id": "order123"}
        bot.rest_client.get_open_orders = AsyncMock(return_value=[open_order])

        await bot._cancel_all_market_orders("test_market")

        bot.order_executor.batch_cancel_orders.assert_called_once()
        args = bot.order_executor.batch_cancel_orders.call_args[0][0]
        assert "order123" in args

    @pytest.mark.asyncio
    async def test_place_quote_risk_rejection(self, bot: MarketMakerBot):
        from src.market_maker.quote_engine import Quote

        bot.risk_manager.inventory_manager.inventory.net_exposure_usd = 9999
        quote = Quote(side="BUY", price=0.5, size=10000, market="m1", token_id="t1")
        await bot._place_quote(quote, "YES")
        assert bot.order_executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_place_quote_success(self, bot: MarketMakerBot):
        from src.market_maker.quote_engine import Quote

        quote = Quote(side="BUY", price=0.45, size=100, market="m1", token_id="t1")
        await bot._place_quote(quote, "YES")

        bot.order_executor.place_order.assert_called_once()
        # New API: place_order takes kwargs not a dict
        call_kwargs = bot.order_executor.place_order.call_args[1]
        assert call_kwargs["token_id"] == "t1"
        assert call_kwargs["price"] == 0.45
        assert call_kwargs["size"] == 100

    @pytest.mark.asyncio
    async def test_cleanup(self, bot: MarketMakerBot):
        bot.running = True
        bot.market_infos = {"m1": {}, "m2": {}}

        await bot.cleanup()

        assert bot.running is False
        assert bot.order_executor.cancel_all_orders.call_count >= 2
        bot.rest_client.close.assert_called_once()
        bot.reward_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cancel_replace_cycle_single_iteration(self, bot: MarketMakerBot):
        market_infos = [
            {"id": "m1", "yes_token_id": "y1", "no_token_id": "n1", "_reward": {"rate_per_day": 5.0, "max_spread_bps": 450, "min_shares": 20}},
            {"id": "m2", "yes_token_id": "y2", "no_token_id": "n2", "_reward": {"rate_per_day": 5.0, "max_spread_bps": 450, "min_shares": 20}},
        ]

        for info in market_infos:
            mid = info["id"]
            bot.orderbooks[mid] = {
                "bids": [{"price": "0.45", "size": "100"}],
                "asks": [{"price": "0.55", "size": "100"}],
            }
            bot._token_map[mid] = {
                "yes_token_id": info["yes_token_id"],
                "no_token_id": info["no_token_id"],
                "neg_risk": False,
            }
            bot.uptime_tracker.register_market(mid)

        bot.running = True

        async def stop_after_one():
            await asyncio.sleep(0.1)
            bot.running = False

        await asyncio.gather(
            bot.run_cancel_replace_cycle(market_infos),
            stop_after_one(),
        )

        assert bot.order_executor.place_order.call_count >= 2

    @pytest.mark.asyncio
    async def test_uptime_tracker_integration(self, bot: MarketMakerBot):
        market_id = "uptime_test"
        market_info = {
            "id": market_id,
            "yes_token_id": "y",
            "no_token_id": "n",
            "_reward": {"rate_per_day": 50.0, "volume_24h": 10000.0,
                        "breakeven_spread_bps": 5.0, "allocated_capital": 2000.0},
        }

        bot.orderbooks[market_id] = {
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        }
        bot._token_map[market_id] = {"yes_token_id": "y", "no_token_id": "n", "neg_risk": False}
        bot.uptime_tracker.register_market(market_id)

        for _ in range(3):
            bot.last_quote_times[market_id] = 0
            await bot.refresh_quotes_for_market(market_id, market_info)

        stats = bot.uptime_tracker.get_stats(market_id)
        assert stats["currently_live"] is True
        assert stats["uptime_pct"] >= 0

    @pytest.mark.asyncio
    async def test_handle_orderbook_update(self, bot: MarketMakerBot):
        """Orderbook updates via WS use asset_id lookup through token map."""
        bot._token_map["m1"] = {"yes_token_id": "yes_tok_123", "no_token_id": "no_tok_123"}
        bot.orderbooks["m1"] = {"bids": [{"price": "0.40", "size": "50"}], "asks": [{"price": "0.60", "size": "50"}]}

        bot._handle_orderbook_update({
            "asset_id": "yes_tok_123",
            "book": {"bids": [{"price": "0.42", "size": "50"}], "asks": [{"price": "0.58", "size": "50"}]},
        })
        assert bot.orderbooks["m1"]["bids"][0]["price"] == "0.42"

    @pytest.mark.asyncio
    async def test_handle_orderbook_update_wrong_asset(self, bot: MarketMakerBot):
        bot._token_map["m1"] = {"yes_token_id": "yes_tok_123", "no_token_id": "no_tok_123"}
        bot.orderbooks["m1"] = {"bids": [{"price": "0.40", "size": "50"}], "asks": []}

        bot._handle_orderbook_update({
            "asset_id": "unknown_token",
            "book": {"bids": [{"price": "0.99", "size": "50"}], "asks": []},
        })
        # m1 should be unchanged
        assert bot.orderbooks["m1"]["bids"][0]["price"] == "0.40"

    @pytest.mark.asyncio
    async def test_fill_detection(self, bot: MarketMakerBot):
        """WebSocket fill detection updates inventory and AS guard."""
        bot.order_signer.get_address.return_value = "0xTestAddress"
        bot._token_map["m1"] = {"yes_token_id": "y1", "no_token_id": "n1", "neg_risk": False}
        bot.orderbooks["m1"] = {"asks": [{"price": "0.55", "size": "100"}]}

        bot._handle_trade_update({
            "maker_address": "0xTestAddress",
            "market": "m1",
            "side": "BUY",
            "outcome": "YES",
            "price": "0.50",
            "size": "100",
        })

        # Inventory should be updated
        assert bot.inventory_manager.pnl.trade_count == 1

    @pytest.mark.asyncio
    async def test_drawdown_pauses_quoting(self, bot: MarketMakerBot):
        """When drawdown limit is hit, quoting should pause."""
        market_id = "dd_test"
        market_info = {"id": market_id, "_reward": {}}
        bot.orderbooks[market_id] = {
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        }
        bot._token_map[market_id] = {"yes_token_id": "y", "no_token_id": "n", "neg_risk": False}
        bot.uptime_tracker.register_market(market_id)

        # Simulate large loss exceeding daily limit
        bot.inventory_manager.pnl.realized_pnl = -300.0
        bot.inventory_manager.daily_loss_limit_usd = 200.0

        await bot.refresh_quotes_for_market(market_id, market_info)
        assert bot.order_executor.place_order.call_count == 0
