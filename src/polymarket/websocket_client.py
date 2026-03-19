"""
Polymarket CLOB WebSocket client with auto-reconnect and re-subscribe.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import structlog
import websockets

from src.config import Settings

logger = structlog.get_logger(__name__)

MAX_RECONNECT_DELAY = 30.0
INITIAL_RECONNECT_DELAY = 1.0


class PolymarketWebSocketClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ws_url = settings.polymarket_ws_url
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.message_handlers: dict[str, list[Callable]] = {}
        self.running = False

        # Track subscriptions for re-subscribe on reconnect
        self._subscriptions: list[dict[str, str]] = []
        self._reconnect_delay = INITIAL_RECONNECT_DELAY

    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type. Multiple handlers allowed."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        if handler not in self.message_handlers[message_type]:
            self.message_handlers[message_type].append(handler)

    async def connect(self):
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            logger.info("websocket_connected", url=self.ws_url)
            self.running = True
            self._reconnect_delay = INITIAL_RECONNECT_DELAY
        except Exception as e:
            logger.error("websocket_connection_failed", error=str(e))
            raise

    async def _reconnect(self):
        """Reconnect and re-subscribe to all channels."""
        logger.info("websocket_reconnecting", delay=self._reconnect_delay)
        await asyncio.sleep(self._reconnect_delay)

        # Exponential backoff
        self._reconnect_delay = min(self._reconnect_delay * 2, MAX_RECONNECT_DELAY)

        try:
            await self.connect()
            # Re-subscribe to all tracked subscriptions
            for sub in self._subscriptions:
                await self._send(sub)
                logger.info("websocket_resubscribed", channel=sub.get("channel"), market=sub.get("market", ""))
        except Exception as e:
            logger.error("websocket_reconnect_failed", error=str(e))

    async def _send(self, message: dict[str, Any]):
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(json.dumps(message))

    async def subscribe_orderbook(self, token_id: str):
        """Subscribe to L2 orderbook updates for a token."""
        if not self.websocket:
            await self.connect()

        sub = {
            "type": "subscribe",
            "channel": "book",
            "assets_id": token_id,
        }
        await self._send(sub)
        # Track for reconnection
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)
        logger.info("orderbook_subscribed", token_id=token_id)

    async def subscribe_trades(self, market_id: str):
        """Subscribe to trade updates for a market (for fill detection)."""
        if not self.websocket:
            await self.connect()

        sub = {
            "type": "subscribe",
            "channel": "trades",
            "market": market_id,
        }
        await self._send(sub)
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)
        logger.info("trades_subscribed", market_id=market_id)

    async def subscribe_user(self, market_id: str):
        """Subscribe to user-specific events (fills, order updates)."""
        if not self.websocket:
            await self.connect()

        sub = {
            "type": "subscribe",
            "channel": "user",
            "market": market_id,
        }
        await self._send(sub)
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)
        logger.info("user_channel_subscribed", market_id=market_id)

    async def listen(self):
        """Main listen loop with auto-reconnect and re-subscribe."""
        if not self.websocket:
            await self.connect()

        while self.running:
            try:
                message = await self.websocket.recv()

                # Skip non-JSON messages (heartbeats, pings, empty frames)
                if not message or not isinstance(message, str):
                    continue
                message = message.strip()
                if not message or not message.startswith(("{", "[")):
                    continue

                data = json.loads(message)

                msg_type = data.get("type") or data.get("channel") or data.get("event_type", "")

                if msg_type and msg_type in self.message_handlers:
                    for handler in self.message_handlers[msg_type]:
                        try:
                            result = handler(data)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.error("handler_error", type=msg_type, error=str(e))

            except websockets.exceptions.ConnectionClosed:
                logger.warning("websocket_connection_closed")
                if self.running:
                    await self._reconnect()
            except json.JSONDecodeError:
                # Non-JSON message — skip silently
                continue
            except Exception as e:
                logger.error("websocket_listen_error", error=str(e))
                if self.running:
                    await self._reconnect()

    async def unsubscribe_all(self):
        """Unsubscribe from all channels."""
        if self.websocket:
            for sub in self._subscriptions:
                unsub = {**sub, "type": "unsubscribe"}
                try:
                    await self._send(unsub)
                except Exception:
                    pass
        self._subscriptions.clear()

    async def close(self):
        self.running = False
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
