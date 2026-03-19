"""
Order executor — uses OrderSigner for EIP-712 signing and PolymarketRestClient for placement.
"""
from __future__ import annotations

import time
from typing import Any

import structlog

from src.config import Settings
from src.polymarket.order_signer import OrderSigner
from src.polymarket.rest_client import PolymarketRestClient

logger = structlog.get_logger(__name__)


class OrderExecutor:
    def __init__(self, settings: Settings, order_signer: OrderSigner, rest_client: PolymarketRestClient):
        self.settings = settings
        self.order_signer = order_signer
        self.rest_client = rest_client

    async def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        neg_risk: bool = False,
        fee_rate_bps: int = 0,
    ) -> dict[str, Any]:
        """Build, sign, and place an order on the CLOB.

        Args:
            token_id: Outcome token ID
            price: Price per share (0-1)
            size: Number of shares
            side: "BUY" or "SELL"
            neg_risk: Whether this is a neg-risk market
            fee_rate_bps: Fee rate in basis points

        Returns:
            API response with orderID, status, etc.
        """
        signed_order = self.order_signer.build_order(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            fee_rate_bps=fee_rate_bps,
            neg_risk=neg_risk,
        )

        result = await self.rest_client.place_order(signed_order)

        logger.info(
            "order_placed",
            order_id=result.get("orderID"),
            status=result.get("status"),
            side=side,
            price=price,
            size=size,
            token_id=token_id[:16] + "...",
        )

        return result

    async def place_hedge_order(
        self,
        market_id: str,
        token_id: str,
        side: str,
        size: float,
        price: float,
        neg_risk: bool = False,
    ) -> dict[str, Any] | None:
        """Place a hedge order. Side is YES/NO, converted to BUY on correct token."""
        try:
            result = await self.place_order(
                token_id=token_id,
                price=price,
                size=size,
                side="BUY",  # Hedging always buys the opposite side
                neg_risk=neg_risk,
            )

            logger.info(
                "hedge_order_placed",
                market_id=market_id,
                hedge_side=side,
                price=price,
                size=size,
                order_id=result.get("orderID"),
            )
            return result

        except Exception as e:
            logger.error("hedge_order_failed", market_id=market_id, side=side, error=str(e))
            return None

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self.rest_client.cancel_order(order_id)
            logger.info("order_cancelled", order_id=order_id)
            return True
        except Exception as e:
            logger.error("order_cancellation_failed", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self, market_id: str) -> int:
        """Cancel all orders for a market."""
        try:
            result = await self.rest_client.cancel_market_orders(market_id)
            cancelled = result.get("canceled", 0) if isinstance(result, dict) else 0
            logger.info("orders_cancelled", market_id=market_id, count=cancelled)
            return cancelled
        except Exception as e:
            logger.error("cancel_all_orders_failed", market_id=market_id, error=str(e))
            return 0

    async def batch_cancel_orders(self, order_ids: list[str]) -> int:
        """Cancel multiple orders by ID."""
        if not order_ids:
            return 0

        try:
            await self.rest_client.cancel_orders(order_ids)
            logger.info("batch_orders_cancelled", count=len(order_ids))
            return len(order_ids)
        except Exception as e:
            logger.error("batch_cancel_failed", error=str(e))
            # Fallback: cancel one by one
            cancelled = 0
            for oid in order_ids:
                if await self.cancel_order(oid):
                    cancelled += 1
            return cancelled
