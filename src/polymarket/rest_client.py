"""
Polymarket CLOB REST client with proper auth, retry, and circuit breaker.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import structlog

from src.config import Settings
from src.polymarket.order_signer import OrderSigner

logger = structlog.get_logger(__name__)

# Retry config
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.5  # seconds
RATE_LIMIT_DELAY = 2.0  # seconds on 429


class CircuitBreaker:
    """Simple circuit breaker to stop hammering a failing API."""

    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                "circuit_breaker_opened",
                failures=self.failure_count,
                reset_in_s=self.reset_timeout_s,
            )

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        # Check if reset timeout has passed
        if time.time() - self.last_failure_time > self.reset_timeout_s:
            self.is_open = False
            self.failure_count = 0
            logger.info("circuit_breaker_reset")
            return True
        return False


class PolymarketRestClient:
    """REST client for Polymarket CLOB API with L2 HMAC auth and retry logic."""

    def __init__(self, settings: Settings, order_signer: OrderSigner | None = None):
        self.settings = settings
        self.base_url = settings.polymarket_api_url
        self.signer = order_signer
        self.client = httpx.AsyncClient(timeout=30.0)
        self.circuit_breaker = CircuitBreaker()

    def _get_l2_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """Get L2 HMAC auth headers. Returns empty dict if signer not configured."""
        if not self.signer or not self.signer.api_key:
            return {}
        return self.signer.create_l2_headers(method, path, body)

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        auth: bool = False,
    ) -> Any:
        """Make an API request with retry, backoff, and circuit breaker."""
        if not self.circuit_breaker.can_proceed():
            raise RuntimeError("Circuit breaker is open — API temporarily unavailable")

        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}

        body_str = ""
        if json_body is not None:
            body_str = json.dumps(json_body, separators=(",", ":"))

        if auth:
            auth_headers = self._get_l2_headers(method, path, body_str)
            headers.update(auth_headers)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.request(
                    method,
                    url,
                    params=params,
                    content=body_str if json_body is not None else None,
                    headers=headers,
                )

                if response.status_code == 429:
                    # Rate limited — wait and retry
                    delay = RATE_LIMIT_DELAY * (attempt + 1)
                    logger.warning("rate_limited", path=path, retry_in=delay)
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                self.circuit_breaker.record_success()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                if status in (400, 401, 403, 404):
                    # Client errors — don't retry
                    logger.error("api_client_error", path=path, status=status, body=e.response.text[:200])
                    raise
                # Server errors — retry with backoff
                self.circuit_breaker.record_failure()
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("api_server_error", path=path, status=status, attempt=attempt + 1, retry_in=delay)
                await asyncio.sleep(delay)

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = e
                self.circuit_breaker.record_failure()
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("api_connection_error", path=path, error=str(e), attempt=attempt + 1, retry_in=delay)
                await asyncio.sleep(delay)

        raise last_error or RuntimeError(f"Request to {path} failed after {MAX_RETRIES} retries")

    # ── Public endpoints (no auth) ──────────────────────────────────

    async def get_markets(self, active: bool = True, closed: bool = False) -> list[dict[str, Any]]:
        params = {"active": str(active).lower(), "closed": str(closed).lower()}
        return await self._request("GET", "/markets", params=params)

    async def get_market_info(self, condition_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/markets/{condition_id}")

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Get orderbook for a specific token ID."""
        return await self._request("GET", "/book", params={"token_id": token_id})

    async def get_midpoint(self, token_id: str) -> dict[str, Any]:
        return await self._request("GET", "/midpoint", params={"token_id": token_id})

    async def get_spread(self, token_id: str) -> dict[str, Any]:
        return await self._request("GET", "/spread", params={"token_id": token_id})

    async def get_server_time(self) -> dict[str, Any]:
        return await self._request("GET", "/time")

    # ── Authenticated endpoints (L2 HMAC) ───────────────────────────

    async def get_open_orders(self, market: str | None = None, asset_id: str | None = None) -> list[dict[str, Any]]:
        """Get open orders. Returns list from paginated 'data' field."""
        params: dict[str, str] = {}
        if market:
            params["market"] = market
        if asset_id:
            params["asset_id"] = asset_id

        result = await self._request("GET", "/data/orders", params=params, auth=True)

        if isinstance(result, dict):
            return result.get("data", [])
        return result

    async def get_trades(
        self, market: str | None = None, after: int | None = None
    ) -> list[dict[str, Any]]:
        """Get trade/fill history."""
        params: dict[str, str] = {}
        if market:
            params["market"] = market
        if after:
            params["after"] = str(after)

        result = await self._request("GET", "/data/trades", params=params, auth=True)

        if isinstance(result, dict):
            return result.get("data", [])
        return result

    async def get_balance_allowance(
        self, asset_type: str = "COLLATERAL", token_id: str = ""
    ) -> dict[str, Any]:
        """Get balance and allowance for USDC or outcome tokens."""
        params: dict[str, str] = {"asset_type": asset_type}
        if token_id:
            params["token_id"] = token_id
        return await self._request("GET", "/balance-allowance", params=params, auth=True)

    async def place_order(self, signed_order: dict[str, Any]) -> dict[str, Any]:
        """Place a signed order on the CLOB."""
        return await self._request("POST", "/order", json_body=signed_order, auth=True)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a single order."""
        body = {"orderID": order_id}
        return await self._request("DELETE", "/order", json_body=body, auth=True)

    async def cancel_orders(self, order_ids: list[str]) -> dict[str, Any]:
        """Cancel multiple orders."""
        return await self._request("DELETE", "/orders", json_body=order_ids, auth=True)

    async def cancel_all(self) -> dict[str, Any]:
        """Cancel all open orders."""
        return await self._request("DELETE", "/cancel-all", auth=True)

    async def cancel_market_orders(self, market: str, asset_id: str = "") -> dict[str, Any]:
        """Cancel all orders for a specific market."""
        body: dict[str, str] = {"market": market}
        if asset_id:
            body["asset_id"] = asset_id
        return await self._request("DELETE", "/cancel-market-orders", json_body=body, auth=True)

    # ── API key management ──────────────────────────────────────────

    async def create_api_key(self) -> dict[str, str]:
        """Create a new API key using L1 auth."""
        if not self.signer:
            raise RuntimeError("OrderSigner required for API key creation")

        headers = self.signer.create_l1_auth_headers()
        headers["Content-Type"] = "application/json"

        response = await self.client.post(
            f"{self.base_url}/auth/api-key",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def derive_api_key(self) -> dict[str, str]:
        """Derive existing API key using L1 auth."""
        if not self.signer:
            raise RuntimeError("OrderSigner required for API key derivation")

        headers = self.signer.create_l1_auth_headers()

        response = await self.client.get(
            f"{self.base_url}/auth/derive-api-key",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()
