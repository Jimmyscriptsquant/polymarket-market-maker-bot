"""
Polymarket CLOB order signing — wraps the official py-clob-client for signing,
adds HMAC auth header generation.
"""
from __future__ import annotations

import base64
import hashlib
import hmac as hmac_lib
import time
from typing import Any

from eth_account import Account
from eth_account.signers.local import LocalAccount
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
from web3 import Web3
import structlog

logger = structlog.get_logger(__name__)

CHAIN_ID = 137  # Polygon mainnet


class OrderSigner:
    """Signs Polymarket CLOB orders using the official py-clob-client."""

    def __init__(self, private_key: str, proxy_address: str = ""):
        self.account: LocalAccount = Account.from_key(private_key)
        self.proxy_address: str = proxy_address

        # Official client handles EIP-712 signing correctly
        self._clob = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=CHAIN_ID,
            funder=proxy_address if proxy_address else None,
            signature_type=2 if proxy_address else 0,  # POLY_GNOSIS_SAFE if proxy
        )

        # API credentials
        self.api_key: str = ""
        self.api_secret: str = ""
        self.api_passphrase: str = ""

    def set_api_credentials(self, api_key: str, secret: str, passphrase: str) -> None:
        self.api_key = api_key
        self.api_secret = secret
        self.api_passphrase = passphrase

    def derive_api_credentials(self) -> dict[str, str]:
        """Derive API credentials using L1 auth via official client."""
        creds = self._clob.derive_api_key()
        self._clob.set_api_creds(creds)
        self.api_key = creds.api_key
        self.api_secret = creds.api_secret
        self.api_passphrase = creds.api_passphrase
        return {
            "apiKey": creds.api_key,
            "secret": creds.api_secret,
            "passphrase": creds.api_passphrase,
        }

    # ── L2: HMAC Auth Signature ─────────────────────────────────────

    def build_hmac_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        secret_bytes = base64.urlsafe_b64decode(self.api_secret)
        message = timestamp + method.upper() + request_path
        if body:
            message += body
        h = hmac_lib.new(secret_bytes, message.encode("utf-8"), hashlib.sha256)
        return base64.urlsafe_b64encode(h.digest()).decode("utf-8")

    def create_l2_headers(
        self, method: str, request_path: str, body: str = ""
    ) -> dict[str, str]:
        timestamp = str(int(time.time()))
        signature = self.build_hmac_signature(timestamp, method, request_path, body)

        return {
            "POLY_ADDRESS": self.account.address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_API_KEY": self.api_key,
            "POLY_PASSPHRASE": self.api_passphrase,
        }

    # ── Order Building & Signing ────────────────────────────────────

    def build_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        neg_risk: bool = False,
    ) -> dict[str, Any]:
        """Build a complete signed order using the official client.

        Returns the signed order object ready for placement via the official client.
        """
        signed_order = self._clob.create_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.upper(),
            ),
            options=PartialCreateOrderOptions(neg_risk=neg_risk),
        )
        return signed_order

    def place_order_via_client(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        neg_risk: bool = False,
    ) -> dict[str, Any]:
        """Build, sign, and place an order using the official client (synchronous)."""
        signed_order = self.build_order(token_id, price, size, side, neg_risk)
        result = self._clob.post_order(signed_order, OrderType.GTC)
        return result

    def cancel_order_via_client(self, order_id: str) -> dict[str, Any]:
        """Cancel an order using the official client."""
        return self._clob.cancel(order_id)

    def cancel_all_via_client(self) -> dict[str, Any]:
        """Cancel all orders using the official client."""
        return self._clob.cancel_all()

    def get_address(self) -> str:
        """Return the trading address (proxy if set, otherwise EOA)."""
        return self.proxy_address if self.proxy_address else self.account.address

    def get_eoa_address(self) -> str:
        return self.account.address
