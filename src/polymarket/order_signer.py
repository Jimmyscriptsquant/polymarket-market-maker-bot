"""
Polymarket CLOB order signing — EIP-712 typed data + HMAC API auth.

Two auth levels:
- L1: EIP-712 signed messages (for creating API keys, signing orders)
- L2: HMAC-SHA256 (for authenticated REST endpoints)
"""
from __future__ import annotations

import base64
import hashlib
import hmac as hmac_lib
import time
from typing import Any

from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_abi import encode as abi_encode
from web3 import Web3
import structlog

logger = structlog.get_logger(__name__)

# ── Polymarket contract addresses (Polygon mainnet) ─────────────────
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_EXCHANGE_ADDRESS = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
COLLATERAL_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC on Polygon
CONDITIONAL_TOKENS_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

CHAIN_ID = 137  # Polygon mainnet

# ── EIP-712 Type Hashes ─────────────────────────────────────────────

# Domain for CLOB auth messages
CLOB_AUTH_DOMAIN = {
    "name": "ClobAuthDomain",
    "version": "1",
    "chainId": CHAIN_ID,
}

# Domain for CTF Exchange orders
CTF_EXCHANGE_DOMAIN = {
    "name": "Polymarket CTF Exchange",
    "version": "1",
    "chainId": CHAIN_ID,
    "verifyingContract": EXCHANGE_ADDRESS,
}

NEG_RISK_CTF_EXCHANGE_DOMAIN = {
    "name": "Polymarket CTF Exchange",
    "version": "1",
    "chainId": CHAIN_ID,
    "verifyingContract": NEG_RISK_EXCHANGE_ADDRESS,
}

# EIP-712 type definitions for order signing
ORDER_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "salt", "type": "uint256"},
        {"name": "maker", "type": "address"},
        {"name": "signer", "type": "address"},
        {"name": "taker", "type": "address"},
        {"name": "tokenId", "type": "uint256"},
        {"name": "makerAmount", "type": "uint256"},
        {"name": "takerAmount", "type": "uint256"},
        {"name": "expiration", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "feeRateBps", "type": "uint256"},
        {"name": "side", "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
    ],
}

CLOB_AUTH_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
    ],
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "message", "type": "string"},
    ],
}

# Side encoding: BUY=0, SELL=1
SIDE_BUY = 0
SIDE_SELL = 1

# Signature type: EOA=0, POLY_PROXY=1, POLY_GNOSIS_SAFE=2
SIG_TYPE_EOA = 0
SIG_TYPE_POLY_PROXY = 1

# USDC has 6 decimals
USDC_DECIMALS = 6
USDC_MULTIPLIER = 10 ** USDC_DECIMALS


class OrderSigner:
    """Signs Polymarket CLOB orders using EIP-712 and provides HMAC auth."""

    def __init__(self, private_key: str):
        self.account: LocalAccount = Account.from_key(private_key)
        self.web3 = Web3()

        # API credentials (set after calling create_or_derive_api_key)
        self.api_key: str = ""
        self.api_secret: str = ""
        self.api_passphrase: str = ""

    def set_api_credentials(self, api_key: str, secret: str, passphrase: str) -> None:
        """Set L2 API credentials for HMAC-authenticated endpoints."""
        self.api_key = api_key
        self.api_secret = secret
        self.api_passphrase = passphrase

    # ── L1: EIP-712 Auth Signature ──────────────────────────────────

    def create_l1_auth_headers(self, nonce: int = 0) -> dict[str, str]:
        """Create L1 auth headers using EIP-712 signed message."""
        timestamp = str(int(time.time()))
        address = self.account.address

        message_data = {
            "address": address,
            "timestamp": timestamp,
            "nonce": nonce,
            "message": "This message attests that I control the given wallet",
        }

        signable = {
            "types": CLOB_AUTH_TYPES,
            "primaryType": "ClobAuth",
            "domain": CLOB_AUTH_DOMAIN,
            "message": message_data,
        }

        signed = self.account.sign_typed_data(
            signable["domain"],
            {"ClobAuth": CLOB_AUTH_TYPES["ClobAuth"]},
            signable["message"],
        )

        return {
            "POLY_ADDRESS": address,
            "POLY_SIGNATURE": signed.signature.hex(),
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

    # ── L2: HMAC Auth Signature ─────────────────────────────────────

    def build_hmac_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        """Build HMAC-SHA256 signature for L2 authenticated endpoints."""
        secret_bytes = base64.urlsafe_b64decode(self.api_secret)
        message = timestamp + method.upper() + request_path
        if body:
            message += body
        h = hmac_lib.new(secret_bytes, message.encode("utf-8"), hashlib.sha256)
        return base64.urlsafe_b64encode(h.digest()).decode("utf-8")

    def create_l2_headers(
        self, method: str, request_path: str, body: str = ""
    ) -> dict[str, str]:
        """Create L2 HMAC auth headers for trading endpoints."""
        timestamp = str(int(time.time()))
        signature = self.build_hmac_signature(timestamp, method, request_path, body)

        return {
            "POLY_ADDRESS": self.account.address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_API_KEY": self.api_key,
            "POLY_PASSPHRASE": self.api_passphrase,
        }

    # ── Order Signing (EIP-712) ─────────────────────────────────────

    def sign_order(
        self,
        order: dict[str, Any],
        neg_risk: bool = False,
    ) -> str:
        """Sign a Polymarket CLOB order using EIP-712 typed data.

        Args:
            order: Order dict with fields: salt, maker, signer, taker,
                   tokenId, makerAmount, takerAmount, expiration, nonce,
                   feeRateBps, side, signatureType
            neg_risk: Whether this is a neg-risk market

        Returns:
            Hex-encoded signature string
        """
        domain = NEG_RISK_CTF_EXCHANGE_DOMAIN if neg_risk else CTF_EXCHANGE_DOMAIN

        # Ensure all values are proper ints for EIP-712
        typed_order = {
            "salt": int(order["salt"]),
            "maker": Web3.to_checksum_address(order["maker"]),
            "signer": Web3.to_checksum_address(order["signer"]),
            "taker": Web3.to_checksum_address(order.get("taker", "0x0000000000000000000000000000000000000000")),
            "tokenId": int(order["tokenId"]),
            "makerAmount": int(order["makerAmount"]),
            "takerAmount": int(order["takerAmount"]),
            "expiration": int(order.get("expiration", 0)),
            "nonce": int(order.get("nonce", 0)),
            "feeRateBps": int(order.get("feeRateBps", 0)),
            "side": int(order["side"]),
            "signatureType": int(order.get("signatureType", SIG_TYPE_EOA)),
        }

        signed = self.account.sign_typed_data(
            domain,
            {"Order": ORDER_TYPES["Order"]},
            typed_order,
        )

        return signed.signature.hex()

    def build_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        fee_rate_bps: int = 0,
        neg_risk: bool = False,
        expiration: int = 0,
        nonce: int = 0,
    ) -> dict[str, Any]:
        """Build a complete signed order ready for the CLOB API.

        Args:
            token_id: The outcome token ID
            price: Price per share (0-1)
            size: Number of shares
            side: "BUY" or "SELL"
            fee_rate_bps: Fee rate in basis points
            neg_risk: Whether this is a neg-risk market
            expiration: Order expiration timestamp (0 = no expiry)
            nonce: Order nonce

        Returns:
            Complete order dict with signature, ready for POST /order
        """
        side_int = SIDE_BUY if side == "BUY" else SIDE_SELL
        salt = int(time.time() * 1000)
        address = self.account.address

        # Calculate amounts in USDC units (6 decimals)
        if side == "BUY":
            # Maker pays USDC, receives outcome tokens
            maker_amount = int(size * price * USDC_MULTIPLIER)
            taker_amount = int(size * USDC_MULTIPLIER)
        else:
            # Maker sends outcome tokens, receives USDC
            maker_amount = int(size * USDC_MULTIPLIER)
            taker_amount = int(size * price * USDC_MULTIPLIER)

        raw_order = {
            "salt": salt,
            "maker": address,
            "signer": address,
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": token_id,
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "expiration": expiration,
            "nonce": nonce,
            "feeRateBps": fee_rate_bps,
            "side": side_int,
            "signatureType": SIG_TYPE_EOA,
        }

        signature = self.sign_order(raw_order, neg_risk=neg_risk)

        return {
            "order": {
                "salt": salt,
                "maker": address,
                "signer": address,
                "taker": "0x0000000000000000000000000000000000000000",
                "tokenId": token_id,
                "makerAmount": str(maker_amount),
                "takerAmount": str(taker_amount),
                "expiration": str(expiration),
                "nonce": str(nonce),
                "feeRateBps": str(fee_rate_bps),
                "side": "BUY" if side == "BUY" else "SELL",
                "signature": signature,
                "signatureType": SIG_TYPE_EOA,
            },
            "owner": self.api_key,
            "orderType": "GTC",
        }

    def get_address(self) -> str:
        return self.account.address
