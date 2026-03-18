from __future__ import annotations

import os
import pytest

# Set required env vars before any Settings import
os.environ.setdefault("PRIVATE_KEY", "0x" + "ab" * 32)
os.environ.setdefault("PUBLIC_ADDRESS", "0x" + "cd" * 20)
os.environ.setdefault("MARKET_ID", "0x" + "ef" * 32)

from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        private_key="0x" + "ab" * 32,
        public_address="0x" + "cd" * 20,
        market_id="0x" + "ef" * 32,
        default_size=100.0,
        min_spread_bps=10,
        max_exposure_usd=10000.0,
        min_exposure_usd=-10000.0,
        target_inventory_balance=0.0,
        inventory_skew_limit=0.3,
        max_position_size_usd=5000.0,
        reward_adjusted_spreads=True,
        target_markets_count=5,
        market_rotation_interval_s=3600,
        min_reward_rate_daily=1.0,
        max_volatility_threshold=0.15,
        capital_pool_usd=50000.0,
        uptime_target_pct=95.0,
        wide_spread_multiplier=3.0,
        volatility_widen_threshold=0.05,
    )
