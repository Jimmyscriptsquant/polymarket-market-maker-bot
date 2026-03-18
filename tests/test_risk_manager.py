from __future__ import annotations

import pytest

from src.config import Settings
from src.inventory.inventory_manager import InventoryManager
from src.risk.risk_manager import RiskManager


@pytest.fixture
def inventory_mgr(settings: Settings) -> InventoryManager:
    return InventoryManager(
        settings.max_exposure_usd,
        settings.min_exposure_usd,
        settings.target_inventory_balance,
    )


@pytest.fixture
def risk_mgr(settings: Settings, inventory_mgr: InventoryManager) -> RiskManager:
    return RiskManager(settings, inventory_mgr)


class TestRiskManager:
    def test_validate_order_ok(self, risk_mgr: RiskManager):
        ok, reason = risk_mgr.validate_order("BUY", 100.0)
        assert ok is True
        assert reason == "OK"

    def test_validate_order_size_exceeded(self, risk_mgr: RiskManager):
        ok, reason = risk_mgr.validate_order("BUY", 6000.0)
        assert ok is False
        assert "size" in reason.lower()

    def test_validate_order_exposure_exceeded_buy(self, risk_mgr: RiskManager):
        risk_mgr.inventory_manager.inventory.net_exposure_usd = 9500
        ok, reason = risk_mgr.validate_order("BUY", 1000.0)
        assert ok is False
        assert "exposure" in reason.lower()

    def test_validate_order_exposure_exceeded_sell(self, risk_mgr: RiskManager):
        risk_mgr.inventory_manager.inventory.net_exposure_usd = -9500
        ok, reason = risk_mgr.validate_order("SELL", 1000.0)
        assert ok is False
        assert "exposure" in reason.lower()

    def test_check_exposure_buy_ok(self, risk_mgr: RiskManager):
        assert risk_mgr.check_exposure_limits(100.0, "BUY") is True

    def test_check_exposure_sell_ok(self, risk_mgr: RiskManager):
        assert risk_mgr.check_exposure_limits(100.0, "SELL") is True

    def test_check_position_size_ok(self, risk_mgr: RiskManager):
        assert risk_mgr.check_position_size(4999.0) is True

    def test_check_position_size_exceeded(self, risk_mgr: RiskManager):
        assert risk_mgr.check_position_size(5001.0) is False

    def test_check_inventory_skew_ok(self, risk_mgr: RiskManager):
        assert risk_mgr.check_inventory_skew() is True

    def test_check_inventory_skew_exceeded(self, risk_mgr: RiskManager):
        inv = risk_mgr.inventory_manager.inventory
        inv.update(1000, 0, 0.5)  # Fully skewed
        assert risk_mgr.check_inventory_skew() is False

    def test_should_stop_trading_false(self, risk_mgr: RiskManager):
        assert risk_mgr.should_stop_trading() is False

    def test_should_stop_trading_near_limit(self, risk_mgr: RiskManager):
        risk_mgr.inventory_manager.inventory.net_exposure_usd = 9500
        assert risk_mgr.should_stop_trading() is True

    def test_should_stop_trading_negative_near_limit(self, risk_mgr: RiskManager):
        risk_mgr.inventory_manager.inventory.net_exposure_usd = -9500
        assert risk_mgr.should_stop_trading() is True
