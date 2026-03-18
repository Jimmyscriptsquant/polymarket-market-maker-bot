from __future__ import annotations

import pytest

from src.inventory.inventory_manager import Inventory, InventoryManager


class TestInventory:
    def test_initial_state(self):
        inv = Inventory()
        assert inv.yes_position == 0.0
        assert inv.no_position == 0.0
        assert inv.net_exposure_usd == 0.0
        assert inv.total_value_usd == 0.0

    def test_update_yes_buy(self):
        inv = Inventory()
        inv.update(yes_delta=100, no_delta=0, price=0.5)
        assert inv.yes_position == 100
        assert inv.net_exposure_usd == 50.0  # 100 * 0.5
        assert inv.total_value_usd == 50.0

    def test_update_no_buy(self):
        inv = Inventory()
        inv.update(yes_delta=0, no_delta=100, price=0.5)
        assert inv.no_position == 100
        assert inv.net_exposure_usd == -50.0  # 0 - 100*0.5
        assert inv.total_value_usd == 50.0

    def test_update_both_sides(self):
        inv = Inventory()
        inv.update(yes_delta=100, no_delta=100, price=0.5)
        assert inv.yes_position == 100
        assert inv.no_position == 100
        assert inv.net_exposure_usd == 0.0  # Balanced
        assert inv.total_value_usd == 100.0

    def test_get_skew_balanced(self):
        inv = Inventory()
        inv.update(100, 100, 0.5)
        assert inv.get_skew() == 0.0

    def test_get_skew_imbalanced(self):
        inv = Inventory()
        inv.update(200, 0, 0.5)
        skew = inv.get_skew()
        assert skew == 1.0  # Fully skewed

    def test_get_skew_empty(self):
        inv = Inventory()
        assert inv.get_skew() == 0.0

    def test_is_balanced(self):
        inv = Inventory()
        inv.update(100, 80, 0.5)
        # net = 50 - 40 = 10, total = 90, skew = 10/90 ≈ 0.11
        assert inv.is_balanced(0.3) is True
        assert inv.is_balanced(0.05) is False


class TestInventoryManager:
    def test_can_quote_yes_within_limit(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        assert mgr.can_quote_yes(100.0) is True

    def test_can_quote_yes_exceeds_limit(self):
        mgr = InventoryManager(100, -100, 0.0)
        mgr.inventory.net_exposure_usd = 90
        assert mgr.can_quote_yes(20.0) is False

    def test_can_quote_no_within_limit(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        assert mgr.can_quote_no(100.0) is True

    def test_can_quote_no_exceeds_limit(self):
        mgr = InventoryManager(100, -100, 0.0)
        mgr.inventory.net_exposure_usd = -90
        assert mgr.can_quote_no(20.0) is False

    def test_get_quote_size_yes_normal(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        size = mgr.get_quote_size_yes(100.0, 0.5)
        assert size == 100.0

    def test_get_quote_size_yes_above_target(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        mgr.inventory.net_exposure_usd = 500  # Above target of 0
        size = mgr.get_quote_size_yes(100.0, 0.5)
        assert size == 50.0

    def test_get_quote_size_yes_at_limit(self):
        mgr = InventoryManager(1000, -1000, 0.0)
        mgr.inventory.net_exposure_usd = 950
        size = mgr.get_quote_size_yes(100.0, 0.5)
        assert size == pytest.approx(100.0)  # min(100, 50/0.5) = 100

    def test_get_quote_size_no_normal(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        size = mgr.get_quote_size_no(100.0, 0.5)
        assert size == 100.0

    def test_get_quote_size_no_below_target(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        mgr.inventory.net_exposure_usd = -500
        size = mgr.get_quote_size_no(100.0, 0.5)
        assert size == 50.0

    def test_should_rebalance(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        mgr.inventory.update(200, 0, 0.5)  # Fully skewed
        assert mgr.should_rebalance(0.3) is True

    def test_should_not_rebalance(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        mgr.inventory.update(100, 100, 0.5)  # Balanced
        assert mgr.should_rebalance(0.3) is False

    def test_get_rebalance_target_small_skew(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        # No skew
        yes_d, no_d = mgr.get_rebalance_target()
        assert yes_d == 0.0
        assert no_d == 0.0

    def test_update_inventory_logs(self):
        mgr = InventoryManager(10000, -10000, 0.0)
        mgr.update_inventory(50, 30, 0.6)
        assert mgr.inventory.yes_position == 50
        assert mgr.inventory.no_position == 30
