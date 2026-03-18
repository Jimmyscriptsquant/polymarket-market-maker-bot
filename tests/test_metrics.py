from __future__ import annotations

import pytest

from src.services.metrics import (
    record_order_placed,
    record_order_filled,
    record_order_cancelled,
    record_inventory,
    record_exposure,
    record_spread,
    record_profit,
    record_quote_latency,
    record_reward_efficiency,
    record_reward_rate,
    record_reward_spread_adj,
    record_active_markets,
    record_allocated_capital,
    record_uptime,
    record_avg_uptime,
    record_longest_gap,
)


class TestMetrics:
    """Smoke tests — verify all metric recording functions run without error."""

    def test_record_order_placed(self):
        record_order_placed("BUY", "YES")
        record_order_placed("BUY", "NO")

    def test_record_order_filled(self):
        record_order_filled("BUY", "YES")

    def test_record_order_cancelled(self):
        record_order_cancelled()

    def test_record_inventory(self):
        record_inventory("yes", 100.0)
        record_inventory("no", 50.0)

    def test_record_exposure(self):
        record_exposure(5000.0)

    def test_record_spread(self):
        record_spread(15.0)

    def test_record_profit(self):
        record_profit(250.0)

    def test_record_quote_latency(self):
        record_quote_latency(42.0)

    def test_record_reward_efficiency(self):
        record_reward_efficiency("m1", 150.0)

    def test_record_reward_rate(self):
        record_reward_rate("m1", 500.0)

    def test_record_reward_spread_adj(self):
        record_reward_spread_adj("m1", 25.0)

    def test_record_active_markets(self):
        record_active_markets(10)

    def test_record_allocated_capital(self):
        record_allocated_capital("m1", 5000.0)

    def test_record_uptime(self):
        record_uptime("m1", 97.5)

    def test_record_avg_uptime(self):
        record_avg_uptime(95.0)

    def test_record_longest_gap(self):
        record_longest_gap("m1", 30.0)
