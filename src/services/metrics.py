from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

orders_placed_counter = Counter(
    "pm_mm_orders_placed_total", "Total orders placed", ["side", "outcome"]
)
orders_filled_counter = Counter(
    "pm_mm_orders_filled_total", "Total orders filled", ["side", "outcome"]
)
orders_cancelled_counter = Counter(
    "pm_mm_orders_cancelled_total", "Total orders cancelled"
)
inventory_gauge = Gauge(
    "pm_mm_inventory", "Current inventory positions", ["type"]
)
exposure_gauge = Gauge("pm_mm_exposure_usd", "Current net exposure in USD")
spread_gauge = Gauge("pm_mm_spread_bps", "Current spread in basis points")
profit_gauge = Gauge("pm_mm_profit_usd", "Cumulative profit in USD")
quote_latency_histogram = Histogram(
    "pm_mm_quote_latency_ms",
    "Quote generation and placement latency in milliseconds",
    buckets=[10, 50, 100, 250, 500, 1000],
)

# Reward program metrics
reward_efficiency_gauge = Gauge(
    "pm_mm_reward_efficiency", "Reward efficiency score per market", ["market_id"]
)
reward_rate_gauge = Gauge(
    "pm_mm_reward_rate_daily", "Daily reward rate USD per market", ["market_id"]
)
reward_spread_adjustment_gauge = Gauge(
    "pm_mm_reward_spread_adj_bps", "Reward-adjusted spread in bps", ["market_id"]
)
markets_active_gauge = Gauge(
    "pm_mm_markets_active", "Number of actively quoted markets"
)
allocated_capital_gauge = Gauge(
    "pm_mm_allocated_capital_usd", "Capital allocated per market", ["market_id"]
)

# Uptime metrics
uptime_gauge = Gauge(
    "pm_mm_uptime_pct", "Quote uptime percentage per market", ["market_id"]
)
uptime_avg_gauge = Gauge(
    "pm_mm_uptime_avg_pct", "Average quote uptime across all markets"
)
longest_gap_gauge = Gauge(
    "pm_mm_longest_gap_s", "Longest quote gap in seconds per market", ["market_id"]
)


def start_metrics_server(host: str, port: int) -> None:
    start_http_server(port, addr=host)


def record_order_placed(side: str, outcome: str) -> None:
    orders_placed_counter.labels(side=side, outcome=outcome).inc()


def record_order_filled(side: str, outcome: str) -> None:
    orders_filled_counter.labels(side=side, outcome=outcome).inc()


def record_order_cancelled() -> None:
    orders_cancelled_counter.inc()


def record_inventory(inventory_type: str, value: float) -> None:
    inventory_gauge.labels(type=inventory_type).set(value)


def record_exposure(exposure_usd: float) -> None:
    exposure_gauge.set(exposure_usd)


def record_spread(spread_bps: float) -> None:
    spread_gauge.set(spread_bps)


def record_profit(profit_usd: float) -> None:
    profit_gauge.set(profit_usd)


def record_quote_latency(latency_ms: float) -> None:
    quote_latency_histogram.observe(latency_ms)


def record_reward_efficiency(market_id: str, efficiency: float) -> None:
    reward_efficiency_gauge.labels(market_id=market_id).set(efficiency)


def record_reward_rate(market_id: str, rate: float) -> None:
    reward_rate_gauge.labels(market_id=market_id).set(rate)


def record_reward_spread_adj(market_id: str, spread_bps: float) -> None:
    reward_spread_adjustment_gauge.labels(market_id=market_id).set(spread_bps)


def record_active_markets(count: int) -> None:
    markets_active_gauge.set(count)


def record_allocated_capital(market_id: str, capital: float) -> None:
    allocated_capital_gauge.labels(market_id=market_id).set(capital)


def record_uptime(market_id: str, uptime_pct: float) -> None:
    uptime_gauge.labels(market_id=market_id).set(uptime_pct)


def record_avg_uptime(avg_pct: float) -> None:
    uptime_avg_gauge.set(avg_pct)


def record_longest_gap(market_id: str, gap_s: float) -> None:
    longest_gap_gauge.labels(market_id=market_id).set(gap_s)

