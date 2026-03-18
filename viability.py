"""Viability analysis using verified live reward data."""

markets = [
    ("Hungary PM - Peter Magyar",  200, 3.5, 200),
    ("Rubio 2028 GOP nom",         70, 3.5, 200),
    ("CA wealth tax 2026",          50, 3.5, 200),
    ("Massie KY-04 GOP nom",       40, 4.5, 200),
    ("Cepeda Colombia pres",        40, 3.5, 200),
    ("St Johns NCAA",               20, 3.5, 200),
    ("Nice mayor - Estrosi",        10, 3.5, 200),
    ("Arsenal EPL",                 10, 4.5, 200),
    ("Freddie Mac IPO",              3, 3.5, 200),
    ("Hottest year 2026",            3, 3.5, 200),
]

print("=" * 85)
print("  VIABILITY ANALYSIS  -  VERIFIED REWARD DATA FROM LIVE API")
print("=" * 85)

total_daily = 0
total_capital = 0

print()
hdr = f"{'#':<3} {'Market':<28} {'Rwd/d':>7} {'MaxSprd':>8} {'MinSz':>6} {'CapReq':>8} {'NetPnL/d':>9} {'ROI/d':>8}"
print(hdr)
print("-" * 85)

for i, (name, rate, spread, min_sz) in enumerate(markets):
    cap_needed = min_sz * 0.50 * 2  # both sides at ~$0.50 mid
    capture = rate * 0.80  # 80% capture rate
    daily_as_cost = cap_needed * 0.05 / 30  # 5%/month adverse selection
    net_daily = capture - daily_as_cost
    daily_roi = (net_daily / cap_needed * 100) if cap_needed > 0 else 0
    total_daily += net_daily
    total_capital += cap_needed
    print(f"{i+1:<3} {name:<28} ${rate:>5}/d {spread:>6.1f}%  {min_sz:>5} ${cap_needed:>6,.0f} ${net_daily:>7.1f} {daily_roi:>6.1f}%")

print("-" * 85)
gross = sum(r[1] for r in markets)
capture_80 = sum(r[1] * 0.8 for r in markets)
as_cost = total_capital * 0.05 / 30

print(f"""
  PORTFOLIO SUMMARY
  ==================================================
  Total capital required:   ${total_capital:>8,.0f}
  Gross daily reward:       ${gross:>8,.0f}
  Est capture (80%):        ${capture_80:>8,.1f}
  Est adverse sel cost:     ${as_cost:>8,.1f}
  NET daily estimate:       ${total_daily:>8,.1f}

  Monthly estimate:         ${total_daily * 30:>8,.1f}
  Annual estimate:          ${total_daily * 365:>8,.1f}
  Annual ROI:               {total_daily * 365 / total_capital * 100:>7.0f}%

  REALITY CHECK
  ==================================================
  Top market: Hungary PM pays $200/day, needs $200 capital
  - min_size: 200 shares on each side (YES + NO)
  - At $0.50 mid price, that is $100 per side = $200 total
  - max_spread: 3.5% = your bid/ask must be within 3.5% of mid
  - Verified orderbook: $0.63 bid / $0.65 ask (310bp spread)
  - Real depth: 339K bid shares, 330K ask shares
  - This market has REAL liquidity and REAL rewards

  HOW IT WORKS
  ==================================================
  1. Place limit BUY at $0.63 for 200 YES shares ($126 locked)
  2. Place limit BUY at $0.35 for 200 NO shares ($70 locked)
  3. Maintain both orders 24/7 within max_spread
  4. Polymarket scores your competitiveness daily
  5. You receive share of $200/day reward pool
  6. More competitive (tighter spread, more size) = bigger share

  RISKS
  ==================================================
  - If event resolves YES: your NO position = $0 (lose ~$70)
  - If event resolves NO: your YES position = $0 (lose ~$126)
  - Competition: more LPs = smaller reward share
  - Rate changes: Polymarket adjusts reward rates regularly
  - Adverse fills: informed traders pick off stale quotes

  MITIGATIONS
  ==================================================
  - Spread across 10 markets (diversify event risk)
  - Use min_size only (minimize capital at risk per market)
  - Widen spread during news (don't pull quotes, maintain uptime)
  - YES + NO = $1: holding both sides hedges perfectly
  - Daily reward income offsets occasional adverse fills

  VERDICT: VIABLE
  ==================================================
  Conservative scenario (50% capture, higher AS):
    ~$150-200/day on $2,000 capital = 2,700-3,650%/yr

  Realistic scenario (80% capture):
    ~$350/day on $2,000 capital = 6,400%/yr

  The returns look extreme because:
    1. Very few LPs are competing for these rewards
    2. Markets are low-volume (Polymarket subsidizes liquidity)
    3. Capital requirements are tiny ($200/market)
    4. This is a temporary inefficiency that will close
""")
print("=" * 85)
