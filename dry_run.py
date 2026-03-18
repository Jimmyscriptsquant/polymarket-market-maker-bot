"""
Full strategy scanner: hits both Gamma API (liquid/high-volume markets)
and CLOB sampling API (reward-farm markets), fetches real orderbooks,
scores everything, and outputs a ranked recommendation.
NO orders placed.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
CAPITAL = 50_000.0
MIN_SPREAD_BPS = 10
SEM_LIMIT = 15  # max concurrent book fetches


@dataclass
class M:
    cid: str = ""
    question: str = ""
    slug: str = ""
    tokens: list = field(default_factory=list)
    reward_rate: float = 0.0
    min_size: float = 0.0
    max_spread: float = 0.0
    min_tick: float = 0.01
    neg_risk: bool = False
    volume_24h: float = 0.0
    liquidity: float = 0.0
    end_date: str = ""
    source: str = ""
    # computed
    depth: float = 0.0
    bb: float = 0.0
    ba: float = 0.0
    mid: float = 0.0
    spread_bps: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    # scoring
    efficiency: float = 0.0
    capital: float = 0.0
    adj_spread: float = 0.0
    sim_bid: float = 0.0
    sim_ask: float = 0.0
    sim_size: float = 0.0
    tier: str = ""
    est_daily_pnl: float = 0.0
    risk_score: float = 0.0  # 0=safe, 10=risky


# ── Fetch ─────────────────────────────────────────────────────────────

async def fetch_clob_markets(client: httpx.AsyncClient) -> list[dict]:
    out = []
    cursor = "MA=="
    for _ in range(3):
        try:
            r = await client.get(f"{CLOB}/sampling-markets?next_cursor={cursor}")
            r.raise_for_status()
            page = r.json()
            out.extend(page.get("data", []))
            cursor = page.get("next_cursor", "")
            if not cursor or cursor == "LTE=":
                break
        except Exception:
            break
    return out


async def fetch_gamma_markets(client: httpx.AsyncClient) -> list[dict]:
    """High-volume markets from Gamma API."""
    out = []
    for offset in (0, 100):
        try:
            r = await client.get(f"{GAMMA}/markets", params={
                "active": "true", "closed": "false",
                "limit": 100, "offset": offset,
                "order": "volume24hr", "ascending": "false",
            })
            r.raise_for_status()
            out.extend(r.json() if isinstance(r.json(), list) else [])
        except Exception:
            break
    return out


async def fetch_book(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                     token_id: str) -> tuple[float, float, float, float]:
    """Returns (best_bid, best_ask, bid_depth_5lvl, ask_depth_5lvl)."""
    async with sem:
        try:
            r = await client.get(f"{CLOB}/book", params={"token_id": token_id})
            r.raise_for_status()
            d = r.json()
            bids = d.get("bids", [])
            asks = d.get("asks", [])
            bb = float(bids[0]["price"]) if bids else 0.0
            ba = float(asks[0]["price"]) if asks else 0.0
            bd = sum(float(b.get("size", 0)) for b in bids[:5])
            ad = sum(float(a.get("size", 0)) for a in asks[:5])
            return bb, ba, bd, ad
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


# ── Parse ─────────────────────────────────────────────────────────────

def parse_clob(raw: dict) -> M | None:
    if not raw.get("enable_order_book") or not raw.get("active") or raw.get("closed"):
        return None
    if not raw.get("accepting_orders"):
        return None
    cid = raw.get("condition_id", "")
    if not cid:
        return None
    rewards = raw.get("rewards") or {}
    rates = rewards.get("rates") or []
    rate = sum(float(r.get("rewards_daily_rate", 0) or 0) for r in rates)
    return M(
        cid=cid, question=raw.get("question", "")[:80],
        slug=raw.get("market_slug", ""),
        tokens=raw.get("tokens", []),
        reward_rate=rate,
        min_size=float(rewards.get("min_size", 0) or 0),
        max_spread=float(rewards.get("max_spread", 0) or 0),
        min_tick=float(raw.get("minimum_tick_size", 0.01) or 0.01),
        neg_risk=raw.get("neg_risk", False),
        end_date=(raw.get("end_date_iso") or "")[:10],
        source="clob",
    )


def parse_gamma(raw: dict) -> M | None:
    cid = raw.get("conditionId") or raw.get("condition_id", "")
    if not cid:
        return None
    if raw.get("closed") or not raw.get("active", True):
        return None

    # Gamma uses clobTokenIds as a JSON string sometimes
    clob_ids = raw.get("clobTokenIds", "")
    tokens = []
    if isinstance(clob_ids, str) and clob_ids.startswith("["):
        import json
        try:
            ids = json.loads(clob_ids)
            for i, tid in enumerate(ids):
                tokens.append({"token_id": tid, "outcome": "Yes" if i == 0 else "No"})
        except Exception:
            pass
    elif isinstance(clob_ids, list):
        for i, tid in enumerate(clob_ids):
            tokens.append({"token_id": tid, "outcome": "Yes" if i == 0 else "No"})

    # Also try outcomes field
    if not tokens:
        outcomes = raw.get("outcomes", "")
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = []

    vol = float(raw.get("volume24hr") or raw.get("volume_24hr") or 0)
    liq = float(raw.get("liquidityClob") or raw.get("liquidity") or 0)

    return M(
        cid=cid, question=raw.get("question", "")[:80],
        slug=raw.get("slug", ""),
        tokens=tokens,
        volume_24h=vol,
        liquidity=liq,
        end_date=(raw.get("endDate") or raw.get("end_date_iso") or "")[:10],
        source="gamma",
    )


# ── Enrich & Score ────────────────────────────────────────────────────

async def enrich(client: httpx.AsyncClient, sem: asyncio.Semaphore, m: M):
    best_bb, best_ba, best_bd, best_ad = 0.0, 0.0, 0.0, 0.0
    for tok in m.tokens:
        tid = tok.get("token_id", "")
        if not tid:
            continue
        bb, ba, bd, ad = await fetch_book(client, sem, tid)
        if bb > 0 and ba > 0 and bb < ba:
            spread = ba - bb
            best_spread = best_ba - best_bb if best_bb > 0 else 999
            if spread < best_spread or best_bb == 0:
                best_bb, best_ba, best_bd, best_ad = bb, ba, bd, ad
    m.bb, m.ba, m.bid_depth, m.ask_depth = best_bb, best_ba, best_bd, best_ad
    m.depth = best_bd + best_ad
    if m.bb > 0 and m.ba > 0 and m.bb < m.ba:
        m.mid = round((m.bb + m.ba) / 2, 4)
        m.spread_bps = round(((m.ba - m.bb) / m.mid) * 10000, 1)


def score(m: M):
    """Assign tier, efficiency, risk, and estimated daily PnL."""
    if m.mid <= 0:
        return

    # Tier
    if m.spread_bps < 500:
        m.tier = "T1-LIQUID"
    elif m.spread_bps < 5000:
        m.tier = "T2-MEDIUM"
    else:
        m.tier = "T3-FARM"

    # Risk score (0-10)
    # Wide spread = higher risk (adverse selection), near-dated = higher risk
    spread_risk = min(m.spread_bps / 2000, 5)
    depth_safety = min(m.depth / 10000, 3)  # more depth = safer
    m.risk_score = round(max(spread_risk - depth_safety + 2, 0), 1)

    # Efficiency
    if m.tier == "T1-LIQUID":
        # For liquid: reward + spread capture - competition
        competition = max(m.depth / 1000, 1)  # more depth = more LPs
        m.efficiency = (m.reward_rate + m.volume_24h * 0.001) / competition
    elif m.tier == "T3-FARM":
        # For farms: pure reward / risk
        m.efficiency = m.reward_rate / max(m.risk_score, 0.5)
    else:
        m.efficiency = (m.reward_rate + m.volume_24h * 0.0005) / max(m.risk_score, 0.5)

    # Estimated daily PnL
    if m.tier == "T1-LIQUID":
        # Spread capture: assume we fill 5% of volume on each side
        fill_volume = m.volume_24h * 0.05
        spread_pnl = fill_volume * (m.spread_bps / 10000) * 0.5  # half spread per fill
        m.est_daily_pnl = spread_pnl + m.reward_rate
    elif m.tier == "T3-FARM":
        # Pure reward, minimal fills, tiny adverse selection
        m.est_daily_pnl = m.reward_rate * 0.9  # 90% capture rate estimate
    else:
        fill_volume = m.volume_24h * 0.02
        spread_pnl = fill_volume * (m.spread_bps / 10000) * 0.3
        m.est_daily_pnl = spread_pnl + m.reward_rate * 0.8

    # Simulate quotes
    if m.reward_rate > 0 and m.tier == "T1-LIQUID":
        fills_est = max(m.volume_24h * 0.05 / 100, 1)
        subsidy = (m.reward_rate / fills_est / 100) * 10000
        m.adj_spread = max(int(MIN_SPREAD_BPS - subsidy), MIN_SPREAD_BPS)
    elif m.tier == "T3-FARM":
        # Quote at max_spread allowed by reward program
        m.adj_spread = max(int(m.max_spread * 100), MIN_SPREAD_BPS) if m.max_spread > 0 else max(int(m.spread_bps * 0.8), MIN_SPREAD_BPS)
    else:
        m.adj_spread = MIN_SPREAD_BPS

    m.sim_bid = round(m.mid * (1 - m.adj_spread / 10000), 4)
    m.sim_ask = round(m.mid * (1 + m.adj_spread / 10000), 4)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    t0 = time.time()
    print("=" * 100)
    print("  POLYMARKET FULL STRATEGY SCANNER")
    print("  Mode: READ-ONLY | Both APIs | All tiers")
    print("=" * 100)

    sem = asyncio.Semaphore(SEM_LIMIT)
    async with httpx.AsyncClient(timeout=20.0) as client:

        # 1) Fetch from both APIs concurrently
        print("\n[1/5] Fetching from CLOB + Gamma APIs concurrently...")
        clob_raw, gamma_raw = await asyncio.gather(
            fetch_clob_markets(client),
            fetch_gamma_markets(client),
        )
        print(f"      CLOB: {len(clob_raw)} markets | Gamma: {len(gamma_raw)} markets")

        # 2) Parse
        print("[2/5] Parsing & deduplicating...")
        seen_cids = set()
        all_markets: list[M] = []

        # Gamma first (higher quality data for liquid markets)
        for raw in gamma_raw:
            m = parse_gamma(raw)
            if m and m.cid not in seen_cids and m.tokens:
                seen_cids.add(m.cid)
                all_markets.append(m)

        # Then CLOB (fills in reward-farm markets)
        for raw in clob_raw:
            m = parse_clob(raw)
            if m and m.cid not in seen_cids:
                seen_cids.add(m.cid)
                all_markets.append(m)
            elif m and m.cid in seen_cids:
                # Merge reward data into existing
                for existing in all_markets:
                    if existing.cid == m.cid:
                        if m.reward_rate > existing.reward_rate:
                            existing.reward_rate = m.reward_rate
                        if m.max_spread > existing.max_spread:
                            existing.max_spread = m.max_spread
                        if m.min_size > existing.min_size:
                            existing.min_size = m.min_size
                        if not existing.tokens and m.tokens:
                            existing.tokens = m.tokens
                        break

        has_tokens = [m for m in all_markets if m.tokens]
        has_rewards = [m for m in all_markets if m.reward_rate > 0]
        print(f"      {len(all_markets)} unique markets, {len(has_tokens)} with token IDs, {len(has_rewards)} with rewards")

        # 3) Select candidates: top rewards + top volume
        print("[3/5] Selecting candidates for orderbook fetch...")
        by_reward = sorted([m for m in has_tokens if m.reward_rate > 0],
                           key=lambda x: x.reward_rate, reverse=True)[:30]
        by_volume = sorted([m for m in has_tokens if m.volume_24h > 0],
                           key=lambda x: x.volume_24h, reverse=True)[:30]

        candidates_set = set()
        candidates = []
        for m in by_reward + by_volume:
            if m.cid not in candidates_set:
                candidates_set.add(m.cid)
                candidates.append(m)

        print(f"      {len(candidates)} candidates ({len(by_reward)} by reward, {len(by_volume)} by volume)")

        # 4) Fetch orderbooks concurrently
        print(f"[4/5] Fetching {len(candidates)} orderbooks (concurrency={SEM_LIMIT})...")
        await asyncio.gather(*(enrich(client, sem, m) for m in candidates))

        tradeable = [m for m in candidates if m.mid > 0]
        print(f"      {len(tradeable)} with valid orderbooks")

        # 5) Score & rank
        print("[5/5] Scoring all markets...\n")
        for m in tradeable:
            score(m)

        # Split by tier
        t1 = sorted([m for m in tradeable if m.tier == "T1-LIQUID"],
                     key=lambda x: x.est_daily_pnl, reverse=True)
        t2 = sorted([m for m in tradeable if m.tier == "T2-MEDIUM"],
                     key=lambda x: x.est_daily_pnl, reverse=True)
        t3 = sorted([m for m in tradeable if m.tier == "T3-FARM"],
                     key=lambda x: x.est_daily_pnl, reverse=True)

        # Allocate capital: 50% T1, 20% T2, 30% T3
        t1_cap = CAPITAL * 0.50
        t2_cap = CAPITAL * 0.20
        t3_cap = CAPITAL * 0.30

        def alloc(markets, pool):
            tot_eff = sum(m.efficiency for m in markets) or 1
            for m in markets:
                m.capital = round(pool * m.efficiency / tot_eff, 2)
                m.sim_size = round(max(min(m.capital * 0.1, 200), m.min_size or 5), 0)

        alloc(t1[:10], t1_cap)
        alloc(t2[:5], t2_cap)
        alloc(t3[:10], t3_cap)

        elapsed = time.time() - t0

        # ── Report ────────────────────────────────────────────────────

        def print_table(markets, label, cap):
            print(f"\n{'=' * 110}")
            print(f"  {label}  |  Capital: ${cap:,.0f}  |  {len(markets)} markets")
            print("=" * 110)
            if not markets:
                print("  (none found)")
                return 0.0

            print(f"{'#':<3} {'Market':<42} {'Mid':>6} {'Sprd':>7} {'Depth':>8} {'Rwd$/d':>8} {'Vol24h':>9} {'Est PnL':>8} {'Risk':>5} {'Cap$':>7}")
            print("-" * 110)
            total_pnl = 0.0
            for i, m in enumerate(markets):
                q = (m.question[:39] + "...") if len(m.question) > 39 else m.question
                rwd = f"${m.reward_rate:,.0f}" if m.reward_rate > 0 else "  --"
                vol = f"${m.volume_24h:,.0f}" if m.volume_24h > 0 else "  --"
                risk_bar = "*" * int(m.risk_score)
                print(f"{i+1:<3} {q:<42} {m.mid:>6.3f} {m.spread_bps:>5.0f}bp {m.depth:>7.0f} {rwd:>8} {vol:>9} ${m.est_daily_pnl:>6.1f} {m.risk_score:>4.1f}{risk_bar} ${m.capital:>6,.0f}")
                total_pnl += m.est_daily_pnl
            print("-" * 110)
            print(f"  Subtotal est. daily PnL: ${total_pnl:.2f}")
            return total_pnl

        print(f"\n{'#' * 110}")
        print(f"  SCAN COMPLETE  |  {len(tradeable)} tradeable markets  |  {elapsed:.1f}s")
        print(f"{'#' * 110}")

        pnl1 = print_table(t1[:10], "TIER 1 — LIQUID (spread < 5%): Active LP, spread capture + rewards", t1_cap)
        pnl2 = print_table(t2[:5], "TIER 2 — MEDIUM (5-50% spread): Wider quotes, moderate reward farm", t2_cap)
        pnl3 = print_table(t3[:10], "TIER 3 — FARM (>50% spread): Be-the-book, pure reward capture", t3_cap)

        # Simulated quotes for top picks
        all_top = (t1[:10] + t2[:5] + t3[:10])
        all_top.sort(key=lambda x: x.est_daily_pnl, reverse=True)

        print(f"\n{'=' * 110}")
        print("  SIMULATED QUOTES — TOP 15 BY EST. DAILY PnL")
        print("=" * 110)
        print(f"{'#':<3} {'Tier':<8} {'Market':<38} {'Bid':>8} {'Ask':>8} {'Size':>6} {'AdjSprd':>8} {'Est$/d':>8}")
        print("-" * 95)
        for i, m in enumerate(all_top[:15]):
            q = (m.question[:35] + "...") if len(m.question) > 35 else m.question
            print(f"{i+1:<3} {m.tier:<8} {q:<38} {m.sim_bid:>8.4f} {m.sim_ask:>8.4f} {m.sim_size:>5.0f}  {m.adj_spread:>5.0f}bp ${m.est_daily_pnl:>6.1f}")

        # Final summary
        total_pnl = pnl1 + pnl2 + pnl3
        print(f"\n{'=' * 110}")
        print("  RECOMMENDATION SUMMARY")
        print("=" * 110)
        print(f"""
  CAPITAL ALLOCATION:
    Tier 1 (Liquid):  ${t1_cap:>8,.0f}  ({len(t1[:10])} mkts)  —  est ${pnl1:>7.1f}/day
    Tier 2 (Medium):  ${t2_cap:>8,.0f}  ({len(t2[:5])} mkts)  —  est ${pnl2:>7.1f}/day
    Tier 3 (Farm):    ${t3_cap:>8,.0f}  ({len(t3[:10])} mkts)  —  est ${pnl3:>7.1f}/day
    -----------------------------------------------
    TOTAL:            ${CAPITAL:>8,.0f}            est ${total_pnl:>7.1f}/day

  ANNUALIZED (before adverse selection):
    Daily:    ${total_pnl:>8.1f}
    Monthly:  ${total_pnl * 30:>8.1f}
    Yearly:   ${total_pnl * 365:>8,.1f}  ({(total_pnl * 365 / CAPITAL) * 100:.1f}% on ${CAPITAL:,.0f})

  KEY RISKS:
    - Tier 3: adverse selection on event resolution (binary outcome)
    - Tier 1: competitive, may get squeezed on spread
    - All: Polymarket counterparty risk, smart contract risk, USDC depeg

  NEXT STEPS:
    1. Set up wallet + fund with USDC on Polygon
    2. Start with Tier 3 farm (lowest competition, highest reward/effort)
    3. Add Tier 1 liquid LP once you have price history for vol estimation
    4. Target 95%+ uptime — widen spreads during volatility, never pull

  DRY RUN COMPLETE — Zero orders placed
""")
        print("=" * 110)


if __name__ == "__main__":
    asyncio.run(main())
