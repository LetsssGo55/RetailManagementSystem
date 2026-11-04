"""
ArthaMart — AI retail assistant (rule-based, grounded in live data).

Design goal: *every* question gets a useful answer.
  1. explicit intents (reorder / markdown / bundles / profit / …)
  2. a product mentioned anywhere → full product brief
  3. otherwise → store KPIs + capability hints (never a dead end)
"""
from __future__ import annotations

import re

SUGGESTIONS = [
    "Which products need reordering?",
    "What should I markdown?",
    "Best bundle deals",
    "Forecast for Rice Basmati",
    "Price of Tata Tea Premium",
    "Stock of Potato",
    "How much profit did the system make?",
    "Who are my VIP customers?",
]

HELP = ("I answer from your live store data. Try:\n"
        + "\n".join(f"- **{s}**" for s in SUGGESTIONS))


def _conn():
    import sqlite3
    from ..config import DB_PATH
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------------------
# product detection (fuzzy — handles typos & partial names)
# ---------------------------------------------------------------------------

def _find_product(conn, q: str):
    """Detect a product mentioned anywhere in the question.
    Fuzzy ratios catch typos; chunk/bigram containment catches short or
    partial mentions ("parle-g", "basmati", "tata tea")."""
    from rapidfuzz import fuzz
    rows = conn.execute("SELECT product_id, name FROM products").fetchall()
    ql = re.sub(r"[^a-z0-9 ]", " ", q.lower())
    best, best_score = None, 0.0
    for r in rows:
        name = re.sub(r"[^a-z0-9 ]", " ", r["name"].lower())
        score = max(fuzz.partial_ratio(name, ql),
                    fuzz.token_set_ratio(name, ql))
        toks = name.split()
        for chunk in toks:                       # "parle", "basmati"…
            if len(chunk) >= 4 and chunk in ql:
                score = max(score, 85.0)
        for a, b in zip(toks, toks[1:]):         # "tata tea", "toor dal"…
            if f"{a} {b}" in ql:
                score = max(score, 90.0)
        if score > best_score:
            best, best_score = r, score
    return best if best_score >= 78 else None


# ---------------------------------------------------------------------------
# answer building blocks
# ---------------------------------------------------------------------------

def _product_brief(conn, pid: int, name: str) -> str:
    p = conn.execute("SELECT * FROM products WHERE product_id = ?",
                     (pid,)).fetchone()
    inv = conn.execute("SELECT stock_on_hand, reorder_level FROM "
                       "inventory_current WHERE product_id = ?",
                       (pid,)).fetchone()
    fc = conn.execute("SELECT SUM(predicted_qty) s, AVG(mape) m FROM "
                      "ml_forecast_output WHERE product_id = ?",
                      (pid,)).fetchone()
    tier = conn.execute("SELECT tier FROM ml_cluster_assignment WHERE "
                        "product_id = ?", (pid,)).fetchone()
    from ..pricing import engine as pe
    price = pe.price_product(conn, pid)
    stock = inv["stock_on_hand"] if inv else 0
    rop = inv["reorder_level"] if inv else 0
    flag = ("🚨 below reorder point — reorder now"
            if stock <= rop else "✅ healthy stock")
    lines = [f"📦 **{name}** [{p['regime']} · "
             f"{tier['tier'] if tier else '—'} tier]"]
    lines.append(f"- Price: **₹{price.price:.2f}** "
                 f"(base ₹{price.base_price:.2f}) — {price.reason}")
    lines.append(f"- Stock: **{stock}** vs ROP {rop} → {flag}")
    if fc and fc["s"] is not None:
        lines.append(f"- Next 7 days forecast: **≈{fc['s']:.0f} units** "
                     f"(MAPE {fc['m']:.1f}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# intents
# ---------------------------------------------------------------------------

def _intent_reorder(conn) -> str:
    from ..inventory.reorder import compute_plan
    due = compute_plan(conn)
    due = due[due["status"].isin(["REORDER", "CRITICAL"])]
    if due.empty:
        return ("✅ Nothing is below its reorder point right now — no "
                "purchase orders needed.")
    lines = "\n".join(f"- **{r.name}** — stock {int(r.stock_on_hand)} ≤ ROP "
                      f"{r.rop:.0f} → order ~{int(round(r.eoq))} units"
                      for r in due.head(6).itertuples())
    return f"📦 **{len(due)} product(s)** need reordering:\n{lines}"


def _intent_markdown(conn) -> str:
    from ..decisions import engine as dec
    cards = [c for c in dec.generate_all(conn)
             if c.card_type == "URGENT_MARKDOWN"]
    if not cards:
        return "🎉 No near-expiry stock needs a markdown today."
    return ("🚨 Expiry markdowns to act on:\n" + "\n".join(
        f"- **{c.product_name}** — {c.action} (₹{c.impact_inr:,.0f} "
        "recoverable)" for c in cards[:6]))


def _intent_bundles(conn) -> str:
    rows = conn.execute(
        "SELECT b.action, b.lift, b.weekly_impact, pa.name a, pc.name c "
        "FROM bundle_recommendations b "
        "JOIN products pa ON pa.product_id=b.antecedent_id "
        "JOIN products pc ON pc.product_id=b.consequent_id "
        "ORDER BY b.weekly_impact DESC LIMIT 5").fetchall()
    if not rows:
        return "No association rules yet — run `python main.py train`."
    return ("🎁 Top bundle / cross-sell opportunities:\n" + "\n".join(
        f"- **{r['a']}** → **{r['c']}** · {r['action']} · lift "
        f"{r['lift']:.2f} · ₹{r['weekly_impact']:,.0f}/wk" for r in rows))


def _intent_profit(conn) -> str:
    t = {r["policy"]: r["np"] for r in conn.execute(
        "SELECT policy, SUM(net_profit) np FROM simulation_results "
        "GROUP BY policy")}
    if not t:
        return "No simulation yet — run `python main.py simulate`."
    d = t.get("system", 0) - t.get("naive", 0)
    pct = d / abs(t.get("naive", 1)) * 100
    return (f"⚖️ Over 180 identical demand days: naive "
            f"**₹{t.get('naive', 0):,.0f}** vs system "
            f"**₹{t.get('system', 0):,.0f}** — the intelligent policy adds "
            f"**₹{d:+,.0f} ({pct:+.1f}%)**.")


def _intent_customers(conn) -> str:
    from ..personalisation import engine as pers
    df = pers.compute_segments(conn)
    c = df["segment"].value_counts()
    top = df.sort_values("spend", ascending=False).head(3)
    lines = "\n".join(f"- **{r.name}** — {r.segment}, ₹{r.spend:,.0f}, "
                      f"{int(r.visits)} orders" for r in top.itertuples())
    return (f"👥 Segments: {c.get('VIP', 0)} VIP · "
            f"{c.get('REGULAR', 0)} regular · {c.get('NEW', 0)} new · "
            f"{c.get('AT_RISK', 0)} at-risk.\nTop customers:\n{lines}")


def _intent_stock(conn, prod) -> str:
    inv = conn.execute("SELECT * FROM inventory_current WHERE product_id = ?",
                       (prod["product_id"],)).fetchone()
    if not inv:
        return f"No stock record for {prod['name']}."
    s, r = inv["stock_on_hand"], inv["reorder_level"]
    flag = "🚨 reorder now" if s <= r else "✅ healthy"
    return (f"📦 **{prod['name']}**: stock **{s}** (reorder level {r}) → "
            f"{flag}.")


def _intent_kpis(conn) -> str:
    n_prod = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    n_ord = conn.execute("SELECT COUNT(*) FROM customer_orders").fetchone()[0]
    val = conn.execute("SELECT COALESCE(SUM(total_value),0) FROM "
                       "customer_orders").fetchone()[0]
    rules = conn.execute("SELECT COUNT(*) FROM bundle_recommendations") \
        .fetchone()[0]
    return (f"📊 **Store KPIs** — {n_prod} products · {n_ord} orders worth "
            f"₹{val:,.0f} · {rules} basket rules · 7-day forecast horizon · "
            "K-Means k=3 · service level 95%.")


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

_INTENTS = [
    (("reorder", "restock", "purchase order", "buy more", "order now"),
     _intent_reorder),
    (("markdown", "expiry", "expire", "discount", "clearance"),
     _intent_markdown),
    (("bundle", "cross", "promo", "pair", "basket", "combo"),
     _intent_bundles),
    (("profit", "simulation", "naive", "vs system", "which policy"),
     _intent_profit),
    (("vip", "customer", "segment", "loyal", "at risk", "at-risk"),
     _intent_customers),
    (("kpi", "overview", "summary", "dashboard stats", "how is my store"),
     _intent_kpis),
]


def answer(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return HELP
    try:
        with _conn() as conn:
            if any(k in q for k in ("help", "what can you", "hello", " hi")):
                return HELP
            for keys, fn in _INTENTS:
                if any(k in q for k in keys):
                    return fn(conn)
            # stock questions with a product
            prod = _find_product(conn, q)
            if any(k in q for k in ("stock", "inventory", "how much left",
                                    "available")) and prod:
                return _intent_stock(conn, prod)
            if any(k in q for k in ("price", "cost", "mrp", "rate")) and prod:
                return _product_brief(conn, prod["product_id"], prod["name"])
            if any(k in q for k in ("forecast", "predict", "demand",
                                    "next week")) and prod:
                return _product_brief(conn, prod["product_id"], prod["name"])
            # any product mention → full brief (never a dead end)
            if prod:
                return _product_brief(conn, prod["product_id"], prod["name"])
            return (f"I couldn't map that to a product or topic. "
                    f"{HELP}")
    except Exception as e:                    # never crash the chat box
        return (f"⚠️ I hit an internal error ({e.__class__.__name__}). "
                "Please try again.")
