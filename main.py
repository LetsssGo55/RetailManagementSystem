#!/usr/bin/env python3
"""
ArthaMart — AI-Driven Retail Intelligence & Profit Optimization System
Design and Development of an Intelligent Retailing Application:
Reinventing Retail Operations with Intelligent Forecasting and Personalisation

Usage:
    python main.py setup       Generate synthetic data + build database
    python main.py train       Train all ML models
    python main.py simulate    Run naive vs system policy simulation
    python main.py ocr-demo    Test OCR pipeline on a sample slip
    python main.py make-slips  Generate sample order-slip images
    python main.py order ...   Order intake, queue & fulfilment (Phase 3)
        order demo                  Full OCR → draft → confirm → fulfil demo
        order new <slip> [cust]     OCR a slip into a draft order
        order list | show <id>      Inspect orders
        order queue                 Prioritised dispatch queue
        order pending               Review queue (unresolved lines)
        order resolve <line> <prod> Resolve a REVIEW/UNMATCHED line
        order confirm <id>          Confirm order (availability scored)
        order fulfil <id>           Ship it — deducts stock
    python main.py inventory ... EOQ/ROP/reorder planning (Phase 5)
        inventory plan [N]          Top-N reorder plan (EOQ, ROP, priority)
        inventory reorder           Run cycle → purchase orders + alerts
        inventory alerts | po       View alerts / recommended POs
    python main.py price ...     Dual-regime pricing (Phase 6)
        price all                   Price everything + log markdowns
        price show <id>             Inspect one pricing decision
    python main.py invoice ...   Invoice & procurement PDFs (Phase 7)
        invoice gen <id|demo>       Generate PDFs for an order
        invoice list                Show generated invoices
    python main.py cards [N]     Decision cards sorted by ₹ impact (Phase 8)
    python main.py personalise   Customer segments + recommendations
    python main.py dashboard [port] THE dashboard — new shadcn UI, 9 pages (8501)
    python main.py streamlit   Classic Streamlit dashboard (8502)
    python main.py webui [port] shadcn UI on a custom port (default 8503)
    python main.py all         setup → train → simulate → dashboard
    python main.py status      Show database & model status
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import database  # noqa: E402


def cmd_setup(args):
    """Generate synthetic data and build the database."""
    from src.data.generate_synthetic import run_all
    run_all()


def cmd_train(args):
    """Train all ML models (Phase 4)."""
    _require_db()
    from src.ml import basket_analysis, clustering, forecasting

    conn = database.get_connection()
    try:
        # ── 4.1 Demand forecasting ──────────────────────────────────────
        print("=" * 70)
        print("4.1 DEMAND FORECASTING — Random Forest vs baselines")
        print("=" * 70)
        rep = forecasting.train_all(conn, save=not args.no_save)
        print(f"  {'model':<16} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8}")
        for name, m in (("random_forest", rep.rf),
                        ("naive-7d", rep.naive),
                        ("holt-winters", rep.hw)):
            print(f"  {name:<16} {m['rmse']:>8.3f} {m['mae']:>8.3f} "
                  f"{m['mape']:>8.2f}")
        verdict = "✅ ACCEPTED — RF beats both baselines" if rep.accepted \
            else "❌ REJECTED — RF failed to beat a baseline"
        print(f"  {verdict} on RMSE/MAE/MAPE across {rep.n_products} products.")
        if not args.no_save:
            print("  → next-7-day forecasts written to ml_forecast_output")

        # ── 4.2 Product segmentation ────────────────────────────────────
        print("\n" + "=" * 70)
        print("4.2 PRODUCT SEGMENTATION — K-Means (k=3)")
        print("=" * 70)
        crep, cdf = clustering.train_clusters(conn, save=not args.no_save)
        print("  k-scan: ", end="")
        print("  ".join(f"k={k}: sil={crep.silhouette_by_k[k]:.3f}"
                        for k in sorted(crep.silhouette_by_k)))
        print(f"  overall silhouette: {crep.silhouette_overall:.3f}")
        print("  tiers: " + "  ".join(
            f"{t}={crep.tier_sizes.get(t, 0)}"
            for t in ("FAST", "MEDIUM", "SLOW")))
        print(f"  ARI vs ground truth: {crep.ari:.3f}")
        print("  confusion (rows=truth, cols=predicted, SLOW/MED/FAST):")
        for row in crep.confusion:
            print("    " + "  ".join(f"{c:>5}" for c in row))
        if not args.no_save:
            print("  → labels written to ml_cluster_assignment")

        # ── 4.3 Market basket analysis ─────────────────────────────────
        print("\n" + "=" * 70)
        print("4.3 MARKET BASKET ANALYSIS — FP-Growth")
        print("=" * 70)
        brep, bdf = basket_analysis.run_basket_analysis(
            conn, save=not args.no_save)
        print(f"  baskets: {brep.n_baskets}   frequent itemsets: "
              f"{brep.n_frequent}   raw rules: {brep.n_rules_raw}")
        print(f"  actionable rules kept: {brep.n_rules_kept} "
              f"({brep.actions})")
        print(f"  projected weekly impact: ₹{brep.total_weekly_impact:,.0f}")
        if not bdf.empty:
            names = dict(conn.execute("SELECT product_id, name FROM products"))
            top = bdf.sort_values("weekly_impact", ascending=False).head(5)
            print("  top rules:")
            for _, r in top.iterrows():
                print(f"    {names[r['antecedent_id']]} → "
                      f"{names[r['consequent_id']]}   "
                      f"lift {r['lift']:.2f}  conf {r['confidence']:.0%}  "
                      f"{r['action']}  ₹{r['weekly_impact']:.0f}/wk")
        if not args.no_save:
            print("  → rules written to bundle_recommendations")
    finally:
        conn.close()


def cmd_simulate(args):
    """Run naive vs system policy simulation (Phase 9)."""
    _require_db()
    from src.simulation import runner

    conn = database.get_connection()
    try:
        print("⏳ Simulating 180 days — identical demand signal, "
              "two decision policies...")
        rep = runner.run_simulation(conn, save=True)
        tn, ts = rep["naive_totals"], rep["system_totals"]

        print(f"\n{'component':<16} {'naive ₹':>14} {'system ₹':>14} "
              f"{'Δ ₹':>14}")
        print("-" * 62)
        for k in ("revenue", "cogs", "holding_cost", "markdown_loss",
                  "stockout_loss", "net_profit"):
            print(f"{k:<16} {tn[k]:>14,.0f} {ts[k]:>14,.0f} "
                  f"{ts[k] - tn[k]:>+14,.0f}")
        print("-" * 62)
        sign = "📈" if rep["delta_inr"] > 0 else "📉"
        print(f"{sign} System policy net profit advantage: "
              f"₹{rep['delta_inr']:+,.0f} ({rep['delta_pct']:+.1f}%) "
              f"over {rep['n_days']} days")
        print("\n   → day-by-day CSVs in data/simulation/, "
              "persisted to simulation_results.")
    finally:
        conn.close()


def cmd_make_slips(args):
    """Generate sample order-slip images for the OCR demo."""
    from src.ocr.slip_generator import generate_all
    paths = generate_all()
    print("🧾 Sample slips generated:")
    for name, p in paths.items():
        print(f"   {name:6s} → {p}")


def cmd_ocr_demo(args):
    """Demonstrate the full OCR intake pipeline on sample slips."""
    from src.ocr import engine, matcher, pipeline
    from src.ocr.slip_generator import SLIP_DIR, generate_all

    if not engine.tesseract_available():
        print("❌ Tesseract OCR binary not found.")
        print(engine.install_hint())
        return

    # ensure sample slips exist
    images = sorted(SLIP_DIR.glob("*.png")) if SLIP_DIR.exists() else []
    if getattr(args, "image", None):
        images = [Path(args.image)]
    elif not images:
        print("🧾 No slips found — generating samples...")
        generate_all()
        images = sorted(SLIP_DIR.glob("*.png"))

    conn = database.get_connection()
    grand = {"matched": 0, "review": 0, "unmatched": 0}
    try:
        for img in images:
            print("=" * 70)
            print(f"SLIP: {img.name}")
            print("=" * 70)
            res = pipeline.process_slip(img, conn=conn)
            s = res.summarise()
            print(f"  lines parsed: {s['items']} items "
                  f"(deskew {s['deskew_angle']:+.1f}°)")
            print(f"  {'OCR text':<34} {'qty':>5}  "
                  f"{'match':<34} {'score':>5}  status")
            print("  " + "-" * 92)
            for m in res.matches:
                icon = {"MATCHED": "✅", "REVIEW": "🔍",
                        "UNMATCHED": "❌"}[m.status]
                print(f"  {m.query_raw[:34]:<34} {m.quantity:>5.0f}  "
                      f"{(m.matched_name or '—')[:34]:<34} "
                      f"{m.score:>5.0f}  {icon} {m.status}")
                grand[m.status.lower()] += 1
            for sk in res.parse.skipped_lines:
                if sk.skip_reason == "no-letters" and sk.raw:
                    print(f"  ⚠ skipped noise: {sk.raw!r}")
            print()
    finally:
        conn.close()

    print("-" * 70)
    print(f"TOTAL → ✅ matched: {grand['matched']}   "
          f"🔍 review: {grand['review']}   ❌ unmatched: {grand['unmatched']}")


def _require_db():
    if not config.DB_PATH.exists():
        print("⚠️  Database not found. Run: python main.py setup")
        sys.exit(1)


def cmd_order(args):
    """Order processing & fulfilment (Phase 3)."""
    _require_db()
    from src.orders import availability, fulfilment, intake, queue
    from src.orders.models import LineStatus

    action = args.action
    rest = list(args.args or [])
    args.slip = args.customer = args.order_id = args.line_id = None
    args.product_id = None

    def need(n, usage):
        if len(rest) < n:
            print(f"Usage: python main.py order {action} {usage}")
            sys.exit(1)

    if action == "new":
        need(1, "<slip-image> [customer_id]")
        args.slip = rest[0]
        args.customer = int(rest[1]) if len(rest) > 1 else None
    elif action in ("show", "confirm", "fulfil"):
        need(1, "<order_id>")
        args.order_id = int(rest[0])
    elif action == "resolve":
        need(2, "<line_id> <product_id>")
        args.line_id, args.product_id = int(rest[0]), int(rest[1])

    conn = database.get_connection()
    try:
        if action == "new":
            from src.ocr import engine, pipeline
            if not engine.tesseract_available():
                print("❌ Tesseract not found:\n" + engine.install_hint())
                return
            res = pipeline.process_slip(args.slip, conn=conn)
            oid = intake.create_order_from_slip(
                conn, res, customer_id=args.customer, source="OCR")
            print(f"🧾 OCR intake → draft order #{oid} "
                  f"({res.matched_count} matched, {res.review_count} review, "
                  f"{res.unmatched_count} unmatched)")

        elif action == "list":
            orders = fulfilment.list_orders(conn)
            if not orders:
                print("No orders yet.")
                return
            print(f"{'#':>4} {'date':<11} {'source':<9} {'status':<15} "
                  f"{'fulfil%':>8} {'value ₹':>10}")
            for o in orders:
                print(f"{o['order_id']:>4} {o['order_date']:<11} "
                      f"{o['source']:<9} {o['status']:<15} "
                      f"{o['fulfilment_pct']:>7.0f}% {o['total_value']:>10,.2f}")

        elif action == "show":
            o = fulfilment.get_order(conn, args.order_id)
            h = o["header"]
            print(f"Order #{h['order_id']} — {h['status']}  "
                  f"(customer {h['customer_id']}, {h['order_date']}, "
                  f"source {h['source']})")
            print(f"  fulfilment: {h['fulfilment_pct']:.0f}%   "
                  f"total: ₹{h['total_value']:,.2f}")
            print(f"  {'status':<14} {'product':<32} {'qty':>6} "
                  f"{'avail':>6} {'₹/unit':>8} {'line ₹':>9}")
            for ln in o["lines"]:
                name = ln.get("product_name") or f"[?] {ln['raw_text'][:26]}"
                avail = ln["available_qty"]
                print(f"  {ln['status']:<14} {name[:32]:<32} "
                      f"{ln['ordered_qty']:>6} "
                      f"{(f'{avail:g}' if avail is not None else '-'):>6} "
                      f"{ln['unit_price']:>8.2f} {ln['line_total']:>9.2f}")

        elif action == "queue":
            # score any unscored pending orders first so the queue is fresh
            for q in queue.build_queue(conn):
                if q.status == "DRAFT":
                    try:
                        availability.check_order(conn, q.order_id)
                    except ValueError:
                        pass
            entries = queue.build_queue(conn)
            if not entries:
                print("Queue empty — no pending orders.")
                return
            print(f"{'rk':>3} {'order':>6} {'status':<15} {'fulfil%':>8} "
                  f"{'value ₹':>10} {'lines':>6}  dispatch")
            for e in entries:
                flag = {"DISPATCH": "🚚 DISPATCH", "PARTIAL": "⚠️ PARTIAL",
                        "HOLD": "⛔ HOLD"}[e.dispatch_flag]
                print(f"{e.rank:>3} {e.order_id:>6} {e.status:<15} "
                      f"{e.fulfilment_pct:>7.0f}% {e.total_value:>10,.2f} "
                      f"{e.lines_total:>6}  {flag}")

        elif action == "pending":
            items = intake.pending_lines(conn)
            if not items:
                print("✅ Review queue empty — nothing needs a human.")
                return
            for it in items:
                print(f"line {it['line_id']} (order {it['order_id']}) "
                      f"[{it['status']}] qty {it['ordered_qty']:g}: "
                      f"{it['raw_text']}")
                for pid, name, s in it["candidates"]:
                    mark = "→" if pid == it["suggested_product_id"] else " "
                    print(f"    {mark} {s:5.1f}  #{pid:<3} {name}")

        elif action == "resolve":
            intake.resolve_line(conn, args.line_id, args.product_id)
            print(f"✅ Line {args.line_id} resolved to product "
                  f"#{args.product_id} ({LineStatus.CONFIRMED.value}).")

        elif action == "confirm":
            s = fulfilment.confirm_order(conn, args.order_id)
            print(f"✅ Order #{args.order_id} confirmed — "
                  f"fulfilment {s['fulfilment_pct']:.0f}% "
                  f"(✅ {s['ready']} ready, ⚠️ {s['partial']} partial, "
                  f"❌ {s['unavailable']} unavailable)")

        elif action == "fulfil":
            s = fulfilment.fulfil_order(conn, args.order_id)
            print(f"🚚 Order #{args.order_id} fulfilled — "
                  f"{s['lines_shipped']}/{s['lines_orderable']} lines shipped, "
                  f"fulfilment {s['fulfilment_pct']:.0f}%.")
            if s["low_stock_products"]:
                print(f"   ⚠️  Low stock now (≤ reorder level): products "
                      f"{s['low_stock_products']} → Phase 5 will raise "
                      "procurement alerts.")

        elif action == "demo":
            _order_demo(conn)

        else:
            print(f"Unknown order action: {action}")
    finally:
        conn.close()


def _order_demo(conn):
    """Full Phase-2→3 showcase: OCR slips → draft → resolve → confirm → fulfil."""
    from src.ocr import engine, pipeline
    from src.ocr.slip_generator import SLIP_DIR, generate_all
    from src.orders import availability, fulfilment, intake, queue

    if not engine.tesseract_available():
        print("❌ Tesseract not found:\n" + engine.install_hint())
        return

    # clean previous demo orders so the demo is repeatable —
    # restore any stock that a prior run shipped, then drop the rows
    old = conn.execute(
        "SELECT order_id FROM customer_orders WHERE source = 'OCR-DEMO'"
    ).fetchall()
    for r in old:
        shipped = conn.execute(
            "SELECT product_id, available_qty FROM order_line_items "
            "WHERE order_id = ? AND status = 'FULFILLED' "
            "  AND product_id IS NOT NULL", (r["order_id"],)).fetchall()
        for ln in shipped:
            conn.execute(
                "UPDATE inventory_current "
                "SET stock_on_hand = stock_on_hand + ? WHERE product_id = ?",
                (float(ln["available_qty"] or 0), ln["product_id"]))
        conn.execute("DELETE FROM order_line_items WHERE order_id = ?",
                     (r["order_id"],))
        conn.execute("DELETE FROM customer_orders WHERE order_id = ?",
                     (r["order_id"],))
    conn.commit()
    if old:
        print(f"🧹 Cleared {len(old)} previous demo order(s) "
              "(restored shipped stock).\n")

    if not list(SLIP_DIR.glob("*.png")):
        generate_all()

    slips = sorted(SLIP_DIR.glob("*.png"))
    for slip in slips:
        res = pipeline.process_slip(slip, conn=conn)
        oid = intake.create_order_from_slip(conn, res, source="OCR-DEMO")
        print(f"🧾 {slip.name:<16} → draft order #{oid:<3} "
              f"(✅ {res.matched_count}  🔍 {res.review_count}  "
              f"❌ {res.unmatched_count})")

    # a deliberately oversized "stress" order to exercise PARTIAL ranking
    from src.ocr.matcher import MATCHED as _M, MatchResult as _MR
    from src.ocr.parser import ParseResult as _PR
    low = conn.execute("SELECT i.product_id, p.name, i.stock_on_hand "
                       "FROM inventory_current i JOIN products p USING "
                       "(product_id) ORDER BY i.stock_on_hand ASC LIMIT 1"
                       ).fetchone()
    high = conn.execute("SELECT i.product_id, p.name, i.stock_on_hand "
                        "FROM inventory_current i JOIN products p USING "
                        "(product_id) ORDER BY i.stock_on_hand DESC LIMIT 1"
                        ).fetchone()
    stress = pipeline.SlipResult(
        image_path=None, raw_text="stress", parse=_PR(),
        matches=[
            _MR(f"{low['name']} x{low['stock_on_hand'] + 50}",
                float(low["stock_on_hand"] + 50), low["product_id"],
                low["name"], 100.0, _M),
            _MR(f"{high['name']} x2", 2.0, high["product_id"],
                high["name"], 100.0, _M),
        ], matched_count=2)
    stress_id = intake.create_order_from_slip(conn, stress, source="OCR-DEMO")
    print(f"🧾 stress order      → draft order #{stress_id:<3} "
          f"(oversized qty on #{low['product_id']} {low['name']})")

    # resolve whatever the review queue holds (accept top candidate)
    pend = intake.pending_lines(conn)
    for it in pend:
        pid = it["suggested_product_id"] or (
            it["candidates"][0][0] if it["candidates"] else None)
        if pid:
            intake.resolve_line(conn, it["line_id"], pid)
            print(f"🔧 Resolved line {it['line_id']} "
                  f"({it['raw_text'][:30]!r}) → product #{pid}")

    # availability-score every draft so the queue below is meaningful
    print("\n" + "=" * 70)
    print("PRIORITISED DISPATCH QUEUE  (before fulfilment)")
    print("=" * 70)
    for r in conn.execute("SELECT order_id FROM customer_orders "
                          "WHERE source='OCR-DEMO' AND status='DRAFT'"):
        availability.check_order(conn, r["order_id"])
    for e in queue.build_queue(conn):
        flag = {"DISPATCH": "🚚 DISPATCH", "PARTIAL": "⚠️ PARTIAL",
                "HOLD": "⛔ HOLD"}[e.dispatch_flag]
        print(f"  #{e.rank:<2} order {e.order_id:<4} "
              f"{e.fulfilment_pct:5.1f}%  ₹{e.total_value:>9,.2f}  "
              f"{e.lines_total} lines  {flag}")

    # confirm + fulfil every demo order
    print("\n" + "=" * 70)
    print("CONFIRM → FULFIL")
    print("=" * 70)
    demo_orders = [r["order_id"] for r in conn.execute(
        "SELECT order_id FROM customer_orders WHERE source='OCR-DEMO' "
        "ORDER BY order_id")]
    for oid in demo_orders:
        s = fulfilment.confirm_order(conn, oid)
        f = fulfilment.fulfil_order(conn, oid)
        print(f"order #{oid:<4} fulfilment {s['fulfilment_pct']:5.1f}%  "
              f"→ shipped {f['lines_shipped']}/{f['lines_orderable']} lines"
              + (f"   ⚠️ low stock: {f['low_stock_products']}"
                 if f["low_stock_products"] else ""))
    print("\n✅ Phase 3 demo complete — stock deducted, orders closed.")


def cmd_inventory(args):
    """Inventory optimisation — EOQ / ROP / reorder planning (Phase 5)."""
    _require_db()
    from src.inventory import reorder

    action = args.action
    conn = database.get_connection()
    try:
        if action == "plan":
            plan = reorder.compute_plan(conn)
            top = int(args.args[0]) if args.args else 15
            print(f"{'pid':>4} {'product':<30} {'fc7':>6} {'σ':>6} "
                  f"{'SS':>6} {'ROP':>7} {'EOQ':>7} {'stock':>6} "
                  f"{'prio':>7}  status")
            for r in plan.head(top).itertuples():
                icon = {"OK": "✅", "REORDER": "📦", "CRITICAL": "🚨"}[r.status]
                print(f"{r.product_id:>4} {r.name[:30]:<30} "
                      f"{r.forecast7:>6.1f} {r.sigma:>6.2f} "
                      f"{r.safety_stock:>6.1f} {r.rop:>7.1f} {r.eoq:>7.1f} "
                      f"{int(r.stock_on_hand):>6} {r.priority:>7.1f}  "
                      f"{icon} {r.status}")
            n_due = plan["status"].isin(["REORDER", "CRITICAL"]).sum()
            print(f"\n{len(plan)} products planned · {n_due} at/below ROP "
                  f"(showing top {min(top, len(plan))} by priority).")

        elif action == "reorder":
            s = reorder.run_reorder_cycle(conn, save=True)
            print(f"📋 Reorder cycle complete — {s['products_planned']} "
                  f"products planned.")
            print(f"   🚨 critical: {s['critical']}   "
                  f"📦 reorder: {s['reorder']}   "
                  f"total POs recommended: {s['orders_recommended']}")
            if len(s["due"]):
                print(f"\n{'pid':>4} {'product':<30} {'stock':>6} {'ROP':>7} "
                      f"{'PO qty':>7} {'priority':>8}")
                for r in s["due"].head(15).itertuples():
                    print(f"{r.product_id:>4} {r.name[:30]:<30} "
                          f"{int(r.stock_on_hand):>6} {r.rop:>7.1f} "
                          f"{int(r.qty):>7} {r.priority:>8.1f}")
            print("\n   → purchase_orders + procurement_alerts updated.")

        elif action == "alerts":
            rows = conn.execute(
                "SELECT a.alert_type, p.name, a.message, a.created_at "
                "FROM procurement_alerts a JOIN products p USING (product_id) "
                "ORDER BY a.alert_id DESC LIMIT 30").fetchall()
            if not rows:
                print("✅ No procurement alerts.")
                return
            for r in rows:
                icon = "🚨" if r["alert_type"] == "CRITICAL" else "📦"
                print(f"{icon} [{r['alert_type']:<8}] {r['message']}")

        elif action == "po":
            rows = conn.execute(
                "SELECT po.*, p.name FROM purchase_orders po "
                "JOIN products p USING (product_id) "
                "ORDER BY po.priority DESC").fetchall()
            if not rows:
                print("No purchase orders yet — run: python main.py "
                      "inventory reorder")
                return
            print(f"{'po':>4} {'product':<30} {'qty':>6} {'eoq':>7} "
                  f"{'rop':>7} {'prio':>8}  status")
            for r in rows:
                print(f"{r['po_id']:>4} {r['name'][:30]:<30} {r['qty']:>6} "
                      f"{r['eoq']:>7} {r['rop']:>7} {r['priority']:>8}  "
                      f"{r['status']}")
        else:
            print(f"Unknown inventory action: {action}")
    finally:
        conn.close()


def cmd_price(args):
    """Dual-regime pricing engine (Phase 6)."""
    _require_db()
    from src.pricing import engine

    conn = database.get_connection()
    try:
        if args.action == "show":
            if not args.args:
                print("Usage: python main.py price show <product_id>")
                return
            r = engine.price_product(conn, int(args.args[0]))
            print(f"#{r.product_id} {r.name}  [{r.regime}]")
            base_lbl = "MRP" if r.regime == "MRP" else "cost"
            print(f"  {base_lbl:<5}: ₹{r.base_price:>9.2f}")
            print(f"  price : ₹{r.price:>9.2f}", end="")
            if r.regime == "MRP" and r.discount_pct:
                print(f"   ({r.discount_pct:.0f}% markdown)", end="")
            if r.regime == "BULK":
                print(f"   (margin {r.margin_pct:.1f}%)", end="")
            if r.floor_applied:
                print("   🛡️ cost floor", end="")
            print()
            print(f"  reason: {r.reason}")

        elif args.action == "all":
            results = engine.price_all(conn, save=True)
            md = [r for r in results
                  if r.regime == "MRP" and r.discount_pct > 0]
            bulk = [r for r in results if r.regime == "BULK"]
            floors = [r for r in results if r.floor_applied]
            print(f"💰 Priced {len(results)} products "
                  f"({len(results) - len(bulk)} MRP · {len(bulk)} BULK)")
            print(f"   markdowns active : {len(md)}")
            print(f"   cost floors hit  : {len(floors)}")
            if md:
                print(f"\n{'pid':>4} {'product':<30} {'MRP':>8} {'now':>8} "
                      f"{'off':>4}  reason")
                for r in sorted(md, key=lambda x: -x.discount_pct)[:12]:
                    print(f"{r.product_id:>4} {r.name[:30]:<30} "
                          f"{r.base_price:>8.2f} {r.price:>8.2f} "
                          f"{r.discount_pct:>3.0f}%  {r.reason[:44]}")
            print("\n   → price_history_log + markdown_alerts_log updated.")
        else:
            print(f"Unknown price action: {args.action}")
    finally:
        conn.close()


def cmd_invoice(args):
    """Invoice & procurement PDF generation (Phase 7)."""
    _require_db()
    from src.invoices import generator

    conn = database.get_connection()
    try:
        if args.action == "list":
            rows = conn.execute(
                "SELECT * FROM invoices_generated ORDER BY invoice_id DESC"
            ).fetchall()
            if not rows:
                print("No invoices yet — run: python main.py invoice gen <order_id>")
                return
            print(f"{'id':>4} {'order':>6} {'date':<11} {'total ₹':>10}  path")
            for r in rows:
                print(f"{r['invoice_id']:>4} {r['order_id']:>6} "
                      f"{r['invoice_date']:<11} {r['total_amount']:>10,.2f}  "
                      f"{r['pdf_path']}")
            return

        # action == gen
        if not args.args:
            print("Usage: python main.py invoice gen <order_id|demo>")
            return
        target = args.args[0]
        if target == "demo":
            row = conn.execute(
                "SELECT order_id FROM customer_orders "
                "WHERE source = 'OCR-DEMO' AND status = 'FULFILLED' "
                "ORDER BY order_id DESC LIMIT 1").fetchone()
            if row is None:
                print("No fulfilled demo order — run: python main.py order demo")
                return
            order_ids = [row["order_id"]]
        else:
            order_ids = [int(target)]

        for oid in order_ids:
            res = generator.generate_for_order(conn, oid)
            print(f"🧾 Invoice for order #{oid}: {res['invoice']}")
            print(f"   {res['line_count']} billable lines · "
                  f"total ₹{res['total_amount']:,.2f}")
            if res["procurement_note"]:
                print(f"📋 Procurement note: {res['procurement_note']}")
            else:
                print("📋 Procurement note: not needed (nothing unavailable)")
        print("\n   → recorded in invoices_generated.")
    finally:
        conn.close()


def cmd_personalise(args):
    """Segment customers + build recommendation profiles."""
    _require_db()
    from src.personalisation import engine as pers

    conn = database.get_connection()
    try:
        df = pers.build_profiles(conn, save=True)
        counts = df["segment"].value_counts()
        print("👥 Customer segments:")
        for t in ("VIP", "REGULAR", "NEW", "AT_RISK"):
            print(f"   {t:<8}: {int(counts.get(t, 0))}")
        top = int(args.args[0]) if args.args else 3
        for cust in df.sort_values("spend", ascending=False) \
                .head(top).itertuples():
            recs = pers.recommend(conn, int(cust.customer_id))
            print(f"\n   {cust.name} [{cust.segment}] — "
                  f"₹{cust.spend:,.0f}, {int(cust.visits)} orders")
            for r in recs[:5]:
                print(f"      → {r['name']:<32} {r['reason']}")
        print("\n   → personalisation_profiles + customers updated.")
    finally:
        conn.close()


def cmd_cards(args):
    """Decision cards (Phase 8)."""
    _require_db()
    from src.decisions import engine as dec_engine

    top = int(args.args[0]) if args.args else 12
    conn = database.get_connection()
    try:
        cards = dec_engine.generate_all(conn)
        if not cards:
            print("No decision cards — run `python main.py train` and "
                  "`python main.py inventory reorder` first.")
            return
        print(f"{'=' * 74}")
        print(f"DECISION CARDS — top {min(top, len(cards))} of {len(cards)} "
              f"(sorted by ₹ impact)")
        print(f"{'=' * 74}")
        for i, c in enumerate(cards[:top], 1):
            print(f"#{i:<2} {c.render()}")
            print()
        counts = {}
        for c in cards:
            counts[c.title] = counts.get(c.title, 0) + 1
        print("-" * 74)
        print("  ".join(f"{t}: {n}" for t, n in counts.items()))
    finally:
        conn.close()


def cmd_webui(args):
    """Built-in shadcn-style web UI, connected to the live DB (Phase 10+)."""
    _require_db()
    from src.webui import server as ws
    port = int(args.args[0]) if args.args else 8503
    ws.serve(port)


def _bootstrap_if_empty():
    """First-run friendliness: if the DB is missing/empty, generate data +
    train + simulate so the dashboard never opens blank."""
    import argparse
    need_setup = not config.DB_PATH.exists()
    if not need_setup:
        conn = database.get_connection()
        try:
            np_ = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            nf = conn.execute("SELECT COUNT(*) FROM ml_forecast_output") \
                .fetchone()[0]
            ns = conn.execute("SELECT COUNT(*) FROM simulation_results") \
                .fetchone()[0]
        finally:
            conn.close()
        need_setup = np_ == 0
        if not need_setup and (nf == 0 or ns == 0):
            print("⏳ First run: training models & running simulation so the "
                  "dashboard has data (one-time, ~1 min)...")
            cmd_train(argparse.Namespace(no_save=False))
            cmd_simulate(argparse.Namespace())
            cmd_price(argparse.Namespace(action="all", args=[]))
            cmd_inventory(argparse.Namespace(action="reorder", args=[]))
            cmd_personalise(argparse.Namespace(args=["1"]))
            return
    if need_setup:
        print("⏳ First run: building database + models + simulation "
              "(one-time, ~1-2 min)...")
        cmd_all(argparse.Namespace())


def cmd_dashboard(args):
    """THE dashboard = new shadcn-style UI (9 pages, live data)."""
    _bootstrap_if_empty()
    from src.webui import server as ws
    port = int(args.args[0]) if args.args else 8501
    ws.serve(port)


def cmd_streamlit(args):
    """Classic Streamlit dashboard (legacy UI) on port 8502."""
    app = ROOT / "src" / "dashboard" / "app.py"
    if not app.exists():
        print("⚠️  Dashboard not yet built (Phase 10).")
        return
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                    "--server.headless=true", "--server.port=8502"],
                   cwd=str(ROOT))


def cmd_status(args):
    """Show database and project status."""
    print("=" * 60)
    print("ArthaMart — Project Status")
    print("=" * 60)
    print(f"Database: {config.DB_PATH}")
    if config.DB_PATH.exists():
        conn = database.get_connection()
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            print(f"\n  {len(tables)} tables created:")
            for t in tables:
                n = database.table_count(conn, t)
                print(f"    {t:30s} {n:>6,} rows")
        finally:
            conn.close()
    else:
        print("\n  ⚠️  Database not found. Run: python main.py setup")
    print()


def cmd_all(args):
    """One-command showcase: setup → train → simulate → price → reorder →
    personalise → OCR demo orders → invoices. OCR steps self-skip when
    Tesseract is missing so the chain never breaks."""
    import argparse
    cmd_setup(args)
    cmd_train(argparse.Namespace(no_save=False))
    cmd_simulate(args)
    cmd_price(argparse.Namespace(action="all", args=[]))
    cmd_inventory(argparse.Namespace(action="reorder", args=[]))
    cmd_personalise(argparse.Namespace(args=["1"]))
    try:
        from src.ocr import engine as _oe
        if _oe.tesseract_available():
            cmd_order(argparse.Namespace(action="demo", args=[]))
            cmd_invoice(argparse.Namespace(action="gen", args=["demo"]))
        else:
            print("⏭️  OCR demo skipped — Tesseract not installed.")
    except Exception as e:                      # demo state is optional
        print(f"⚠️  demo step skipped: {e}")
    print("\n🎉 ArthaMart ready — launch with: python main.py dashboard")


def main():
    parser = argparse.ArgumentParser(
        description="ArthaMart — AI-Driven Retail Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("setup", help="Generate data + build database")
    p_train = sub.add_parser("train", help="Train ML models")
    p_train.add_argument("--no-save", action="store_true",
                         help="Evaluate only; don't write model outputs to DB")
    sub.add_parser("simulate", help="Run policy simulation")
    p_ocr = sub.add_parser("ocr-demo", help="Test OCR pipeline")
    p_ocr.add_argument("image", nargs="?", default=None,
                       help="Optional slip image path (default: all samples)")
    sub.add_parser("make-slips", help="Generate sample order slips")

    p_ord = sub.add_parser("order", help="Order processing & fulfilment")
    p_ord.add_argument("action",
                       choices=["new", "list", "show", "queue", "pending",
                                "resolve", "confirm", "fulfil", "demo"],
                       help="Order action")
    p_ord.add_argument("args", nargs="*",
                       help="action args: slip path / order id / line+product")
    p_inv = sub.add_parser("inventory", help="Inventory optimisation (EOQ/ROP)")
    p_inv.add_argument("action", choices=["plan", "reorder", "alerts", "po"],
                       help="Inventory action")
    p_inv.add_argument("args", nargs="*", help="action args (plan: top N)")
    p_price = sub.add_parser("price", help="Dual-regime pricing engine")
    p_price.add_argument("action", choices=["all", "show"],
                         help="Pricing action")
    p_price.add_argument("args", nargs="*", help="action args (show: product id)")
    p_inv2 = sub.add_parser("invoice", help="Invoice/procurement PDFs")
    p_inv2.add_argument("action", choices=["gen", "list"], help="Invoice action")
    p_inv2.add_argument("args", nargs="*",
                        help="gen: order id or 'demo'")
    p_cards = sub.add_parser("cards", help="Decision cards")
    p_cards.add_argument("args", nargs="*", help="top N (default 12)")
    p_web = sub.add_parser("webui", help="shadcn-style connected web UI")
    p_web.add_argument("args", nargs="*", help="optional port (default 8503)")
    p_dash = sub.add_parser("dashboard",
                            help="NEW shadcn dashboard — 9 pages (default 8501)")
    p_dash.add_argument("args", nargs="*", help="optional port")
    sub.add_parser("streamlit", help="Classic Streamlit dashboard (8502)")
    p_pers = sub.add_parser("personalise", help="Customer segments + recs")
    p_pers.add_argument("args", nargs="*", help="top customers to show")
    sub.add_parser("all", help="setup → train → simulate")
    sub.add_parser("status", help="Show status")

    args = parser.parse_args()
    commands = {
        "setup": cmd_setup, "train": cmd_train, "simulate": cmd_simulate,
        "ocr-demo": cmd_ocr_demo, "make-slips": cmd_make_slips,
        "order": cmd_order, "inventory": cmd_inventory, "price": cmd_price,
        "invoice": cmd_invoice, "cards": cmd_cards,
        "personalise": cmd_personalise, "webui": cmd_webui,
        "dashboard": cmd_dashboard, "streamlit": cmd_streamlit,
        "all": cmd_all, "status": cmd_status,
    }
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
