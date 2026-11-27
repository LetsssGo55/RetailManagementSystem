"""
ArthaMart — shadcn-style UI, now FULLY CONNECTED to the real project.

Serves index.html + a JSON API backed by the live ArthaMart backend
(SQLite DB + all modules: decisions, inventory, pricing, personalisation,
simulation, OCR pipeline, invoices, auth).

Run:  python server.py            (port 8503)
"""
from __future__ import annotations

import json
import secrets
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_UI = Path(__file__).resolve().parent          # <root>/src/webui
ARTHAMART = ROOT_UI.parent.parent                  # ArthaMart project root
sys.path.insert(0, str(ARTHAMART))

from src import config  # noqa: E402
from src.data import database  # noqa: E402

PORT = 8503
SESSIONS: dict[str, dict] = {}          # token -> user dict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def q(fn):
    """Run fn(conn) on a fresh connection."""
    conn = database.get_connection()
    try:
        return fn(conn)
    finally:
        conn.close()


def j(data):
    return json.dumps(data, default=str).encode()


# ---------------------------------------------------------------------------
# API builders (real project modules)
# ---------------------------------------------------------------------------

def api_health():
    def f(c):
        def n(t):
            return c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        products = n("products")
        forecasts = n("ml_forecast_output")
        sim = n("simulation_results")
        return {"status": "ok", "version": config.get("project.version"),
                "db": str(config.DB_PATH),
                "products": products, "forecasts": forecasts,
                "simulation": sim,
                "ready": products > 0 and forecasts > 0 and sim > 0}
    return q(f)


def api_kpis():
    def f(c):
        rev = c.execute("SELECT COALESCE(SUM(total_value),0) FROM "
                        "customer_orders").fetchone()[0]
        open_o = c.execute("SELECT COUNT(*) FROM customer_orders WHERE "
                           "status IN ('DRAFT','CONFIRMED','PENDING_REVIEW')"
                           ).fetchone()[0]
        alerts = c.execute("SELECT COUNT(*) FROM inventory_current i JOIN "
                           "products p USING(product_id) WHERE "
                           "i.stock_on_hand <= i.reorder_level").fetchone()[0]
        crit = c.execute("SELECT COUNT(*) FROM purchase_orders WHERE "
                         "status='RECOMMENDED'").fetchone()[0]
        np_ = {r["policy"]: r["n"] for r in c.execute(
            "SELECT policy, SUM(net_profit) n FROM simulation_results "
            "GROUP BY policy")}
        return {"revenue": rev, "open_orders": open_o, "stock_alerts": alerts,
                "pos": crit, "naive": np_.get("naive", 0),
                "system": np_.get("system", 0)}
    return q(f)


def api_cards():
    from src.decisions import engine as dec
    def f(c):
        return [{"type": cd.title, "icon": cd.icon, "product": cd.product_name,
                 "action": cd.action, "reason": cd.reason,
                 "impact": cd.impact_inr}
                for cd in dec.generate_all(c)]
    return q(f)


def api_orders():
    from src.orders import queue as oq
    def f(c):
        rows = c.execute(
            "SELECT o.order_id, COALESCE(cu.name,'Walk-in') customer, "
            "o.total_value, o.fulfilment_pct, o.status FROM customer_orders o "
            "LEFT JOIN customers cu USING(customer_id) "
            "ORDER BY o.order_id DESC LIMIT 12").fetchall()
        return {"recent": [dict(r) for r in rows],
                "queue": [{"rank": e.rank, "order": e.order_id,
                           "fulfilment": e.fulfilment_pct,
                           "value": e.total_value,
                           "flag": e.dispatch_flag}
                          for e in oq.build_queue(c)]}
    return q(f)


def api_inventory():
    from src.inventory.reorder import compute_plan
    def f(c):
        plan = compute_plan(c)
        return [dict(product=r.name, stock=int(r.stock_on_hand),
                     rop=round(float(r.rop), 1), eoq=round(float(r.eoq), 1),
                     status=r.status)
                for r in plan.head(10).itertuples()]
    return q(f)


def api_customers():
    from src.personalisation import engine as pers
    def f(c):
        df = pers.compute_segments(c)
        counts = df["segment"].value_counts().to_dict()
        top = df.sort_values("spend", ascending=False).head(8)
        return {"counts": counts,
                "top": [dict(customer_id=int(r.customer_id), name=r.name,
                             segment=r.segment,
                             spend=float(r.spend), visits=int(r.visits))
                        for r in top.itertuples()]}
    return q(f)


def api_models():
    def f(c):
        fc = c.execute("SELECT AVG(rmse) r, AVG(mae) m, AVG(mape) p FROM "
                       "ml_forecast_output").fetchone()
        tiers = dict(c.execute("SELECT tier, COUNT(*) FROM "
                               "ml_cluster_assignment GROUP BY tier"))
        sil = c.execute("SELECT AVG(silhouette) FROM ml_cluster_assignment"
                        ).fetchone()[0]
        rules = c.execute("SELECT COUNT(*), MAX(lift), SUM(weekly_impact) "
                          "FROM bundle_recommendations").fetchone()
        return {"rmse": fc["r"], "mae": fc["m"], "mape": fc["p"],
                "tiers": tiers, "silhouette": sil,
                "rules": rules[0], "max_lift": rules[1],
                "weekly_impact": rules[2]}
    return q(f)


def api_product_segments():
    """Product-level K-Means assignments joined to current sales features."""
    from src.ml.clustering import product_features

    def f(c):
        feature_df = product_features(c)
        features = {
            int(row.product_id): {
                "avg_daily_sales": float(row.avg_daily_sales),
                "velocity": float(row.velocity),
                "revenue": float(row.revenue),
                "days_with_sales": int(row.days_with_sales),
            }
            for row in feature_df.itertuples()
        } if not feature_df.empty else {}
        assignments = {
            int(row["product_id"]): dict(row)
            for row in c.execute(
                "SELECT product_id, cluster, tier, silhouette, assigned_at "
                "FROM ml_cluster_assignment")
        }
        stock = {
            int(row["product_id"]): int(row["stock_on_hand"])
            for row in c.execute(
                "SELECT product_id, stock_on_hand FROM inventory_current")
        }
        products = c.execute(
            "SELECT product_id, name, category FROM products "
            "WHERE is_active = 1 ORDER BY name").fetchall()
        result = []
        for product in products:
            pid = int(product["product_id"])
            feat = features.get(pid, {})
            assignment = assignments.get(pid, {})
            result.append({
                "product_id": pid, "name": product["name"],
                "category": product["category"],
                "tier": assignment.get("tier"),
                "cluster": assignment.get("cluster"),
                "avg_daily_sales": feat.get("avg_daily_sales"),
                "velocity": feat.get("velocity"),
                "revenue": feat.get("revenue"),
                "days_with_sales": feat.get("days_with_sales", 0),
                "silhouette": assignment.get("silhouette"),
                "stock_on_hand": stock.get(pid),
                "assigned_at": assignment.get("assigned_at"),
            })
        return result
    return q(f)


def api_basket_products():
    """All active products, annotated with their persisted rule count."""
    def f(c):
        return [dict(row) for row in c.execute(
            """
            SELECT p.product_id, p.name, COUNT(DISTINCT r.rec_id) AS rule_count
            FROM products p
            LEFT JOIN bundle_recommendations r
              ON r.antecedent_id = p.product_id
              OR r.consequent_id = p.product_id
            WHERE p.is_active = 1
            GROUP BY p.product_id, p.name
            ORDER BY rule_count DESC, p.name
            """)]
    return q(f)


def api_basket_rules(product_id: int):
    """Persisted FP-Growth rules in which the selected product participates."""
    def f(c):
        product = c.execute(
            "SELECT name FROM products WHERE product_id = ? AND is_active = 1",
            (product_id,)).fetchone()
        if product is None:
            return {"product": None, "rules": []}
        rows = c.execute(
            """
            SELECT r.rec_id, a.name AS antecedent, b.name AS consequent,
                   CASE WHEN r.antecedent_id = ? THEN b.name ELSE a.name END
                       AS associated_product,
                   CASE WHEN r.antecedent_id = ?
                        THEN 'Selected product → associated product'
                        ELSE 'Associated product → selected product' END
                       AS direction,
                   r.support, r.confidence, r.lift, r.action, r.weekly_impact
            FROM bundle_recommendations r
            JOIN products a ON a.product_id = r.antecedent_id
            JOIN products b ON b.product_id = r.consequent_id
            WHERE r.antecedent_id = ? OR r.consequent_id = ?
            ORDER BY r.lift DESC, r.weekly_impact DESC, r.rec_id
            """, (product_id, product_id, product_id, product_id)).fetchall()
        return {"product": product["name"], "rules": [dict(row) for row in rows]}
    return q(f)


def api_inventory_ledger(product_id=None, limit=100):
    """GET /api/inventory-ledger — stock audit trail (newest first)."""
    def f(c):
        from src.inventory import management
        pid = int(product_id) if product_id is not None else None
        return management.get_ledger(c, product_id=pid, limit=int(limit))
    return q(f)


def api_inventory_add(data):
    """POST /api/inventory/add  {product_id, qty, user_email, user_name, reason}"""
    from src.inventory import management
    def f(c):
        return management.add_stock(
            c, int(data["product_id"]), int(data["qty"]),
            data.get("user_email", "store-admin@local"),
            data.get("user_name"), data.get("reason"))
    return q(f)


def api_inventory_remove(data):
    """POST /api/inventory/remove  {product_id, qty, user_email, user_name, reason}"""
    from src.inventory import management
    def f(c):
        return management.remove_stock(
            c, int(data["product_id"]), int(data["qty"]),
            data.get("user_email", "store-admin@local"),
            data.get("user_name"), data.get("reason"))
    return q(f)


def api_inventory_set(data):
    """POST /api/inventory/set  {product_id, qty, user_email, user_name, reason}"""
    from src.inventory import management
    def f(c):
        return management.set_stock(
            c, int(data["product_id"]), int(data["qty"]),
            data.get("user_email", "store-admin@local"),
            data.get("user_name"), data.get("reason"))
    return q(f)


def api_scan(file_bytes: bytes, filename: str):
    """POST /api/scan  — scan a receipt image/PDF and match lines to stock."""
    from pathlib import Path
    import tempfile
    from src.ocr import scanner
    def f(c):
        suffix = Path(filename).suffix.lower() or ".png"
        tmp = Path(tempfile.mkdtemp(prefix="ui_scan_")) / ("scan" + suffix)
        tmp.write_bytes(file_bytes)
        return scanner.scan_file(tmp, conn=c)
    return q(f)


def api_scan_confirm(data):
    """POST /api/scan/confirm  {items, user_email, user_name} — deduct scanned stock."""
    from src.ocr import scanner
    def f(c):
        return scanner.confirm_scanned_order(
            c, data["items"],
            data.get("user_email", "store-admin@local"),
            data.get("user_name"))
    return q(f)


def api_simulation():
    def f(c):
        rows = c.execute("SELECT policy, day, net_profit FROM "
                         "simulation_results ORDER BY policy, day").fetchall()
        series: dict = {}
        for r in rows:
            series.setdefault(r["policy"], []).append(
                [r["day"], r["net_profit"]])
        # cumulative
        for k in series:
            run = 0.0
            for pt in series[k]:
                run += pt[1]
                pt[1] = round(run, 2)
        pnl = c.execute(
            "SELECT policy, SUM(revenue) rev, SUM(cogs) cogs, "
            "SUM(holding_cost) hold, SUM(markdown_loss) md, "
            "SUM(stockout_loss) so FROM simulation_results "
            "GROUP BY policy").fetchall()
        return {"series": series, "pnl": [dict(r) for r in pnl]}
    return q(f)


def api_forecast():
    def f(c):
        prods = [dict(r) for r in c.execute(
            "SELECT p.product_id, p.name, SUM(f.predicted_qty) next7 "
            "FROM ml_forecast_output f JOIN products p USING(product_id) "
            "WHERE p.is_active = 1 GROUP BY p.product_id ORDER BY next7 DESC")]
        return prods
    return q(f)


def api_forecast_series(pid: int):
    def f(c):
        hist = [dict(r) for r in c.execute(
            "SELECT date, qty_sold FROM daily_sales_history WHERE "
            "product_id = ? ORDER BY date", (pid,))]
        fcst = [dict(r) for r in c.execute(
            "SELECT forecast_date date, predicted_qty FROM ml_forecast_output "
            "WHERE product_id = ? ORDER BY forecast_date", (pid,))]
        metrics = c.execute(
            "SELECT AVG(rmse) rmse, AVG(mae) mae, AVG(mape) mape, "
            "MAX(model) model FROM ml_forecast_output WHERE product_id = ?",
            (pid,)).fetchone()
        return {"history": hist[-60:], "forecast": fcst,
                "metrics": dict(metrics) if metrics else {}}
    return q(f)


def api_assistant(question: str) -> dict:
    from src.dashboard import assistant
    return {"answer": assistant.answer(question or "")}


def api_recommend(cid: int):
    from src.personalisation import engine as pers
    def f(c):
        return pers.recommend(c, cid)
    return q(f)


def api_pending():
    from src.orders import intake
    def f(c):
        return intake.pending_lines(c)
    return q(f)


def api_order_lines(oid: int):
    def f(c):
        return [dict(r) for r in c.execute(
            "SELECT l.line_id, l.status, l.raw_text, p.name product, "
            "l.ordered_qty, l.available_qty, l.unit_price, l.line_total "
            "FROM order_line_items l LEFT JOIN products p USING(product_id) "
            "WHERE l.order_id = ? ORDER BY l.line_id", (oid,))]
    return q(f)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                      # quiet
        pass

    # ---- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, j(obj))

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def _user(self):
        tok = (self.headers.get("Authorization") or "").replace("Bearer ", "")
        return SESSIONS.get(tok)

    # ---- GET -------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (ROOT_UI / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
            return
        if not path.startswith("/api/"):
            self._send(404, b"not found", "text/plain")
            return
        if path != "/api/health" and not path.startswith("/api/auth/"):
            if not self._user():
                return self._json({"error": "unauthorized"}, 401)
        try:
            if path == "/api/health":
                return self._json(api_health())
            if path == "/api/kpis":
                return self._json(api_kpis())
            if path == "/api/cards":
                return self._json(api_cards())
            if path == "/api/orders":
                return self._json(api_orders())
            if path == "/api/inventory":
                return self._json(api_inventory())
            if path == "/api/inventory-ledger":
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                pid = qs.get("product", [None])[0]
                limit = int(qs.get("limit", ["100"])[0])
                return self._json(api_inventory_ledger(pid, limit))
            if path == "/api/customers":
                return self._json(api_customers())
            if path == "/api/models":
                return self._json(api_models())
            if path == "/api/product-segments":
                return self._json(api_product_segments())
            if path == "/api/basket-products":
                return self._json(api_basket_products())
            if path.startswith("/api/basket-rules/"):
                return self._json(api_basket_rules(int(path.rsplit("/", 1)[1])))
            if path == "/api/simulation":
                return self._json(api_simulation())
            if path == "/api/forecast":
                return self._json(api_forecast())
            if path.startswith("/api/forecast/"):
                return self._json(api_forecast_series(int(path.rsplit("/", 1)[1])))
            if path.startswith("/api/recommend/"):
                return self._json(api_recommend(int(path.rsplit("/", 1)[1])))
            if path == "/api/pending":
                return self._json(api_pending())
            if path.startswith("/api/order/"):
                return self._json(api_order_lines(int(path.rsplit("/", 1)[1])))
            if path.startswith("/api/invoice/"):
                return self._invoice(int(path.rsplit("/", 1)[1]))
        except Exception as e:
            return self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)
        self._json({"error": "not found"}, 404)

    # ---- POST ------------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/auth/login":
                return self._login()
            if path == "/api/auth/signup":
                return self._signup()
            if path == "/api/auth/google-demo":
                return self._google_demo()
            if not self._user():
                return self._json({"error": "unauthorized"}, 401)
            if path == "/api/auth/logout":
                tok = self.headers.get("Authorization", "").replace("Bearer ", "")
                SESSIONS.pop(tok, None)
                return self._json({"ok": True})
            if path.startswith("/api/fulfil/"):
                return self._fulfil(int(path.rsplit("/", 1)[1]))
            if path == "/api/ocr":
                return self._ocr()
            if path == "/api/assistant":
                data = json.loads(self._body() or b"{}")
                return self._json(api_assistant(data.get("q", "")))
            if path == "/api/resolve":
                from src.orders import intake
                data = json.loads(self._body() or b"{}")
                def f(c):
                    return intake.resolve_line(c, int(data["line_id"]),
                                               int(data["product_id"]))
                q(f)
                return self._json({"ok": True})
            if path == "/api/inventory/add":
                data = json.loads(self._body() or b"{}")
                return self._json(api_inventory_add(data))
            if path == "/api/inventory/remove":
                data = json.loads(self._body() or b"{}")
                return self._json(api_inventory_remove(data))
            if path == "/api/inventory/set":
                data = json.loads(self._body() or b"{}")
                return self._json(api_inventory_set(data))
            if path == "/api/scan":
                filename = self.headers.get("X-Filename", "scan.png")
                return self._json(api_scan(self._body(), filename))
            if path == "/api/scan/confirm":
                data = json.loads(self._body() or b"{}")
                return self._json(api_scan_confirm(data))
        except Exception as e:
            return self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)
        self._json({"error": "not found"}, 404)

    # ---- auth ------------------------------------------------------------
    def _login(self):
        from src.auth import store
        data = json.loads(self._body() or b"{}")
        email = (data.get("email") or "").strip().lower()
        if store.is_locked(email):
            return self._json(
                {"error": f"Too many attempts. Try again in "
                          f"{store.lock_remaining(email)}s."}, 429)
        def f(c):
            store.ensure_table(c)
            return store.check_local_login(c, email,
                                           data.get("password", ""))
        user = q(f)
        if not user:
            store.record_fail(email)
            return self._json({"error": "Invalid email or password."}, 401)
        store.clear_fails(email)
        tok = secrets.token_hex(16)
        SESSIONS[tok] = user
        return self._json({"token": tok, "user": user})

    def _signup(self):
        from src.auth import store
        data = json.loads(self._body() or b"{}")
        try:
            def f(c):
                store.ensure_table(c)
                return store.create_local_user(
                    c, data.get("email", ""), data.get("name", ""),
                    data.get("password", ""))
            user = q(f)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        tok = secrets.token_hex(16)
        SESSIONS[tok] = user
        return self._json({"token": tok, "user": user})

    def _google_demo(self):
        from src.auth import store, google_auth
        def f(c):
            store.ensure_table(c)
            return store.upsert_oauth_user(
                c, **{**google_auth.demo_google_user(),
                      "provider": "google-demo"})
        user = q(f)
        tok = secrets.token_hex(16)
        SESSIONS[tok] = user
        return self._json({"token": tok, "user": user})

    # ---- actions ----------------------------------------------------------
    def _fulfil(self, oid: int):
        from src.orders import fulfilment
        def f(c):
            try:
                s = fulfilment.confirm_order(c, oid)
            except Exception:
                s = None
            res = fulfilment.fulfil_order(c, oid)
            inv = None
            try:
                from src.invoices import generator
                inv = generator.generate_for_order(c, oid)["invoice"]
            except Exception:
                pass
            return {"fulfilment": res, "invoice": inv, "confirm": s}
        return self._json(q(f))

    def _invoice(self, oid: int):
        from src.invoices import generator
        def f(c):
            return generator.generate_for_order(c, oid)["invoice"]
        pdf = q(f)
        self._send(200, Path(pdf).read_bytes(), "application/pdf")

    def _ocr(self):
        from src.ocr import engine as oe, pipeline
        if not oe.tesseract_available():
            return self._json({
                "error": "Tesseract OCR is not installed on this machine. "
                         "Windows: run  winget install --id UB-Mannheim."
                         "TesseractOCR  (or download the installer from "
                         "github.com/UB-Mannheim/tesseract/wiki), then "
                         "RESTART this server and try again."}, 501)
        raw = self._body()
        tmp = Path(tempfile.mkdtemp(prefix="ui_ocr_")) / "slip.png"
        tmp.write_bytes(raw)
        def f(c):
            res = pipeline.process_slip(tmp, conn=c)
            return [{"raw": m.query_raw, "qty": m.quantity,
                     "match": m.matched_name, "score": round(m.score, 1),
                     "status": m.status, "pid": m.product_id}
                    for m in res.matches]
        return self._json(q(f))


if __name__ == "__main__":
    print(f"🛒 ArthaMart connected UI on http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def make_server(port: int = PORT) -> ThreadingHTTPServer:
    """Factory used by `python main.py webui [port]` and the test-suite."""
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def serve(port: int = PORT) -> None:
    """Start the UI; if the port is taken (e.g. an old dashboard still
    running), automatically pick the next free one and print the real URL."""
    httpd, used = None, port
    for p in range(port, port + 8):
        try:
            httpd = make_server(p)
            used = p
            break
        except OSError:
            continue
    if httpd is None:
        print(f"❌ No free port in {port}..{port + 7} — close old dashboards.")
        return
    print(f"🛒 ArthaMart NEW dashboard → open http://localhost:{used} "
          "(Ctrl+C to stop)", flush=True)
    if used != port:
        print(f"   (port {port} was busy — using {used}; close the old "
              "dashboard to free it)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 stopped")
