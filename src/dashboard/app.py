"""
ArthaMart — Streamlit Dashboard (Phase 10 + Phase 12 polish)
==============================================================
Sidebar navigation across five pages, ArthaMart branding with logo,
and a dark/light theme toggle (persisted in the URL).

Pages:
    1. Decision Dashboard   — quantified decision cards
    2. Order Queue          — fulfilment queue + review resolution
    3. OCR Intake           — slip upload → OCR → matched order
    4. Model Performance    — forecasting, clustering, basket analysis
    5. Simulation Results   — naive vs system policy comparison

Launch:  python main.py dashboard
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dashboard import data, branding, assistant  # noqa: E402

st.set_page_config(page_title="ArthaMart — Intelligent Retailing",
                   page_icon="🛒", layout="wide")

if not data.db_exists():
    st.error("Database not found — run `python main.py setup` first.")
    st.stop()

# theme preference (sidebar toggle / ?theme= URL param)
if "dark" not in st.session_state:
    st.session_state.dark = st.query_params.get("theme") == "dark"
DARK = st.session_state.dark
branding.apply_css(DARK)

# ---------------------------------------------------------------------------
# authentication gate (local accounts + Google OAuth / demo)
# ---------------------------------------------------------------------------
from src.auth import store as auth_store, ui as auth_ui  # noqa: E402

with data.conn() as _ac:
    USER = auth_ui.render(_ac, DARK)
if not USER:
    st.stop()

PAGE, DARK = branding.sidebar(USER)
branding.apply_css(DARK)
branding.hero()

BRAND_SEQ = [branding.palette(DARK)["danger"],
             branding.palette(DARK)["primary"],
             branding.palette(DARK)["accent"],
             branding.palette(DARK)["warn"]]


# ===========================================================================
# PAGE 1 — DECISION DASHBOARD
# ===========================================================================
def page_decisions():
    cards = data.load_cards()
    if not cards:
        st.info("No cards yet — run `python main.py train` and "
                "`python main.py inventory reorder`.")
        return

    c1, c2, c3, c4 = st.columns(4)
    md_cards = [c for c in cards if c.card_type == "URGENT_MARKDOWN"]
    ro_cards = [c for c in cards if c.card_type == "INVEST_REORDER"]
    bd_cards = [c for c in cards if c.card_type == "BUNDLE"]
    c1.metric("🚨 Markdown cards", len(md_cards),
              f"₹{sum(c.impact_inr for c in md_cards):,.0f} recoverable")
    c2.metric("📦 Reorder cards", len(ro_cards),
              f"₹{sum(c.impact_inr for c in ro_cards):,.0f} protected")
    c3.metric("🎁 Bundle cards", len(bd_cards),
              f"₹{sum(c.impact_inr for c in bd_cards):,.0f}/week")
    c4.metric("Total ₹ impact", f"₹{sum(c.impact_inr for c in cards):,.0f}",
              f"{len(cards)} cards")

    types = sorted({c.title for c in cards})
    sel = st.multiselect("Filter by card type", types, default=types)
    shown = [c for c in cards if c.title in sel]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        for c in shown[:15]:
            st.markdown(branding.render_card(c, DARK),
                        unsafe_allow_html=True)
    with col_b:
        df = pd.DataFrame(
            [{"type": c.title, "₹ impact": c.impact_inr} for c in shown])
        if not df.empty:
            agg = df.groupby("type")["₹ impact"].sum().reset_index()
            fig = px.bar(agg, x="type", y="₹ impact",
                         title="₹ impact by card type", color="type",
                         color_discrete_sequence=BRAND_SEQ)
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2 — ORDER QUEUE
# ===========================================================================
def page_orders():
    st.subheader("Prioritised dispatch queue")
    queue = data.load_queue()
    if queue:
        qdf = pd.DataFrame([{
            "rank": e.rank, "order": e.order_id, "status": e.status,
            "fulfilment %": e.fulfilment_pct, "value ₹": e.total_value,
            "lines": e.lines_total, "dispatch": e.dispatch_flag}
            for e in queue])
        st.dataframe(qdf, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export queue CSV",
                           qdf.to_csv(index=False).encode(),
                           file_name="dispatch_queue.csv", mime="text/csv")
    else:
        st.info("Queue empty — no pending orders.")

    st.subheader("Review queue — lines needing a human")
    pend = data.load_pending_lines()
    if pend:
        for it in pend:
            st.markdown(
                f"**line {it['line_id']}** (order {it['order_id']}) "
                f"`{it['status']}` — “{it['raw_text']}” "
                f"× {it['ordered_qty']:g}")
            opts = {f"#{pid} {name} ({score:.0f})": pid
                    for pid, name, score in it["candidates"]}
            ccol1, ccol2 = st.columns([3, 1])
            with ccol1:
                choice = st.selectbox("Assign product", list(opts),
                                      key=f"pq_{it['line_id']}")
            with ccol2:
                st.write("")
                if st.button("Resolve", key=f"rb_{it['line_id']}"):
                    from src.orders import intake
                    with data.conn() as c:
                        intake.resolve_line(c, it["line_id"], opts[choice])
                    st.cache_data.clear()
                    st.rerun()
    else:
        st.success("Nothing needs a human — all lines resolved. ✅")

    st.subheader("Order detail")
    orders = data.load_orders()
    if not orders.empty:
        oid = st.selectbox("Select order", orders["order_id"].tolist())
        lines = data.load_order_lines(int(oid))
        st.dataframe(lines, use_container_width=True, hide_index=True)
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🚚 Confirm + Fulfil order"):
                from src.orders import fulfilment
                try:
                    with data.conn() as c:
                        fulfilment.confirm_order(c, int(oid))
                        res = fulfilment.fulfil_order(c, int(oid))
                    st.cache_data.clear()
                    st.success(
                        f"Shipped {res['lines_shipped']}/"
                        f"{res['lines_orderable']} lines — fulfilment "
                        f"{res['fulfilment_pct']:.0f}%.")
                except Exception as e:
                    st.error(str(e))
        with bcol2:
            if st.button("🧾 Generate invoice PDF"):
                from src.invoices import generator
                with data.conn() as c:
                    res = generator.generate_for_order(c, int(oid))
                st.cache_data.clear()
                st.success(f"Invoice: {res['invoice']}")
                with open(res["invoice"], "rb") as f:
                    st.download_button("Download invoice", f.read(),
                                       file_name=f"invoice_{oid}.pdf",
                                       mime="application/pdf")


# ===========================================================================
# PAGE 3 — OCR INTAKE
# ===========================================================================
def page_ocr():
    st.subheader("Upload an order slip")
    up = st.file_uploader("Slip image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Drop `assets/sample_slips/slip_noisy.png` here to see the "
                "full OCR → matching → order pipeline.")
        return

    # never trust the client-supplied filename (path traversal / collisions):
    # keep the basename only and write into a private temp dir
    import uuid as _uuid
    safe_name = Path(up.name).name.replace(" ", "_")
    tmp = Path(tempfile.mkdtemp(prefix="arthamart_")) / (
        _uuid.uuid4().hex[:8] + "_" + safe_name)
    tmp.write_bytes(up.getbuffer())

    with st.spinner("Preprocessing + OCR + fuzzy matching…"):
        from src.ocr import engine as ocr_engine, pipeline
        if not ocr_engine.tesseract_available():
            st.error("Tesseract OCR binary not installed on this machine.")
            st.stop()
        with data.conn() as c:
            res = pipeline.process_slip(tmp, conn=c)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**Preprocessed (denoised · binarised · deskewed)**")
        from src.ocr.preprocess import preprocess
        prep = preprocess(tmp)
        st.image(prep["image"], use_container_width=True)
        st.caption(f"deskew angle: {prep['angle']:+.2f}°")
    with pcol2:
        st.markdown("**Raw OCR text**")
        st.code(res.raw_text or "(empty)")

    rows = [{"OCR text": m.query_raw, "qty": m.quantity,
             "matched product": m.matched_name or "—",
             "score": m.score, "status": m.status} for m in res.matches]
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf.style.map(branding.status_css, subset=["status"]),
                 use_container_width=True, hide_index=True)

    if st.button("🛒 Create draft order from this slip"):
        from src.orders import intake
        with data.conn() as c:
            oid = intake.create_order_from_slip(c, res, source="OCR")
        st.cache_data.clear()
        st.success(f"Draft order #{oid} created — see Order Queue page.")


# ===========================================================================
# PAGE 4 — MODEL PERFORMANCE
# ===========================================================================
def page_models():
    b = branding.palette(DARK)
    st.subheader("4.1 Demand forecasting — Random Forest vs baselines")
    fm = data.load_forecast_metrics()
    if fm.empty:
        st.info("No forecasts — run `python main.py train`.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean RMSE", f"{fm['rmse'].mean():.2f}")
        m2.metric("Mean MAE", f"{fm['mae'].mean():.2f}")
        m3.metric("Mean MAPE", f"{fm['mape'].mean():.1f}%")
        st.caption("Baselines (training run): naive-7d RMSE 4.48 / MAE 3.51 /"
                   " MAPE 51.3 · Holt–Winters 4.56 / 3.63 / 49.4 — RF beats "
                   "both on all metrics ✅")

        hist, fc = data.load_forecast_series(int(
            st.selectbox(
                "Product (history + next-7-day forecast)",
                fm["product_id"].tolist(),
                format_func=lambda pid:
                fm.loc[fm["product_id"] == pid, "name"].iloc[0])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["qty_sold"],
                                 name="actual (180d)",
                                 line=dict(color=b["primary"])))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["predicted_qty"],
                                 name="forecast (+7d)",
                                 line=dict(color=b["danger"], dash="dash")))
        fig.update_layout(xaxis_title="date", yaxis_title="units/day",
                          height=320, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        top_rmse = fm.sort_values("rmse", ascending=False).head(12)
        fig2 = px.bar(top_rmse, x="name", y="rmse",
                      title="Hardest-to-forecast products (test RMSE)")
        fig2.update_layout(xaxis_tickangle=-45, height=320,
                           margin=dict(t=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("4.2 Product segmentation — K-Means")
    report, cdf = data.load_clusters()
    if cdf.empty:
        st.info("No clusters — run `python main.py train`.")
    else:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from src.ml.clustering import FEATURES
        X = StandardScaler().fit_transform(cdf[FEATURES].values)
        emb = PCA(n_components=2).fit_transform(X)
        cdf = cdf.assign(pc1=emb[:, 0], pc2=emb[:, 1])
        s1, s2, s3 = st.columns(3)
        s1.metric("Silhouette", f"{report.silhouette_overall:.2f}")
        s2.metric("ARI vs ground truth", f"{report.ari:.2f}")
        s3.metric("Tiers",
                  " / ".join(f"{t}:{report.tier_sizes.get(t, 0)}"
                             for t in ("FAST", "MEDIUM", "SLOW")))
        fig3 = px.scatter(cdf, x="pc1", y="pc2", color="tier",
                          hover_name="product_id",
                          color_discrete_map={
                              "FAST": b["danger"], "MEDIUM": b["warn"],
                              "SLOW": b["primary"]},
                          title="Products in PCA space (coloured by tier)")
        fig3.update_layout(height=360, margin=dict(t=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("4.3 Market basket — FP-Growth rules")
    rules = data.load_basket_rules()
    if rules.empty:
        st.info("No rules — run `python main.py train`.")
    else:
        rules_view = rules.drop(columns=["antecedent_id", "consequent_id"])
        st.dataframe(rules_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export rules CSV",
                           rules_view.to_csv(index=False).encode(),
                           file_name="basket_rules.csv", mime="text/csv")
        fig4 = px.scatter(rules, x="confidence", y="lift",
                          size="weekly_impact", color="action",
                          hover_name="consequent",
                          title="Rule map — confidence vs lift "
                                "(size = ₹/week)")
        fig4.update_layout(height=360, margin=dict(t=40))
        st.plotly_chart(fig4, use_container_width=True)


# ===========================================================================
# PAGE 5 — SIMULATION RESULTS
# ===========================================================================
def page_sim():
    b = branding.palette(DARK)
    sim = data.load_simulation()
    if sim is None:
        st.info("No simulation results — run `python main.py simulate`.")
        return

    sim["cum_profit"] = sim.groupby("policy")["net_profit"].cumsum()
    naive_t = sim[sim["policy"] == "naive"].sum(numeric_only=True)
    system_t = sim[sim["policy"] == "system"].sum(numeric_only=True)
    delta = system_t["net_profit"] - naive_t["net_profit"]
    pct = delta / abs(naive_t["net_profit"]) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Naive net profit", f"₹{naive_t['net_profit']:,.0f}")
    m2.metric("System net profit", f"₹{system_t['net_profit']:,.0f}")
    m3.metric("System advantage", f"₹{delta:+,.0f}", f"{pct:+.1f}%")

    fig = px.line(sim, x="day", y="cum_profit", color="policy",
                  color_discrete_map={"naive": b["muted"],
                                      "system": b["accent"]},
                  title="Cumulative net profit over 180 days "
                        "(identical demand signal)")
    fig.update_layout(xaxis_title="day", yaxis_title="₹ cumulative",
                      height=380, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    comps = ["revenue", "cogs", "holding_cost", "markdown_loss",
             "stockout_loss", "net_profit"]
    comp_df = pd.DataFrame({
        "component": comps * 2,
        "₹": [naive_t[k] for k in comps] + [system_t[k] for k in comps],
        "policy": ["naive"] * len(comps) + ["system"] * len(comps)})
    fig2 = px.bar(comp_df, x="component", y="₹", color="policy",
                  barmode="group",
                  color_discrete_map={"naive": b["muted"],
                                      "system": b["accent"]},
                  title="P&L component breakdown (₹, 180 days)")
    fig2.update_layout(height=380, margin=dict(t=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.download_button("Download day-by-day CSV",
                       sim.to_csv(index=False).encode(),
                       file_name="simulation_results.csv",
                       mime="text/csv")


# ===========================================================================
# PAGE — PERSONALISATION
# ===========================================================================
def page_personalisation():
    from src.personalisation import engine as pers

    st.subheader("👥 Customer segments & recommendations")
    seg = data.load_segments()
    if seg.empty:
        st.info("No customers yet — run `python main.py setup`.")
        return

    if st.button("🔄 Refresh profiles"):
        with data.conn() as c:
            pers.build_profiles(c, save=True)
        st.cache_data.clear()
        st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    counts = seg["segment"].value_counts()
    m1.metric("VIP", int(counts.get("VIP", 0)))
    m2.metric("Regular", int(counts.get("REGULAR", 0)))
    m3.metric("New", int(counts.get("NEW", 0)))
    m4.metric("At risk", int(counts.get("AT_RISK", 0)))

    col_l, col_r = st.columns([2, 3])
    with col_l:
        fig = px.pie(seg, names="segment", hole=0.55,
                     title="Customer base by segment",
                     color_discrete_sequence=BRAND_SEQ)
        fig.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            seg.sort_values("spend", ascending=False).head(10)
               [["customer_id", "name", "segment", "spend", "visits"]]
               .rename(columns={"customer_id": "id"}),
            use_container_width=True, hide_index=True)
        st.download_button("️ Export segments CSV",
                           seg.to_csv(index=False).encode(),
                           file_name="customer_segments.csv",
                           mime="text/csv")

    with col_r:
        label = {i: f"{r['name']}  ·  {r['segment']}  ·  ₹{r['spend']:,.0f}"
                 for i, r in seg.iterrows()}
        pick = st.selectbox("Customer", list(label),
                            format_func=lambda i: label[i])
        cust = seg.loc[pick]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spend", f"₹{cust['spend']:,.0f}")
        c2.metric("Orders", int(cust["visits"]))
        c3.metric("Inactive (days)",
                  int(cust["recency_days"]) if cust["recency_days"] < 9000
                  else "—")
        c4.metric("Loves", str(cust["preferred_cat"] or "—"))

        with data.conn() as c:
            recs = pers.recommend(c, int(cust["customer_id"]))
        if recs:
            rdf = pd.DataFrame(recs)[["name", "reason"]]
            st.markdown("**Recommended next basket**")
            st.dataframe(rdf, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough catalogue data for recommendations yet.")


# ===========================================================================
# PAGE 6 — AI ASSISTANT
# ===========================================================================
def page_assistant():
    st.markdown(
        '<div class="dcard" style="--dc:#4F46E5">'
        '<div class="dhead"><span class="dtitle">🤖 ArthaMart Assistant'
        '</span><span class="dval">live data</span></div>'
        '<div class="drow">Ask anything about reorders, markdowns, bundles, '
        "prices, stock, forecasts, customers or profit — answers come "
        "straight from your store's data.</div></div>",
        unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "assistant",
             "content": "Hi! I'm your retail copilot. Tap a suggestion "
                        "below or type your own question 👇"}]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # suggestion chips + clear
    st.caption("Quick questions")
    chip_cols = st.columns(4)
    for i, s in enumerate(assistant.SUGGESTIONS):
        with chip_cols[i % 4]:
            if st.button(s, key=f"chip_{i}", use_container_width=True):
                st.session_state.chat.append({"role": "user", "content": s})
                st.session_state.chat.append(
                    {"role": "assistant", "content": assistant.answer(s)})
                st.rerun()
    if st.button("🗑️ Clear conversation", key="clear_chat"):
        st.session_state.chat = []
        st.rerun()

    if prompt := st.chat_input("Ask about reorders, markdowns, bundles, "
                               "prices, stock, forecasts…"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.session_state.chat.append(
            {"role": "assistant", "content": assistant.answer(prompt)})
        st.rerun()


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
PAGES = {
    "Decision Dashboard": page_decisions,
    "Order Queue": page_orders,
    "OCR Intake": page_ocr,
    "Personalisation": page_personalisation,
    "Model Performance": page_models,
    "Simulation Results": page_sim,
    "AI Assistant": page_assistant,
}
PAGES[PAGE]()

# footer with live system stats
try:
    with data.conn() as _fc:
        _rows = _fc.execute(
            "SELECT (SELECT COUNT(*) FROM daily_sales_history) + "
            "(SELECT COUNT(*) FROM order_line_items) AS n").fetchone()[0]
        _np = {r["policy"]: r["np"] for r in _fc.execute(
            "SELECT policy, SUM(net_profit) np FROM simulation_results "
            "GROUP BY policy")}
        _delta = ((_np.get("system", 0) - _np.get("naive", 0))
                  / abs(_np["naive"]) * 100) if _np.get("naive") else None
    branding.footer(db_rows=_rows, sim_delta_pct=_delta, dark=DARK)
except Exception:
    branding.footer(dark=DARK)
