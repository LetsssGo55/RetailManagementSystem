# ArthaMart v1.4.1 — FINAL Release Notes

### v1.4.1 (this zip)
- Feature parity in the shadcn UI: AI Assistant chat, per-customer
  recommendations, review-queue resolution, order-line detail, CSV exports
- shadcn-style connected web UI is now **part of the main project**:
  `src/webui/` (server.py + index.html) → `python main.py webui [port]`
- Login / signup / Google-demo + Bearer-token API over the live DB
  (KPIs, cards, orders, inventory, customers, models, simulation,
  forecast, OCR upload, fulfil + invoice PDF download)
- 166/166 tests including in-process webui API smoke tests

### v1.3.2
- Full re-test: 161/161 tests + every CLI command + every API endpoint verified
- Connected shadcn UI (`ArthaMart_ShadcnUI/`) now included: login/signup/
  Google-demo, live KPIs, cards, orders, inventory, OCR, invoices, simulation
- Login hardened: signup on the login card + sandbox-safe token storage

**Design and Development of an Intelligent Retailing Application:
Reinventing Retail Operations with Intelligent Forecasting and
Personalisation**

B.Tech Major Project · Department of CSE · Dr. A.P.J. Abdul Kalam Technical
University, Lucknow

---

## What ships

A complete, working AI decision-support system for Indian kirana /
mid-size retail, spanning **12 build phases** and **~8,000 lines** of
tested Python. **160/160 tests pass** (with Tesseract installed).

### v1.3 highlights (final)
- 🚀 `python main.py all` is now a complete one-command showcase:
  setup → train → simulate → price → reorder → personalise → OCR demo
  orders → invoices (OCR steps self-skip without Tesseract)
- 🐞 Fixed the OCR Intake crash (pandas Styler CSS declarations) + regression test
- 🔢 Version badge on login screen & footer (v1.3.0) so builds are identifiable
- ✅ 161/161 tests; acceptance criteria 10/10

### v1.2 highlights
- 🔐 Login / sign-up (local + Google OAuth) gating the dashboard
- 👥 Personalisation engine (VIP/REGULAR/NEW/AT_RISK + cross-sell recs)
- 🤖 AI assistant chat grounded in live data
- 🧭 Professional fixed sidebar, dark/light mode, branded footer
- 🧾 Overlap-free invoices with logo letterhead; CSV exports

| # | Capability | Status |
|---|---|---|
| 1 | 18-table SQLite foundation + realistic synthetic data | ✅ |
| 2 | OCR order intake (deskew → Tesseract → parse → fuzzy match) | ✅ |
| 3 | Order processing, review queue, prioritised dispatch | ✅ |
| 4 | Random Forest forecasting · K-Means segmentation · FP-Growth | ✅ |
| 5 | EOQ / ROP / safety-stock inventory planning | ✅ |
| 6 | Dual-regime pricing (MRP markdown ladder + bulk cost-plus) | ✅ |
| 7 | Invoice & procurement PDFs (ReportLab) | ✅ |
| 8 | Quantified decision cards (4 types) | ✅ |
| 9 | Naive-vs-system policy simulation | ✅ |
| 10 | Streamlit 5-tab dashboard | ✅ |
| 11 | Testing & QA (149 tests) | ✅ |
| 12 | Documentation & polish | ✅ |

## Headline numbers

- **149 / 149** automated tests passing
- **Random Forest ACCEPTED** — RMSE 4.15 / MAE 3.26 / MAPE 48.2%, beating
  naive (4.48/3.51/51.3) and Holt–Winters (4.56/3.63/49.4) on every metric
- **K-Means** ARI **0.63** vs ground truth; 12 FAST / 46 MEDIUM / 7 SLOW
- **FP-Growth** 16 actionable rules, lifts up to **4.96**
- **OCR** 16/16 items matched across clean / noisy / typo'd slips
- **Policy simulation: +₹377,315 (+12.0%)** net profit for the intelligent
  policy over 180 identical demand days

## Run it

```bash
pip install -r requirements.txt && sudo apt install -y tesseract-ocr
python main.py all        # setup → train → simulate
python main.py dashboard  # 5-tab dashboard on :8501
python main.py cards      # decision cards sorted by ₹ impact
pytest tests/ -v          # full suite
```

See `DEMO_SCRIPT.md` for the guided 3-minute walkthrough and viva Q&A.

## Configuration

Every business rule — pricing ladders, EOQ parameters, forecast
hyper-parameters, simulation elasticity — lives in `config.yaml` (also
mirrored into the `config_parameters` table). No magic numbers in code.
