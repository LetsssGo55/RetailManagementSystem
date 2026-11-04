# 🛒 ArthaMart

> **Design and Development of an Intelligent Retailing Application:
> Reinventing Retail Operations with Intelligent Forecasting and Personalisation**

ArthaMart is an AI-powered decision-support system for small and mid-sized Indian
retailers — kirana stores, neighbourhood marts and mini-distributors. It unifies
OCR-based order intake, dual-regime pricing (MRP-locked vs market-linked bulk),
three complementary ML techniques and classical inventory optimisation into one
working pipeline. Every prediction is delivered as a quantified **decision card**
with a clear action, a plain-language reason and a rupee impact.

---

## ✨ Key Features

| Module | Technology | Business decision |
|---|---|---|
| **OCR Order Intake** ✅ | Tesseract + OpenCV + RapidFuzz | Converts photographed order slips to line items |
| **Demand Forecasting** | Random Forest (scikit-learn) | Predicts next-7-day demand per product |
| **Product Segmentation** | K-Means (k=3) | FAST / MEDIUM / SLOW velocity tiers |
| **Market Basket Analysis** | FP-Growth (mlxtend) | Bundle & cross-promote recommendations |
| **Inventory Optimisation** ✅ | EOQ + ROP + Safety Stock | When and how much to reorder |
| **Dual-Regime Pricing** ✅ | MRP markdown ladder + cost-plus | Price per regime, expiry and demand |
| **Policy Simulation** | Controlled A/B over 180 days | ₹ profit: naive vs system policy |
| **Personalisation** | Collaborative filtering | Per-customer recommendations & segments |
| **Dashboard** | Streamlit (5 tabs) | Decision cards, orders, OCR, ML, simulation |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│  L1  PRESENTATION  (Streamlit 5-tab dashboard)        │
├──────────────────────────────────────────────────────┤
│  L2  APPLICATION / BUSINESS LOGIC                     │
│      Order Processing · OCR · Pricing · Invoices      │
├──────────────────────────────────────────────────────┤
│  L3  ML / INTELLIGENCE                                │
│      Random Forest · K-Means · FP-Growth · Simulation │
├──────────────────────────────────────────────────────┤
│  L4  DATA  (SQLite · 16 normalised tables)            │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone and enter
cd ArthaMart

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Tesseract OCR (system-level, required for OCR module)
#    Ubuntu/Debian:  sudo apt install tesseract-ocr
#    macOS:          brew install tesseract
#    Windows:        https://github.com/UB-Mannheim/tesseract/wiki

# 5. Generate database and synthetic data
python main.py setup

# 6. Check status
python main.py status

# 7. (Phase 4 ✅) Train models
python main.py train

# 8. (Phase 9+) Run simulation
python main.py simulate

# 9. Launch a dashboard (Phase 10 ✅)
python main.py dashboard      # NEW shadcn-style UI — 9 pages (port 8501)
python main.py streamlit      # classic Streamlit dashboard (port 8502)

# Or run everything:
python main.py all
```

---

## 📂 Project Structure

```
ArthaMart/
├── main.py                     # CLI entry point
├── config.yaml                 # All business rules (no magic numbers)
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py               # Config loader
│   ├── data/
│   │   ├── database.py         # SQLite schema (16+ tables)
│   │   └── generate_synthetic.py
│   ├── ocr/                    # Phase 2 ✅
│   │   ├── preprocess.py       #   grayscale→upscale→denoise→Otsu→deskew
│   │   ├── engine.py           #   Tesseract wrapper (graceful fallback)
│   │   ├── parser.py           #   OCR text → structured line items
│   │   ├── matcher.py          #   RapidFuzz fuzzy product matching
│   │   ├── pipeline.py         #   end-to-end process_slip()
│   │   └── slip_generator.py   #   renders clean/noisy/typo sample slips
│   ├── orders/                 # Phase 3 ✅
│   │   ├── models.py           #   Order/Line status state machine
│   │   ├── intake.py           #   slip→draft order + review-queue resolution
│   │   ├── availability.py     #   READY/PARTIAL/N-A + fulfilment score
│   │   ├── queue.py            #   prioritised dispatch queue
│   │   └── fulfilment.py       #   confirm → ship → deduct stock
│   ├── ml/                     # Phase 4 ✅
│   │   ├── features.py         #   leakage-free lag/rolling/calendar features
│   │   ├── baselines.py        #   naive-7d + Holt–Winters
│   │   ├── metrics.py          #   RMSE / MAE / MAPE
│   │   ├── forecasting.py      #   global Random Forest demand forecaster
│   │   ├── clustering.py       #   K-Means → FAST/MEDIUM/SLOW tiers
│   │   └── basket_analysis.py  #   FP-Growth → BUNDLE/CROSS-PROMOTE/PROMOTE
│   ├── inventory/              # Phase 5 ✅
│   │   ├── eoq.py              #   EOQ = √(2DS/H), forecast-driven demand
│   │   ├── rop.py              #   safety stock + reorder point
│   │   └── reorder.py          #   priority plan → POs + procurement alerts
│   ├── pricing/                # Phase 6 ✅
│   │   ├── mrp_pricing.py      #   expiry markdown ladder (tier-shifted)
│   │   ├── bulk_pricing.py     #   market-linked cost-plus + caps/floor
│   │   └── engine.py           #   regime dispatcher + logging
│   ├── invoices/               # Phase 7 ✅
│   │   └── generator.py        #   ReportLab invoice + procurement PDFs
│   ├── decisions/              # Phase 8 ✅
│   │   ├── cards.py            #   DecisionCard model (4 types)
│   │   └── engine.py           #   card generation, ₹-impact sorting
│   ├── simulation/             # Phase 9 ✅
│   │   ├── naive_policy.py     #   fixed rules baseline
│   │   ├── system_policy.py    #   ML-driven policy
│   │   └── runner.py           #   identical-signal A/B simulator
│   └── dashboard/              # Phase 10 ✅
│       ├── app.py              #   Streamlit 5-tab app (Plotly)
│       └── data.py             #   cached data loaders
│
├── tests/
├── data/                       # Generated SQLite DB & CSVs
└── assets/sample_slips/        # Test order slip images
```

---

## 🧪 Testing

```bash
pytest tests/ -v        # 149 tests
```

Suite covers: schema/config, OCR (parser, matcher, end-to-end), order
state machine + queue, ML (leakage-free features, acceptance criteria,
tiers, rules), EOQ/ROP, pricing boundaries, invoices, decision cards,
simulation determinism, dashboard smoke tests and QA edge cases
(zero demand, negative-stock guards, duplicate lines, threshold
boundaries, idempotent setup).

---

## 🔁 Full Pipeline Map

```
slip photo ──OCR──▶ draft order ──availability──▶ dispatch queue ──▶ fulfil
                                                        │              │
        daily_sales_history ─▶ Random Forest forecast   │         stock −
        order_line_items ────▶ FP-Growth rules          ▼          deducted
                                    │            EOQ/ROP plan ─▶ POs+alerts
        K-Means tiers ◀─────────────┴─────┬──────────────────────┘
                                          ▼
                        dual-regime pricing ─▶ invoices + decision cards
                                          ▼
                     naive vs system simulation: **+12% net profit**
```

---

## ✅ Acceptance Criteria (Definition of Done)

1. ✅ `python main.py all` runs setup → train → simulate without errors
2. ✅ Random Forest beats naive AND Holt–Winters on RMSE, MAE and MAPE
3. ✅ K-Means k=3 validated with elbow + silhouette + ARI (0.63)
4. ✅ FP-Growth produces bundle/cross-promote rules with ₹ impact
5. ✅ OCR matches ≥72% threshold on sample slips (16/16 live)
6. ✅ System policy shows higher net profit than naive (+12%)
7. ✅ Every model output surfaces as a decision card with reason + ₹ impact
8. ✅ All 149 tests pass
9. ✅ Dashboard has all 5 tabs and runs without a GPU
10. ✅ No magic numbers — every business rule lives in `config.yaml`

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Database:** SQLite3 (18 normalised tables)
- **ML:** scikit-learn, statsmodels, mlxtend
- **OCR:** Tesseract, OpenCV, RapidFuzz
- **Dashboard:** Streamlit, Plotly
- **PDFs:** ReportLab
- **Testing:** pytest

---

## 📊 CLI Commands

| Command | Description |
|---|---|
| `python main.py setup` | Generate synthetic data + build database |
| `python main.py train` | Train all ML models |
| `python main.py simulate` | Run naive vs system simulation |
| `python main.py ocr-demo` | Test OCR pipeline on sample slips (or pass an image path) |
| `python main.py make-slips` | Generate sample order-slip images |
| `python main.py order demo` | Full OCR → draft → confirm → fulfil walkthrough |
| `python main.py order queue` | Prioritised dispatch queue |
| `python main.py order pending` | Review queue (lines needing a human) |
| `python main.py inventory plan [N]` | Top-N EOQ / ROP / priority plan |
| `python main.py inventory reorder` | Run reorder cycle → POs + alerts |
| `python main.py price all` | Price every product (markdowns + cost-plus) |
| `python main.py price show <id>` | Inspect one pricing decision |
| `python main.py invoice gen <id\|demo>` | Generate invoice + procurement PDFs |
| `python main.py invoice list` | Show generated invoices |
| `python main.py cards [N]` | Decision cards sorted by ₹ impact |
| `python main.py personalise` | Customer segments + recommendations |
| `python main.py webui [port]` | Built-in shadcn-style connected web UI |
| `python main.py dashboard` | Launch Streamlit dashboard |
| `python main.py all` | setup → train → simulate |
| `python main.py status` | Show database & model status |

---

## 👥 Team

- Tushar Patel (2300140100114)
- Anadi Singh (2300140100013)
- Lalit Singh (2300140100058)
- Vishal Gangwar (2300140100123)

**Guide:** Dr. Jyoti Aggarwal, Professor, Department of CSE

**College:** Shri Ram Murti Smarak College of Engineering & Technology, Bareilly
**University:** Dr. A.P.J. Abdul Kalam Technical University, Lucknow

---

## 🧾 OCR Intake Pipeline (Phase 2)

```
slip photo → grayscale → upscale → denoise → Otsu binarise → deskew
           → Tesseract OCR → line parser → RapidFuzz matcher → order draft
```

- **Match tiers:** score ≥ 72 → `MATCHED` (auto-accept) · 55–72 → `REVIEW` · < 55 → `UNMATCHED`
- **Parser rules:** quantity = last pure number on the line; unit-suffixed numbers
  (`800g`, `1L`, `750ml`, `pack-12`) stay in the product name; handles `x4` notation,
  decimal bulk quantities and qty/rate/amount price runs.
- **Try it:** `python main.py make-slips && python main.py ocr-demo`

## 📦 Order Processing & Fulfilment (Phase 3)

```
OCR slip → DRAFT order ──► review queue (REVIEW/UNMATCHED → resolve)
                              │
                              ▼
              confirm → READY / PARTIAL / NOT_AVAILABLE per line
                        fulfilment score = Σavailable / Σordered
                              │
                              ▼
        prioritised queue: fulfilment ↓ · value ↓ · FIFO
                              │
                              ▼
              fulfil → stock deducted · low-stock flags raised
```

- **Line lifecycle:** `MATCHED/REVIEW/UNMATCHED → CONFIRMED → READY/PARTIAL/NOT_AVAILABLE → FULFILLED`
- **Dispatch flags:** `DISPATCH` (100%) · `PARTIAL` (0–100%) · `HOLD` (0%)
- **Try it:** `python main.py order demo` — runs 3 OCR slips + an oversized
  stress order through the whole flow.

---

## 🧠 ML Intelligence (Phase 4)

| Model | Result | Where it lands |
|---|---|---|
| **Random Forest forecaster** (global, 65 products pooled) | RMSE **4.15**, MAE **3.26**, MAPE **48.2%** — beats naive (4.48/3.51/51.3) *and* Holt–Winters (4.56/3.63/49.4) on every metric ✅ | `ml_forecast_output` (65 × next-7-days) |
| **K-Means segmentation** (k=3, StandardScaler) | silhouette 0.35, ARI **0.63** vs generation ground truth; 12 FAST / 46 MEDIUM / 7 SLOW | `ml_cluster_assignment` |
| **FP-Growth basket analysis** | rediscovered all 15 seeded co-purchase pairs (lifts 3.0–5.0); **16 actionable rules** — 1 BUNDLE, 5 CROSS-PROMOTE, 10 PROMOTE; ₹5,200/week projected | `bundle_recommendations` |

- Leakage-free features: lags t-1/7/14, rolling 7/14/28 mean+std (all shifted one day), calendar + festival flags, `prod_scale` expanding mean.
- Time-based split (first 144 days train, last 36 test) — no shuffle, no leakage.
- **Run:** `python main.py train` (add `--no-save` to evaluate without writing).

---

## 📦 Inventory Optimisation (Phase 5)

Classical inventory theory driven by the ML forecast:

```
EOQ = √(2·D·S / H)          D = Σ(7-day forecast) × 52
SS  = z · σ_demand · √LT    σ = forecast RMSE (fallback: 28-day std)
ROP = avg_daily·LT + SS     priority = misses×2 + Σ(7-day forecast)
```

- Products at/below **ROP** → `RECOMMENDED` purchase order (EOQ-sized) + alert.
- Stock ≤ **safety stock** → `CRITICAL` alert. Cycle is idempotent.
- **Try:** `python main.py inventory plan` then `inventory reorder`.

---

## 💰 Dual-Regime Pricing (Phase 6)

| Regime | Rule |
|---|---|
| **MRP-locked (packaged)** | Expiry markdown ladder **0% / 10% / 20% / 35%** at >14 / 8–14 / 4–7 / 0–3 days; cluster tier shifts timing (SLOW −4d, FAST +2d); never below **90% of cost** |
| **Market-linked (bulk)** | `cost × (1 + margin)`; margin = base ± demand adjustment (+1pp per +5% forecast uplift) + festival (+2pp), clamped to ±8pp; wholesale spikes pass through, daily move capped ±15% |

- Dispatcher routes by `products.regime`; every decision logged to
  `price_history_log` + `markdown_alerts_log` (idempotent per day).
- **Try:** `python main.py price all` or `price show <id>`.

---

## 🧾 Invoices & Procurement (Phase 7)

ReportLab renders two PDFs per order (Unicode ₹ via embedded DejaVu):

- **Tax invoice** — shop header + GSTIN, customer, item table, subtotal,
  round-off and grand total. Bills only the **ship-able** lines
  (`available_qty > 0`) at their unit prices.
- **Procurement note** — only when something was unavailable: product,
  category, unmet qty, suggested supplier (per-category mandi map) and the
  reference EOQ from the Phase-5 planner.

Recorded idempotently in `invoices_generated` (one row per order).
**Try:** `python main.py invoice gen demo` then `invoice list`.

---

## 🎯 Decision Cards (Phase 8)

Every ML insight lands as a quantified card — **icon, action, reason, ₹ value**
(no bare numbers, per the synopsis). Sorted by impact, descending:

| Card | Trigger | ₹ figure |
|---|---|---|
| 🚨 **Urgent Markdown** | near-expiry MRP item, SLOW/MEDIUM tier | recovered value of at-risk stock |
| 📦 **Invest-Reorder** | FAST mover at/below ROP | 7-day forecast revenue protected |
| ⏸️ **Hold** | SLOW mover above ROP | monthly holding cost of excess stock |
| 🎁 **Bundle Opportunity** | FP-Growth BUNDLE / CROSS-PROMOTE rule | weekly ₹ impact |

**Try:** `python main.py cards` (optionally `cards 5`).

---

## ⚖️ Policy Simulation (Phase 9)

Controlled A/B: **naive** vs **system** policy over the *identical* 180-day
demand signal (same stock, costs, lead times, price elasticity −1.2 — only
decisions differ).

| ₹ over 180 days | Naive | System | Δ |
|---|---:|---:|---:|
| Revenue | 15,582,099 | 16,246,436 | **+664,337** |
| COGS | 12,072,440 | 12,470,911 | +398,470 |
| Holding | 66,581 | 203,242 | +136,661 |
| Markdown loss | 153,702 | 39,631 | **−114,071** |
| Stockout loss | 138,193 | 5,701 | **−132,492** |
| **Net profit** | **3,151,182** | **3,526,951** | **+375,769 (+11.9%)** |

Intelligence that produces the gap: shelf-life-capped EOQ, surplus-gated
markdowns (no discounts when stock sells through before expiry), cost
pass-through on mandi spikes, bundle demand uplift. Day-by-day CSVs in
`data/simulation/`, persisted to `simulation_results`. **Run:** `python main.py simulate`.

---

## 📊 Streamlit Dashboard (Phase 10)

Five Plotly-interactive tabs over cached data loaders:

1. **🎯 Decision Dashboard** — all four card types, ₹ impact metrics + bar chart, filter by type
2. **📋 Order Queue** — dispatch queue, review-queue resolution UI, order detail with confirm/fulfil + invoice buttons
3. **🧾 Invoice / OCR Intake** — upload a slip → preprocessed image, raw OCR text, colour-coded matches → create draft order
4. **📈 Model Performance** — forecast history+future per product, RMSE leaderboard, K-Means PCA scatter, basket rule map
5. **⚖️ Simulation Results** — cumulative profit curves (naive vs system), P&L component bars, CSV export

**Run:** `python main.py dashboard` → opens on port 8501.

### v1.1 — auth, assistant & UI upgrades

- 🔐 **Login / Sign-up gate** — local accounts (PBKDF2-HMAC-SHA256 salted
  hashes in `app_users`) plus **Google OAuth 2.0** (set
  `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` for real mode; a labelled
  demo button works out of the box).
- 🧭 **Professional sidebar** — logo, signed-in user chip with avatar
  initials + sign-out, grouped navigation (Operations / Intelligence),
  dark-mode toggle persisted in the URL.
- 🤖 **AI Assistant page** — chat grounded in live data: reorders,
  markdowns, bundles, forecasts, prices, tiers, KPIs, simulation.
- 🧾 **Invoice PDFs fixed** — wrapping cells (no overlap) + ArthaMart logo
  in the letterhead.
- 👥 **Personalisation engine** (completes the project title): VIP /
  REGULAR / NEW / AT_RISK segmentation + lift-weighted cross-sell
  recommendations, persisted to `personalisation_profiles`; dashboard page
  + `python main.py personalise`.
- ⬇️ CSV export buttons on dispatch queue & basket rules.
- 🦶 **Redesigned footer** — two-column brand / system card.
- 🖥️ **Built-in web UI (v1.4)** — the shadcn/blocks-style dashboard is now
  *inside* the main project: `src/webui/` + `python main.py webui`.
  Login / signup / Google-demo, live KPIs, cards, orders, inventory, OCR
  upload, fulfil+invoice PDF download — all wired to the real DB.

---

*Phase 10 complete — 135/135 tests passing. Full decision-support loop now visible end to end in the browser. Next: Phase 11 (testing & QA hardening).*


## Comparative Algorithm Evaluation (Research Benchmark)

ArthaMart includes a reproducible research benchmark comparing:

- **Product segmentation:** K-Means vs Agglomerative Ward vs Gaussian Mixture.
- **Market-basket mining:** reference FP-Growth vs Apriori vs ECLAT.

The experiment reads the generated SQLite database and writes JSON/CSV evidence under `reports/algorithm_comparison/`. The benchmark is separate from the production application: the existing K-Means and MLxtend FP-Growth modules are not changed by running it.

Run from the project root:

```powershell
python experiments/run_algorithm_comparison.py --db data/arthamart.db --repeats 5
python -m pytest tests/test_algorithm_comparison.py -q
```

Read [the comparative evaluation](reports/algorithm_comparison/COMPARATIVE_EVALUATION.md) for the experimental design, measured scores, conclusions, limitations and references. The SQLite database remains excluded from Git; regenerate it using `python main.py setup` before rerunning and compare its SHA-256 with the report to confirm the same snapshot.
