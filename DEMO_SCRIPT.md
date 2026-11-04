# ArthaMart — 3-Minute Viva Demo Script

> Design and Development of an Intelligent Retailing Application:
> Reinventing Retail Operations with Intelligent Forecasting and
> Personalisation

Run each block from the project root (`cd ArthaMart`). Every command is
safe to repeat.

---

## 0 · One-line setup (only on a fresh machine)

```bash
pip install -r requirements.txt && sudo apt install -y tesseract-ocr
python main.py all          # setup → train → simulate (~1 minute)
```

**Say:** *“This builds a 19-table SQLite database with 180 days of
synthetic kirana data, trains all three ML models, and runs the policy
simulation.”*

## 1 · OCR order intake (Phase 2–3) — ~45 s

```bash
python main.py ocr-demo
```

Point out: noisy slip deskewed automatically, typo'd names fuzzy-matched
(“Biscult Parle-G” → 95), **16/16 items matched**.

```bash
python main.py order demo
```

Point out: slips become draft orders → prioritised dispatch queue
(stress order ranked PARTIAL behind DISPATCH orders) → stock deducted.

## 2 · Intelligence (Phase 4–6) — ~45 s

```bash
python main.py train          # skip if `all` already ran
python main.py inventory plan 10
python main.py price show 36
```

Key numbers to read out:
- RF forecaster **ACCEPTED**: RMSE 4.15 vs naive 4.48, HW 4.56 (beats both
  on RMSE/MAE/MAPE)
- K-Means: ARI 0.63 vs ground truth, 12 FAST / 46 MEDIUM / 7 SLOW
- FP-Growth: 16 actionable rules, lifts up to 4.96
- EOQ/ROP plan, then live bulk price reacting to the day's mandi cost

## 3 · Decisions & documents (Phase 7–8) — ~30 s

```bash
python main.py cards 6
python main.py invoice gen demo
```

Show one of each card type, then open `data/invoices/invoice_*.pdf`
(₹ totals with round-off) — mention procurement notes for unfulfilled
items.

## 4 · The money slide (Phase 9) — ~30 s

```bash
python main.py simulate
```

**Say:** *“Both policies face the identical 180-day demand signal; only
decisions differ. The intelligent policy earns **+12% net profit** —
mainly by avoiding stockouts and passing mandi cost spikes through.”*

## 5 · Dashboard (Phase 10) — ~30 s

```bash
python main.py dashboard
```

Walk the five tabs left → right: Decision Dashboard → Order Queue → OCR
upload (drop `assets/sample_slips/slip_noisy.png`) → Model Performance →
Simulation curves.

---

## Likely viva questions → answers

| Question | Answer |
|---|---|
| Why Random Forest over ARIMA? | Handles calendar/festival features + short series without stationarity assumptions; validated against 2 baselines |
| Leakage prevention? | All lags/rolling features shifted one day; strict time-based split (no shuffle) |
| Why EOQ? | Classic trade-off of ordering vs holding cost; demand input comes from the forecast, not gut feel |
| MRP vs BULK pricing rationale? | Packaged goods are price-locked until expiry pressure; loose goods follow the mandi daily |
| Simulation fairness? | Same demand draw, costs, lead times, elasticity (−1.2); only decisions differ |
| How would it work on real data? | Swap the synthetic generator for a CSV/API loader — every module reads from the same 19-table schema |
