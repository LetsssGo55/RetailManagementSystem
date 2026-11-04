"""
ArthaMart — Demand Forecasting: Random Forest (Phase 4, Step 4.1)
===================================================================
One GLOBAL RandomForestRegressor trained on all 65 products pooled
(~9,000 leakage-free rows). Per-product models overfit short series;
the pooled model learns shared weekly/festival/trend dynamics while the
`prod_scale` feature (expanding mean of past sales) keeps predictions
calibrated to each product's magnitude.

Evaluation protocol (per roadmap):
    • time-based split: first (n − test_days) days train, last 36 test
    • NO shuffle — chronology preserved, every feature uses past data only
    • baselines: naive rolling-7-day mean, Holt–Winters
    • acceptance: RF beats BOTH baselines on RMSE, MAE and MAPE

After evaluation the module refits on the full history and writes the
next `forecast_horizon` days per product into `ml_forecast_output`.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import get
from .baselines import exp_smooth_forecast, naive_test_predictions
from .features import build_features, drop_warmup, feature_columns
from .metrics import all_metrics

MODEL_NAME = "random_forest"
WITH_SCALE = True


@dataclass
class ForecastReport:
    n_products: int = 0
    rf: dict = field(default_factory=dict)        # aggregate metrics
    naive: dict = field(default_factory=dict)
    hw: dict = field(default_factory=dict)
    beats_naive: bool = False
    beats_hw: bool = False
    per_product: list = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.beats_naive and self.beats_hw


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def load_series(conn: sqlite3.Connection, product_id: int) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT date, qty_sold FROM daily_sales_history "
        "WHERE product_id = ? ORDER BY date", (product_id,)).fetchall()
    return pd.DataFrame({"date": [r["date"] for r in rows],
                         "y": [float(r["qty_sold"]) for r in rows]})


def _product_ids(conn) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT product_id FROM daily_sales_history ORDER BY 1")]


def _frame(conn, product_id: int) -> pd.DataFrame:
    hist = load_series(conn, product_id)
    df = drop_warmup(
        build_features(hist["y"], hist["date"], with_scale=WITH_SCALE))
    df["product_id"] = product_id
    return df


# ---------------------------------------------------------------------------
# global model
# ---------------------------------------------------------------------------

def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(get("forecasting.n_estimators", 200)),
        max_depth=int(get("forecasting.max_depth", 12)),
        random_state=int(get("forecasting.random_state", 42)),
        n_jobs=-1,
    )


def pooled_dataset(conn) -> pd.DataFrame:
    return pd.concat([_frame(conn, pid) for pid in _product_ids(conn)],
                     ignore_index=True)


def train_global(conn, test_days: int | None = None):
    """Fit the global RF on the train window. Returns (model, cutoff_date)."""
    test_days = test_days or int(get("forecasting.test_days", 36))
    pool = pooled_dataset(conn)
    cutoff = pool["date"].iloc[-test_days]
    train = pool[pool["date"] < cutoff]
    model = _new_model()
    model.fit(train[feature_columns(WITH_SCALE)], train["y"])
    return model, cutoff


def evaluate_global(conn, model, cutoff, test_days: int | None = None):
    """Score RF + both baselines over the test window, per product."""
    test_days = test_days or int(get("forecasting.test_days", 36))
    feats = feature_columns(WITH_SCALE)
    out = []
    for pid in _product_ids(conn):
        df = _frame(conn, pid)
        test = df[df["date"] >= cutoff]
        if test.empty:
            continue
        rf_preds = np.clip(model.predict(test[feats]), 0.0, None)

        hist = load_series(conn, pid)
        series = hist["y"]
        naive_preds = naive_test_predictions(series, test_days)
        hw_preds = exp_smooth_forecast(series.iloc[:-test_days], test_days)

        actual = test["y"].values
        out.append((pid, {
            "rf": all_metrics(actual, rf_preds),
            "naive": all_metrics(actual, naive_preds),
            "hw": all_metrics(actual, hw_preds),
        }))
    return out


def predict_future(conn, product_id: int, model,
                   horizon: int | None = None, last_date: str | None = None):
    """Recursive multi-step forecast for the next `horizon` days."""
    horizon = horizon or int(get("forecasting.forecast_horizon", 7))
    hist = load_series(conn, product_id)
    series = list(hist["y"])
    dates = list(pd.to_datetime(hist["date"]))
    start = pd.to_datetime(last_date) if last_date else dates[-1]
    feats = feature_columns(WITH_SCALE)

    out = []
    for h in range(1, horizon + 1):
        next_day = start + pd.Timedelta(days=h)
        ext = pd.DataFrame({"date": dates + [next_day],
                            "y": series + [np.nan]})
        row = drop_warmup(
            build_features(ext["y"], ext["date"],
                           with_scale=WITH_SCALE)).iloc[[-1]]
        pred = float(max(model.predict(row[feats])[0], 0.0))
        out.append((next_day.strftime("%Y-%m-%d"), round(pred, 2)))
        series.append(pred)                       # recursive feedback
        dates.append(next_day)
    return out


# ---------------------------------------------------------------------------
# full training run
# ---------------------------------------------------------------------------

def train_all(conn: sqlite3.Connection, save: bool = True) -> ForecastReport:
    """Train + evaluate the global model, persist next-7-day forecasts."""
    test_days = int(get("forecasting.test_days", 36))
    horizon = int(get("forecasting.forecast_horizon", 7))

    model, cutoff = train_global(conn, test_days)
    per_product = evaluate_global(conn, model, cutoff, test_days)

    report = ForecastReport(n_products=len(per_product))
    report.per_product = per_product
    agg = {k: {"rmse": [], "mae": [], "mape": []}
           for k in ("rf", "naive", "hw")}
    for _, m in per_product:
        for k in agg:
            for met, val in m[k].items():
                if np.isfinite(val):
                    agg[k][met].append(val)

    if save:
        # refit on ALL history before forecasting the future
        pool = pooled_dataset(conn)
        model = _new_model()
        model.fit(pool[feature_columns(WITH_SCALE)], pool["y"])
        conn.execute("DELETE FROM ml_forecast_output")
        for pid, m in per_product:
            rows = predict_future(conn, pid, model, horizon)
            met = m["rf"]
            conn.executemany(
                "INSERT INTO ml_forecast_output "
                "(product_id, forecast_date, predicted_qty, model, "
                " rmse, mae, mape) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(pid, d, q, MODEL_NAME, met["rmse"], met["mae"],
                  met["mape"]) for d, q in rows])
        conn.commit()

    for k in agg:
        target = {"rf": report.rf, "naive": report.naive, "hw": report.hw}[k]
        for met, vals in agg[k].items():
            target[met] = float(np.mean(vals)) if vals else float("nan")

    rf, naive, hw = report.rf, report.naive, report.hw
    report.beats_naive = all(rf[k] < naive[k] for k in ("rmse", "mae", "mape"))
    report.beats_hw = all(rf[k] < hw[k] for k in ("rmse", "mae", "mape"))
    return report
