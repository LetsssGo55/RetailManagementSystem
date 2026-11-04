"""Phase 4 — forecasting, clustering, basket analysis tests.

These run against the real synthetic database (read-only, save=False),
so `python main.py setup` must have been run first.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import database
from src.ml import baselines, basket_analysis, clustering, forecasting
from src.ml.features import build_features, drop_warmup, feature_columns
from src.ml.metrics import all_metrics, mae, mape, rmse


@pytest.fixture(scope="module")
def conn():
    assert config.DB_PATH.exists(), "run `python main.py setup` first"
    c = database.get_connection()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_metrics_known_values():
    actual = [10, 20, 30]
    pred = [12, 18, 33]
    assert rmse(actual, pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert mae(actual, pred) == pytest.approx((2 + 2 + 3) / 3)
    assert mape(actual, pred) == pytest.approx(
        np.mean([0.2, 0.1, 0.1]) * 100)


def test_mape_skips_zero_actuals():
    assert mape([0, 10], [5, 12]) == pytest.approx(20.0)
    assert np.isnan(mape([0, 0], [5, 5]))


def test_all_metrics_keys():
    m = all_metrics([1, 2, 3], [1, 2, 3])
    assert set(m) == {"rmse", "mae", "mape"}
    assert m["rmse"] == 0 and m["mae"] == 0 and m["mape"] == 0


# ---------------------------------------------------------------------------
# features — leakage checks
# ---------------------------------------------------------------------------

def test_lag_features_use_only_past():
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4, dtype=float)
    dates = pd.date_range("2026-01-01", periods=len(y), freq="D")
    df = build_features(y, dates)
    # lag_1 at row t must equal y[t-1]
    assert df["lag_1"].iloc[5] == y.iloc[4]
    assert df["lag_7"].iloc[10] == y.iloc[3]


def test_rolling_features_exclude_current_day():
    y = pd.Series([10.0] * 20)
    y.iloc[10] = 100.0                       # spike on day 10
    dates = pd.date_range("2026-01-01", periods=len(y), freq="D")
    df = build_features(y, dates)
    # day 10's roll7_mean must NOT feel the spike (uses days 3..9 only)
    assert df["roll7_mean"].iloc[10] == pytest.approx(10.0)
    # day 11's roll7_mean DOES include it
    assert df["roll7_mean"].iloc[11] > 10.0


def test_scale_feature_is_past_only():
    y = pd.Series([1.0] * 10 + [100.0])
    dates = pd.date_range("2026-01-01", periods=len(y), freq="D")
    df = build_features(y, dates, with_scale=True)
    assert df["prod_scale"].iloc[-1] == pytest.approx(1.0)  # spike unseen


def test_calendar_features():
    dates = pd.Series(pd.date_range("2026-01-01", periods=14, freq="D"))
    y = pd.Series([1.0] * 14)
    df = build_features(y, dates)
    assert df["is_festival"].iloc[0] == 1      # January is a festival month
    assert df["is_weekend"].sum() == 4         # 2 weekends in 14 days


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def test_naive_forecast_is_last_window_mean():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    fc = baselines.naive_forecast(s, horizon=3, window=7)
    assert len(fc) == 3
    assert fc[0] == pytest.approx(np.mean([4, 5, 6, 7, 8, 9, 10]))
    assert np.allclose(fc, fc[0])


def test_naive_test_predictions_no_lookahead():
    s = pd.Series(np.arange(20, dtype=float))
    preds = baselines.naive_test_predictions(s, test_days=3, window=7)
    # prediction for day 17 uses days 10..16 only
    assert preds[0] == pytest.approx(np.mean(np.arange(10, 17)))


def test_exp_smooth_falls_back_on_short_series():
    s = pd.Series([1.0, 2.0, 3.0])
    fc = baselines.exp_smooth_forecast(s, horizon=2)
    assert len(fc) == 2 and np.all(np.isfinite(fc))


# ---------------------------------------------------------------------------
# forecasting — end to end (acceptance criterion from the roadmap)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(conn):
    """One shared training run for the forecasting tests."""
    report = forecasting.train_all(conn, save=False)
    model, cutoff = forecasting.train_global(conn)
    return report, model, cutoff


def test_forecast_beats_baselines(trained):
    report, _, _ = trained
    assert report.n_products >= 60
    for k in ("rmse", "mae", "mape"):
        assert np.isfinite(report.rf[k])
        assert report.rf[k] < report.naive[k], f"RF lost to naive on {k}"
        assert report.rf[k] < report.hw[k], f"RF lost to HW on {k}"
    assert report.accepted


def test_predict_future_horizon(conn, trained):
    _, model, _ = trained
    rows = forecasting.predict_future(conn, 1, model, horizon=7)
    assert len(rows) == 7
    dates = [r[0] for r in rows]
    assert dates == sorted(dates) and len(set(dates)) == 7
    assert all(q >= 0 for _, q in rows)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def test_product_features_shape(conn):
    df = clustering.product_features(conn)
    assert len(df) >= 60
    assert set(clustering.FEATURES) <= set(df.columns)
    assert (df["avg_daily_sales"] >= 0).all()


def test_clusters_and_tiers(conn):
    report, df = clustering.train_clusters(conn, save=False)
    assert set(df["tier"]) == {"FAST", "MEDIUM", "SLOW"}
    assert len(df) >= 60
    # deterministic labelling: FAST centroid must beat SLOW on avg sales
    means = df.groupby("tier")["avg_daily_sales"].mean()
    assert means["FAST"] > means["MEDIUM"] > means["SLOW"]
    assert 0.0 <= report.silhouette_overall <= 1.0
    # meaningful agreement with generation ground truth
    assert report.ari > 0.3
    # elbow scan recorded for k = 2..6
    assert set(report.silhouette_by_k) == {2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# basket analysis
# ---------------------------------------------------------------------------

def test_baskets_loaded(conn):
    baskets = basket_analysis.load_baskets(conn)
    assert len(baskets) >= 2000
    assert all(len(b) >= 2 for b in baskets)


def test_basket_rules(conn):
    report, df = basket_analysis.run_basket_analysis(conn, save=False)
    assert report.n_rules_kept >= 5
    assert set(df["action"]) <= {"BUNDLE", "CROSS-PROMOTE", "PROMOTE"}
    assert (df["lift"] >= 1.2).all()
    assert (df["confidence"] >= 0.3).all()
    assert (df["weekly_impact"] >= 0).all()


def test_seeded_pair_discovered(conn):
    """The generator seeds Rice Basmati + Toor Dal co-purchases — FP-Growth
    must rediscover the association with strong lift."""
    _, df = basket_analysis.run_basket_analysis(conn, save=False)
    names = dict(conn.execute("SELECT product_id, name FROM products"))
    pairs = {(names[r.antecedent_id], names[r.consequent_id])
             for r in df.itertuples()}
    assert ("Rice Basmati (loose)", "Toor Dal (loose)") in pairs \
        or ("Toor Dal (loose)", "Rice Basmati (loose)") in pairs


# ---------------------------------------------------------------------------
# persisted outputs (written by `python main.py train`)
# ---------------------------------------------------------------------------

def test_ml_outputs_persisted(conn):
    n_fc = conn.execute("SELECT COUNT(*) FROM ml_forecast_output").fetchone()[0]
    n_cl = conn.execute("SELECT COUNT(*) FROM ml_cluster_assignment").fetchone()[0]
    n_br = conn.execute("SELECT COUNT(*) FROM bundle_recommendations").fetchone()[0]
    horizon = config.get("forecasting.forecast_horizon", 7)
    n_products = conn.execute(
        "SELECT COUNT(DISTINCT product_id) FROM daily_sales_history"
    ).fetchone()[0]
    assert n_fc == n_products * horizon
    assert n_cl == n_products
    assert n_br >= 1
