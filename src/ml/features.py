"""
ArthaMart — Forecast Feature Engineering (Phase 4, Step 4.1)
==============================================================
Per-product daily features built ONLY from past data (no leakage):

    lags      sales on t-1, t-7, t-14            (config: forecasting.lags)
    rolling   mean/std over t-1 … t-w windows    (config: forecasting.windows)
    calendar  day-of-week, month, is_weekend, is_festival, day_index (trend)

Every rolling/lag value is computed on the series shifted by one day, so the
feature row for day t never contains information from day t itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get

FESTIVAL_MONTHS = set(get("calendar.festival_months", [8, 10, 11, 1]))

FEATURE_LAGS = [f"lag_{i}" for i in get("forecasting.lags", [1, 7, 14])]


def _roll_cols(w: int) -> list[str]:
    return [f"roll{w}_mean", f"roll{w}_std"]


def feature_columns(with_scale: bool = False) -> list[str]:
    cols = list(FEATURE_LAGS)
    for w in get("forecasting.windows", [7, 14, 28]):
        cols += _roll_cols(w)
    cols += ["dow", "month", "is_weekend", "is_festival", "day_index"]
    if with_scale:
        cols += ["prod_scale"]
    return cols


def build_features(sales: pd.Series, dates: pd.Series,
                   with_scale: bool = False) -> pd.DataFrame:
    """
    Build the feature matrix for one product.

    Parameters
    ----------
    sales : daily qty_sold, indexed 0..n-1 (chronological, no gaps)
    dates : the corresponding dates
    with_scale : add `prod_scale` — expanding mean of all sales BEFORE day t
                 (leakage-free product magnitude signal for global models)

    Returns a DataFrame with one row per day (warm-up rows containing NaNs
    are included; callers drop them).
    """
    n = len(sales)
    y = sales.astype(float)
    shifted = y.shift(1)                      # everything lags by one day

    df = pd.DataFrame(index=range(n))
    for lag in get("forecasting.lags", [1, 7, 14]):
        df[f"lag_{lag}"] = y.shift(lag)
    for w in get("forecasting.windows", [7, 14, 28]):
        df[f"roll{w}_mean"] = shifted.rolling(w, min_periods=w).mean()
        df[f"roll{w}_std"] = shifted.rolling(w, min_periods=w).std()

    d = pd.Series(pd.to_datetime(dates))       # accepts Series or Index
    df["dow"] = d.dt.dayofweek.values
    df["month"] = d.dt.month.values
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_festival"] = df["month"].isin(FESTIVAL_MONTHS).astype(int)
    df["day_index"] = np.arange(n)            # linear trend proxy
    if with_scale:
        df["prod_scale"] = shifted.expanding(min_periods=1).mean()
    df["y"] = y.values
    df["date"] = d.values
    return df


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=feature_columns()).reset_index(drop=True)
