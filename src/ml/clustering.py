"""
ArthaMart — Product Segmentation: K-Means (Phase 4, Step 4.2)
================================================================
Features per product:
    avg_daily_sales   mean units sold per day
    velocity          std/mean of daily sales (coefficient of variation)
    revenue           total ₹ over the window
    days_with_sales   how many of the 180 days recorded a sale

Pipeline: StandardScaler → KMeans(k=3, seed=42). k is validated with the
elbow method (inertia) + silhouette over k=2..6. Clusters are labelled
deterministically by ranking centroids on avg daily sales:
    slowest → SLOW, middle → MEDIUM, fastest → FAST.

Validation: Adjusted Rand Index against ground-truth tiers derived from
the data generator's category velocities (independent of the clustering
features' noise), plus a confusion matrix.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler

from ..config import get

# Mirrors src/data/generate_synthetic.py::_base_velocity — the generation
# ground truth. Buckets: ≤4 → SLOW, 5–8 → MEDIUM, ≥9 → FAST.
CATEGORY_VELOCITY = {
    "Snacks": 8, "Confectionery": 5, "Namkeen": 6, "Household": 4,
    "Personal Care": 5, "Beverages": 7, "Grocery": 6, "Dairy": 9,
    "Pet Care": 2, "Grain": 12, "Pulses": 6, "Oil": 5,
    "Spice": 3, "Dry Fruit": 2, "Vegetable": 15, "Fruit": 10,
}

FEATURES = ["avg_daily_sales", "velocity", "revenue", "days_with_sales"]


@dataclass
class ClusterReport:
    k: int = 3
    inertia: dict = field(default_factory=dict)      # k → inertia (elbow)
    silhouette_by_k: dict = field(default_factory=dict)
    silhouette_overall: float = 0.0
    tier_sizes: dict = field(default_factory=dict)
    ari: float = 0.0
    confusion: list = field(default_factory=list)


def ground_truth_tier(category: str) -> str:
    v = CATEGORY_VELOCITY.get(category, 5)
    if v <= 4:
        return "SLOW"
    if v <= 8:
        return "MEDIUM"
    return "FAST"


def product_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per product with the four segmentation features."""
    rows = conn.execute(
        """
        SELECT s.product_id,
               AVG(s.qty_sold)                          AS avg_daily_sales,
               AVG(s.qty_sold * 1.0)                    AS mean2,
               AVG(s.qty_sold * s.qty_sold * 1.0)       AS mean_sq,
               SUM(s.revenue)                           AS revenue,
               SUM(CASE WHEN s.qty_sold > 0 THEN 1 ELSE 0 END) AS days_with_sales
        FROM daily_sales_history s
        GROUP BY s.product_id
        ORDER BY s.product_id
        """
    ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    # velocity = std/mean (population std via E[x²] − E[x]²)
    var = (df["mean_sq"] - df["mean2"] ** 2).clip(lower=0)
    std = np.sqrt(var)
    df["velocity"] = np.where(df["avg_daily_sales"] > 0,
                              std / df["avg_daily_sales"], 0.0)
    return df[["product_id", *FEATURES]]


def choose_k(X: np.ndarray, k_range=range(2, 7),
             random_state: int = 42) -> tuple[dict, dict]:
    """Elbow (inertia) + silhouette scan. Returns (inertia, silhouette) by k."""
    inertia, sil = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertia[k] = float(km.inertia_)
        sil[k] = float(silhouette_score(X, labels)) if k < len(X) else 0.0
    return inertia, sil


def train_clusters(conn: sqlite3.Connection, save: bool = True) -> tuple:
    """Run K-Means segmentation; persist labels; return (report, df)."""
    k = int(get("clustering.k", 3))
    seed = int(get("clustering.random_state", 42))
    tiers = list(get("clustering.tiers", ["SLOW", "MEDIUM", "FAST"]))

    df = product_features(conn)
    X = StandardScaler().fit_transform(df[FEATURES].values)

    report = ClusterReport(k=k)
    report.inertia, report.silhouette_by_k = choose_k(X, random_state=seed)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    labels = km.labels_
    sil_samples = silhouette_samples(X, labels)
    report.silhouette_overall = float(silhouette_score(X, labels))

    # deterministic tier labels: rank clusters by centroid avg_daily_sales
    feat_idx = FEATURES.index("avg_daily_sales")
    order = np.argsort(km.cluster_centers_[:, feat_idx])     # slow → fast
    cluster_to_tier = {int(c): tiers[i] for i, c in enumerate(order)}

    df["cluster"] = labels
    df["tier"] = [cluster_to_tier[int(c)] for c in labels]
    df["silhouette"] = sil_samples
    report.tier_sizes = df["tier"].value_counts().to_dict()

    # ---- validation vs generation ground truth -------------------------
    cats = dict(conn.execute(
        "SELECT product_id, category FROM products").fetchall())
    truth = [ground_truth_tier(cats.get(pid, "")) for pid in df["product_id"]]
    report.ari = float(adjusted_rand_score(truth, df["tier"]))
    report.confusion = confusion_matrix(
        truth, df["tier"], labels=["SLOW", "MEDIUM", "FAST"]).tolist()

    if save:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("DELETE FROM ml_cluster_assignment")
        conn.executemany(
            "INSERT INTO ml_cluster_assignment "
            "(product_id, cluster, tier, velocity, silhouette, assigned_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(int(r.product_id), int(r.cluster), r.tier,
              round(float(r.velocity), 6), round(float(r.silhouette), 4),
              now) for r in df.itertuples()])
        conn.commit()

    return report, df
