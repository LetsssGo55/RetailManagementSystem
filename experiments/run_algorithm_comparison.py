"""Reproducible ArthaMart algorithm comparison.

Compares three clustering algorithms on the product-feature table and three
frequent-itemset miners on POS transactions. All methods receive the same
inputs and thresholds. The script writes JSON/CSV evidence under
reports/algorithm_comparison/.

Run from the repository root:
    python experiments/run_algorithm_comparison.py
    python experiments/run_algorithm_comparison.py --db data/arthamart.db --repeats 5

The association-mining implementations below are small, transparent reference
implementations for benchmarking. The production FP-Growth pipeline remains
unchanged and continues to use mlxtend.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sqlite3
import statistics
import time
import tracemalloc
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "arthamart.db"
OUT_DIR = ROOT / "reports" / "algorithm_comparison"
SEED = 42
N_CLUSTERS = 3
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.20

# Mirrors src/ml/clustering.py generation labels. These are synthetic-data
# reference labels, not independent human annotations from a real retailer.
CATEGORY_VELOCITY = {
    "Snacks": 8, "Confectionery": 5, "Namkeen": 6, "Household": 4,
    "Personal Care": 5, "Beverages": 7, "Grocery": 6, "Dairy": 9,
    "Pet Care": 2, "Grain": 12, "Pulses": 6, "Oil": 5,
    "Spice": 3, "Dry Fruit": 2, "Vegetable": 15, "Fruit": 10,
}
FEATURES = ["avg_daily_sales", "velocity", "revenue", "days_with_sales"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ground_truth_tier(category: str) -> str:
    velocity = CATEGORY_VELOCITY.get(category, 5)
    if velocity <= 4:
        return "SLOW"
    if velocity <= 8:
        return "MEDIUM"
    return "FAST"


def load_product_features(conn: sqlite3.Connection) -> tuple[pd.DataFrame, list[str]]:
    rows = conn.execute(
        """
        SELECT s.product_id,
               AVG(s.qty_sold) AS avg_daily_sales,
               AVG(s.qty_sold * 1.0) AS mean2,
               AVG(s.qty_sold * s.qty_sold * 1.0) AS mean_sq,
               SUM(s.revenue) AS revenue,
               SUM(CASE WHEN s.qty_sold > 0 THEN 1 ELSE 0 END) AS days_with_sales,
               p.category
        FROM daily_sales_history s
        JOIN products p USING (product_id)
        GROUP BY s.product_id, p.category
        ORDER BY s.product_id
        """
    ).fetchall()
    frame = pd.DataFrame([dict(row) for row in rows])
    variance = (frame["mean_sq"] - frame["mean2"] ** 2).clip(lower=0)
    std = np.sqrt(variance)
    frame["velocity"] = np.where(
        frame["avg_daily_sales"] > 0,
        std / frame["avg_daily_sales"],
        0.0,
    )
    frame = frame[["product_id", "category", *FEATURES]]
    truth = [ground_truth_tier(category) for category in frame["category"]]
    if len(frame) < N_CLUSTERS + 1:
        raise ValueError("Not enough products for three-cluster comparison.")
    if frame[FEATURES].isna().any().any():
        raise ValueError("Product feature table contains missing values.")
    return frame, truth


def map_clusters_to_retail_tiers(labels: np.ndarray, sales: np.ndarray) -> np.ndarray:
    """Assign cluster labels by increasing mean sales: SLOW, MEDIUM, FAST."""
    unique = np.unique(labels)
    ordered = sorted(unique, key=lambda label: float(np.mean(sales[labels == label])))
    tier_names = ["SLOW", "MEDIUM", "FAST"]
    if len(ordered) != 3:
        # A degenerate model is still represented, but gets deterministic labels.
        names = tier_names[: len(ordered)]
    else:
        names = tier_names
    mapping = {label: names[index] for index, label in enumerate(ordered)}
    return np.asarray([mapping[label] for label in labels], dtype=object)


def _fit_predict_clusterer(name: str, X: np.ndarray, seed: int) -> tuple[np.ndarray, object]:
    if name == "K-Means":
        model = KMeans(n_clusters=3, random_state=seed, n_init=10, max_iter=500)
        labels = model.fit_predict(X)
    elif name == "Agglomerative (Ward)":
        model = AgglomerativeClustering(n_clusters=3, linkage="ward")
        labels = model.fit_predict(X)
    elif name == "Gaussian Mixture":
        model = GaussianMixture(
            n_components=3,
            covariance_type="full",
            random_state=seed,
            n_init=10,
            max_iter=500,
            reg_covar=1e-6,
        )
        labels = model.fit_predict(X)
    else:
        raise ValueError(name)
    return np.asarray(labels), model


def _cluster_stability(name: str, X: np.ndarray, seeds: list[int]) -> float:
    assignments = []
    for seed in seeds:
        labels, _ = _fit_predict_clusterer(name, X, seed)
        assignments.append(labels)
    scores = [
        adjusted_rand_score(assignments[i], assignments[j])
        for i in range(len(assignments))
        for j in range(i + 1, len(assignments))
    ]
    return float(np.mean(scores)) if scores else 1.0


def benchmark_clustering(conn: sqlite3.Connection, repeats: int) -> tuple[list[dict], dict]:
    frame, truth = load_product_features(conn)
    scaler = StandardScaler()
    X = scaler.fit_transform(frame[FEATURES].to_numpy(dtype=float))
    y_true = np.asarray(truth)
    sales = frame["avg_daily_sales"].to_numpy(dtype=float)
    results = []
    for name in ("K-Means", "Agglomerative (Ward)", "Gaussian Mixture"):
        timings = []
        peak_bytes = []
        final_labels = None
        final_model = None
        for repeat in range(repeats):
            seed = SEED + repeat
            tracemalloc.start()
            start = time.perf_counter()
            labels, model = _fit_predict_clusterer(name, X, seed)
            timings.append(time.perf_counter() - start)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_bytes.append(peak)
            if repeat == 0:
                final_labels, final_model = labels, model
        assert final_labels is not None
        tier_labels = map_clusters_to_retail_tiers(final_labels, sales)
        unique_count = len(np.unique(final_labels))
        row = {
            "algorithm": name,
            "n_products": int(len(frame)),
            "n_features": len(FEATURES),
            "requested_clusters": N_CLUSTERS,
            "actual_clusters": int(unique_count),
            "silhouette": float(silhouette_score(X, final_labels)) if unique_count > 1 else None,
            "davies_bouldin": float(davies_bouldin_score(X, final_labels)) if unique_count > 1 else None,
            "calinski_harabasz": float(calinski_harabasz_score(X, final_labels)) if unique_count > 1 else None,
            "ari_vs_synthetic_reference": float(adjusted_rand_score(y_true, tier_labels)),
            "nmi_vs_synthetic_reference": float(normalized_mutual_info_score(y_true, tier_labels)),
            "seed_stability_pairwise_ari": _cluster_stability(name, X, list(range(SEED, SEED + max(5, repeats)))),
            "fit_time_median_seconds": float(statistics.median(timings)),
            "fit_time_min_seconds": float(min(timings)),
            "python_tracemalloc_peak_median_mib": float(statistics.median(peak_bytes) / (1024 * 1024)),
            "cluster_sizes": {str(k): int(v) for k, v in Counter(tier_labels).items()},
        }
        if name == "Gaussian Mixture":
            row["gmm_aic"] = float(final_model.aic(X))
            row["gmm_bic"] = float(final_model.bic(X))
            row["gmm_converged"] = bool(final_model.converged_)
            row["gmm_n_iter"] = int(final_model.n_iter_)
        else:
            row["gmm_aic"] = None
            row["gmm_bic"] = None
            row["gmm_converged"] = None
            row["gmm_n_iter"] = None
        results.append(row)
    metadata = {
        "product_count": int(len(frame)),
        "feature_names": FEATURES,
        "feature_scaling": "StandardScaler fitted once on the complete 65-product feature matrix; unsupervised descriptive benchmark, not a predictive holdout task.",
        "cluster_count": N_CLUSTERS,
        "reference_label_source": "Synthetic generator category-velocity mapping mirrored from src/ml/clustering.py; not independently human-labelled retail ground truth.",
        "reference_label_counts": dict(Counter(y_true)),
    }
    return results, metadata


@dataclass
class FPNode:
    item: int | None
    count: int
    parent: "FPNode | None"
    children: dict[int, "FPNode"]


def _fp_build_tree(weighted_transactions: list[tuple[tuple[int, ...], int]], min_count: int):
    counts: Counter[int] = Counter()
    for items, weight in weighted_transactions:
        for item in set(items):
            counts[item] += weight
    frequent = {item: count for item, count in counts.items() if count >= min_count}
    if not frequent:
        return None, {}
    order = sorted(frequent, key=lambda item: (-frequent[item], item))
    rank = {item: index for index, item in enumerate(order)}
    root = FPNode(None, 0, None, {})
    headers: dict[int, list] = {item: [frequent[item], []] for item in frequent}
    for items, weight in weighted_transactions:
        filtered = [item for item in set(items) if item in frequent]
        filtered.sort(key=lambda item: rank[item])
        node = root
        for item in filtered:
            child = node.children.get(item)
            if child is None:
                child = FPNode(item, 0, node, {})
                node.children[item] = child
                headers[item][1].append(child)
            child.count += weight
            node = child
    return root, headers


def mine_fpgrowth(transactions: list[frozenset[int]], min_support: float) -> dict[frozenset[int], int]:
    min_count = max(1, math.ceil(min_support * len(transactions) - 1e-12))
    weighted = [(tuple(tx), 1) for tx in transactions]
    results: dict[frozenset[int], int] = {}

    def mine(weighted_tx: list[tuple[tuple[int, ...], int]], suffix: frozenset[int]):
        root, headers = _fp_build_tree(weighted_tx, min_count)
        if root is None:
            return
        for item in sorted(headers, key=lambda value: (headers[value][0], value)):
            support_count = int(headers[item][0])
            pattern = suffix | frozenset([item])
            results[pattern] = support_count
            conditional_base: list[tuple[tuple[int, ...], int]] = []
            for node in headers[item][1]:
                path = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional_base.append((tuple(reversed(path)), node.count))
            if conditional_base:
                mine(conditional_base, pattern)

    mine(weighted, frozenset())
    return results


def mine_apriori(transactions: list[frozenset[int]], min_support: float) -> dict[frozenset[int], int]:
    n = len(transactions)
    min_count = max(1, math.ceil(min_support * n - 1e-12))
    counts = Counter(item for tx in transactions for item in tx)
    current = {frozenset([item]): count for item, count in counts.items() if count >= min_count}
    results = dict(current)
    k = 2
    while current:
        previous = set(current)
        ordered = sorted(previous, key=lambda s: tuple(sorted(s)))
        candidates: set[frozenset[int]] = set()
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                union = ordered[i] | ordered[j]
                if len(union) != k:
                    continue
                if all(frozenset(subset) in previous for subset in itertools.combinations(union, k - 1)):
                    candidates.add(union)
        if not candidates:
            break
        candidate_counts = Counter()
        for tx in transactions:
            present = [candidate for candidate in candidates if candidate.issubset(tx)]
            candidate_counts.update(present)
        current = {itemset: count for itemset, count in candidate_counts.items() if count >= min_count}
        results.update(current)
        k += 1
    return results


def mine_eclat(transactions: list[frozenset[int]], min_support: float) -> dict[frozenset[int], int]:
    min_count = max(1, math.ceil(min_support * len(transactions) - 1e-12))
    vertical: dict[int, set[int]] = defaultdict(set)
    for tid, tx in enumerate(transactions):
        for item in tx:
            vertical[item].add(tid)
    items = [(item, tids) for item, tids in sorted(vertical.items()) if len(tids) >= min_count]
    results: dict[frozenset[int], int] = {}

    def extend(prefix: tuple[int, ...], suffix_items: list[tuple[int, set[int]]]):
        for index, (item, tids) in enumerate(suffix_items):
            new_prefix = prefix + (item,)
            results[frozenset(new_prefix)] = len(tids)
            next_suffix = []
            for other, other_tids in suffix_items[index + 1:]:
                intersection = tids & other_tids
                if len(intersection) >= min_count:
                    next_suffix.append((other, intersection))
            if next_suffix:
                extend(new_prefix, next_suffix)

    extend(tuple(), items)
    return results


def mine_rules(itemsets: dict[frozenset[int], int], n_transactions: int):
    singleton_counts = {
        next(iter(itemset)): count
        for itemset, count in itemsets.items()
        if len(itemset) == 1
    }
    rules = []
    for itemset, count in itemsets.items():
        if len(itemset) != 2:
            continue
        a, b = sorted(itemset)
        support = count / n_transactions
        for antecedent, consequent in ((a, b), (b, a)):
            antecedent_count = singleton_counts.get(antecedent, 0)
            consequent_count = singleton_counts.get(consequent, 0)
            confidence = count / antecedent_count if antecedent_count else 0.0
            consequent_support = consequent_count / n_transactions if n_transactions else 0.0
            lift = confidence / consequent_support if consequent_support else 0.0
            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support_train": support,
                    "confidence_train": confidence,
                    "lift_train": lift,
                })
    return rules


def evaluate_rules(rules: list[dict], validation: list[frozenset[int]]) -> dict:
    if not validation or not rules:
        return {"validation_rules_evaluated": 0, "validation_rule_coverage": 0.0,
                "validation_mean_support": None, "validation_mean_confidence": None,
                "validation_mean_lift": None, "validation_weighted_cooccurrence_precision": None}
    n = len(validation)
    item_counts = Counter(item for tx in validation for item in tx)
    rows = []
    covered_tids: set[int] = set()
    weighted_both = 0
    weighted_antecedent = 0
    for rule in rules:
        a, b = rule["antecedent"], rule["consequent"]
        a_count = 0
        both_count = 0
        for tid, tx in enumerate(validation):
            if a in tx:
                a_count += 1
                covered_tids.add(tid)
                if b in tx:
                    both_count += 1
        if a_count == 0:
            continue
        support = both_count / n
        confidence = both_count / a_count
        b_support = item_counts[b] / n
        lift = confidence / b_support if b_support else 0.0
        rows.append((support, confidence, lift))
        weighted_both += both_count
        weighted_antecedent += a_count
    return {
        "validation_rules_evaluated": len(rows),
        "validation_rule_coverage": len(covered_tids) / n,
        "validation_mean_support": float(np.mean([r[0] for r in rows])) if rows else None,
        "validation_mean_confidence": float(np.mean([r[1] for r in rows])) if rows else None,
        "validation_mean_lift": float(np.mean([r[2] for r in rows])) if rows else None,
        "validation_weighted_cooccurrence_rate": (weighted_both / weighted_antecedent) if weighted_antecedent else None,
    }


def load_pos_baskets(conn: sqlite3.Connection):
    """Use only completed synthetic POS baskets; exclude OCR demo orders."""
    order_rows = conn.execute(
        "SELECT order_id, order_date FROM customer_orders "
        "WHERE LOWER(source) = 'pos' AND status = 'COMPLETED' "
        "ORDER BY order_date, order_id"
    ).fetchall()
    baskets_by_order: dict[int, set[int]] = {int(r["order_id"]): set() for r in order_rows}
    line_rows = conn.execute(
        "SELECT li.order_id, li.product_id FROM order_line_items li "
        "JOIN customer_orders o USING(order_id) "
        "WHERE LOWER(o.source) = 'pos' AND o.status = 'COMPLETED' "
        "AND li.product_id IS NOT NULL"
    ).fetchall()
    for row in line_rows:
        baskets_by_order[int(row["order_id"])].add(int(row["product_id"]))
    order_ids = [int(r["order_id"]) for r in order_rows]
    dates = {int(r["order_id"]): str(r["order_date"]) for r in order_rows}
    baskets = [(dates[oid], oid, frozenset(baskets_by_order[oid])) for oid in order_ids]
    baskets = [row for row in baskets if row[2]]
    baskets.sort(key=lambda row: (row[0], row[1]))
    cutoff = int(len(baskets) * 0.75)
    train = [row[2] for row in baskets[:cutoff]]
    validation = [row[2] for row in baskets[cutoff:]]
    return train, validation, {
        "total_completed_pos_baskets": len(baskets),
        "training_baskets": len(train),
        "validation_baskets": len(validation),
        "split": "chronological 75% train / 25% validation by order_date, then order_id",
        "order_status_filter": "customer_orders.source='pos' AND status='COMPLETED'",
        "ocr_demo_orders_excluded": True,
        "unique_products_in_baskets": len(set().union(*(row[2] for row in baskets))) if baskets else 0,
        "first_order_date": baskets[0][0] if baskets else None,
        "last_order_date": baskets[-1][0] if baskets else None,
        "average_basket_size": float(np.mean([len(row[2]) for row in baskets])) if baskets else 0.0,
    }


def _timed_miner(miner: Callable, transactions, support: float, repeats: int):
    elapsed = []
    peaks = []
    outputs = []
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        result = miner(transactions, support)
        elapsed.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        outputs.append(result)
    if any(output != outputs[0] for output in outputs[1:]):
        raise AssertionError("Algorithm produced non-deterministic itemsets across repeats.")
    return outputs[0], float(statistics.median(elapsed)), float(statistics.median(peaks) / (1024 * 1024))


def benchmark_association(conn: sqlite3.Connection, repeats: int):
    train, validation, metadata = load_pos_baskets(conn)
    if len(train) < 10 or len(validation) < 2:
        raise ValueError("Insufficient POS transactions for train/validation comparison.")
    miners = {
        "FP-Growth (reference FP-tree)": mine_fpgrowth,
        "Apriori (reference)": mine_apriori,
        "ECLAT (reference)": mine_eclat,
    }
    results = []
    all_itemsets = {}
    all_rules = {}
    for name, miner in miners.items():
        patterns, median_seconds, peak_mib = _timed_miner(miner, train, MIN_SUPPORT, repeats)
        rules = mine_rules(patterns, len(train))
        holdout = evaluate_rules(rules, validation)
        results.append({
            "algorithm": name,
            "training_baskets": len(train),
            "validation_baskets": len(validation),
            "min_support": MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "min_lift": MIN_LIFT,
            "frequent_itemsets_total": len(patterns),
            "frequent_singletons": sum(1 for p in patterns if len(p) == 1),
            "frequent_pairs": sum(1 for p in patterns if len(p) == 2),
            "frequent_itemsets_size_3_plus": sum(1 for p in patterns if len(p) >= 3),
            "association_rules_kept": len(rules),
            "runtime_median_seconds": median_seconds,
            "python_tracemalloc_peak_mib": peak_mib,
            **holdout,
        })
        all_itemsets[name] = patterns
        all_rules[name] = rules
    reference = all_itemsets["FP-Growth (reference FP-tree)"]
    exact_equivalence = {
        name: {
            "same_itemset_keys": set(patterns) == set(reference),
            "same_support_counts": patterns == reference,
            "itemset_count": len(patterns),
        }
        for name, patterns in all_itemsets.items()
    }
    if not all(value["same_support_counts"] for value in exact_equivalence.values()):
        raise AssertionError(f"Frequent-itemset algorithms disagree: {exact_equivalence}")
    # All algorithms must generate the same rule set from the same counts.
    normalized_rules = {
        name: {(r["antecedent"], r["consequent"]) for r in rules}
        for name, rules in all_rules.items()
    }
    rule_equivalence = all(v == normalized_rules["FP-Growth (reference FP-tree)"] for v in normalized_rules.values())
    if not rule_equivalence:
        raise AssertionError("Algorithms produced different rules despite equal itemsets.")
    metadata.update({
        "support_count_threshold": math.ceil(MIN_SUPPORT * len(train) - 1e-12),
        "association_rule_filter": f"single-item -> single-item; confidence >= {MIN_CONFIDENCE}; lift >= {MIN_LIFT}",
        "frequent_itemset_equivalence_check": exact_equivalence,
        "rule_equivalence_check_passed": rule_equivalence,
        "validation_interpretation": "Held-out basket co-occurrence diagnostics. These are not next-item prediction metrics because the baskets are unordered.",
        "timing_memory_note": "All three miners are transparent Python reference implementations executed in the same process; tracemalloc captures Python-tracked allocations, not total process RSS or native allocations.",
    })
    return results, metadata, all_itemsets


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to the ArthaMart SQLite database")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repetitions (median reported)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least 3 repeats; 5 is recommended.")
    if not args.db.exists():
        parser.error(f"Database not found: {args.db}. Run `python main.py setup` first.")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        clustering_rows, clustering_meta = benchmark_clustering(conn, args.repeats)
        association_rows, association_meta, itemsets = benchmark_association(conn, args.repeats)
        db_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("products", "daily_sales_history", "customer_orders", "order_line_items")
        }
    finally:
        conn.close()

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "clustering_comparison.csv", clustering_rows)
    write_csv(args.out / "association_comparison.csv", association_rows)
    # Itemset-level detail is useful for reproducibility and discrepancy review.
    fp_patterns = itemsets["FP-Growth (reference FP-tree)"]
    product_names_conn = sqlite3.connect(args.db)
    name_map = {int(pid): name for pid, name in product_names_conn.execute("SELECT product_id, name FROM products")}
    product_names_conn.close()
    itemset_rows = []
    total_train = association_meta["training_baskets"]
    for pattern, count in sorted(fp_patterns.items(), key=lambda pair: (len(pair[0]), -pair[1], tuple(sorted(pair[0])))):
        itemset_rows.append({
            "itemset_size": len(pattern),
            "product_ids": sorted(pattern),
            "product_names": [name_map.get(pid, str(pid)) for pid in sorted(pattern)],
            "support_count": count,
            "support": count / total_train,
        })
    write_csv(args.out / "frequent_itemsets.csv", itemset_rows)

    payload = {
        "experiment": "ArthaMart clustering and frequent-itemset algorithm comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "random_seed": SEED,
        "timing_repeats": args.repeats,
        "database": {
            "filename": args.db.name,
            "sha256": sha256_file(args.db),
            "table_counts": db_counts,
        },
        "clustering": {"metadata": clustering_meta, "results": clustering_rows},
        "association_mining": {"metadata": association_meta, "results": association_rows},
        "limitations": [
            "The database is synthetic; results do not establish real-kirana-store performance.",
            "Clustering reference labels are generated from synthetic category velocities, not independent human labels.",
            "Association validation uses unordered held-out baskets; validation co-occurrence is not a temporal next-item prediction metric.",
            "Reference Python miner timings are implementation-specific and should not be interpreted as a universal ranking of optimised library implementations.",
            "tracemalloc reports Python-tracked peak allocations, not complete process resident memory.",
        ],
    }
    (args.out / "comparison_results.json").write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("CLUSTERING COMPARISON")
    print(pd.DataFrame(clustering_rows)[[
        "algorithm", "silhouette", "davies_bouldin", "calinski_harabasz",
        "ari_vs_synthetic_reference", "nmi_vs_synthetic_reference", "fit_time_median_seconds"
    ]].to_string(index=False))
    print("\nASSOCIATION MINING COMPARISON")
    print(pd.DataFrame(association_rows)[[
        "algorithm", "frequent_itemsets_total", "association_rules_kept",
        "runtime_median_seconds", "python_tracemalloc_peak_mib",
        "validation_rule_coverage", "validation_mean_confidence", "validation_mean_lift"
    ]].to_string(index=False))
    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
