# ArthaMart Comparative Algorithm Evaluation

**Study:** Product segmentation and market-basket frequent-itemset mining  
**Project baseline:** ArthaMart v1.4.6, commit 39b68a0  
**Experiment date:** 4 November 2026  
**Dataset:** bundled synthetic SQLite snapshot; 65 products, 11,700 product-day sales rows, 3,000 completed synthetic POS baskets  
**Database SHA-256:** d9d40d7fc2102b1fc967da0515b9524deb4ba0d023a46e74b259e798453187b3

## Executive summary

This study compares K-Means, Agglomerative Ward and Gaussian Mixture for product segmentation, and reference implementations of FP-Growth, Apriori and ECLAT for frequent-itemset mining.

- **Retain K-Means for the current product-tier implementation.** It has the strongest agreement with ArthaMart's synthetic FAST/MEDIUM/SLOW reference labels (ARI 0.6266; NMI 0.5864). Agglomerative Ward has better geometric separation (silhouette 0.4111; Davies-Bouldin 0.7741).
- **Retain the existing MLxtend FP-Growth production path for now.** All three mining implementations returned the same 72 frequent itemsets and 12 qualifying rules. ECLAT was fastest among the reference implementations, but this does not establish that a production ECLAT library outperforms the existing MLxtend FP-Growth code.
- No production algorithm was replaced. The benchmark is an isolated research/evaluation addition.

## 1. Experimental design

### 1.1 Product segmentation

The 65-product feature matrix uses average daily sales, coefficient of variation of daily sales, total revenue and number of days with sales. A single StandardScaler transformation is applied. All candidates use three clusters to preserve the business output categories SLOW, MEDIUM and FAST. Labels are ordered by cluster mean daily sales.

Candidates:
- K-Means: k=3, n_init=10, max_iter=500, seed 42.
- Agglomerative clustering: Ward linkage, Euclidean distance, k=3.
- Gaussian Mixture: 3 components, full covariance, n_init=10, max_iter=500, reg_covar=1e-6, seed 42.

Reported fit time is the median of five runs. Stability is mean pairwise ARI across seeds 42–46.

### 1.2 Market-basket mining

The basket experiment includes only customer_orders with source='pos' and status='COMPLETED'. Four OCR demonstration orders are excluded. The 3,000 completed synthetic POS baskets are sorted by order date and order ID, then divided chronologically into 2,250 training baskets and 750 validation baskets.

All miners use the same thresholds:
- Minimum support: 0.02 (45 of 2,250 training baskets).
- Minimum confidence: 0.30.
- Minimum lift: 1.20.
- Rule shape: one product antecedent to one product consequent.

The validation metrics are held-out co-occurrence diagnostics, not next-item prediction measures, because baskets are unordered.

## 2. Product segmentation results

| Metric | K-Means | Agglomerative Ward | Gaussian Mixture |
|---|---:|---:|---:|
| Silhouette (higher is better) | 0.3465 | **0.4111** | 0.3366 |
| Davies-Bouldin (lower is better) | 0.9807 | **0.7741** | 1.2497 |
| Calinski-Harabasz (higher is better) | **37.7809** | 36.4053 | 31.6785 |
| ARI vs synthetic reference (higher is better) | **0.6266** | 0.5179 | 0.5208 |
| NMI vs synthetic reference (higher is better) | **0.5864** | 0.5172 | 0.5308 |
| Pairwise seed-stability ARI | 0.9668 | **1.0000** | **1.0000** |
| Median fit time (seconds) | 0.0113 | **0.0011** | 0.3927 |
| Python-tracked peak allocation (MiB) | 0.0187 | 0.0303 | 0.0248 |
| Cluster sizes (SLOW / MEDIUM / FAST) | 7 / 46 / 12 | 7 / 52 / 6 | 9 / 39 / 17 |

The GMM converged in seven iterations (AIC 510.1532; BIC 605.8263). AIC/BIC are GMM-specific diagnostics and are not directly comparable with K-Means or Ward.

**Interpretation:** Ward has the strongest geometric separation, while K-Means best matches the synthetic generator's business reference labels and has the highest Calinski-Harabasz score. The evidence supports retaining K-Means for now, with Ward as a credible alternative if geometric separation is the primary objective.

**Important limitation:** ARI and NMI reference labels are generated from the synthetic category-velocity mapping; they are not independent labels from a retailer or domain expert. They measure agreement with the generator's intended labels, not proven real-world correctness.

The business requirement fixes k=3. A separate existing K-Means diagnostic gave silhouette 0.5375 at k=2 versus 0.3465 at k=3. Three tiers are therefore a business-driven requirement, not the silhouette-optimal cluster count.

## 3. Frequent-itemset mining results

| Metric | FP-Growth (reference FP-tree) | Apriori (reference) | ECLAT (reference) |
|---|---:|---:|---:|
| Training / validation baskets | 2,250 / 750 | 2,250 / 750 | 2,250 / 750 |
| Frequent itemsets | 72 | 72 | 72 |
| Frequent singletons / pairs | 65 / 7 | 65 / 7 | 65 / 7 |
| Itemsets of size 3+ | 0 | 0 | 0 |
| Qualifying rules | 12 | 12 | 12 |
| Median runtime (seconds) | 0.0540 | 0.2277 | **0.0080** |
| Python-tracked peak allocation (MiB) | 1.2974 | 0.6840 | **0.5978** |
| Held-out rule coverage | 0.5200 | 0.5200 | 0.5200 |
| Mean held-out support | 0.0203 | 0.0203 | 0.0203 |
| Mean held-out confidence | 0.3006 | 0.3006 | 0.3006 |
| Mean held-out lift | 4.2852 | 4.2852 | 4.2852 |
| Weighted held-out co-occurrence rate | 0.3128 | 0.3128 | 0.3128 |

All three reference implementations returned identical itemset keys, support counts and rule sets. No algorithm had a higher rule-quality score in this experiment; the measured differences were computational.

### Frequent pairs

| Pair | Training count | Support |
|---|---:|---:|
| Rice Basmati (loose) + Toor Dal (loose) | 90 | 4.00% |
| Aashirvaad Atta 5kg + Fortune Sunflower Oil 1L | 62 | 2.76% |
| Red Chilli Powder (loose) + Turmeric Powder (loose) | 57 | 2.53% |
| Dairy Milk Silk 150g + Nescafe Classic 100g | 55 | 2.44% |
| Aashirvaad Atta 5kg + Amul Butter 500g | 48 | 2.13% |
| Potato (loose) + Tomato (loose) | 48 | 2.13% |
| Colgate MaxFresh 300g + Dove Cream Beauty 125g | 46 | 2.04% |

At 2% support, no itemsets of size three or greater were discovered in the training split. This reflects this synthetic sample and threshold; it does not imply that the algorithms cannot find larger sets at other thresholds or on other data.

**Implementation caveat:** FP-Growth timing here is for a transparent reference FP-tree implementation, not the MLxtend FP-Growth function used by the production ArthaMart module. ECLAT's observed runtime advantage is therefore implementation-specific. A like-for-like production-library benchmark on a larger transaction dataset is required before considering a production replacement. The production FP-Growth code was left unchanged.

## 4. Reproducibility and correctness

- Five timing repetitions were used; median runtime is reported.
- Clustering stability was checked across seeds 42–46.
- Itemset keys and exact support counts were compared across all three miners.
- Fifty seeded random basket datasets also produced identical support-count maps.
- Two new automated tests passed: one known-basket test and fifty seeded randomized equivalence cases (2 passed).
- The SQLite database is intentionally excluded from Git; its SHA-256 identifies the exact experimental snapshot.

## 5. Limitations

1. The dataset is synthetic; results do not establish performance on actual kirana-store data.
2. Clustering reference labels are generated by category-velocity rules, not independent human labels.
3. Unordered baskets support co-occurrence analysis, not temporal next-item prediction or causal claims.
4. Runtime and tracemalloc measurements are machine-, interpreter- and implementation-dependent. tracemalloc is not total process RSS.
5. FP-Growth timing is from a reference implementation, not the production MLxtend implementation.
6. Only single-item-to-single-item rules and one support/confidence/lift threshold configuration were evaluated.

## 6. Conclusion

For this synthetic dataset and the three-tier business requirement, K-Means remains the most defensible current segmentation choice because it most closely matches the generated retail tiers; Agglomerative Ward separates the scaled points more cleanly. FP-Growth, Apriori and ECLAT discovered the same frequent patterns and rules at the selected thresholds. ECLAT was fastest among the reference implementations, but a production change is not justified until the actual MLxtend FP-Growth path is compared with a production-grade ECLAT implementation on a larger dataset.

These findings apply to this dataset and experiment only; they do not establish universal algorithm superiority or real-retailer performance.

## References

1. J. B. MacQueen, “Some Methods for Classification and Analysis of Multivariate Observations,” Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Vol. 1, 1967, pp. 281–297.
2. J. H. Ward Jr., “Hierarchical Grouping to Optimize an Objective Function,” Journal of the American Statistical Association, Vol. 58, No. 301, 1963, pp. 236–244. doi:10.1080/01621459.1963.10500845.
3. C. Fraley and A. E. Raftery, “Model-Based Clustering, Discriminant Analysis, and Density Estimation,” Journal of the American Statistical Association, Vol. 97, No. 458, 2002, pp. 611–631.
4. R. Agrawal and R. Srikant, “Fast Algorithms for Mining Association Rules in Large Databases,” VLDB, 1994, pp. 487–499.
5. J. Han, J. Pei, and Y. Yin, “Mining Frequent Patterns without Candidate Generation,” ACM SIGMOD, 2000, pp. 1–12. doi:10.1145/335191.335372.
6. M. J. Zaki, “Scalable Algorithms for Association Mining,” IEEE Transactions on Knowledge and Data Engineering, Vol. 12, No. 3, 2000, pp. 372–390. doi:10.1109/69.846291.

## Reproduction

Run from the repository root after installing requirements and generating the ArthaMart database:

    python experiments/run_algorithm_comparison.py --db data/arthamart.db --repeats 5
    python -m pytest tests/test_algorithm_comparison.py -q

Outputs are written to reports/algorithm_comparison/: comparison_results.json, clustering_comparison.csv, association_comparison.csv and frequent_itemsets.csv.
