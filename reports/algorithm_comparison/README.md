# Algorithm comparison research outputs

Run from the repository root:

    python experiments/run_algorithm_comparison.py --db data/arthamart.db --repeats 5
    python -m pytest tests/test_algorithm_comparison.py -q

The benchmark compares K-Means, Agglomerative Ward and Gaussian Mixture on the 65-product feature matrix, and reference FP-Growth, Apriori and ECLAT on completed synthetic POS baskets.

Read COMPARATIVE_EVALUATION.md for the method, measured scores, interpretation, limitations and references. The DOCX report is supplied as a separate formatted deliverable.

The SQLite database is intentionally not stored in Git. Regenerate the synthetic data using the project's setup procedure before rerunning the experiment. Compare the generated database SHA-256 with the report to verify that you are using the same snapshot.
