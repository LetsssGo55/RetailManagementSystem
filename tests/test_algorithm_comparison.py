"""Correctness tests for the experimental frequent-itemset implementations."""
import random

from experiments.run_algorithm_comparison import (
    mine_apriori,
    mine_eclat,
    mine_fpgrowth,
)


def test_three_miners_return_same_support_counts_on_known_baskets():
    transactions = [
        frozenset({1, 2, 3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
        frozenset({1, 2, 3}),
    ]
    support = 0.40  # minimum count = 2 of 5
    apriori = mine_apriori(transactions, support)
    eclat = mine_eclat(transactions, support)
    fpgrowth = mine_fpgrowth(transactions, support)

    assert apriori == eclat == fpgrowth
    assert apriori[frozenset({1})] == 4
    assert apriori[frozenset({1, 2})] == 3
    assert apriori[frozenset({1, 2, 3})] == 2


def test_three_miners_agree_on_seeded_small_random_baskets():
    for seed in range(50):
        rng = random.Random(seed)
        transactions = []
        for _ in range(24):
            basket = frozenset(item for item in range(1, 9) if rng.random() < 0.35)
            transactions.append(basket)
        support = 0.125  # at least 3 of 24 transactions
        apriori = mine_apriori(transactions, support)
        eclat = mine_eclat(transactions, support)
        fpgrowth = mine_fpgrowth(transactions, support)
        assert apriori == eclat == fpgrowth, f"miner mismatch for seed={seed}"
