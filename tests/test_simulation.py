"""Phase 9 — policy simulation tests."""
import numpy as np
import pytest

from src import config
from src.data import database
from src.simulation import runner
from src.simulation.naive_policy import NaivePolicy
from src.simulation.system_policy import SystemPolicy


# ---------------------------------------------------------------------------
# policy unit behaviour
# ---------------------------------------------------------------------------

def test_naive_price_is_static():
    p = NaivePolicy()
    # bulk price never changes with cost or month
    pr1, _ = p.price("BULK", 100, 0, None, 50.0, 20.0, 60.0)
    pr2, _ = p.price("BULK", 100, 0, None, 90.0, 20.0, 60.0)
    assert pr1 == pr2 == 60.0
    # MRP full price until the 3-day fire sale
    assert p.price("MRP", 100, 100.0, 10, 70, 20, 60)[0] == 100.0
    fire, md = p.price("MRP", 100, 100.0, 3, 70, 20, 60)
    assert fire == pytest.approx(65.0) and md is True


def test_naive_reorder_rule():
    p = NaivePolicy()
    assert p.reorder(19) == 50          # below threshold 20 → fixed 50
    assert p.reorder(20) == 0
    assert p.reorder(500) == 0


def test_system_reorder_at_rop():
    plans = {1: {"rop": 30.0, "eoq": 120.0}}
    p = SystemPolicy(plans, {})
    assert p.reorder(1, 30) == 120      # at ROP → EOQ
    assert p.reorder(1, 10) == 120
    assert p.reorder(1, 31) == 0        # above ROP → nothing
    assert p.reorder(999, 0) == 0       # unknown product


def test_system_markdown_gate():
    p = SystemPolicy({}, {})
    # surplus at risk → ladder fires
    price, md = p.price("MRP", 100, 100.0, 5, 70, 20, 60,
                        tier="MEDIUM", stock=100, avg_daily=2)
    assert md and price < 100
    # stock sells through naturally before expiry → hold full price
    price2, md2 = p.price("MRP", 100, 100.0, 5, 70, 20, 60,
                          tier="MEDIUM", stock=8, avg_daily=2)
    assert not md2 and price2 == 100.0


def test_system_tier_shift_in_sim():
    p = SystemPolicy({}, {})
    # surplus case: SLOW (−4d) discounts earlier than FAST (+2d)
    slow, md_s = p.price("MRP", 100, 100.0, 16, 70, 20, 60,
                         tier="SLOW", stock=1000, avg_daily=1)
    fast, md_f = p.price("MRP", 100, 100.0, 16, 70, 20, 60,
                         tier="FAST", stock=1000, avg_daily=1)
    assert md_s and slow < 100          # 16−4 = 12d → 10% rung
    assert not md_f and fast == 100.0   # 16+2 = 18d → full price


def test_elasticity_direction():
    e = float(config.get("simulation.demand_elasticity", -1.2))
    assert (0.8) ** e > 1.0             # cheaper → more demand
    assert (1.2) ** e < 1.0             # pricier → less demand


# ---------------------------------------------------------------------------
# runner integration on the real DB
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report(request):
    assert config.DB_PATH.exists(), "run setup first"
    conn = database.get_connection()
    rep = runner.run_simulation(conn, save=True)
    request.addfinalizer(conn.close)
    return rep, conn


def test_both_policies_full_horizon(report):
    rep, _ = report
    assert rep["n_days"] == 180
    assert len(rep["naive"]) == 180 and len(rep["system"]) == 180


def test_system_beats_naive(report):
    rep, _ = report
    assert rep["delta_inr"] > 0
    assert rep["system_totals"]["net_profit"] > rep["naive_totals"]["net_profit"]
    assert rep["delta_pct"] > 0


def test_system_avoids_stockouts(report):
    rep, _ = report
    assert rep["system_totals"]["stockout_loss"] < \
        rep["naive_totals"]["stockout_loss"]


def test_daily_rows_sum_to_totals(report):
    rep, _ = report
    for key in ("naive", "system"):
        df = rep[key]
        t = rep[f"{key}_totals"]
        for col in ("revenue", "cogs", "holding_cost", "markdown_loss",
                    "stockout_loss", "net_profit"):
            assert df[col].sum() == pytest.approx(t[col], rel=1e-6)


def test_results_persisted(report):
    _, conn = report
    n = conn.execute("SELECT COUNT(*) FROM simulation_results").fetchone()[0]
    assert n == 360                      # 2 policies × 180 days
    pols = {r[0] for r in conn.execute(
        "SELECT DISTINCT policy FROM simulation_results")}
    assert pols == {"naive", "system"}


def test_csv_outputs_exist(report):
    assert (runner.SIM_DIR / "naive_daily.csv").exists()
    assert (runner.SIM_DIR / "system_daily.csv").exists()


def test_identical_demand_signal_deterministic():
    conn = database.get_connection()
    try:
        s1 = runner.build_demand_signal(conn)
        s2 = runner.build_demand_signal(conn)
        assert set(s1) == set(s2)
        for pid in s1:
            assert np.array_equal(s1[pid], s2[pid])
    finally:
        conn.close()
