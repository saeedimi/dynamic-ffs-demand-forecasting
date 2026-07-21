from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_zero_start_is_best_raw_sales_in_each_group():
    ranking = pd.read_csv(ROOT / "results/full_benchmark_ranking.csv")
    for group in ["Overall", "Low Sale", "High Sale"]:
        raw = ranking[(ranking["sales_group"] == group) & (ranking["target_or_recovery"] == "Raw Sales")]
        winner = raw.sort_values("wape_percent").iloc[0]
        assert winner["model"] == "GPU XGBoost — zero start"


def test_timesnet_not_claimed_as_reproduced():
    import json
    manifest = json.loads((ROOT / "results/run_manifest.json").read_text())
    assert manifest["timesnet_latent_demand_recovery_reproduced"] is False
