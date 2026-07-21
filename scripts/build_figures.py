from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

ranking = pd.read_csv(RESULTS / "full_benchmark_ranking.csv")

for group, slug in [("Overall", "overall"), ("Low Sale", "low_sale"), ("High Sale", "high_sale")]:
    frame = ranking[ranking["sales_group"] == group].sort_values("wape_percent")
    labels = [f"{m}\n({r})" for m, r in zip(frame["model"], frame["target_or_recovery"])]
    plt.figure(figsize=(9, 5))
    plt.barh(labels, frame["wape_percent"])
    plt.xlabel("WAPE (%) — lower is better")
    plt.title(f"{group}: paper and XGBoost comparison")
    for y, value in enumerate(frame["wape_percent"]):
        plt.text(value + 0.08, y, f"{value:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / f"wape_{slug}.png", dpi=180, bbox_inches="tight")
    plt.close()

horizon = pd.read_csv(RESULTS / "horizon_comparison_uncensored_overall.csv")
plt.figure(figsize=(8, 5))
plt.plot(horizon["horizon"], horizon["wape_percent_mandatory_start"], marker="o", label="Mandatory start")
plt.plot(horizon["horizon"], horizon["wape_percent_zero_start"], marker="o", label="Zero start")
plt.xlabel("Forecast horizon (day)")
plt.ylabel("WAPE (%)")
plt.title("Overall uncensored evaluation by forecast horizon")
plt.xticks(range(1, 8))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "horizon_wape.png", dpi=180, bbox_inches="tight")
plt.close()
