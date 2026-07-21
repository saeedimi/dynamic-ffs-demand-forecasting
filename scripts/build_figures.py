from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def model_labels(frame: pd.DataFrame) -> list[str]:
    return [
        f"{model}\n({target})"
        for model, target in zip(frame["model"], frame["target_or_recovery"])
    ]


def save_horizontal_ranking(
    frame: pd.DataFrame,
    value_column: str,
    xlabel: str,
    title: str,
    output_name: str,
) -> None:
    labels = model_labels(frame)
    plt.figure(figsize=(9, 5))
    plt.barh(labels, frame[value_column])
    plt.xlabel(xlabel)
    plt.title(title)
    offset = max(float(frame[value_column].max()) * 0.01, 0.04)
    for y, value in enumerate(frame[value_column]):
        plt.text(value + offset, y, f"{value:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / output_name, dpi=180, bbox_inches="tight")
    plt.close()


ranking = pd.read_csv(RESULTS / "full_benchmark_ranking.csv")

for group, slug in [
    ("Overall", "overall"),
    ("Low Sale", "low_sale"),
    ("High Sale", "high_sale"),
]:
    group_frame = ranking[ranking["sales_group"] == group].copy()

    wape_frame = group_frame.sort_values("wape_percent", ascending=False)
    save_horizontal_ranking(
        wape_frame,
        value_column="wape_percent",
        xlabel="WAPE (%) — lower is better",
        title=f"{group}: paper and XGBoost WAPE comparison",
        output_name=f"wape_{slug}.png",
    )

    bias_frame = group_frame.sort_values("absolute_wpe_percent", ascending=False)
    save_horizontal_ranking(
        bias_frame,
        value_column="absolute_wpe_percent",
        xlabel="Absolute WPE (%) — lower is less biased",
        title=f"{group}: absolute aggregate-bias comparison",
        output_name=f"absolute_bias_{slug}.png",
    )

horizon = pd.read_csv(RESULTS / "horizon_comparison_uncensored_overall.csv")

plt.figure(figsize=(8, 5))
plt.plot(
    horizon["horizon"],
    horizon["wape_percent_mandatory_start"],
    marker="o",
    label="Mandatory start",
)
plt.plot(
    horizon["horizon"],
    horizon["wape_percent_zero_start"],
    marker="o",
    label="Zero start",
)
plt.xlabel("Forecast horizon (day)")
plt.ylabel("WAPE (%)")
plt.title("Overall uncensored evaluation by forecast horizon")
plt.xticks(range(1, 8))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "horizon_wape.png", dpi=180, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    horizon["horizon"],
    horizon["wpe_percent_mandatory_start"],
    marker="o",
    label="Mandatory start",
)
plt.plot(
    horizon["horizon"],
    horizon["wpe_percent_zero_start"],
    marker="o",
    label="Zero start",
)
plt.axhline(0.0, linewidth=1)
plt.xlabel("Forecast horizon (day)")
plt.ylabel("WPE (%)")
plt.title("Overall uncensored aggregate bias by forecast horizon")
plt.xticks(range(1, 8))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "horizon_wpe.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"Generated 8 figures in {FIGURES}")
