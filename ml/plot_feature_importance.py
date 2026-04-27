"""
ml/plot_feature_importance.py
==============================
Plots horizontal bar graphs for Top 5 feature importances
from both Random Forest and XGBoost (saved during Phase 3 training).

Run:
    python ml/plot_feature_importance.py
"""

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves PNG without needing a display
import matplotlib.pyplot as plt
import pandas as pd

ML_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load CSVs ──────────────────────────────────────────────────────────────────
rf_path  = os.path.join(ML_DIR, "feature_importance_random_forest.csv")
xgb_path = os.path.join(ML_DIR, "feature_importance_xgboost.csv")

rf_df  = pd.read_csv(rf_path).head(5)
xgb_df = pd.read_csv(xgb_path).head(5)

# Reverse so most important appears at top
rf_df  = rf_df.iloc[::-1].reset_index(drop=True)
xgb_df = xgb_df.iloc[::-1].reset_index(drop=True)

# ── Pretty feature name mapping ────────────────────────────────────────────────
rename = {
    "arrival_delay_min"       : "Arrival Delay (min)",
    "connection_time_min"     : "Connection Time (min)",
    "departure.scheduled"     : "Departure Scheduled",
    "arrival.scheduled"       : "Arrival Scheduled",
    "arrival.iata"            : "Arrival Airport (IATA)",
    "is_intl_hub"             : "Is International Hub",
    "day_of_week"             : "Day of Week",
    "immigration_time_min"    : "Immigration Time (min)",
    "departure_hour"          : "Departure Hour",
    "scheduled_flight_min"    : "Scheduled Flight (min)",
}

rf_df["feature"]  = rf_df["feature"].map(lambda x: rename.get(x, x))
xgb_df["feature"] = xgb_df["feature"].map(lambda x: rename.get(x, x))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")

COLORS = {
    "RF":  ["#4CAF50", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9"],
    "XGB": ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"],
}

for ax, df, title, palette in zip(
    axes,
    [rf_df, xgb_df],
    ["Random Forest  —  Top 5 Features", "XGBoost  —  Top 5 Features"],
    [COLORS["RF"], COLORS["XGB"]],
):
    bars = ax.barh(
        df["feature"],
        df["importance"],
        color=palette,
        edgecolor="none",
        height=0.55,
    )

    # Value labels inside bars
    for bar, val in zip(bars, df["importance"]):
        ax.text(
            bar.get_width() - 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="right",
            fontsize=10, fontweight="bold",
            color="white",
        )

    ax.set_facecolor("#1a1d2e")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Feature Importance Score", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("#aaaaaa")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_tick_params(color="#444")
    ax.set_xlim(0, df["importance"].max() * 1.18)
    ax.grid(axis="x", color="#333355", linewidth=0.6, linestyle="--")

fig.suptitle(
    "Smart Connection Feasibility Predictor\nFeature Importance — Top 5",
    color="white", fontsize=15, fontweight="bold", y=1.03
)
plt.tight_layout()

out_path = os.path.join(ML_DIR, "feature_importance_top5.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nChart saved -> {out_path}")
