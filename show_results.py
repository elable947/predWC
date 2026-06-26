import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

PREDICTIONS = "data/knockout_predictions.csv"


LOCAL_COLOR = "#3498db"
AWAY_COLOR = "#e67e22"


def plot_advancement(df):
    if df.is_empty():
        print("  No data to plot")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    matches = [m.replace(" vs ", "\nvs\n") for m in df["match"]]
    y = np.arange(len(matches))
    width = 0.35
    ax.barh(y + width / 2, df["local_advance_pct"], width, label="Local", color=LOCAL_COLOR)
    ax.barh(y - width / 2, df["away_advance_pct"], width, label="Visitante", color=AWAY_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(matches, fontsize=8)
    ax.set_xlabel("Probabilidad de avance (%)")
    ax.set_title("Avance a Octavos de Final — Mundial 2026")
    ax.legend()
    ax.set_xlim(0, 105)
    for i, (l, a) in enumerate(zip(df["local_advance_pct"], df["away_advance_pct"])):
        ax.text(l + 1, i + width / 2, f"{l:.0f}%", va="center", fontsize=7, color=LOCAL_COLOR)
        ax.text(a + 1, i - width / 2, f"{a:.0f}%", va="center", fontsize=7, color=AWAY_COLOR)
    fig.tight_layout()
    fig.savefig("data/advancement.png")
    print("  Saved data/advancement.png")
    plt.close(fig)


def plot_match_probabilities(df):
    if df.is_empty():
        print("  No data to plot")
        return
    n = len(df)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3 * rows))
    axes = axes.flatten()
    colors = [LOCAL_COLOR, "#f39c12", AWAY_COLOR]
    for i, row in enumerate(df.iter_rows(named=True)):
        ax = axes[i]
        labels = ["Local", "Empate", "Visitante"]
        vals = [row["local_win_pct"], row["draw_pct"], row["away_win_pct"]]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title(row["match"], fontsize=8)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Probabilidades por partido — 16avos Mundial 2026", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("data/probabilities.png")
    print("  Saved data/probabilities.png")
    plt.close(fig)


def plot_confidence_gauge(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    df = df.with_columns(
        (pl.max_horizontal("local_win_pct", "away_win_pct")).alias("max_prob")
    )
    matches_short = [m.replace(" vs ", "\n") for m in df["match"]]
    colors = ["#2ecc71" if l > a else "#e74c3c" for l, a in zip(df["local_advance_pct"], df["away_advance_pct"])]
    bars = ax.bar(range(len(df)), df["max_prob"], color=colors, width=0.6)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(matches_short, fontsize=7)
    ax.set_ylabel("Confianza (%)")
    ax.set_title("Confianza del modelo por partido")
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, df["max_prob"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig("data/confidence.png")
    print("  Saved data/confidence.png")
    plt.close(fig)


def main():
    print("=" * 50)
    print("  SHOW RESULTS — World Cup 2026 Predictions")
    print("=" * 50)
    print(f"\nReading {PREDICTIONS}...")
    try:
        df = pl.read_csv(PREDICTIONS)
    except FileNotFoundError:
        print(f"  ERROR: {PREDICTIONS} not found. Run stacking_model.py first.")
        return
    print(f"  {df.height} matches loaded")
    print()
    print("Generating charts...")
    plot_advancement(df)
    plot_match_probabilities(df)
    plot_confidence_gauge(df)
    print("\nAll charts saved to data/")


if __name__ == "__main__":
    main()
