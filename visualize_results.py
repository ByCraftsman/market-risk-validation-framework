from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_RESULTS_DIR = PROJECT_ROOT / "results" / "reference_run"
FIGURES_DIR = PROJECT_ROOT / "figures"

STATIC_RESULTS_PATH = REFERENCE_RESULTS_DIR / "static_var_es_summary.csv"
BACKTEST_RESULTS_PATH = REFERENCE_RESULTS_DIR / "backtesting_summary.csv"
ROLLING_FORECASTS_PATH = REFERENCE_RESULTS_DIR / "rolling_var_forecasts.csv"

MODEL_ORDER = ["Historical", "Parametric", "Monte Carlo", "EWMA", "GARCH", "FHS"]


def load_reference_results():
    required_files = [STATIC_RESULTS_PATH, BACKTEST_RESULTS_PATH, ROLLING_FORECASTS_PATH]

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing reference result files:\n{missing}")

    static_results = pd.read_csv(STATIC_RESULTS_PATH, index_col="Model")
    backtest_results = pd.read_csv(BACKTEST_RESULTS_PATH)
    rolling_forecasts = pd.read_csv(ROLLING_FORECASTS_PATH, index_col="Date", parse_dates=True)

    return static_results, backtest_results, rolling_forecasts


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")




#----Static VaR & ES comparison----    
def plot_static_var_es(static_results):
    plot_data = static_results.reindex(
        ["Historical", "Parametric", "Monte Carlo"]
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_data.plot(
        kind="bar",
        ax=ax,
        color=["#4472C4", "#C00000"],
        width=0.72,
    )

    ax.set_title("Static 5-Day 99% VaR and Expected Shortfall")
    ax.set_xlabel("")
    ax.set_ylabel("Portfolio loss (USD)")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"${value / 1_000:.0f}k")
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "static_var_es_comparison.png")




#----Overall violation rates----
def plot_violation_rates(backtest_results):
    plot_data = (
        backtest_results
        .set_index("Model")
        .reindex(MODEL_ORDER)
    )

    rates = (
        plot_data[[
            "Violation Rate",
            "Non-overlapping Violation Rate",
        ]]
        * 100
    )
    rates.columns = ["Overlapping", "Non-overlapping"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rates.plot(
        kind="bar",
        ax=ax,
        color=["#4472C4", "#A5A5A5"],
        width=0.75,
    )

    ax.axhline(
        1.0,
        color="#C00000",
        linestyle="--",
        linewidth=1.5,
        label="Expected rate (1%)",
    )
    ax.set_title("VaR Violation Rates by Model")
    ax.set_xlabel("")
    ax.set_ylabel("Violation rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "violation_rate_comparison.png")




#----Traffic light overall performance----
def plot_traffic_light_distribution(backtest_results):
    plot_data = (
        backtest_results
        .set_index("Model")
        .reindex(MODEL_ORDER)
    )

    zone_ratios = (
        plot_data[[
            "Traffic Green Ratio",
            "Traffic Yellow Ratio",
            "Traffic Red Ratio",
        ]]
        * 100
    )
    zone_ratios.columns = ["Green", "Yellow", "Red"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    zone_ratios.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#70AD47", "#FFC000", "#C00000"],
        width=0.72,
    )

    ax.set_title("Rolling Basel Traffic-Light Distribution", pad=35)
    ax.set_xlabel("")
    ax.set_ylabel("Share of rolling windows")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)

    fig.tight_layout()
    save_figure(fig, "traffic_light_distribution.png")




#----Overall model performance comparison----    
def plot_rolling_var_backtests(rolling_forecasts):
    pnl_column = "Realized 5-day PnL"

    var_columns = {
        "Historical": "Historical VaR",
        "Parametric": "Parametric VaR",
        "Monte Carlo": "Monte Carlo VaR",
        "EWMA": "EWMA VaR",
        "GARCH": "GARCH VaR",
        "FHS": "FHS VaR",
    }

    PNL_COLOR = "#A5A5A5"
    VAR_THRESHOLD_COLOR = "#4472C4"
    BREACH_COLOR = "#C00000"

    fig, axes = plt.subplots(nrows=3,ncols=2, figsize=(15, 13), sharex=True, sharey=True)

    for ax, model in zip(axes.flat, MODEL_ORDER):
        var_column = var_columns[model]
        aligned = rolling_forecasts[[pnl_column, var_column]].dropna()
        threshold = -aligned[var_column]
        violations = (aligned[pnl_column] < threshold)
        
        ax.plot(
            aligned.index,
            aligned[pnl_column],
            color=PNL_COLOR,
            linewidth=0.6,
            alpha=0.50,
            label="Realized 5-day PnL",
            )
        
        ax.plot(
            aligned.index,
            threshold,
            color=VAR_THRESHOLD_COLOR,
            linewidth=1.0,
            label="99% VaR threshold (-VaR)",
            )

        ax.scatter(
            aligned.index[violations],
            aligned.loc[violations, pnl_column],
            color=BREACH_COLOR,
            s=12,
            zorder=3,
            label="VaR breach",
            )

        ax.axhline(0, color="black", linewidth=0.5) # axhline = axis horizontal line
        ax.set_title(f"{model} ({violations.sum()} VaR breaches)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1_000:.0f}k"))
        handles, labels = (axes.flat[0].get_legend_handles_labels())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=13,
        )

    fig.suptitle(
        "Rolling 5-Day 99% VaR Backtest by Model",
        fontsize=18,
        y=0.995,
        )

    fig.text(0.5, 0.965, 
             ("A breach occurs when realized PnL falls below the VaR threshold."),
             ha="center", 
             color="#595959",
             fontsize=13,
             )

    fig.supylabel("Realized 5-day PnL and VaR threshold (USD)", fontsize=15)

    fig.tight_layout(rect=[0.03, 0.065, 1, 0.945])

    save_figure(fig,"rolling_var_backtests.png")    




def main():
    plt.style.use("seaborn-v0_8-whitegrid")

    static_results, backtest_results, rolling_forecasts = (
        load_reference_results()
    )

    plot_static_var_es(static_results)
    plot_violation_rates(backtest_results)
    plot_traffic_light_distribution(backtest_results)
    plot_rolling_var_backtests(rolling_forecasts)


if __name__ == "__main__":
    main()
