"""Estimate and validate market risk for a multi-asset portfolio.

The framework compares Historical, Parametric, and Monte Carlo VaR/ES
and extends them with rolling backtesting and volatility-based models.

The reference configuration uses an equal-weighted USD 1 million
portfolio, a five-day holding period, and a 99% confidence level.
Short-horizon normal models assume zero expected return, and FX risk
is excluded.
"""


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from arch import arch_model
from pathlib import Path
import json
import sys

# Reproducibility
np.random.seed(42)
start_date = dt.datetime(2006, 1, 1)
end_date = dt.datetime(2026, 1, 1)


# Reference Data Path
PROJECT_ROOT = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

DATA_DIR = PROJECT_ROOT / "data"
PRICE_SNAPSHOT_PATH = (DATA_DIR / "market_prices_2006-01-01_2025-12-31.csv")

DATA_DIR.mkdir(parents=True, exist_ok=True)

tickers = [
    '^KS11',      # KOSPI Composite Index
    '^KQ11',      # Kosdaq Composite Index
    'IEF',        # iShares 7-10 Year Treasury Bond ETF
    '^GSPC'       # S&P 500
]


def fetch_prices(tickers, start_date, end_date):
    """
    Download and align daily market prices from Yahoo Finance.

    Close prices are used for equity indices. Adjusted Close is used
    for IEF to account for distributions and other price adjustments.
    """

    df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            raise ValueError(f"No price data returned for {ticker}.")

        price_column = (
            "Adj Close"
            if ticker == "IEF"
            else "Close"
        )

        series = data[price_column]

        # Handles MultiIndex output from recent yfinance versions.
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        df[ticker] = series

    return df.dropna()


def load_or_create_price_snapshot(tickers, start_date, end_date, snapshot_path):
    """Load the saved price snapshot or download and save it if absent."""
    
    if snapshot_path.exists():
        prices = pd.read_csv(
            snapshot_path,
            index_col="Date",
            parse_dates=True,
        )

        missing_tickers = set(tickers) - set(prices.columns)

        if missing_tickers:
            raise ValueError(
                f"Price snapshot is missing tickers: "
                f"{sorted(missing_tickers)}"
            )

        prices = prices.loc[:, tickers]
        data_source = "saved snapshot"

    else:
        prices = fetch_prices(
            tickers,
            start_date,
            end_date,
        )

        prices.to_csv(
            snapshot_path,
            index_label="Date",
        )

        data_source = "Yahoo Finance download"

    print(f"Data source: {data_source}")
    print(f"Observations: {len(prices):,}")
    print(
        f"Sample: {prices.index.min().date()} "
        f"to {prices.index.max().date()}"
    )

    return prices


prices = load_or_create_price_snapshot(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    snapshot_path=PRICE_SNAPSHOT_PATH,
)

weights = np.array([1/len(tickers)]*len(tickers))


def compute_log_returns(price_df):
    """Compute continuously compounded daily returns."""

    returns = np.log(price_df / price_df.shift(1)) 
    
    return returns.dropna()

log_returns = compute_log_returns(prices)


def compute_portfolio_returns(log_returns, weights):
    """Aggregate asset returns using fixed portfolio weights."""
    
    return (log_returns * weights).sum(axis=1)

portfolio_returns = compute_portfolio_returns(log_returns, weights)




# Reference Run Configuration
REFERENCE_RESULTS_DIR = (
    PROJECT_ROOT / "results" / "reference_run"
)

REFERENCE_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_CONFIG_PATH = (
    REFERENCE_RESULTS_DIR / "run_config.json"
)

run_config = {
    "reference_run_id": "market-risk-2006-2025-seed42",
    "requested_sample": {
        "start_date": start_date.date().isoformat(),
        "end_date_exclusive": end_date.date().isoformat(),
    },
    "actual_sample": {
        "start_date": prices.index.min().date().isoformat(),
        "end_date": prices.index.max().date().isoformat(),
        "observations": len(prices),
    },
    "portfolio": {
        "tickers": tickers,
        "weights": weights.tolist(),
        "portfolio_value": 1_000_000,
        "currency": "USD",
        "fx_risk_included": False,
    },
    "risk_settings": {
        "holding_period_days": 5,
        "confidence_level": 0.99,
        "estimation_window": 1_000,
        "random_seed": 42,
    },
    "simulation_settings": {
        "monte_carlo_simulations": 10_000,
        "fhs_simulations": 2_000,
        "ewma_lambda": 0.94,
        "garch_specification": "GARCH(1,1)",
    },
    "data_snapshot": (
        PRICE_SNAPSHOT_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    ),
    "environment": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "yfinance": yf.__version__,
    },
}

with RUN_CONFIG_PATH.open(
    "w",
    encoding="utf-8",
) as config_file:
    json.dump(
        run_config,
        config_file,
        indent=4,
        ensure_ascii=False,
    )

print(f"Run configuration saved to: {RUN_CONFIG_PATH}")




# Historical VaR
def compute_rolling_pnl(returns, value, horizon):
    """Convert rolling cumulative portfolio returns into monetary PnL."""
    
    rolling = returns.rolling(horizon).sum().dropna()
    return rolling * value


def compute_historical_VaR(pnl, confidence):
    """Estimate VaR from the empirical PnL distribution."""
    
    return -np.percentile(pnl, (1 - confidence) * 100)

rolling_pnl = compute_rolling_pnl(portfolio_returns, 1000000, 5)
historical_VaR = compute_historical_VaR(rolling_pnl, 0.99)

print(historical_VaR)




# Parametric VaR
def compute_parametric_VaR(
        log_returns, 
        weights, 
        value=1000000, 
        horizon=5, 
        confidence=0.99
        ):
    """Estimate variance-covariance VaR under normality.
    
    Daily portfolio volatility is scaled to the holding period using
    the square-root-of-time rule. Expected return is assumed to be zero.
    """
    
    cov_matrix = log_returns.cov()
    portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights) 
    
    para_var = (
          value
        * portfolio_std
        * norm.ppf(confidence)
        * np.sqrt(horizon)
          )
    
    return portfolio_std , para_var

portfolio_std, parametric_VaR = compute_parametric_VaR(log_returns, weights)

print(parametric_VaR)




# Monte Carlo VaR
def compute_monte_carlo_VaR(
        log_returns, 
        weights, 
        value=1000000, 
        horizon=5, 
        confidence=0.99, 
        simulations=10000
        ):
    """Monte Carlo VaR under multivariate normality."""
    
    cov_matrix = log_returns.cov()
    num_assets = len(weights)
    mu = np.zeros(num_assets) 

    # Simulate asset returns from a multivariate normal distribution
    simulated_returns = np.random.multivariate_normal(
        mean=mu,
        cov=cov_matrix,
        size=simulations
    )

    portfolio_sim_returns = simulated_returns @ weights
    portfolio_sim_returns *= np.sqrt(horizon)
    scenario_pnl = value * portfolio_sim_returns
    mc_var = -np.percentile(scenario_pnl, (1 - confidence) * 100)

    return scenario_pnl, mc_var

scenario_pnl, monte_carlo_VaR = compute_monte_carlo_VaR(log_returns, weights)

print(monte_carlo_VaR)





# VaR Summary
VaR_summary = pd.DataFrame({
    "VaR": [historical_VaR, parametric_VaR, monte_carlo_VaR]
}, index=["Historical", "Parametric", "Monte Carlo"])

print(VaR_summary)




# Monte Carlo convergence analysis
simulation_sizes = [500, 3000, 10000, 50000]
mc_var_estimates = []

for n in simulation_sizes:
    _, var_estimate = compute_monte_carlo_VaR(
        log_returns, weights, simulations=n
    )
    mc_var_estimates.append(var_estimate)

mc_convergence = pd.DataFrame({
    "Simulations": simulation_sizes,
    "Monte Carlo VaR": mc_var_estimates
})

print(mc_convergence)




# VaR distribution plots
def generate_parametric_pnl(std, value, horizon, simulations=10000):
    """Generate normal PnL scenarios for distribution visualization."""
    
    # Simulated PnL used only for visual comparison
    simulated_returns = np.random.normal(0, std * np.sqrt(horizon), simulations)  

    return simulated_returns * value

parametric_pnl = generate_parametric_pnl(portfolio_std, 1000000, 5)


def plot_VaR_distribution(data, var_value, title, xlim=None, ylim=None):
    """Plot a PnL distribution and its VaR threshold."""
    
    plt.figure()
    plt.hist(data, bins=100, density=True)
    plt.axvline(-var_value, linestyle='dashed', linewidth=1, label='VaR')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

tail_xlim = (-80000, 80000)

plot_VaR_distribution(rolling_pnl, historical_VaR, 'Historical VaR', tail_xlim)
plot_VaR_distribution(parametric_pnl, parametric_VaR, 'Parametric VaR', tail_xlim)
plot_VaR_distribution(scenario_pnl, monte_carlo_VaR, 'Monte Carlo VaR', tail_xlim)




# Expected Shortfall
value = 1000000
horizon = 5
confidence = 0.99

historical_ES = -rolling_pnl[rolling_pnl <= -historical_VaR].mean()

# Closed-form ES under zero-mean normally distributed returns.
z = norm.ppf(confidence)
pdf_z = norm.pdf(z)
parametric_ES = (
    value
    * portfolio_std
    * (pdf_z / (1 - confidence))
    * np.sqrt(horizon)
)

monte_carlo_ES = -scenario_pnl[scenario_pnl <= -monte_carlo_VaR].mean()


# ES Summary
ES_summary = pd.DataFrame({
    "ES": [historical_ES, parametric_ES, monte_carlo_ES]
}, index=["Historical", "Parametric", "Monte Carlo"])

print(ES_summary)




# Rolling VaR Backtesting
def compute_forward_pnl(returns, value, horizon):
    """Compute forward-looking PnL for VaR backtesting.
    
    Each value at time t aggregates portfolio returns from t through 
    t + horizon - 1, aligning the realized outcome with the VaR 
    forecast made at the same forecast origin.
    """

    # Future cumulative return over the holding period
    future_returns = returns.rolling(horizon).sum().shift(-horizon + 1)

    pnl = future_returns * value

    return pnl.dropna()

forward_pnl = compute_forward_pnl(portfolio_returns, 1000000, 5)




def rolling_historical_VaR(pnl_series, window=1000, confidence=0.99):
    """Estimate rolling Historical VaR from trailing PnL windows."""

    var_list = []
    index = []

    for i in range(window, len(pnl_series)):

        pnl_sample = pnl_series.iloc[i-window:i]
        var = compute_historical_VaR(pnl_sample, confidence)

        var_list.append(var)
        index.append(pnl_series.index[i])

    return pd.Series(var_list, index=index)

historical_var_series = rolling_historical_VaR(rolling_pnl)




def rolling_parametric_VaR(log_returns, weights, window=1000,
                           value=1000000, horizon=5, confidence=0.99):
    """Estimate rolling normal VaR from trailing return windows."""

    var_list = []
    index = []

    for i in range(window, len(log_returns)-horizon):

        sample_returns = log_returns.iloc[i-window:i]

        _, var = compute_parametric_VaR(
            sample_returns,
            weights,
            value=value,
            horizon=horizon,
            confidence=confidence
        )

        var_list.append(var)
        index.append(log_returns.index[i])

    return pd.Series(var_list, index=index)

parametric_var_series = rolling_parametric_VaR(log_returns, weights)




def rolling_mc_VaR(log_returns, weights, window=1000,
                   value=1000000, horizon=5, confidence=0.99):
    """Estimate rolling Monte Carlo VaR from trailing return windows."""

    var_list = []
    index = []

    for i in range(window, len(log_returns)-horizon):

        sample_returns = log_returns.iloc[i-window:i]

        _, var = compute_monte_carlo_VaR(
            sample_returns,
            weights,
            value=value,
            horizon=horizon,
            confidence=confidence
        )

        var_list.append(var)
        index.append(log_returns.index[i])

    return pd.Series(var_list, index=index)

mc_var_series = rolling_mc_VaR(log_returns, weights)




# Backtesting Sample Alignment
# Retain dates shared by every VaR forecast and the realized forward PnL.
common_index = (
    historical_var_series.index
    .intersection(parametric_var_series.index)
    .intersection(mc_var_series.index)
    .intersection(forward_pnl.index)
)

historical_var_series = historical_var_series.loc[common_index]
parametric_var_series = parametric_var_series.loc[common_index]
mc_var_series = mc_var_series.loc[common_index]
pnl_test = forward_pnl.loc[common_index]




# Kupiec Unconditional Coverage Test
def kupiec_test(var, pnl, confidence=0.99):
    """Test whether the observed VaR breach rate matches its expected rate.

    The null hypothesis is that the unconditional breach probability
    equals one minus the VaR confidence level. The input series must
    already be aligned to the same forecast dates.
    """
    
    violations = pnl < -var
    x = violations.sum()
    n = len(pnl)
    p = 1 - confidence
    
    # Clip the empirical breach rate to keep the log-likelihood finite.
    eps = 1e-10
    p_hat = x / n
    p_hat = max(min(p_hat, 1 - eps), eps)

    likelihood_ratio = -2 * (
        (n-x)*np.log(1-p) + x*np.log(p)
        - ((n-x)*np.log(1-p_hat) + x*np.log(p_hat))
    )
    
    p_value = chi2.sf(likelihood_ratio, df=1)

    return likelihood_ratio, x, p_value

his_kupiec_LR, his_kupiec_x, his_kupiec_p = kupiec_test(historical_var_series, pnl_test)
para_kupiec_LR, para_kupiec_x, para_kupiec_p = kupiec_test(parametric_var_series, pnl_test)
mc_kupiec_LR, mc_kupiec_x, mc_kupiec_p = kupiec_test(mc_var_series, pnl_test)

kupiec_test_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Statistic": [his_kupiec_LR, para_kupiec_LR, mc_kupiec_LR],
    "p-value": [his_kupiec_p, para_kupiec_p, mc_kupiec_p],
    "Violations (x)": [his_kupiec_x, para_kupiec_x, mc_kupiec_x],
})

print(kupiec_test_results)




# Basel-Style Traffic Light
def traffic_light_rolling(var, pnl, window=250):
    """Classify rolling windows by their number of VaR breaches.
    
    Windows with at most 4 breaches are classified as Green, those
    with 5–9 breaches as Yellow, and those with at least 10 as Red.

    This framework uses the thresholds as a Basel-style diagnostic
    for five-day VaR rather than as a formal regulatory backtest.
    """

    zones = []
    counts = []
    index_list = []

    for i in range(window, len(var)):
        var_window = var.iloc[i - window:i]
        pnl_window = pnl.iloc[i - window:i]

        violations = (pnl_window < -var_window).sum()

        if violations <= 4:
            zone = "Green"
        elif violations <= 9:
            zone = "Yellow"
        else:
            zone = "Red"

        zones.append(zone)
        counts.append(violations)
        index_list.append(var.index[i])

    return pd.DataFrame({
        "Violations": counts,
        "Zone": zones
    }, index=index_list)

traffic_hist = traffic_light_rolling(historical_var_series, pnl_test)
traffic_para = traffic_light_rolling(parametric_var_series, pnl_test)
traffic_mc = traffic_light_rolling(mc_var_series, pnl_test)

def summarize_traffic(df):
    """Count rolling windows assigned to each traffic-light zone."""
    
    return df["Zone"].value_counts().reindex(
        ["Green", "Yellow", "Red"],
        fill_value=0
    )

traffic_summary = pd.DataFrame({
    "Historical": summarize_traffic(traffic_hist),
    "Parametric": summarize_traffic(traffic_para),
    "Monte Carlo": summarize_traffic(traffic_mc)
})

traffic_summary_ratio = traffic_summary.div(traffic_summary.sum())

traffic_violations_avg = pd.DataFrame({
    "Historical": traffic_hist["Violations"].mean(),
    "Parametric": traffic_para["Violations"].mean(),
    "Monte Carlo": traffic_mc["Violations"].mean()
}, index=["Avg Violations"])

print(traffic_summary)
print(traffic_summary_ratio)
print(traffic_violations_avg)




# Non-Overlapping Backtesting Sample
def make_non_overlapping_sample(var, pnl, horizon=5, start=0):
    """Construct a non-overlapping sample of aligned VaR and PnL.

    Every horizon-th observation is retained to reduce the mechanical
    dependence caused by overlapping multi-day PnL windows. The start
    parameter determines the sampling offset.
    """

    paired = pd.DataFrame({
        "VaR": var,
        "PnL": pnl
    }).dropna()

    sampled = paired.iloc[start::horizon].copy()

    return sampled["VaR"], sampled["PnL"]


hist_var_nonoverlap, hist_pnl_nonoverlap = make_non_overlapping_sample(
    historical_var_series, pnl_test, horizon=5, start=0)

para_var_nonoverlap, para_pnl_nonoverlap = make_non_overlapping_sample(
    parametric_var_series, pnl_test, horizon=5, start=0)

mc_var_nonoverlap, mc_pnl_nonoverlap = make_non_overlapping_sample(
    mc_var_series, pnl_test, horizon=5, start=0)




# Kupiec Test on the Non-Overlapping Sample
his_kupiec_LR_NO, his_kupiec_x_NO, his_kupiec_p_NO = kupiec_test(
    hist_var_nonoverlap, hist_pnl_nonoverlap
    )

para_kupiec_LR_NO, para_kupiec_x_NO, para_kupiec_p_NO = kupiec_test(
    para_var_nonoverlap, para_pnl_nonoverlap
    )

mc_kupiec_LR_NO, mc_kupiec_x_NO, mc_kupiec_p_NO = kupiec_test(
    mc_var_nonoverlap, mc_pnl_nonoverlap
    )


kupiec_nonoverlap_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Statistic (Non-overlapping)": [his_kupiec_LR_NO, para_kupiec_LR_NO, mc_kupiec_LR_NO],
    "p-value": [his_kupiec_p_NO, para_kupiec_p_NO, mc_kupiec_p_NO,],
    "Violations (x)": [his_kupiec_x_NO, para_kupiec_x_NO, mc_kupiec_x_NO]
})

print(kupiec_nonoverlap_results)




# Christoffersen Independence Test
def christoffersen_independence_test(var, pnl):
    """Test whether VaR breaches are independent over time.

    The test compares a constant breach probability under the null
    hypothesis with a first-order Markov alternative in which the
    probability depends on the previous breach state.
    """
    
    # Encode breaches as a binary state sequence.
    violations = (pnl < -var).astype(int)

    # Count transitions between adjacent breach states.
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(violations)): 
        prev = violations.iloc[i - 1]
        curr = violations.iloc[i]

        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Clip probabilities to keep both log-likelihoods finite.
    eps = 1e-10
    pi01 = max(min(pi01, 1 - eps), eps)
    pi11 = max(min(pi11, 1 - eps), eps)
    pi = max(min(pi, 1 - eps), eps)

    # Compare the independent and first-order Markov log-likelihoods.
    LR_ind = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - (
            n00 * np.log(1 - pi01)
            + n01 * np.log(pi01)
            + n10 * np.log(1 - pi11)
            + n11 * np.log(pi11)
        )
    )

    p_value = chi2.sf(LR_ind, df=1)

    return LR_ind, (n00, n01, n10, n11), p_value

his_ind_lr, his_trans, his_ind_p = christoffersen_independence_test(
    hist_var_nonoverlap, hist_pnl_nonoverlap
    )

para_ind_lr, para_trans, para_ind_p = christoffersen_independence_test(
    para_var_nonoverlap, para_pnl_nonoverlap
    )

mc_ind_lr, mc_trans, mc_ind_p = christoffersen_independence_test(
    mc_var_nonoverlap, mc_pnl_nonoverlap
    )

independence_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Independence (Non-overlapping)": [his_ind_lr, para_ind_lr, mc_ind_lr],
    "p-value": [his_ind_p, para_ind_p, mc_ind_p,]
    })

print(independence_results)
print(his_trans)
print(para_trans)
print(mc_trans)




# Christoffersen Conditional Coverage Test
def conditional_coverage_test(lr_uc, lr_ind):
    """Combine coverage and independence into a joint LR test.

    Under the joint null hypothesis of correct unconditional coverage
    and independent breaches, the statistic follows an asymptotic
    chi-square distribution with two degrees of freedom.
    """

    lr_cc = lr_uc + lr_ind
    p_value = chi2.sf(lr_cc, df=2)

    return lr_cc, p_value

his_cc_lr, his_cc_p = conditional_coverage_test(his_kupiec_LR_NO, his_ind_lr)
para_cc_lr, para_cc_p = conditional_coverage_test(para_kupiec_LR_NO, para_ind_lr)
mc_cc_lr, mc_cc_p = conditional_coverage_test(mc_kupiec_LR_NO, mc_ind_lr)

cc_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR_CC (Non-overlapping)": [his_cc_lr, para_cc_lr, mc_cc_lr],
    "p-value": [his_cc_p, para_cc_p, mc_cc_p],
})

print(cc_results)




# EWMA VaR
def compute_ewma_volatility(returns, lam=0.94, init_window=60):
    """Estimate daily conditional volatility using EWMA.

    The variance recursion uses the previous variance estimate and
    squared return, with the initial variance estimated from the first
    init_window observations. The default decay factor of 0.94 follows
    the RiskMetrics convention for daily returns.
    """

    ewma_variance = np.full(len(returns), np.nan)

    ewma_variance[init_window - 1] = returns.iloc[:init_window].var()

    for t in range(init_window, len(returns)):
        
        ewma_variance[t] = (
            lam * ewma_variance[t - 1]           
            + (1 - lam) * returns.iloc[t - 1]**2 
        )
        
    return pd.Series(np.sqrt(ewma_variance), index=returns.index)


def compute_ewma_VaR_series(returns, value=1000000, horizon=5,
    confidence=0.99, lam=0.94, init_window=60):
    """Convert EWMA volatility into multi-period monetary VaR.

    The model assumes zero expected return and normal innovations.
    Daily conditional volatility is scaled to the holding period
    using the square-root-of-time rule.
    """

    ewma_volatility = compute_ewma_volatility(
        returns,
        lam=lam,
        init_window=init_window
    )

    z = norm.ppf(confidence)

    ewma_VaR_series = value * z * ewma_volatility * np.sqrt(horizon)
    
    return ewma_VaR_series, ewma_volatility


ewma_VaR_series, ewma_volatility = compute_ewma_VaR_series(
    portfolio_returns,
    value=1000000,
    horizon=5,
    confidence=0.99,
    lam=0.94,
    init_window=60
    )


# EWMA Backtesting
# Align forecasts with the common backtesting sample.
ewma_common_index = common_index.intersection(ewma_VaR_series.index)

ewma_VaR_series = ewma_VaR_series.loc[ewma_common_index]
ewma_pnl_test = forward_pnl.loc[ewma_common_index]

ewma_violations = ewma_pnl_test < -ewma_VaR_series

print("EWMA Violations:", ewma_violations.sum())
print("EWMA Violation Rate:", ewma_violations.mean())
print("EWMA Average 5-day VaR:", ewma_VaR_series.mean())


# Overlapping-sample diagnostics
ewma_kupiec_LR, ewma_kupiec_x, ewma_kupiec_p = kupiec_test(
    ewma_VaR_series, ewma_pnl_test
    )

traffic_ewma = traffic_light_rolling(ewma_VaR_series, ewma_pnl_test)

print("EWMA Kupiec LR:", ewma_kupiec_LR)
print("EWMA Avg Violations (250-day window):", traffic_ewma["Violations"].mean())


# Non-overlapping-sample diagnostics
ewma_VaR_nonoverlap, ewma_pnl_nonoverlap = make_non_overlapping_sample(
    ewma_VaR_series, ewma_pnl_test, horizon=5, start=0)

ewma_kupiec_LR_NO, _, ewma_kupiec_p_NO = kupiec_test(
    ewma_VaR_nonoverlap, ewma_pnl_nonoverlap
    )

ewma_ind_lr, _, ewma_ind_p = christoffersen_independence_test(
    ewma_VaR_nonoverlap, ewma_pnl_nonoverlap
    )

ewma_cc_lr, ewma_cc_p = conditional_coverage_test(
    ewma_kupiec_LR_NO, ewma_ind_lr
    )

print("EWMA Kupiec LR (Non-overlapping):", ewma_kupiec_LR_NO)
print("EWMA Conditional Coverage LR:", ewma_cc_lr)




# GARCH VaR
def rolling_garch_VaR(returns, window=1000, value=1000000, 
                      horizon=5, confidence=0.99, scale=100):
    """Estimate rolling multi-period VaR using a GARCH(1,1) model.

    At each forecast origin, the model is fitted using only the
    preceding estimation window. The holding-period variance is
    obtained by summing the model-implied daily variance forecasts.

    The model assumes a zero conditional mean and normally distributed
    innovations. Returns are scaled during estimation for numerical
    stability, and forecast variances are converted back to the
    original return scale.
    """

    var_list = []
    volatility_list = []
    index_list = []

    z = norm.ppf(confidence)

    for i in range(window, len(returns) - horizon):

        # Use only information available before the forecast origin.
        sample_returns = returns.iloc[i - window:i]
        scaled_sample = sample_returns * scale

        model = arch_model(scaled_sample, mean="Zero", vol="GARCH", 
                           p=1, q=1, dist="normal")

        result = model.fit(disp="off")

        # Forecast daily conditional variances over the holding period.
        variance_forecast = (result.forecast(horizon=horizon, reindex=False)
                             .variance.iloc[-1].to_numpy() / scale**2)

        # Under zero conditional mean, the variance of the cumulative
        # holding-period return is the sum of daily forecast variances.
        cumulative_variance = variance_forecast.sum()

        var_t = value * z * np.sqrt(cumulative_variance)

        var_list.append(var_t)
        volatility_list.append(np.sqrt(variance_forecast[0]))
        index_list.append(returns.index[i])

    garch_var_series = pd.Series(var_list, index=index_list)

    garch_volatility = pd.Series(volatility_list, index=index_list)

    return garch_var_series, garch_volatility


# Estimate portfolio-level rolling GARCH VaR for comparison
# with the other portfolio VaR forecasts.
garch_var_series, garch_vol = rolling_garch_VaR(portfolio_returns,
                                                window=1000,
                                                value=1000000,
                                                horizon=5,
                                                confidence=0.99
                                                )


# GARCH Backtesting
# Align forecasts with the common backtesting sample.
garch_common_index = common_index.intersection(garch_var_series.index)

garch_var_series = garch_var_series.loc[garch_common_index]
garch_pnl_test = forward_pnl.loc[garch_common_index]

garch_violations = garch_pnl_test < -garch_var_series

print("GARCH Violations:", garch_violations.sum())
print("GARCH Violation Rate:", garch_violations.mean())
print("GARCH Average VaR:", garch_var_series.mean())


# Overlapping-sample diagnostics
garch_kupiec_LR, _, garch_kupiec_p = kupiec_test(
    garch_var_series, garch_pnl_test
    )

traffic_garch = traffic_light_rolling(garch_var_series, garch_pnl_test)

print("GARCH Kupiec LR:", garch_kupiec_LR)
print("GARCH Avg Violations (250-day window):", traffic_garch["Violations"].mean())


# Non-overlapping-sample diagnostics
garch_var_nonoverlap, garch_pnl_nonoverlap = make_non_overlapping_sample(
    garch_var_series,
    garch_pnl_test,
    horizon=5,
    start=0)

garch_kupiec_LR_NO, _, garch_kupiec_p_NO = kupiec_test(
    garch_var_nonoverlap, garch_pnl_nonoverlap
    )

garch_ind_lr, _, garch_ind_p = christoffersen_independence_test(
    garch_var_nonoverlap, garch_pnl_nonoverlap
    )

garch_cc_lr, garch_cc_p = conditional_coverage_test(
    garch_kupiec_LR_NO, garch_ind_lr
    )

print("GARCH Kupiec LR (Non-overlapping):", garch_kupiec_LR_NO)
print("GARCH Conditional Coverage LR:", garch_cc_lr)




# Filtered Historical Simulation VaR
def compute_standardized_residuals(returns, vol):
    """Standardize returns using aligned conditional volatility."""

    aligned = pd.DataFrame({
        "returns": returns,
        "vol": vol
    }).dropna()

    return aligned["returns"] / aligned["vol"]


def rolling_fhs_var(returns, window=1000, value=1000000, horizon=5,
                    confidence=0.99, simulations=2000, scale=100):
    """Estimate rolling VaR using filtered historical simulation.
    
    Each estimation window fits a zero-mean GARCH(1,1) model and
    constructs an empirical distribution of standardized residuals.
    Future shocks are resampled from that distribution and propagated
    through the fitted conditional-variance recursion.

    The normal distribution specified during GARCH estimation applies
    only to volatility-filter estimation. Simulated FHS innovations
    are drawn from the empirical standardized residuals.
    """
    
    var_list = []
    index_list = []

    for i in range(window, len(returns) - horizon):
        sample_returns = returns.iloc[i - window:i]

        scaled_sample = sample_returns * scale
        model = arch_model(scaled_sample, mean='Zero', vol='GARCH',
                           p=1, q=1, dist='normal')
        
        result = model.fit(disp='off')

        sigma_hist = result.conditional_volatility / scale
        sigma_hist = pd.Series(sigma_hist, index=sample_returns.index)

        z_hist = compute_standardized_residuals(sample_returns, sigma_hist
                                                ).dropna().values

        # Omega is a variance parameter and must be rescaled by scale squared.
        omega = result.params["omega"] / (scale**2) 
        alpha = result.params["alpha[1]"]
        beta = result.params["beta[1]"]

        last_sigma2 = sigma_hist.iloc[-1] ** 2
        last_return2 = sample_returns.iloc[-1] ** 2

        path_pnl = np.zeros(simulations)

        for s in range(simulations):
            sigma2_t = omega + alpha * last_return2 + beta * last_sigma2
            pnl_path = 0.0

            for _ in range(horizon):
                # Draw a shock from the empirical residual distribution.
                z_draw = np.random.choice(z_hist) 
                r_draw = np.sqrt(sigma2_t) * z_draw
                pnl_path += value * r_draw

                # Propagate conditional variance along the simulated path.
                sigma2_t = omega + alpha * (r_draw**2) + beta * sigma2_t

            path_pnl[s] = pnl_path

        var_t = -np.percentile(path_pnl, (1 - confidence) * 100)

        var_list.append(var_t)
        index_list.append(returns.index[i])

    return pd.Series(var_list, index=index_list)

fhs_var_series = rolling_fhs_var(
    portfolio_returns, 
    window=1000,
    value=1000000,
    horizon=5,
    confidence=0.99,
    simulations=2000
    )


# FHS Backtesting
# Align forecasts with the common backtesting sample.
fhs_common_index = common_index.intersection(fhs_var_series.index)

fhs_var_series = fhs_var_series.loc[fhs_common_index]
fhs_pnl_test = forward_pnl.loc[fhs_common_index]

fhs_violations = fhs_pnl_test < -fhs_var_series

print("FHS Violations:", fhs_violations.sum())
print("FHS Violation Rate:", fhs_violations.mean())
print("FHS Average VaR:", fhs_var_series.mean())


# Overlapping-sample diagnostics
fhs_kupiec_LR, fhs_kupiec_x, fhs_kupiec_p = kupiec_test(
    fhs_var_series, fhs_pnl_test
    )

traffic_fhs = traffic_light_rolling(fhs_var_series, fhs_pnl_test)

print("FHS Kupiec LR:", fhs_kupiec_LR)
print("FHS Avg Violations (250-day window):", traffic_fhs["Violations"].mean())


# Non-overlapping-sample diagnostics
fhs_var_nonoverlap, fhs_pnl_nonoverlap = make_non_overlapping_sample(
    fhs_var_series,
    fhs_pnl_test,
    horizon=5,
    start=0
    )

fhs_kupiec_LR_NO, fhs_kupiec_x_NO, fhs_kupiec_p_NO = kupiec_test(
    fhs_var_nonoverlap, fhs_pnl_nonoverlap
    )

fhs_ind_lr, fhs_trans, fhs_ind_p = christoffersen_independence_test(
    fhs_var_nonoverlap, fhs_pnl_nonoverlap
    )

fhs_cc_lr, fhs_cc_p = conditional_coverage_test(
    fhs_kupiec_LR_NO, fhs_ind_lr
    )

print("FHS Kupiec LR (Non-overlapping):", fhs_kupiec_LR_NO)
print("FHS Conditional Coverage LR:", fhs_cc_lr)




# Save Reference Run Results
static_risk_summary = VaR_summary.join(ES_summary)
static_risk_summary.index.name = "Model"

static_risk_summary.to_csv(
    REFERENCE_RESULTS_DIR / "static_var_es_summary.csv"
)

mc_convergence.to_csv(
    REFERENCE_RESULTS_DIR / "monte_carlo_convergence.csv",
    index=False,
)




# Collect model outputs in a common schema for consolidated reporting.
reference_models = [
    {
        "model": "Historical",
        "var": historical_var_series,
        "pnl": pnl_test,
        "var_nonoverlap": hist_var_nonoverlap,
        "pnl_nonoverlap": hist_pnl_nonoverlap,
        "kupiec_lr": his_kupiec_LR,
        "kupiec_lr_nonoverlap": his_kupiec_LR_NO,
        "independence_lr": his_ind_lr,
        "conditional_coverage_lr": his_cc_lr,
        "traffic": traffic_hist,
        "kupiec_p": his_kupiec_p,
        "kupiec_p_nonoverlap": his_kupiec_p_NO,
        "independence_p": his_ind_p,
        "conditional_coverage_p": his_cc_p,
    },
    {
        "model": "Parametric",
        "var": parametric_var_series,
        "pnl": pnl_test,
        "var_nonoverlap": para_var_nonoverlap,
        "pnl_nonoverlap": para_pnl_nonoverlap,
        "kupiec_lr": para_kupiec_LR,
        "kupiec_lr_nonoverlap": para_kupiec_LR_NO,
        "independence_lr": para_ind_lr,
        "conditional_coverage_lr": para_cc_lr,
        "traffic": traffic_para,
        "kupiec_p": para_kupiec_p,
        "kupiec_p_nonoverlap": para_kupiec_p_NO,
        "independence_p": para_ind_p,
        "conditional_coverage_p": para_cc_p,
    },
    {
        "model": "Monte Carlo",
        "var": mc_var_series,
        "pnl": pnl_test,
        "var_nonoverlap": mc_var_nonoverlap,
        "pnl_nonoverlap": mc_pnl_nonoverlap,
        "kupiec_lr": mc_kupiec_LR,
        "kupiec_lr_nonoverlap": mc_kupiec_LR_NO,
        "independence_lr": mc_ind_lr,
        "conditional_coverage_lr": mc_cc_lr,
        "traffic": traffic_mc,
        "kupiec_p": mc_kupiec_p,
        "kupiec_p_nonoverlap": mc_kupiec_p_NO,
        "independence_p": mc_ind_p,
        "conditional_coverage_p": mc_cc_p,
    },
    {
        "model": "EWMA",
        "var": ewma_VaR_series,
        "pnl": ewma_pnl_test,
        "var_nonoverlap": ewma_VaR_nonoverlap,
        "pnl_nonoverlap": ewma_pnl_nonoverlap,
        "kupiec_lr": ewma_kupiec_LR,
        "kupiec_lr_nonoverlap": ewma_kupiec_LR_NO,
        "independence_lr": ewma_ind_lr,
        "conditional_coverage_lr": ewma_cc_lr,
        "traffic": traffic_ewma,
        "kupiec_p": ewma_kupiec_p,
        "kupiec_p_nonoverlap": ewma_kupiec_p_NO,
        "independence_p": ewma_ind_p,
        "conditional_coverage_p": ewma_cc_p,
    },
    {
        "model": "GARCH",
        "var": garch_var_series,
        "pnl": garch_pnl_test,
        "var_nonoverlap": garch_var_nonoverlap,
        "pnl_nonoverlap": garch_pnl_nonoverlap,
        "kupiec_lr": garch_kupiec_LR,
        "kupiec_lr_nonoverlap": garch_kupiec_LR_NO,
        "independence_lr": garch_ind_lr,
        "conditional_coverage_lr": garch_cc_lr,
        "traffic": traffic_garch,
        "kupiec_p": garch_kupiec_p,
        "kupiec_p_nonoverlap": garch_kupiec_p_NO,
        "independence_p": garch_ind_p,
        "conditional_coverage_p": garch_cc_p,
    },
    {
        "model": "FHS",
        "var": fhs_var_series,
        "pnl": fhs_pnl_test,
        "var_nonoverlap": fhs_var_nonoverlap,
        "pnl_nonoverlap": fhs_pnl_nonoverlap,
        "kupiec_lr": fhs_kupiec_LR,
        "kupiec_lr_nonoverlap": fhs_kupiec_LR_NO,
        "independence_lr": fhs_ind_lr,
        "conditional_coverage_lr": fhs_cc_lr,
        "traffic": traffic_fhs,
        "kupiec_p": fhs_kupiec_p,
        "kupiec_p_nonoverlap": fhs_kupiec_p_NO,
        "independence_p": fhs_ind_p,
        "conditional_coverage_p": fhs_cc_p,
    },
]


backtesting_rows = []

for result in reference_models:
    violations = result["pnl"] < -result["var"]

    nonoverlap_violations = (
        result["pnl_nonoverlap"]
        < -result["var_nonoverlap"]
    )

    traffic_zone_ratio = (
        result["traffic"]["Zone"]
        .value_counts(normalize=True)
        .reindex(
            ["Green", "Yellow", "Red"],
            fill_value=0.0,
        )
    )

    backtesting_rows.append({
        "Model": result["model"],
        "Observations": len(result["var"]),
        "Average VaR": result["var"].mean(),
        "Violations": int(violations.sum()),
        "Violation Rate": violations.mean(),
        "Kupiec LR": result["kupiec_lr"],
        "Kupiec p-value": result["kupiec_p"],
        "Non-overlapping Observations": len(
            result["var_nonoverlap"]
        ),
        "Non-overlapping Violations": int(
            nonoverlap_violations.sum()
        ),
        "Non-overlapping Violation Rate": (
            nonoverlap_violations.mean()
        ),
        "Kupiec LR (Non-overlapping)": (
            result["kupiec_lr_nonoverlap"]
        ),
        "Kupiec p-value (Non-overlapping)": (
            result["kupiec_p_nonoverlap"]
        ),
        "Independence LR (Non-overlapping)": (
            result["independence_lr"]
        ),
        "Independence p-value (Non-overlapping)": (
            result["independence_p"]
        ),
        "Conditional Coverage LR (Non-overlapping)": (
            result["conditional_coverage_lr"]
        ),
        "Conditional Coverage p-value (Non-overlapping)": (
            result["conditional_coverage_p"]
        ),
        "Traffic Light Avg Violations": (
            result["traffic"]["Violations"].mean()
        ),
        "Traffic Green Ratio": traffic_zone_ratio["Green"],
        "Traffic Yellow Ratio": traffic_zone_ratio["Yellow"],
        "Traffic Red Ratio": traffic_zone_ratio["Red"],
    })


backtesting_summary = pd.DataFrame(backtesting_rows)

backtesting_summary.to_csv(
    REFERENCE_RESULTS_DIR / "backtesting_summary.csv",
    index=False,
)


rolling_forecasts = pd.concat(
    {
        "Realized 5-day PnL": forward_pnl,
        "Historical VaR": historical_var_series,
        "Parametric VaR": parametric_var_series,
        "Monte Carlo VaR": mc_var_series,
        "EWMA VaR": ewma_VaR_series,
        "GARCH VaR": garch_var_series,
        "FHS VaR": fhs_var_series,
    },
    axis=1,
)

rolling_forecasts.index.name = "Date"

rolling_forecasts.to_csv(
    REFERENCE_RESULTS_DIR / "rolling_var_forecasts.csv"
)


print("\nReference Run results saved:")
print(static_risk_summary.round(2))
print(backtesting_summary.round(6))
print(f"Output directory: {REFERENCE_RESULTS_DIR}")


