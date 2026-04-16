import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from arch import arch_model

years = 20
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days = 365*years)

tickers = [
    '^KS11',      # KOSPI Composite Index
    '^KQ11',      # Kosdaq Composite Index
    'IEF',        # iShares 7-10 Year Treasury Bond ETF
    '^GSPC'       # S&P 500
]


def fetch_prices(tickers, start_date, end_date):
    
    df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)

        """
        For Indices, 'Close' is used as they don't have dividends or splits.
        For ETFs, 'Adj Close' is used to account for corporate actions.
        """

        df[ticker] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    return df.dropna()

prices = fetch_prices(tickers, start_date, end_date)




"""
[Modeling Assumptions]

1. Log returns are used because they are additive over time,
   which is convenient for multi-period risk measurement.

2. The portfolio is assumed to be equal-weighted across the selected assets.

3. VaR and ES are estimated over a 5-day holding period at the 99% confidence level.

4. For short-horizon parametric and Monte Carlo VaR, mean returns are assumed to be zero, 
   which is standard practice for short-horizon VaR.

5. Portfolio value is assumed to be denominated in USD, and FX risk is excluded.
   (Since the portfolio mixes Korean and US assets, currency risk would be
   material in practice. However, for analytical clarity, this framework
   focuses on market risk only.)
"""

weights = np.array([1/len(tickers)]*len(tickers))

def compute_log_returns(price_df):
    
    returns = np.log(price_df / price_df.shift(1)) 
    
    return returns.dropna()

log_returns = compute_log_returns(prices)

def compute_portfolio_returns(log_returns, weights):
    
    return (log_returns * weights).sum(axis=1)

portfolio_returns = compute_portfolio_returns(log_returns, weights)




#-----Historical VaR Method-----
def compute_rolling_pnl(returns, value, horizon):
    rolling = returns.rolling(horizon).sum().dropna()
    return rolling * value

def compute_historical_VaR(pnl, confidence):
    return -np.percentile(pnl, (1 - confidence) * 100)

rolling_pnl = compute_rolling_pnl(portfolio_returns, 1000000, 5)
historical_VaR = compute_historical_VaR(rolling_pnl, 0.99)

print(historical_VaR)




#-----VaR Parametric Method-----
def compute_parametric_VaR(log_returns, weights, value=1000000, horizon=5, confidence=0.99):
    cov_matrix = log_returns.cov()     # daily covariance
    portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights) # T means Transpose.
    
    para_var = (
          value
        * portfolio_std
        * norm.ppf(confidence) # PPF (percent point function) is the inverse of the CDF.
        * np.sqrt(horizon)
          )
    
    return portfolio_std , para_var

portfolio_std, parametric_VaR = compute_parametric_VaR(log_returns, weights)

print(parametric_VaR)




#-----VaR Monte Carlo Method-----
def compute_monte_carlo_VaR(log_returns, weights, value=1000000, horizon=5, confidence=0.99, simulations=10000):
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
    # Scale returns to T-day horizon (variance scales with time -> std scales with sqrt(T))
    # *= is the Compound Assignment Operator that means a = a * b
    portfolio_sim_returns *= np.sqrt(horizon)
    scenario_pnl = value * portfolio_sim_returns
    mc_var = -np.percentile(scenario_pnl, (1 - confidence) * 100)

    return scenario_pnl, mc_var

scenario_pnl, monte_carlo_VaR = compute_monte_carlo_VaR(log_returns, weights)

print(monte_carlo_VaR)

"""
Monte Carlo VaR is often close to Parametric VaR when returns are
simulated under multivariate normality.

Because both methods rely on the same volatility-covariance structure,
their VaR estimates are typically similar under this assumption.

The gap widens when Monte Carlo simulation incorporates features such as:

    - non-normal return distributions
    - stochastic volatility
    - fat tails or skewness
"""




#-----VaR Summary-----
VaR_summary = pd.DataFrame({
    "VaR": [historical_VaR, parametric_VaR, monte_carlo_VaR]
}, index=["Historical", "Parametric", "Monte Carlo"])

print(VaR_summary)




#-----Simulation Convergence Insight-----
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

"""
Monte Carlo VaR estimates become more stable as the number of simulations increases.

In this setup, the difference between 10,000 and 50,000 simulations is small,
suggesting that 10,000 simulations provide a reasonable balance between
computational efficiency and estimation stability.

Smaller simulation sizes may produce noisier tail estimates, while larger
simulation sizes generally improve the reliability of risk measurement.
"""




#-----VaR Distribution Plots-----
def generate_parametric_pnl(std, value, horizon, simulations=10000):
    # Simulated PnL used only for visual comparison
    simulated_returns = np.random.normal(0, std * np.sqrt(horizon), simulations)  

    return simulated_returns * value

parametric_pnl = generate_parametric_pnl(portfolio_std, 1000000, 5)


def plot_VaR_distribution(data, var_value, title, xlim=None, ylim=None):
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

"""
The parametric and Monte Carlo distributions appear more concentrated than the
historical distribution. This reflects the normality assumption imposed in the
simulated frameworks, which tends to smooth extreme outcomes and underrepresent
the heavier tails present in realized market data.

As a result, normal-based models may underestimate tail risk relative to the
historical distribution.
"""




#-----Expected Shortfall-----
"""
Expected Shortfall (ES) measures the average loss conditional on losses
exceeding the VaR threshold.

    Definition:

        ES_α = -E[X | X ≤ -VaR_α]

For a normal distribution:

    ES = V * σ * φ(z_α) / (1 - α) * sqrt(h)

where:
- φ(z_α): standard normal PDF evaluated at z_α
- z_α: standard normal quantile at confidence level α

Key intuition:

- VaR identifies the minimum loss level within the tail at a given confidence level.
- ES measures the average loss once that tail threshold has been breached.
- Therefore, ES captures the severity of tail losses more directly than VaR.

Why ES is greater than VaR:

- VaR is a cutoff point for tail losses.
- ES is the average of losses beyond that cutoff.
- Since ES averages only losses that are at least as extreme as VaR,
  ES must be greater than or equal to VaR.

For this reason, ES is generally viewed as a more informative measure of
tail risk than VaR.
"""

value = 1000000
horizon = 5
confidence = 0.99

historical_ES = -rolling_pnl[rolling_pnl <= -historical_VaR].mean()

z = norm.ppf(confidence)
pdf_z = norm.pdf(z)
parametric_ES = (
    value
    * portfolio_std
    * (pdf_z / (1 - confidence))  # Tail conditional expectation term
    * np.sqrt(horizon)
)

monte_carlo_ES = -scenario_pnl[scenario_pnl <= -monte_carlo_VaR].mean()


#-----ES Summary-----
ES_summary = pd.DataFrame({
    "ES": [historical_ES, parametric_ES, monte_carlo_ES]
}, index=["Historical", "Parametric", "Monte Carlo"])

print(ES_summary)




#-----Backtesting Methodologies-----
"""
Backtesting methodologies for Value-at-Risk (VaR) models can be grouped into
four broad categories:

1. Coverage Tests
    Coverage tests evaluate whether the observed frequency of VaR exceedances
    matches the expected violation rate implied by the confidence level.

    Example:
    - Kupiec Unconditional Coverage Test

    Limitation:
    - Does not detect clustering of violations


2. Independence Tests
    Independence tests evaluate whether VaR violations occur independently
    over time.

    Example:
    - Christoffersen Independence Test

    Role:
    - Detects whether violations are randomly distributed or clustered


3. Conditional Coverage Tests
    Conditional coverage tests jointly assess both coverage accuracy and
    independence of violations.

    Example:
    - Christoffersen Conditional Coverage Test


4. Distribution Tests (not implemented here)
    Distribution tests evaluate whether the full forecast loss distribution
    matches realized losses.

    Examples:
    - Berkowitz test
    - Kolmogorov-Smirnov test
    - Anderson-Darling test

    Note:
    - These tests require full distributional forecasts rather than only
      VaR quantiles


Regulatory Perspective

    In this framework, the main focus is on exceedance-based backtesting.

    The most relevant tests are:
    - Coverage
    - Independence
    - Conditional Coverage

These tests form the basis for regulatory interpretations such as the Basel traffic light approach.
"""




"""
VaR Backtesting Core Principle

Backtesting compares:

    VaR(t)  vs  PnL(t)

where:

    VaR(t) = risk forecast based on information available at time t
    PnL(t) = realized PnL over the next holding period

Therefore:

    - rolling (backward-looking) PnL is used to estimate historical rolling VaR
    - forward PnL is used for backtesting because it represents the realized
      outcome after the VaR forecast is made

The forward PnL series is also used for index alignment so that each VaR forecast
is matched with the realized PnL over the same forecast origin.
"""

def compute_forward_pnl(returns, value, horizon):

    # Future cumulative return over the holding period
    future_returns = returns.rolling(horizon).sum().shift(-horizon + 1)

    pnl = future_returns * value

    return pnl.dropna()

# Each timestamp is aligned to the future realized PnL used for backtesting.
forward_pnl = compute_forward_pnl(portfolio_returns, 1000000, 5)




def rolling_historical_VaR(pnl_series, window=1000, confidence=0.99):

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

    var_list = []
    index = []

    for i in range(window, len(log_returns)-horizon):

        sample_returns = log_returns.iloc[i-window:i]

        # Portfolio standard deviation is not needed here
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

    var_list = []
    index = []

    for i in range(window, len(log_returns)-horizon):

        sample_returns = log_returns.iloc[i-window:i]

        # Simulated scenario PnL is not needed here
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




"""
All VaR series and the forward PnL series must be aligned to a common index.

This ensures that:

    VaR(t) is compared with PnL(t) at the same forecast origin

Because rolling windows and horizon shifts generate slightly different index
ranges across series, only the overlapping timestamps are retained.
"""

# Keep only timestamps where all VaR series and forward PnL are available
common_index = (
    historical_var_series.index
    .intersection(parametric_var_series.index)
    .intersection(mc_var_series.index)
    .intersection(forward_pnl.index)
)

# Align all series for 1:1 backtesting comparison
historical_var_series = historical_var_series.loc[common_index]
parametric_var_series = parametric_var_series.loc[common_index]
mc_var_series = mc_var_series.loc[common_index]
pnl_test = forward_pnl.loc[common_index]




#-----Important Notes-----
"""
The framework uses a 5-day VaR horizon to reflect holding-period risk.
However, 5-day forward PnL constructed with overlapping windows may induce
mechanical serial dependence in VaR violations.

To address this, different backtesting metrics are evaluated on different samples:

    - Kupiec and traffic-light results are evaluated on the full overlapping sample,
      since they primarily focus on violation frequency.

    - Independence tests are evaluated on a non-overlapping 5-day sample to reduce
      overlap-induced dependence.

    - The Kupiec test is also recomputed on the non-overlapping sample so that
      conditional coverage (CC) results are based on the same dataset.

This separation helps distinguish frequency effects from dependence effects in
backtesting results.
"""




#-----Kupiec Unconditional Coverage Test-----
"""
Proportion of failure Likelihood Ratio (LR) Test

The POF Likelihood Ratio (LR) test statistic is defined as:

    LR = -2 * ln[ ( (1-p)^(n-x) * p^x ) / ( (1-x/n)^(n-x) * (x/n)^x ) ]

where:
- n: total number of observations
- x: number of violations (pnl < -VaR)
- p: expected failure rate (1 - confidence level)

Decision Rule: 
    Reject H0 if LR > 6.63 (at 99% confidence level, df=1)
"""
   
def kupiec_test(var, pnl, confidence=0.99):
    violations = pnl < -var
    x = violations.sum()
    n = len(pnl)
    p = 1 - confidence
    
    # Prevent log(0) by bounding the observed violation probability (p_hat)
    eps = 1e-10
    p_hat = x / n
    p_hat = max(min(p_hat, 1 - eps), eps)

    likelihood_ratio = -2 * (
        (n-x)*np.log(1-p) + x*np.log(p)
        - ((n-x)*np.log(1-p_hat) + x*np.log(p_hat))
    )
    return likelihood_ratio, x

his_kupiec_LR, his_kupiec_x = kupiec_test(historical_var_series, pnl_test)
para_kupiec_LR, para_kupiec_x = kupiec_test(parametric_var_series, pnl_test)
mc_kupiec_LR, mc_kupiec_x = kupiec_test(mc_var_series, pnl_test)

kupiec_test_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Statistic": [his_kupiec_LR, para_kupiec_LR, mc_kupiec_LR],
    "Violations (x)": [his_kupiec_x, para_kupiec_x, mc_kupiec_x],
})

print(kupiec_test_results)

"""
Backtesting insight:
    
Expected violations 3759 × 0.01 ≈ 37.6 (at 99% confidence level)
    
Results:
    
    Historical VaR: 40 violations, LR ≈ 0.153
        
        -> Slightly above the theoretical expectation, but still close enough
           to indicate broadly acceptable unconditional coverage.
    
    Parametric / Monte Carlo VaR: 85 violations, LR ≈ 44.49
        
        -> Far more violations than expected, indicating substantial
           underestimation of tail risk.

Interpretation: 
    
    The historical VaR model is well-calibrated, as its violation frequency
    fairly matches the theoretical expectation.
    
    In contrast, both parametric and Monte Carlo VaR models exhibit excessive
    violations, indicating systematic underestimation of tail risk.
    
    This is likely related to distributional assumptions such as normality,
    which fail to capture fat tails and extreme market movements.
"""




#-----Traffic Light-----
"""
Basel Traffic Light Approach

The Basel traffic light framework classifies VaR model performance based on
the number of exceedances observed within a fixed rolling backtesting window
(typically 250 observations).

Classification rule:

    Green  : ≤ 4 violations
    Yellow : 5–9 violations
    Red    : ≥ 10 violations

Requirement:
    PnL must be forward-looking and properly aligned with VaR forecasts.

Violation condition:
    PnL(t) < -VaR(t)

Unlike statistical tests such as the Kupiec test, the traffic light approach
is rule-based rather than hypothesis-based.

Both methods evaluate exceedance frequency, but their roles differ:
    - Kupiec test provides statistical evidence
    - Traffic light provides a regulatory-style classification
"""

def traffic_light_rolling(var, pnl, window=250):

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

"""
Traffic Light Results Interpretation:
    
                Historical  Parametric  Monte Carlo
Avg Violations    2.710744    5.838131     5.980621

Historical VaR spends most of the time in the green zone, with an average
violation count close to the benchmark implied by a 99% VaR model over a
250-day window.

By contrast, Parametric and Monte Carlo VaR spend a larger share of time in
the yellow and red zones, with higher average violation counts.

Overall, the traffic light results indicate that Historical VaR is more
robust in this framework, while Parametric and Monte Carlo VaR tend to
underestimate tail risk under static normality-based assumptions.
"""




#-----Non-overlapping Sample Construction-----
"""
Because the 5-day forward PnL is constructed using overlapping windows,
consecutive violations may exhibit mechanical serial dependence.

To reduce overlap-induced dependence, Christoffersen independence and
conditional coverage tests are additionally evaluated on a non-overlapping
5-day sample.

The starting offset is set to 0 for simplicity. Alternative offsets may
produce slightly different results.
"""

def make_non_overlapping_sample(var, pnl, horizon=5, start=0):
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




#-----Kupiec Test on Non-overlapping Sample-----
his_kupiec_LR_NO, his_kupiec_x_NO = kupiec_test(hist_var_nonoverlap, hist_pnl_nonoverlap)
para_kupiec_LR_NO, para_kupiec_x_NO = kupiec_test(para_var_nonoverlap, para_pnl_nonoverlap)
mc_kupiec_LR_NO, mc_kupiec_x_NO = kupiec_test(mc_var_nonoverlap, mc_pnl_nonoverlap)

kupiec_nonoverlap_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Statistic (Non-overlapping)": [his_kupiec_LR_NO, para_kupiec_LR_NO, mc_kupiec_LR_NO],
    "Violations (x)": [his_kupiec_x_NO, para_kupiec_x_NO, mc_kupiec_x_NO]
})

print(kupiec_nonoverlap_results)

"""
Kupiec Test Comparison:

The non-overlapping sample is used to evaluate unconditional coverage on a
dataset that is more consistent with the independence and conditional coverage
tests.

Compared with the overlapping sample, the non-overlapping specification reduces
the mechanical dependence induced by overlapping 5-day forward PnL construction.

Interpretation:

    - Historical VaR remains the best-calibrated model under both specifications.

    - Parametric and Monte Carlo VaR continue to show weaker unconditional
      coverage, indicating that tail risk is still underestimated.

    - The comparison between overlapping and non-overlapping results helps
      distinguish pure coverage performance from overlap-related effects in
      backtesting.
"""




#-----Christoffersen Independence Likelihood Ratio (LR) Test-----
"""
The independence test evaluates whether VaR violations occur independently
over time by comparing the observed transition structure of violations with
the structure implied under independence.

Transition probabilities:

    π01 = n01 / (n00 + n01)
    π11 = n11 / (n10 + n11)
    π   = (n01 + n11) / (n00 + n01 + n10 + n11) 

Transition counts:

    n00 : no violation → no violation
    n01 : no violation → violation
    n10 : violation → no violation
    n11 : violation → violation (key count for detecting clustering)

Hypotheses:

    H0: π01 = π11 = π   (violations are independent over time)
    H1: π01 ≠ π11       (violations exhibit dependence)

Under H0:
    LR_ind is asymptotically distributed as χ²(1)

Decision rule:
    Reject H0 if LR_ind > 6.63 (critical value at the 1% significance level, df = 1)

To reduce overlap-induced dependence, this test is evaluated on the
non-overlapping 5-day sample.
"""

def christoffersen_independence_test(var, pnl):
    # Convert Boolean violations to a 0/1 indicator series. violation -> 1, non-violation -> 0
    violations = (pnl < -var).astype(int)

    # Transition counts
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(violations)):  # The first observation has no previous state
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

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Prevent log(0)
    eps = 1e-10
    pi01 = max(min(pi01, 1 - eps), eps)
    pi11 = max(min(pi11, 1 - eps), eps)
    pi = max(min(pi, 1 - eps), eps)

    # Likelihood Ratio test statistic
    # LR_ind = -2 * (log L_H0 - log L_H1)
    LR_ind = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - (
            n00 * np.log(1 - pi01)
            + n01 * np.log(pi01)
            + n10 * np.log(1 - pi11)
            + n11 * np.log(pi11)
        )
    )

    return LR_ind, (n00, n01, n10, n11)

his_ind_lr, his_trans = christoffersen_independence_test(hist_var_nonoverlap, hist_pnl_nonoverlap)
para_ind_lr, para_trans = christoffersen_independence_test(para_var_nonoverlap, para_pnl_nonoverlap)
mc_ind_lr, mc_trans = christoffersen_independence_test(mc_var_nonoverlap, mc_pnl_nonoverlap)

independence_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Independence (Non-overlapping)": [his_ind_lr, para_ind_lr, mc_ind_lr],})

print(independence_results)
print(his_trans)
print(para_trans)
print(mc_trans)

"""
Backtesting insight:

Independence is evaluated on the non-overlapping sample to reduce the
mechanical serial dependence induced by overlapping 5-day forward PnL construction.

In this framework, the independence statistics remain below the 99% rejection
threshold, suggesting that strong residual dependence is not evident once
overlap effects are reduced.

The transition counts provide additional context by showing how often
violations persist from one observation to the next.
"""




#-----Christoffersen Conditional Coverage Test (Non-overlapping)-----
def conditional_coverage_test(lr_uc, lr_ind):
    return lr_uc + lr_ind

his_cc_lr = conditional_coverage_test(his_kupiec_LR_NO, his_ind_lr)
para_cc_lr = conditional_coverage_test(para_kupiec_LR_NO, para_ind_lr)
mc_cc_lr = conditional_coverage_test(mc_kupiec_LR_NO, mc_ind_lr)

cc_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR_CC (Non-overlapping)": [
        his_cc_lr,
        para_cc_lr,
        mc_cc_lr]})

print(cc_results)

"""
Backtesting insight:

The conditional coverage test combines unconditional coverage and independence into a single diagnostic.

Because it is evaluated on the non-overlapping sample, the results are less affected 
by the mechanical serial dependence induced by overlapping forward PnL construction.

Interpretation:

    - Historical VaR remains the most stable model under conditional coverage.

    - Parametric and Monte Carlo VaR may continue to appear weaker because
      deficiencies in unconditional coverage are carried into the joint test.

Overall, the non-overlapping conditional coverage analysis provides a cleaner
joint assessment of violation frequency and dependence structure.
"""




#-----Exponentially Weighted Moving Average-----
"""
Dynamic Volatility Extension:

The static Parametric and Monte Carlo models show weak tail coverage under backtesting. 
EWMA is introduced as a dynamic volatility extension to examine whether time-varying volatility scaling 
improves VaR performance relative to static volatility assumptions.

To keep the comparison internally consistent, EWMA is evaluated using the same:
- 5-day VaR horizon
- 99% confidence level
- portfolio value
- forward 5-day realized PnL for backtesting

EWMA volatility is implemented using the RiskMetrics framework:

    σ_t² = λσ_{t-1}² + (1 - λ)r_{t-1}²

RiskMetrics standard decay factors:
    Daily data:   λ = 0.94
    Weekly data:  λ = 0.97
    Monthly data: λ = 0.99

Because λ is determined by data frequency rather than VaR horizon, λ = 0.94 is appropriate 
for daily returns. The 5-day VaR is then obtained through square-root-of-time scaling.
"""

def compute_ewma_volatility(returns, lam=0.94, init_window=60):
    # Create an array of length len(returns), initialized with NaN,
    # to store the EWMA variance estimates over time
    ewma_variance = np.full(len(returns), np.nan)
    
    # Use the variance of the first init_window returns as the initial seed.
    # It is stored at index init_window - 1 because Python uses zero-based indexing.
    ewma_variance[init_window - 1] = returns.iloc[:init_window].var()

    # Start recursion after the initial variance seed is set
    for t in range(init_window, len(returns)):
        
        ewma_variance[t] = (
            lam * ewma_variance[t - 1]           # persistence from the previous variance estimate
            + (1 - lam) * returns.iloc[t - 1]**2 # new information from the latest squared return shock
        )
    return pd.Series(np.sqrt(ewma_variance), index=returns.index)

def compute_ewma_VaR_series(
    returns,
    value=1000000,
    horizon=5,
    confidence=0.99,
    lam=0.94,
    init_window=60
):
    ewma_volatility = compute_ewma_volatility(
        returns,
        lam=lam,
        init_window=init_window
    )

    z = norm.ppf(confidence)

    # Convert 1-day EWMA volatility into multi-day monetary VaR
    # under normality using square-root-of-time scaling
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

# Align EWMA VaR to the same forward PnL test period used in the other models
ewma_common_index = common_index.intersection(ewma_VaR_series.index)

ewma_VaR_series = ewma_VaR_series.loc[ewma_common_index]
ewma_pnl_test = forward_pnl.loc[ewma_common_index]

# Boolean violation series
ewma_violations = ewma_pnl_test < -ewma_VaR_series

# EWMA summary
print("EWMA Violations:", ewma_violations.sum())
print("EWMA Violation Rate:", ewma_violations.mean())
print("EWMA Average 5-day VaR:", ewma_VaR_series.mean())

# Kupiec and Traffic Light
ewma_kupiec_LR, ewma_kupiec_x = kupiec_test(ewma_VaR_series, ewma_pnl_test)
traffic_ewma = traffic_light_rolling(ewma_VaR_series, ewma_pnl_test)

print("EWMA Kupiec LR:", ewma_kupiec_LR)
print("EWMA Avg Violations (250-day window):", traffic_ewma["Violations"].mean())

# Non-overlapping sample
ewma_VaR_nonoverlap, ewma_pnl_nonoverlap = make_non_overlapping_sample(
    ewma_VaR_series, ewma_pnl_test, horizon=5, start=0)

ewma_kupiec_LR_NO, _ = kupiec_test(
    ewma_VaR_nonoverlap, ewma_pnl_nonoverlap)

print("EWMA Kupiec LR (Non-overlapping):", ewma_kupiec_LR_NO)

"""
EWMA Backtesting Interpretation:

EWMA updates conditional volatility over time, but the VaR construction still
relies on a volatility-scaled normal framework.

In this setup, EWMA produces a much higher violation rate than the 1% level
implied by a 99% VaR model, indicating severe underestimation of 5-day tail risk.

The traffic light results are also unfavorable, with the model frequently
falling into the red zone and producing an average violation count above the
standard 250-day benchmark.

Importantly, the weak coverage remains even on the non-overlapping sample,
suggesting that the problem is not driven only by overlap effects, but by the
model's inability to fully capture fat tails, large jumps, and stress-period
loss dynamics.

Overall, volatility updating alone is not sufficient to recover the true
multi-day tail risk in the present dataset.
"""




#-----Generalized Autoregressive Conditional Heteroskedasticity-----
"""
GARCH is introduced as a more flexible conditional volatility model than EWMA.

As with EWMA, the model is evaluated on the same basis as the previous VaR frameworks:
- 5-day VaR horizon
- 99% confidence level
- same portfolio value
- same forward 5-day realized PnL for backtesting

To maintain comparability with EWMA, this implementation first estimates 1-day conditional
volatility and then converts it to a 5-day VaR using square-root-of-time scaling.

This version uses:
- portfolio-level univariate GARCH(1,1)
- zero conditional mean
- normal innovations

GARCH(1,1) conditional variance dynamics:

    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

where:
- omega : long-run variance component
- alpha : sensitivity to recent shocks
- beta  : volatility persistence

With mean='Zero', the shock term epsilon_t is the portfolio return itself.
The innovation distribution is assumed to be normal.

Rationale for using GARCH(1,1):

- GARCH(1,1) is a standard and parsimonious conditional volatility model.
- Higher-order GARCH specifications do not necessarily improve performance.
  As the number of parameters increases, interpretability declines, while the
  risk of estimation instability and overfitting may rise.
- This makes GARCH(1,1) a practical and interpretable benchmark for VaR comparison.  
"""

def compute_garch_volatility(returns, scale=100):
    
    # GARCH estimation is often numerically more stable when returns are scaled.
    # The conditional volatility is then rescaled back to the original return units.
    scaled_returns = returns * scale

    model = arch_model(
        scaled_returns,
        mean='Zero',
        vol='GARCH',
        p=1,
        q=1,
        dist='normal' # Assume normally distributed innovations
    )

    result = model.fit(disp='off')

    # Rescale conditional volatility back to the original return scale
    garch_vol = result.conditional_volatility / scale

    return pd.Series(garch_vol, index=returns.index), result

def compute_garch_var_series(
    returns,
    value=1000000,
    horizon=5,
    confidence=0.99
):
    garch_vol, garch_result = compute_garch_volatility(returns)
    z = norm.ppf(confidence)

    garch_var_series = value * z * garch_vol * np.sqrt(horizon)

    return garch_var_series, garch_vol, garch_result

# Portfolio-level univariate GARCH is used for simplicity and direct comparability
# with the portfolio-level backtesting framework.
garch_var_series, garch_vol, garch_result = compute_garch_var_series(
    portfolio_returns, 
    value=1000000,
    horizon=5,
    confidence=0.99)

# Align GARCH VaR with the same forward PnL test period used in the other models
garch_common_index = common_index.intersection(garch_var_series.index)

# Retain only timestamps shared by GARCH VaR and the common backtesting sample
garch_var_series = garch_var_series.loc[garch_common_index]
garch_pnl_test = forward_pnl.loc[garch_common_index]

# GARCH results
garch_violations = garch_pnl_test < -garch_var_series

print("GARCH Violations:", garch_violations.sum())
print("GARCH Violation Rate:", garch_violations.mean())
print("GARCH Average VaR:", garch_var_series.mean())

# Kupiec and Traffic Light
garch_kupiec_LR, _ = kupiec_test(garch_var_series, garch_pnl_test)
traffic_garch = traffic_light_rolling(garch_var_series, garch_pnl_test)

print("GARCH Kupiec LR:", garch_kupiec_LR)
print("GARCH Avg Violations (250-day window):", traffic_garch["Violations"].mean())

#-----Non-overlapping Sample-----
garch_var_nonoverlap, garch_pnl_nonoverlap = make_non_overlapping_sample(
    garch_var_series,
    garch_pnl_test,
    horizon=5,
    start=0)

garch_kupiec_LR_NO, _ = kupiec_test(garch_var_nonoverlap, garch_pnl_nonoverlap)
garch_ind_lr, _ = christoffersen_independence_test(garch_var_nonoverlap, garch_pnl_nonoverlap)
garch_cc_lr = conditional_coverage_test(garch_kupiec_LR_NO, garch_ind_lr)

print("GARCH Kupiec LR (Non-overlapping):", garch_kupiec_LR_NO)
print("GARCH Conditional Coverage LR:", garch_cc_lr)

"""
GARCH Backtesting Interpretation:

Relative to EWMA, GARCH(1,1) improves backtesting performance by producing
fewer violations and lower backtesting test statistics.

This suggests that GARCH captures time-varying volatility dynamics more
effectively than the simpler EWMA specification.

However, the model still performs poorly in absolute terms under a 99% VaR
framework, as the observed violation rate remains materially above the level
implied by the confidence threshold.

The non-overlapping results lead to the same general conclusion: GARCH improves
on EWMA, but still fails to provide adequate tail coverage.

Overall, more flexible volatility dynamics help, but are not sufficient on
their own. The remaining weakness likely reflects not only volatility modeling,
but also the normal innovation assumption, which motivates the next extension
toward filtered historical simulation (FHS).
"""




#-----Filtered Historical Simulation-----
"""
FHS combines dynamic volatility filtering with an empirical residual distribution.

Step 1:
    Estimate conditional volatility using GARCH(1,1)

Step 2:
    Standardize returns by the estimated volatility:
        
        z_t = r_t / sigma_t

Step 3:
    Use the historical distribution of standardized residuals as the innovation distribution

Step 4:
    Re-scale sampled residuals using the evolving conditional volatility process 
    to generate filtered future returns

Although the GARCH filter is estimated under normal innovations,
future shocks in the FHS simulation are not drawn from a normal distribution.
Instead, they are resampled from the empirical distribution of standardized residuals.

This allows FHS to retain GARCH-based volatility dynamics while reflecting
fatter tails than a purely normal innovation assumption.
"""

def compute_standardized_residuals(returns, vol):
    # Align returns and volatility on the same timestamps before standardization
    aligned = pd.DataFrame({
        "returns": returns,
        "vol": vol
    }).dropna()

    return aligned["returns"] / aligned["vol"]

# A 1000-day rolling window is used to balance estimation stability
# and a sufficiently rich empirical residual pool.
def rolling_fhs_var(
    returns,
    window=1000, 
    value=1000000,
    horizon=5,
    confidence=0.99,
    simulations=2000,
    scale=100):
    
    var_list = []
    index_list = []

    for i in range(window, len(returns) - horizon):
        sample_returns = returns.iloc[i - window:i]

        scaled_sample = sample_returns * scale
        model = arch_model(
            scaled_sample,
            mean='Zero',
            vol='GARCH',
            p=1,
            q=1,
            dist='normal' # Used for GARCH estimation, not for the final shock draw
        )
        
        result = model.fit(disp='off')

        sigma_hist = result.conditional_volatility / scale
        sigma_hist = pd.Series(sigma_hist, index=sample_returns.index)

        z_hist = compute_standardized_residuals(
            sample_returns,
            sigma_hist
        ).dropna().values

        omega = result.params["omega"] / (scale**2) # omega is a variance parameter, so it must be rescaled by scale**2
        alpha = result.params["alpha[1]"]
        beta = result.params["beta[1]"]

        last_sigma2 = sigma_hist.iloc[-1] ** 2
        last_return2 = sample_returns.iloc[-1] ** 2

        path_pnl = np.zeros(simulations)

        for s in range(simulations):
            sigma2_t = omega + alpha * last_return2 + beta * last_sigma2
            pnl_path = 0.0

            for _ in range(horizon):
                z_draw = np.random.choice(z_hist) # Empirical draw from standardized residuals
                r_draw = np.sqrt(sigma2_t) * z_draw
                pnl_path += value * r_draw

                # Update conditional variance recursively, so volatility changes along each simulated path
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

fhs_common_index = common_index.intersection(fhs_var_series.index)

fhs_var_series = fhs_var_series.loc[fhs_common_index]
fhs_pnl_test = forward_pnl.loc[fhs_common_index]

#-----FHS Results-----
fhs_violations = fhs_pnl_test < -fhs_var_series

print("FHS Violations:", fhs_violations.sum())
print("FHS Violation Rate:", fhs_violations.mean())
print("FHS Average VaR:", fhs_var_series.mean())

fhs_kupiec_LR, fhs_kupiec_x = kupiec_test(fhs_var_series, fhs_pnl_test)
traffic_fhs = traffic_light_rolling(fhs_var_series, fhs_pnl_test)

print("FHS Kupiec LR:", fhs_kupiec_LR)
print("FHS Avg Violations (250-day window):", traffic_fhs["Violations"].mean())

fhs_var_nonoverlap, fhs_pnl_nonoverlap = make_non_overlapping_sample(
    fhs_var_series,
    fhs_pnl_test,
    horizon=5,
    start=0
)

fhs_kupiec_LR_NO, fhs_kupiec_x_NO = kupiec_test(fhs_var_nonoverlap, fhs_pnl_nonoverlap)
fhs_ind_lr, fhs_trans = christoffersen_independence_test(fhs_var_nonoverlap, fhs_pnl_nonoverlap)
fhs_cc_lr = conditional_coverage_test(fhs_kupiec_LR_NO, fhs_ind_lr)

print("FHS Kupiec LR (Non-overlapping):", fhs_kupiec_LR_NO)
print("FHS Conditional Coverage LR:", fhs_cc_lr)

"""
FHS delivers the strongest performance among the dynamic extensions considered in this framework.

Relative to the normal-based dynamic models such as EWMA and GARCH, FHS produces 
fewer violations and more favorable backtesting results.

It also improves materially on the static parametric and Monte Carlo approaches.
However, Historical VaR still remains highly competitive, indicating that the
empirical distribution of realized returns continues to provide strong tail-risk
information in this dataset.




Overall, Historical VaR delivers the strongest backtesting performance in this framework.

Across Kupiec, Traffic Light, and Conditional Coverage diagnostics, Historical VaR
remains the most robust model on this dataset.

FHS performs best among the model extensions and comes closest to Historical VaR,
but Historical VaR still appears to be the strongest overall benchmark.
"""







