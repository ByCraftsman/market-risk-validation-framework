# Work in progress: VaR/ES risk framework under active development

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
years = 20
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days = 365*years)
tickers = [
    '^KS11',      # KOSPI
    '^KQ11',      # KOSDAQ
    'IEF',        # US 7–10Y Treasury ETF
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

        df[ticker] = data['Adj Close'] if 'Adj Close' in data else data['Close']

    return df.dropna()

prices = fetch_prices(tickers, start_date, end_date)


# Log returns are used because they are additive.
def compute_log_returns(price_df):
    
    returns = np.log(price_df / price_df.shift(1)) #shift는 위의 행. 이전 시계열
    
    return returns.dropna()

log_returns = compute_log_returns(prices)


#We assume Equal-weighted Portfolio
weights = np.array([1/len(tickers)]*len(tickers)) #0.25 * 4

def compute_portfolio_returns(log_returns, weights):
    
    return (log_returns * weights).sum(axis=1)

portfolio_returns = compute_portfolio_returns(log_returns, weights)




"""
[Assumptions]

1. 5-day holding period VaR. Therefore, Mean return is assumed to be zero. 

2. Portfolio value is assumed to be denominated in USD. FX risk excluded.
   (Since the portfolio mixes Korean and US assets, currency risk
    would be material in practice. However, for analytical clarity,
    this framework focuses on market risk only.)

3. 99% confidence level is used. In risk management and regulation
 (e.g., Basel frameworks) to capture extreme tail losses conservatively.
"""




#-----Historical VaR Method-----
def compute_rolling_pnl(returns, value, horizon):
    rolling = returns.rolling(horizon).sum().dropna()
    return rolling * value


def compute_historical_VaR(pnl, confidence):
    return -np.percentile(pnl, (1 - confidence) * 100)


rolling_pnl = compute_rolling_pnl(portfolio_returns, 1000000, 5)
historical_VaR = compute_historical_VaR(rolling_pnl, 0.99)

print(rolling_pnl)
print(historical_VaR)




#-----VaR Parametric Method-----
from scipy.stats import norm

def compute_parametric_VaR(log_returns, weights, value=1000000, horizon=5, confidence=0.99):
    cov_matrix = log_returns.cov()     # daily covariance
    portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights) # T means Transpose.
    
    # Mean return is assumed to be zero (This is standard practice for short-horizon VaR)
    var = (
          value
        * portfolio_std
        * norm.ppf(confidence) # This is Inverse CDF (quantile). Returns the z-score for the given confidence level.
        * np.sqrt(horizon)
          )
    
    return var, portfolio_std

parametric_VaR, portfolio_std = compute_parametric_VaR(log_returns, weights)

print(portfolio_std)
print(parametric_VaR)




#-----VaR Monte Carlo Method-----
def compute_monte_carlo_var(log_returns, weights, value=1000000, horizon=5, confidence=0.99, simulations=10000):
    cov_matrix = log_returns.cov()
    num_assets = len(weights)
    # Again, in short-horizon VaR, mean returns are assumed to be zero. 
    mu = np.zeros(num_assets) 

    # Simulate asset returns from a multivariate normal distribution
    simulated_returns = np.random.multivariate_normal(
        mean=mu,
        cov=cov_matrix,
        size=simulations
    )

    portfolio_sim_returns = simulated_returns @ weights
    # Scale returns to T-day horizon (variance scales with time → std scales with sqrt(T))
    # *= is the Compound Assignment Operator that means a = a * b
    portfolio_sim_returns *= np.sqrt(horizon)
    # PnL 
    scenario_pnl = value * portfolio_sim_returns
    # VaR
    mc_var = -np.percentile(scenario_pnl, (1 - confidence) * 100)

    return scenario_pnl, mc_var

scenario_pnl, monte_carlo_VaR = compute_monte_carlo_var(log_returns, weights)

print(monte_carlo_VaR)

"""
Monte Carlo VaR is similar to Parametric VaR.

This similarity arises because when returns are assumed to follow a multivariate normal distribution,

    Parametric VaR ≈ Monte Carlo VaR

as both methods rely on the same distributional assumption.

However, differences arise when:
    
    • non-normal distributions are used
    • stochastic volatility is introduced
    • fat tails or skewness are modeled
"""




#-----VaR Convergence Insight-----
simulation_sizes = [500, 3000, 10000, 50000]
results = []

for n in simulation_sizes:
    _, var_estimate = compute_monte_carlo_var(
        log_returns, weights, simulations=n
    )
    results.append(var_estimate)

mc_convergence = pd.DataFrame({
    "Simulations": simulation_sizes,
    "Monte Carlo VaR": results
})

print(mc_convergence)

"""
Monte Carlo VaR estimates stabilize as the number of simulations increases.
The difference between 10,000 and 50,000 simulations is less than 1%,
indicating that 10,000 simulations provide sufficient accuracy for practical risk measurement.

This suggests that smaller simulation sizes may lead to unstable tail estimates,
while larger samples improve the reliability of risk measurement.
"""




#-----VaR Distribution Plots-----
def generate_parametric_pnl(std, value, horizon, simulations=10000):
    simulated_returns = np.random.normal(0, std * np.sqrt(horizon), simulations) #Rt ~ N(0, σ²T)
    return simulated_returns * value

parametric_pnl = generate_parametric_pnl(portfolio_std, 1000000, 5)


def plot_var_distribution(data, var_value, title, xlim=None, ylim=None): # Optional axis limits allow consistent scaling across plots when needed.
    plt.figure()
    plt.hist(data, bins=100, density=True)
    plt.axvline(-var_value, linestyle='dashed', linewidth=1, label='VaR')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

tail_xlim = (-80000, 80000)

plot_var_distribution(rolling_pnl, historical_VaR, 'Historical VaR', tail_xlim)
plot_var_distribution(parametric_pnl, parametric_VaR, 'Parametric VaR', tail_xlim)
plot_var_distribution(scenario_pnl, monte_carlo_VaR, 'Monte Carlo VaR', tail_xlim)

"""
The parametric and Monte Carlo distributions appear narrower (lower standard deviation)
than the historical distribution. This is expected because the simulations assume a
multivariate normal distribution, which fails to capture the fat tails observed in
real market data, potentially leading to an underestimation of tail risk.
"""




#-----VaR Summary-----
VaR_summary = pd.DataFrame({
    "VaR": [historical_VaR, parametric_VaR, monte_carlo_VaR]
}, index=["Historical", "Parametric", "Monte Carlo"])

print(VaR_summary)




#-----Expected Shortfall-----

"""
Expected Shortfall (ES) measures the average loss beyond VaR.

    Definition:
    
        ES_α = -E[X | X ≤ -VaR_α]

    For normal distribution:
   
        ES = V * σ * φ(z_α) / (1 - α) * sqrt(h)

where:
- φ(z): standard normal PDF
- z_α: quantile at confidence level α


While VaR answers:
   "What loss level will not be exceeded with X% confidence?"

ES answers:
   "If losses exceed VaR, how large are they on average?"

ES is a coherent risk measure and captures tail risk better than VaR.
"""

his_ES = -rolling_pnl[rolling_pnl <= -historical_VaR].mean()

z = norm.ppf(0.99)
pdf_z = norm.pdf(z)
para_ES = (
    1000000
    * portfolio_std
    * (pdf_z / (1 - 0.99)) # Tail conditional expectation term
    * np.sqrt(5)
)

mc_ES = -scenario_pnl[scenario_pnl <= -monte_carlo_VaR].mean()


#ES Summary
ES_summary = pd.DataFrame({
    "ES": [his_ES, para_ES, mc_ES]
}, index=["Historical", "Parametric", "Monte Carlo"])

print(ES_summary)




#-----Backtesting-----
"""
Backtesting methodologies for Value-at-Risk (VaR) models are generally grouped into
four categories:

1. Coverage Tests
   Coverage tests evaluate whether the observed frequency of VaR exceedances
   matches the expected rate implied by the confidence level.

   Example:
   - Kupiec Unconditional Coverage Test

   Purpose:
   Ensures the model produces the correct proportion of tail losses.

   Limitation:
   Does not detect clustering of violations.


2. Independence Tests
   Independence tests assess whether VaR violations occur independently over time.

   Example:
   - Christoffersen Independence Test

   Why is this necessary?

   A model may produce the correct violation frequency but still be inadequate if
   violations cluster during periods of market stress. Clustering indicates that
   the model fails to capture volatility dynamics (e.g., volatility clustering).

   Kupiec test only answers:
       "Is the violation rate correct?"

   Independence test answers:
       "Are violations randomly distributed over time?"

   Detecting clustering is crucial because persistent violations imply that risk
   is systematically underestimated during turbulent periods.


3. Conditional Coverage Tests
   Conditional coverage tests jointly evaluate both correct coverage and independence.

   Example:
   - Christoffersen Conditional Coverage Test

   Why use conditional coverage?

   A robust VaR model must satisfy BOTH:
       
   • Correct exceedance frequency
   • Independence of violations

   Conditional coverage provides a comprehensive assessment by combining both criteria.


4. Distribution Tests (Not implemented here)

   Distribution tests evaluate whether the entire forecast loss distribution
   matches the realized loss distribution.

   Examples:
   - Berkowitz test
   - Kolmogorov–Smirnov test
   - Anderson–Darling test

   Why are distribution tests not used here?

   • They require full density forecasts, not just VaR quantiles.
   • Regulatory frameworks (e.g., Basel) focus primarily on exceedance behavior.
   • VaR is a quantile-based risk measure; therefore, exceedance tests are more
     directly aligned with its objective.




Regulatory Perspective (Basel Framework)

The Basel Committee emphasizes exceedance-based backtesting because:

• Capital requirements depend on tail loss quantiles (VaR/ES).
• The primary regulatory concern is whether extreme losses are underestimated.
• Coverage and independence directly assess model reliability in tail risk.
• Distribution tests provide broader diagnostics but are not essential for
  regulatory capital validation.

In practice, regulators prioritize:
    - Coverage (Kupiec)
    - Independence (Christoffersen)
    - Conditional Coverage

These tests ensure that the model is reliable in estimating extreme losses,
which is the core objective of market risk regulation.

These exceedance tests form the basis of regulatory frameworks such as the Basel traffic light approach.
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
   
def kupiec_test(var, pnl=rolling_pnl, confidence=0.99):
    violations = pnl < -var
    x = violations.sum()
    n = len(pnl)
    p = 1 - confidence

    likelihood_ratio = -2 * (
        (n-x)*np.log(1-p) + x*np.log(p)
        - ((n-x)*np.log(1-x/n) + x*np.log(x/n))
    )
    return likelihood_ratio, x


his_likelihood_ratio, his_x = kupiec_test(historical_VaR)
para_likelihood_ratio, para_x = kupiec_test(parametric_VaR) 
mc_likelihood_ratio, mc_x = kupiec_test(monte_carlo_VaR)


kupiec_test_results = pd.DataFrame({
    "Method": ["Historical VaR", "Parametric VaR", "Monte Carlo VaR"],
    "LR Statistic": [his_likelihood_ratio, para_likelihood_ratio, mc_likelihood_ratio],
    "Violations (x)": [his_x, para_x, mc_x],
})

print(kupiec_test_results)

"""
Backtesting insight:

Historical VaR shows violation frequency consistent with the expected 1% level,
indicating that empirical distributions capture tail risk reasonably well.

Parametric and Monte Carlo VaR (both assuming normality) exhibit excessive
violations (~2.48%), demonstrating that normal distribution assumptions
underestimate fat tails and extreme losses in financial markets.

This result highlights a well-known limitation of normal-based VaR models.
"""




#-----Christoffersen Independence Test-----
"""
Christoffersen Independence Likelihood Ratio (LR) Test

The LR statistic for testing independence of VaR violations is defined as:

LR_ind = -2 ln [
    ((1 − π)^(n00 + n10) * π^(n01 + n11)) /
    ((1 − π01)^(n00) * π01^(n01) * (1 − π11)^(n10) * π11^(n11))
]

Transition probabilities:

π01 = n01 / (n00 + n01)     π11 = n11 / (n10 + n11)
π   = (n01 + n11) / (n00 + n01 + n10 + n11)

Transition counts:

n00 : no violation → no violation
n01 : no violation → violation
n10 : violation → no violation
n11 : violation → violation

Hypotheses:

H₀: π01 = π11 = π   (violations are independent over time)
H₁: π01 ≠ π11       (violation clustering exists)

Under H₀:
LR_ind ~ χ²(1)

Decision Rule:
    Reject H0 if LR_ind > 6.63  (at 99% confidence level, df = 1)
"""

def christoffersen_independence_test(var, pnl=rolling_pnl, confidence=0.99):
    violations = (pnl < -var).astype(int)

    # Transition counts
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(violations)): #첫 관측은 제외함.
        prev = violations.iloc[i-1]
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

    # Likelihood ratio
    LR_ind = -2 * (
        (n00 + n10)*np.log(1 - pi) + (n01 + n11)*np.log(pi)
        - (n00*np.log(1 - pi01) + n01*np.log(pi01)
           + n10*np.log(1 - pi11) + n11*np.log(pi11))
    )

    return LR_ind, (n00, n01, n10, n11)


his_ind_lr, his_trans = christoffersen_independence_test(historical_VaR)
para_ind_lr, para_trans = christoffersen_independence_test(parametric_VaR)
mc_ind_lr, mc_trans = christoffersen_independence_test(monte_carlo_VaR)

independence_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR Independence": [his_ind_lr, para_ind_lr, mc_ind_lr],
})

print(independence_results)
print(his_trans)
print(para_trans)
print(mc_trans)

"""
Backtesting insight:

The extremely large LR statistics from the Christoffersen independence test
indicate strong violation clustering. This is consistent with well-known
features of financial returns such as volatility clustering and regime shifts.

The results suggest that static volatility assumptions (as in normal VaR)
fail to capture time-varying risk, leading to concentrated exceedances
during periods of market stress.
"""




#-----Christoffersen Conditional Coverage test-----
def conditional_coverage_test(lr_uc, lr_ind):
    return lr_uc + lr_ind

his_CCI_test_result = conditional_coverage_test(his_likelihood_ratio, his_ind_lr)
para_CCI_test_result = conditional_coverage_test(para_likelihood_ratio, para_ind_lr)
mc_CCI_test_result = conditional_coverage_test(mc_likelihood_ratio, mc_ind_lr)

CCI_results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "LR_CC": [
        his_CCI_test_result,
        para_CCI_test_result,
        mc_CCI_test_result
    ]
})

print(CCI_results)

"""
Backtesting insight:

[Historical VaR]

Finding: 
    
    Passed UC (LR=0.0027~), but failed CC due to Ind (LR=224.98).

    
[Parametric & Monte Carlo VaR]

Finding: 
    
    Failed both UC and Ind (LR_cc > 500).


[Interpretation]

From a regulatory and risk-management perspective, a high LR_CC statistic is a red flag,
indicating that VaR violations are not only too frequent but also temporally clustered.

This suggests that the model fails to capture time-varying volatility and regime shifts,
leading to an underestimation of risk during periods of market stress.

To address this issue, more advanced approaches such as GARCH-based volatility models
or Filtered Historical Simulation (FHS) can be employed to better account for volatility
dynamics and reduce violation clustering.

This result is consistent with well-documented volatility clustering in financial markets.
"""




"""
Backtesting results (Kupiec, independence, conditional coverage) indicate that the static VaR model
fails to adequately capture violation frequency and independence. To address volatility clustering
and time-varying risk, dynamic volatility models (EWMA, GARCH, FHS) are applied sequentially.

I implemented EWMA to address volatility clustering observed in the Christoffersen independence test.
By allowing time-varying volatility, EWMA helps capture violation clustering and improves VaR reliability.

EWMA volatility is implemented following the RiskMetrics framework developed by J.P. Morgan.
EWMA volatility estimator (RiskMetrics):

    σ_t² = λ σ_{t-1}² + (1 - λ) r_{t-1}²

RiskMetrics standard decay factors:
    Daily data:   λ = 0.94
    Weekly data:  λ = 0.97
    Monthly data: λ = 0.99
    
Because λ is determined by data frequency, not VaR horizon. Since the model uses daily returns, 
λ=0.94 is appropriate. The 5-day VaR is obtained through square-root-of-time scaling.    
"""




#-----EWMA-----
def compute_ewma_volatility(returns, lam=0.94):
    ewma_var = np.zeros(len(returns))
    ewma_var[0] = returns.var() 

    for t in range(1, len(returns)):
        ewma_var[t] = lam * ewma_var[t-1] + (1 - lam) * returns.iloc[t-1]**2

    ewma_vol = np.sqrt(ewma_var)
    return pd.Series(ewma_vol, index=returns.index)

def compute_ewma_var(returns, value=1000000, horizon=1, confidence=0.99, lam=0.94):
    ewma_vol = compute_ewma_volatility(returns, lam)

    z = norm.ppf(confidence)

    ewma_var_series = value * ewma_vol * z * np.sqrt(horizon)

    return ewma_var_series, ewma_vol

ewma_VaR_series, ewma_vol = compute_ewma_var(portfolio_returns)
daily_portfolio_PNL = portfolio_returns*1000000

def ewma_violations(pnl):
    aligned_var = ewma_VaR_series.loc[pnl.index]
    return pnl < -aligned_var

ewma_violations = ewma_violations(daily_portfolio_PNL) 

#result
print("Violations:", ewma_violations.sum())
print("Violation rate:", ewma_violations.mean())
print(ewma_VaR_series.mean())
print(daily_portfolio_PNL.min())




"""
Backtesting insight:
    
The observed violation rate (2.68%) exceeds the expected rate (1%), indicating that the model
underestimates tail risk.

The largest loss (65,350) is more than 4 times the average VaR estimate (15,850), suggesting the presence
of fat tails and extreme market movements that are not captured by the normality assumption.

Although EWMA captures volatility clustering, it still relies on a normal distribution assumption. 
This leads to an underestimation of extreme losses in fat-tailed return distributions.

EWMA assumes symmetric responses to positive and negative shocks, failing to capture leverage effects 
observed in equity markets. This suggests that tail-sensitive models such as GARCH or Filtered Historical Simulation 
may provide more robust risk estimates.
"""








