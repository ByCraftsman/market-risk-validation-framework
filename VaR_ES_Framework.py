#Due to time constraints, this project currently implements VaR and ES calculations only.
#Further extensions are planned within March.

"""
For Indices, 'Close' is used as they don't have dividends or splits.
For ETFs, 'Adj Close' is used to account for corporate actions.

지수는 배당이나 분할 개념이 없기 때문에 Close 가격이 수정 가격과 동일합니다.
따라서 지수는 Close를, ETF는 Adj Close을 사용했습니다.
"""

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




#Historical VaR Method 

"""
Rolling sums of log returns represent cumulative multi-period returns.

[Assumptions]

1. 5-day holding period VaR. Therefore, Mean return is assumed to be zero. 
2. Portfolio value is assumed to be denominated in USD. FX risk excluded.
3. 99% confidence level is used. In risk management and regulation
 (e.g., Basel frameworks) to capture extreme tail losses conservatively.
"""

def compute_rolling_pnl(returns, value, horizon):
    rolling = returns.rolling(horizon).sum().dropna()
    return rolling * value


def compute_historical_VaR(pnl, confidence):
    return -np.percentile(pnl, (1 - confidence) * 100)


rolling_pnl = compute_rolling_pnl(portfolio_returns, 1000000, 5)
historical_VaR = compute_historical_VaR(rolling_pnl, 0.99)

print(rolling_pnl)
print(historical_VaR)




#VaR Parametric Method
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

print(parametric_VaR)




#VaR Monte Carlo Method

"""
When returns are assumed to be multivariate normal:

    Parametric VaR  ≈  Monte Carlo VaR

because both rely on the same distributional assumption.

Differences arise when:
    • non-normal distributions are used
    • stochastic volatility is introduced
    • fat tails or skewness are modeled
"""

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




#Visualization
def plot_var_distribution(data, var_value, title, xlim=None, ylim=None): # Optional axis limits allow consistent scaling across plots when needed.
    plt.figure()
    plt.hist(data, bins=200, density=True)
    plt.axvline(-var_value, linestyle='dashed', linewidth=1, label='VaR')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

x_limits = (
    min(rolling_pnl.min(), scenario_pnl.min()),
    max(rolling_pnl.max(), scenario_pnl.max())
)

print(historical_VaR)
print(parametric_VaR)
print(monte_carlo_VaR)

"""
The Monte Carlo distribution appears narrower than the historical distribution.
This is expected because the simulation assumes a multivariate normal distribution,
which does not capture fat tails observed in real market data. (i.e. underestimates fat tails)
"""

plot_var_distribution(rolling_pnl, historical_VaR, 'Historical VaR', x_limits)
plot_var_distribution(rolling_pnl, parametric_VaR, 'Parametric VaR', x_limits)
plot_var_distribution(scenario_pnl, monte_carlo_VaR, 'Monte Carlo VaR', x_limits)




#VaR Summary
VaR_summary = pd.DataFrame({
    "VaR": [historical_VaR, parametric_VaR, monte_carlo_VaR]
}, index=["Historical", "Parametric", "Monte Carlo"])




#Expected Shortfall

"""
Expected Shortfall (ES) measures the average loss beyond VaR.

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























