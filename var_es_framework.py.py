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




# log ruturns are used because its additive.
def compute_log_returns(price_df):
    
    returns = np.log(price_df / price_df.shift(1)) #shift는 위의 행. 이전 시계열
    
    return returns.dropna()

log_returns = compute_log_returns(prices)




#Equal-weighted portfolio
weights = np.array([1/len(tickers)]*len(tickers)) #0.25 * 4

#로그 수익률 서메이션. This equation has implicit daily rebalancing.
def compute_portfolio_returns(log_returns, weights):
    
    return (log_returns * weights).sum(axis=1)

portfolio_returns = compute_portfolio_returns(log_returns, weights)



#Historical VaR Method 
"""
#Rolling sums of log returns represent cumulative multi-period returns.
#5-day holding period VaR. Mean return is assumed to be zero. (짧은 기간은 보통 mean return을 0으로 가정.)
#Portfolio value is assumed to be denominated in USD. FX risk excluded.
#0.99를 쓴 이유는 Tail Risk때문에 보수적인 수치를 요구하기 때문임.
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

#시각화
plt.hist(rolling_pnl.dropna(), bins=200, density=True)
plt.xlabel('5 Day Portfolio Return (Dollar value)')
plt.ylabel('Frequency')
plt.title('Historical VaR, 5 day')
plt.axvline(-historical_VaR, color = 'r', linestyle = 'dashed', linewidth = 1, label = 'VaR at 99% Confidence Level')
plt.legend()





#VaR Parametric Method
from scipy.stats import norm



def compute_parametric_VaR(log_returns, weights, value=1000000, horizon=5, confidence=0.99):
    cov_matrix = log_returns.cov()     # daily covariance
    portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights) #.T은 Transpose다.
    
    # Mean return is assumed to be zero (This is standard practice for short-horizon VaR)
    var = (
        value
        * portfolio_std
        * norm.ppf(confidence) #Percent Point Function는 CDF (누적분포함수)의 역함수임.
        * np.sqrt(horizon)
          )
    
    return var, portfolio_std


parametric_VaR, portfolio_std = compute_parametric_VaR(log_returns, weights)

print(parametric_VaR)


#시각화
plt.hist(rolling_pnl.dropna(), bins=200, density=True)
plt.axvline(-parametric_VaR, color = 'r', linestyle = 'dashed', linewidth = 1, label = 'VaR at 99% Confidence Level')
plt.xlabel('5 Day Portfolio Return (Dollar value)')
plt.ylabel('Frequency')
plt.title('Parametric VaR, 5 day')
plt.legend()




"""
정규분포를 가정한 다변량 Monte Carlo VaR는 수학적으로 Parametric VaR와 동일한 분포를 가지므로,
 두 결과가 일치하는 것이 정상이다. 더 큰 VaR를 얻기 위해서는 분포 가정을 변경해야 한다.

포트폴리오 VaR에서는 자산 간 상관관계가 리스크의 핵심이기 때문에,
 실무에서는 다변량 분포 기반 VaR가 표준적으로 사용된다.”
"""



#VaR Monte Carlo Method
def compute_monte_carlo_var(log_returns, weights, value=1000000, horizon=5, confidence=0.99, simulations=10000):
    cov_matrix = log_returns.cov()
    num_assets = len(weights)
    mu = np.zeros(num_assets) #평균 수익률 벡터 (보통 VaR에서는 0으로 둔다고 함.)

    #다변량 정규분포에서 자산 수익률 시뮬레이션
    simulated_returns = np.random.multivariate_normal(
        mean=mu,
        cov=cov_matrix,
        size=simulations
    )

    #포트폴리오 수익률 계산  
    portfolio_sim_returns = simulated_returns @ weights
    #기간 조정 (T-day VaR)
    portfolio_sim_returns *= np.sqrt(horizon)
    #PnL 계산
    scenario_pnl = value * portfolio_sim_returns
    #VaR 계산
    mc_var = -np.percentile(scenario_pnl, (1 - confidence) * 100)

    return mc_var, scenario_pnl

monte_carlo_VaR, scenario_pnl = compute_monte_carlo_var(log_returns, weights)

print(monte_carlo_VaR)


#시각화  <- 이 정규분포처럼 생긴 이유를 설명할 주석을 달아야 할듯.
plt.hist(scenario_pnl, bins=200, density=True)
plt.axvline(-monte_carlo_VaR, color = 'r', linestyle = 'dashed', linewidth = 1, label = 'VaR at 99% Confidence Level')
plt.xlabel('5 Day Portfolio Return (Dollar value)')
plt.ylabel('Frequency')
plt.title('Monte Carlo VaR 5 day, Multivariate')
plt.legend()




#VaR Summary
VaR_summary = pd.DataFrame({
    "VaR": [historical_VaR, parametric_VaR, monte_carlo_VaR]
}, index=["Historical", "Parametric", "Monte Carlo"])




#Calculating Expected Shortfall
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



