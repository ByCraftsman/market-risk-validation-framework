#Due to time constraints, this project currently implements VaR and ES calculations only.
#Further extensions are planned within March.


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
years = 20
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days = 365*years)

"""
For Indices, 'Close' is used as they don't have dividends or splits.
For ETFs, 'Adj Close' is used to account for corporate actions.

지수는 배당이나 분할 개념이 없기 때문에 Close 가격이 수정 가격과 동일합니다.
따라서 지수는 Close를, ETF는 Adj Close을 사용했습니다.
"""
tickers = [
    '^KS11',      # KOSPI
    '^KQ11',      # KOSDAQ
    'IEF',        # US 7–10Y Treasury ETF
    '^GSPC'       # S&P 500 
]

# Download
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if 'Adj Close' in data.columns:
        adj_close_df[ticker] = data['Adj Close']
    else:
        adj_close_df[ticker] = data['Close']

adj_close_df = adj_close_df.dropna()
        
# log ruturns are used because its additive.
log_returns = np.log(adj_close_df/adj_close_df.shift(1)) #shift는 위의 행. 이전 시계열
log_returns = log_returns.dropna() 


#Portfolio value is assumed to be denominated in USD. FX risk excluded.
portfolio_value = 1000000

#Equal-weighted portfolio
weights = np.array([1/len(tickers)]*len(tickers)) #0.25*4

#로그 수익률 서메이션. This equation has implicit daily rebalancing.
historical_returns = (log_returns * weights).sum(axis = 1)

#5-day holding period VaR. Mean return is assumed to be zero. (짧은 기간은 보통 mean return을 0으로 가정.)
return_window = 5

#Rolling sums of log returns represent cumulative multi-period returns.
his_range_returns = historical_returns.rolling(window=return_window).sum().dropna()

#0.99를 쓴 이유는 Tail Risk때문에 보수적인 수치를 요구하기 때문임.
confidence_level = 0.99 




#Historical VaR Method 
#이 방법은 데이터를 그대로 사용하기 때문에 moments (적률) 와 correlation이 데이터에 모두 반영되는 특징이 있다.
alpha = 1 - confidence_level
range_returns_dollar = his_range_returns * portfolio_value
his_VaR = -np.percentile(range_returns_dollar, alpha * 100)
print(his_VaR)

#시각화
plt.hist(range_returns_dollar.dropna(), bins=200, density=True)
plt.xlabel(f'{return_window} - Day Portfolio Return (Dollar value)')
plt.ylabel('Frequency')
plt.title(f'Historical VaR, {return_window}-day')
plt.axvline(-his_VaR, color = 'r', linestyle = 'dashed', linewidth = 1, label = f'VaR at {confidence_level:.0%} Confidence Level')
plt.legend()




#VaR Parametric Method
from scipy.stats import norm
cov_matrix = log_returns.cov()          # daily covariance
portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights) #.T은 Transpose다.

# Mean return is assumed to be zero (This is standard practice for short-horizon VaR)
para_VaR = (
    portfolio_value
    * portfolio_std_dev
    * norm.ppf(confidence_level) #Percent Point Function는 CDF (누적분포함수)의 역함수임.
    * np.sqrt(return_window)
)

print(para_VaR)

#시각화
plt.hist(range_returns_dollar.dropna(), bins=200, density=True)
plt.axvline(-para_VaR, color = 'r', linestyle = 'dashed', linewidth = 1, label = f'VaR at {confidence_level:.0%} Confidence Level')
plt.xlabel(f'{return_window} - Day Portfolio Return (Dollar value)')
plt.ylabel('Frequency')
plt.title(f'Parametric VaR, {return_window}-day')
plt.legend()




#VaR Monte Carlo Method
"""
정규분포를 가정한 다변량 Monte Carlo VaR는 수학적으로 Parametric VaR와 동일한 분포를 가지므로,
 두 결과가 일치하는 것이 정상이다. 더 큰 VaR를 얻기 위해서는 분포 가정을 변경해야 한다.

포트폴리오 VaR에서는 자산 간 상관관계가 리스크의 핵심이기 때문에,
 실무에서는 다변량 분포 기반 VaR가 표준적으로 사용된다.”
"""
simulations = 10000

#자산 개수
num_assets = len(weights)

#평균 수익률 벡터 (보통 VaR에서는 0으로 둔다고 함.)
mu = np.zeros(num_assets)

#다변량 정규분포에서 자산 수익률 시뮬레이션
simulated_returns = np.random.multivariate_normal(
    mean=mu,
    cov=cov_matrix,
    size=simulations
)

#포트폴리오 수익률 계산
portfolio_returns = simulated_returns @ weights

#기간 조정 (T-day VaR)
portfolio_returns = portfolio_returns * np.sqrt(return_window)

#PnL 계산
scenario_pnl = portfolio_value * portfolio_returns

#VaR 계산
mc_var = -np.percentile(scenario_pnl, 100 * (1 - confidence_level))
print(mc_var)

#시각화
plt.hist(scenario_pnl, bins=200, density=True)
plt.axvline(-mc_var, color = 'r', linestyle = 'dashed', linewidth = 1, label = f'VaR at {confidence_level:.0%} Confidence Level')
plt.xlabel(f'{return_window} - Day Portfolio Return (Dollar value)')
plt.ylabel('Frequency')
plt.title(f'Monte Carlo VaR ({return_window}-day, Multivariate)')
plt.legend()




#VaR Summary
VaR_summary = pd.DataFrame({
    "VaR": [his_VaR, para_VaR, mc_var]
}, index=["Historical", "Parametric", "Monte Carlo"])




#Calculating Expected Shortfall
his_ES = -range_returns_dollar[range_returns_dollar <= -his_VaR].mean()

z = norm.ppf(confidence_level)
pdf_z = norm.pdf(z)
para_ES = (
    portfolio_value
    * portfolio_std_dev
    * (pdf_z / (1 - confidence_level)) # Tail conditional expectation term
    * np.sqrt(return_window)
)

mc_ES = -scenario_pnl[scenario_pnl <= -mc_var].mean()

#ES Summary
ES_summary = pd.DataFrame({
    "ES": [his_ES, para_ES, mc_ES]
}, index=["Historical", "Parametric", "Monte Carlo"])








