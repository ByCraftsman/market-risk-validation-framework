# Market Risk Framework (VaR, ES, and Backtesting)
- **File:** [`Market_Risk_Framework.py`](./Market_Risk_Framework.py)

Built an end-to-end market risk framework in Python for multi-method VaR/ES estimation, rolling-window forecasting, backtesting, and volatility-based model extensions.

## Portfolio Setting
The test portfolio is intentionally simplified as an equal-weighted cross-asset mix of:

- **KOSPI Composite Index**
- **Kosdaq Composite Index**
- **iShares 7–10 Year Treasury Bond ETF (IEF)**
- **S&P 500 Index**

This portfolio is intended for transparent model comparison rather than replicating an actual trading desk portfolio. 
The goal is to provide a clear setting for comparing model behavior, backtesting performance, and differences in tail-risk sensitivity across market risk approaches.

## Key Features
- Implements **Historical, Parametric, and Monte Carlo VaR/ES**
- Uses a **rolling-window framework** to update VaR forecasts over time
- Applies a structured validation framework:
  - **Kupiec test** (unconditional coverage)
  - **Christoffersen independence test**
  - **Conditional coverage test**
  - **Basel-style traffic light interpretation**
- Compares **overlapping** and **non-overlapping** backtesting results
- Extends the framework with **EWMA, GARCH(1,1), and Filtered Historical Simulation (FHS)**
- Includes supporting analysis such as:
  - Monte Carlo simulation convergence checks
  - interpretation of tail-risk underestimation
  - analysis of overlap-induced violation clustering

## Main Insight
Within this portfolio and sample setting, Historical VaR appears to be the best-calibrated model among the baseline models, while Parametric and Monte Carlo VaR tend to underestimate tail risk under normality-based assumptions.

Non-overlapping tests further suggest that part of the observed violation clustering is mechanically induced by overlapping forward PnL construction, leading to a cleaner interpretation of independence and conditional coverage results.

Among the volatility-based extensions, FHS shows the strongest performance, materially improving tail-risk calibration relative to EWMA and standard GARCH.

Overall, Historical VaR remains the strongest model in this framework, with FHS emerging as the most effective volatility-based extension.

## Project Evolution
This project began as a basic implementation of three VaR methods: Historical, Parametric, and Monte Carlo.

As the framework expanded, the focus moved beyond point estimation toward model validation and backtesting.  
This led to several important extensions:

- moving from single VaR estimates to rolling VaR series
- constructing forward PnL for proper backtesting alignment
- distinguishing estimation inputs from realized backtesting targets
- comparing overlapping and non-overlapping samples
- identifying how overlapping forward PnL can mechanically distort independence test results
- extending static models into volatility-updating and filtered simulation frameworks

Through this process, the project evolved from a basic VaR implementation into a broader market risk validation framework.

## Modeling Assumptions
- The project is designed for risk-model comparison and interpretation
- The framework uses a 5-day holding period and 99% confidence level
- Mean returns are assumed to be zero for short-horizon parametric and simulation-based VaR
- FX risk is excluded for analytical clarity, although the portfolio includes both Korean and US market instruments
