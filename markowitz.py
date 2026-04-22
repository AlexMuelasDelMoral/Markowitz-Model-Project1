import yfinance as yf
import pandas as pd

# List of tickers (stock symbols) — you can change these later
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Download 5 years of daily price data
data = yf.download(tickers, start="2021-01-01", end="2026-01-01")

# Keep only the "Close" price column
prices = data["Close"]

# Show the first 5 rows so we can see it worked
print(prices.head())

# Calculate daily percentage returns
returns = prices.pct_change().dropna()

print("\n--- Daily Returns ---")
print(returns.head())

# Annualize: multiply daily mean by 252 trading days
mean_returns = returns.mean() * 252

# Annualize covariance by multiplying by 252
cov_matrix = returns.cov() * 252

print("\n--- Annualized Expected Returns ---")
print(mean_returns)

print("\n--- Annualized Covariance Matrix ---")
print(cov_matrix)

import numpy as np

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Given weights, return (annual_return, annual_volatility, sharpe_ratio)
    """
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    sharpe = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe


# Test it with equal weights (25% each)
equal_weights = [0.25, 0.25, 0.25, 0.25]
ret, vol, sharpe = portfolio_performance(equal_weights, mean_returns, cov_matrix)

print("\n--- Equal-Weight Portfolio ---")
print(f"Expected Return: {ret:.2%}")
print(f"Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

from scipy.optimize import minimize

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    # Optimizers minimize, so we minimize the NEGATIVE Sharpe
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe

def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Constraint: weights sum to 1
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})

    # Bounds: each weight between 0 and 1 (no shorting)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: equal weights
    initial = num_assets * [1.0 / num_assets]

    result = minimize(negative_sharpe, initial, args=args,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x  # optimal weights


# Run the optimizer
optimal_weights = max_sharpe_portfolio(mean_returns, cov_matrix)

print("\n--- Max Sharpe Portfolio ---")
for ticker, w in zip(mean_returns.index, optimal_weights):
    print(f"{ticker}: {w:.2%}")

ret, vol, sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
print(f"Expected Return: {ret:.2%}")
print(f"Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def min_volatility_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = num_assets * [1.0 / num_assets]
    result = minimize(portfolio_volatility, initial, args=args,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


min_vol_weights = min_volatility_portfolio(mean_returns, cov_matrix)

print("\n--- Min Volatility Portfolio ---")
for ticker, w in zip(mean_returns.index, min_vol_weights):
    print(f"{ticker}: {w:.2%}")

ret, vol, sharpe = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
print(f"Expected Return: {ret:.2%}")
print(f"Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")