import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def get_price_data(tickers, start_date, end_date):
    """Download historical closing prices from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)
    prices = data["Close"]
    # If only one ticker, pandas returns a Series — make it a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna()


def calculate_returns_and_cov(prices):
    """Return annualized mean returns and covariance matrix."""
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    sharpe = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = num_assets * [1.0 / num_assets]
    result = minimize(negative_sharpe, initial, args=args,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def min_volatility_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = num_assets * [1.0 / num_assets]
    result = minimize(portfolio_volatility, initial, args=args,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def efficient_frontier(mean_returns, cov_matrix, num_points=50):
    num_assets = len(mean_returns)
    results = []
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial = num_assets * [1.0 / num_assets]
        result = minimize(portfolio_volatility, initial,
                          args=(mean_returns, cov_matrix),
                          method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            results.append((target, result.fun, result.x))
    return results


def target_return_portfolio(mean_returns, cov_matrix, target):
    """Find min-vol portfolio that achieves a given target return."""
    num_assets = len(mean_returns)
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target}
    )
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = num_assets * [1.0 / num_assets]
    result = minimize(portfolio_volatility, initial,
                      args=(mean_returns, cov_matrix),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None