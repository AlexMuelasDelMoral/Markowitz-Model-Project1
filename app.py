import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

from markowitz import (
    get_price_data,
    calculate_returns_and_cov,
    portfolio_performance,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    efficient_frontier,
    target_return_portfolio,
)

# ---------- Page Setup ----------
st.set_page_config(page_title="Markowitz Portfolio Optimizer", layout="wide")
st.title("📈 Markowitz Portfolio Optimizer")
st.markdown(
    "Build an optimal portfolio using Modern Portfolio Theory. "
    "Enter tickers, choose a date range, and see the efficient frontier."
)

# ---------- Sidebar Inputs ----------
st.sidebar.header("Inputs")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, AMZN, NVDA"
)

start_date = st.sidebar.date_input(
    "Start date",
    value=date.today() - timedelta(days=5 * 365)
)
end_date = st.sidebar.date_input("End date", value=date.today())

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual, decimal)",
    value=0.04, min_value=0.0, max_value=0.20, step=0.005, format="%.3f"
)

run_button = st.sidebar.button("Run Optimization", type="primary")

# ---------- Main Logic ----------
if run_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
        st.stop()

    with st.spinner("Downloading price data..."):
        try:
            prices = get_price_data(tickers, start_date, end_date)
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()

    if prices.empty:
        st.error("No price data returned. Check your tickers or date range.")
        st.stop()

    st.success(f"Downloaded {len(prices)} days of data for {len(prices.columns)} assets.")

    # ---------- Calculations ----------
    mean_returns, cov_matrix = calculate_returns_and_cov(prices)

    # ---------- Display Prices ----------
    st.subheader("Price History")
    st.line_chart(prices)

    # ---------- Portfolios ----------
    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
    min_vol_w = min_volatility_portfolio(mean_returns, cov_matrix)

    ms_ret, ms_vol, ms_sharpe = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix, risk_free_rate)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(min_vol_w, mean_returns, cov_matrix, risk_free_rate)

    # ---------- Metrics Cards ----------
    st.subheader("Optimal Portfolios")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌟 Max Sharpe Portfolio")
        st.metric("Expected Return", f"{ms_ret:.2%}")
        st.metric("Volatility", f"{ms_vol:.2%}")
        st.metric("Sharpe Ratio", f"{ms_sharpe:.2f}")
        ms_df = pd.DataFrame({
            "Ticker": mean_returns.index,
            "Weight": [f"{w:.2%}" for w in max_sharpe_w]
        })
        st.dataframe(ms_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 🛡️ Min Volatility Portfolio")
        st.metric("Expected Return", f"{mv_ret:.2%}")
        st.metric("Volatility", f"{mv_vol:.2%}")
        st.metric("Sharpe Ratio", f"{mv_sharpe:.2f}")
        mv_df = pd.DataFrame({
            "Ticker": mean_returns.index,
            "Weight": [f"{w:.2%}" for w in min_vol_w]
        })
        st.dataframe(mv_df, hide_index=True, use_container_width=True)

    # ---------- Efficient Frontier ----------
    st.subheader("Efficient Frontier")
    with st.spinner("Building efficient frontier..."):
        frontier = efficient_frontier(mean_returns, cov_matrix, num_points=50)

    frontier_returns = [p[0] for p in frontier]
    frontier_vols = [p[1] for p in frontier]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_returns,
        mode="lines", name="Efficient Frontier",
        line=dict(color="blue", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret],
        mode="markers", name="Max Sharpe",
        marker=dict(color="red", size=15, symbol="star")
    ))

    fig.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret],
        mode="markers", name="Min Volatility",
        marker=dict(color="green", size=15, symbol="star")
    ))

    for i, ticker in enumerate(mean_returns.index):
        fig.add_trace(go.Scatter(
            x=[np.sqrt(cov_matrix.iloc[i, i])], y=[mean_returns.iloc[i]],
            mode="markers+text", name=ticker,
            text=[ticker], textposition="top center",
            marker=dict(size=10), showlegend=False
        ))

    fig.update_layout(
        xaxis_title="Volatility (Annualized Std Dev)",
        yaxis_title="Expected Return (Annualized)",
        height=600,
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- Custom Target Return ----------
    st.subheader("🎯 Custom Target Return")
    target = st.slider(
        "Choose a target annual return",
        min_value=float(mean_returns.min()),
        max_value=float(mean_returns.max()),
        value=float(ms_ret),
        step=0.01,
        format="%.2f"
    )
    custom_w = target_return_portfolio(mean_returns, cov_matrix, target)
    if custom_w is not None:
        c_ret, c_vol, c_sharpe = portfolio_performance(custom_w, mean_returns, cov_matrix, risk_free_rate)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Return", f"{c_ret:.2%}")
        col_b.metric("Volatility", f"{c_vol:.2%}")
        col_c.metric("Sharpe", f"{c_sharpe:.2f}")

        custom_df = pd.DataFrame({
            "Ticker": mean_returns.index,
            "Weight": [f"{w:.2%}" for w in custom_w]
        })
        st.dataframe(custom_df, hide_index=True, use_container_width=True)
    else:
        st.warning("Could not find a portfolio for this target return.")

else:
    st.info("👈 Set your inputs in the sidebar and click **Run Optimization**.")