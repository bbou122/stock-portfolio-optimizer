import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #1E3A8A, #3B82F6); padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 28px; font-weight: bold; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
Stock Portfolio Optimizer — 3 Years Real Market Data
</div>
""", unsafe_allow_html=True)

# Load data from CSV in your repo (instant, no API calls)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/bbou122/stock-portfolio-optimizer/main/stock_data.csv"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    return df

prices = load_data()
returns = prices.pct_change().dropna()

st.markdown("Select any stocks from 10 major companies + S&P 500")
tickers = st.multiselect("Choose your stocks", prices.columns.tolist(), default=["AAPL", "NVDA", "JPM"])

if tickers:
    # Function to compute negative Sharpe ratio
    def neg_sharpe(weights, mean_returns, cov_matrix):
        port_ret = np.dot(weights, mean_returns) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        return -sharpe

    # Optimization function
    def optimize_portfolio(tickers, returns):
        mean_returns = returns[tickers].mean()
        cov_matrix = returns[tickers].cov()
        
        num_assets = len(tickers)
        args = (mean_returns, cov_matrix)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        initial_weights = np.array([1. / num_assets] * num_assets)
        
        result = minimize(neg_sharpe, initial_weights, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights

    # Tabs for user-defined and optimized portfolios
    tab1, tab2 = st.tabs(["User-Defined Portfolio", "Optimized Portfolio"])

    with tab1:
        st.subheader("Set Your Portfolio Weights")
        weights = {}
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            with cols[i]:
                weights[t] = st.slider(t, 0, 100, round(100 / len(tickers)), key=f"user_{t}") / 100
        
        total = sum(weights.values())
        if total > 0:
            weights = pd.Series(weights) / total
        else:
            weights = pd.Series({t: 1 / len(tickers) for t in tickers})
        
        # Portfolio math
        port_returns = (returns[tickers] * weights).sum(axis=1)
        cum_returns = (1 + port_returns).cumprod()
        
        annual_return = port_returns.mean() * 252
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Corrected max drawdown calculation
        drawdowns = (cum_returns / cum_returns.cummax()) - 1
        max_drawdown = -drawdowns.min()

        # S&P 500 from same data
        spy_returns = returns["SPY"]
        spy_cum = (1 + spy_returns).cumprod()
        spy_annual = spy_returns.mean() * 252
        spy_vol = spy_returns.std() * np.sqrt(252)
        spy_sharpe = spy_annual / spy_vol if spy_vol > 0 else 0

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annual Return", f"{annual_return:.1%}", f"{annual_return - spy_annual:+.1%}")
        col2.metric("Volatility", f"{annual_vol:.1%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}", f"{sharpe - spy_sharpe:+.2f}")
        col4.metric("Max Drawdown", f"-{max_drawdown:.1%}")

        # Comparison vs S&P 500 (equity curve)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name="Your Portfolio", line=dict(width=4)))
        fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500", line=dict(color="gray", dash="dash")))
        fig.update_layout(title="Portfolio vs S&P 500 ($1 → ?)", template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Portfolio distribution (donut chart)
        fig_donut = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, hole=.5, textinfo='label+percent')])
        fig_donut.update_layout(title="Portfolio Allocation", height=400)
        st.plotly_chart(fig_donut, use_container_width=True)

        # Summary Section
        st.subheader("Portfolio Summary")
        st.info(f"""
        • **Your Annual Return**: {annual_return:.1%} (vs S&P 500: {spy_annual:.1%})  
        • **Risk Level**: Volatility {annual_vol:.1%}, Max Drawdown -{max_drawdown:.1%}  
        • **Performance Score**: Sharpe Ratio {sharpe:.2f} (higher = better return per risk)  
        • **Suggestion**: If Sharpe < 1.0, add more diversification (e.g., bonds or ETFs)
        """)

    with tab2:
        st.subheader("Optimized Portfolio (Max Sharpe Ratio)")
        opt_weights_array = optimize_portfolio(tickers, returns)
        opt_weights = pd.Series(opt_weights_array, index=tickers)
        
        # Display optimized weights
        st.write("Optimized Weights:")
        for t, w in opt_weights.items():
            st.write(f"{t}: {w:.2%}")

        # Portfolio math for optimized
        port_returns_opt = (returns[tickers] * opt_weights).sum(axis=1)
        cum_returns_opt = (1 + port_returns_opt).cumprod()
        
        annual_return_opt = port_returns_opt.mean() * 252
        annual_vol_opt = port_returns_opt.std() * np.sqrt(252)
        sharpe_opt = annual_return_opt / annual_vol_opt if annual_vol_opt > 0 else 0
        
        # Corrected max drawdown calculation
        drawdowns_opt = (cum_returns_opt / cum_returns_opt.cummax()) - 1
        max_drawdown_opt = -drawdowns_opt.min()

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annual Return", f"{annual_return_opt:.1%}", f"{annual_return_opt - spy_annual:+.1%}")
        col2.metric("Volatility", f"{annual_vol_opt:.1%}")
        col3.metric("Sharpe Ratio", f"{sharpe_opt:.2f}", f"{sharpe_opt - spy_sharpe:+.2f}")
        col4.metric("Max Drawdown", f"-{max_drawdown_opt:.1%}")

        # Comparison vs S&P 500 (equity curve)
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(x=cum_returns_opt.index, y=cum_returns_opt, name="Optimized Portfolio", line=dict(width=4)))
        fig_opt.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500", line=dict(color="gray", dash="dash")))
        fig_opt.update_layout(title="Optimized Portfolio vs S&P 500 ($1 → ?)", template="plotly_dark", height=500)
        st.plotly_chart(fig_opt, use_container_width=True)

        # Portfolio distribution (donut chart)
        fig_donut_opt = go.Figure(data=[go.Pie(labels=opt_weights.index, values=opt_weights.values, hole=.5, textinfo='label+percent')])
        fig_donut_opt.update_layout(title="Optimized Portfolio Allocation", height=400)
        st.plotly_chart(fig_donut_opt, use_container_width=True)

        # Summary Section
        st.subheader("Optimized Portfolio Summary")
        st.info(f"""
        • **Optimized Annual Return**: {annual_return_opt:.1%} (vs S&P 500: {spy_annual:.1%})  
        • **Risk Level**: Volatility {annual_vol_opt:.1%}, Max Drawdown -{max_drawdown_opt:.1%}  
        • **Performance Score**: Sharpe Ratio {sharpe_opt:.2f} (higher = better return per risk)  
        • **Suggestion**: If Sharpe < 1.0, add more diversification (e.g., bonds or ETFs)
        """)
else:
    st.info("Select stocks to begin")
    st.balloons()

st.markdown("---")
st.caption("Built quickly by Braden Bourgeois • Master’s in Analytics")

