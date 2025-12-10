import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as gost.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")st.markdown("""<div style="background: linear-gradient(90deg, #1E3A8A, #3B82F6); padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 28px; font-weight: bold; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
Stock Portfolio Optimizer — 3 Years Real Market Data
</div>
""", unsafe_allow_html=True)

Load data from CSV in your repo (instant, no API calls)@st
.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/bbou122/stock-portfolio-optimizer/main/stock_data.csv"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    return dfprices = load_data()
returns = prices.pct_change().dropna()st.markdown("Select any stocks from 10 major companies + S&P 500")
tickers = st.multiselect("Choose your stocks", prices.columns.tolist(), default=["AAPL", "NVDA", "JPM"])if tickers:
    # Allocation sliders
    st.subheader("Set Your Portfolio Weights")
    weights = {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            weights[t] = st.slider(t, 0, 100, round(100/len(tickers)), key=t) / 100total = sum(weights.values())
weights = pd.Series(weights) / total if total > 0 else pd.Series({t:1/len(tickers) for t in tickers})

# Portfolio math
port_returns = (returns[tickers] * weights).sum(axis=1)
cum_returns = (1 + port_returns).cumprod()

annual_return = port_returns.mean() * 252
annual_vol = port_returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol if annual_vol > 0 else 0
max_drawdown = (cum_returns.cummax() - cum_returns).max()

# S&P 500 from same data
spy_cum = (1 + returns["SPY"]).cumprod()
spy_annual = returns["SPY"].mean() * 252
spy_sharpe = spy_annual / (returns["SPY"].std() * np.sqrt(252))

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

# NEW: Summary Section (added as requested)
st.subheader("Portfolio Summary")
st.info(f"""
• **Your Annual Return**: {annual_return:.1%} (vs S&P 500: {spy_annual:.1%})  
• **Risk Level**: Volatility {annual_vol:.1%}, Max Drawdown -{max_drawdown:.1%}  
• **Performance Score**: Sharpe Ratio {sharpe:.2f} (higher = better return per risk)  
• **Suggestion**: If Sharpe < 1.0, add more diversification (e.g., bonds or ETFs)
""")else:
    st.info("Select stocks to begin")
    st.balloons()

st.markdown("---")
st.caption("Built quickly by Braden Bourgeois • Master’s in Analytics")

