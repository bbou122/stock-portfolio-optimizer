import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Stock Portfolio Optimizer Pro", layout="wide")

# Custom header
st.markdown("""
<div style="background: linear-gradient(90deg, #1E3A8A, #3B82F6); padding: 25px; border-radius: 15px; text-align: center; color: white; font-size: 32px; font-weight: bold; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.3);">
    Stock Portfolio Optimizer — 3 Years Real Market Data
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/bbou122/stock-portfolio-optimizer/main/stock_data.csv"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    return df

prices = load_data()
returns = prices.pct_change().dropna()

st.markdown("### Select stocks to build and optimize your portfolio")
tickers = st.multiselect(
    "Choose stocks (includes S&P 500 as 'SPY')",
    options=prices.columns.tolist(),
    default=["AAPL", "NVDA", "MSFT", "JPM", "SPY"]
)

if len(tickers) < 2:
    st.warning("Please select at least 2 stocks to enable optimization.")
    st.stop()

# Risk-free rate slider
risk_free_rate = st.slider("Risk-Free Rate (for Sharpe Ratio)", 0.0, 10.0, 3.0, 0.1) / 100

# Optimization functions
def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.dot(weights, returns.mean()) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0
    return port_return, port_std, sharpe

def neg_sharpe_ratio(weights, returns, cov_matrix):
    _, _, sharpe = portfolio_performance(weights, returns, cov_matrix)
    return -sharpe

def min_volatility(weights, returns, cov_matrix):
    _, port_std, _ = portfolio_performance(weights, returns, cov_matrix)
    return port_std

def optimize_portfolio(tickers, returns, objective='sharpe'):
    mean_returns = returns[tickers].mean()
    cov_matrix = returns[tickers].cov()
    num_assets = len(tickers)
    args = (returns[tickers], cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = np.array([1/num_assets] * num_assets)

    if objective == 'sharpe':
        result = minimize(neg_sharpe_ratio, initial, args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    else:  # min volatility
        result = minimize(min_volatility, initial, args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    return result.x if result.success else initial

# Run optimizations
max_sharpe_weights = optimize_portfolio(tickers, returns, 'sharpe')
min_vol_weights = optimize_portfolio(tickers, returns, 'volatility')

# Calculate performances
max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = portfolio_performance(max_sharpe_weights, returns[tickers], returns[tickers].cov())
min_vol_ret, min_vol_vol, min_vol_sr = portfolio_performance(min_vol_weights, returns[tickers], returns[tickers].cov())

# Equal weight
equal_weights = np.array([1/len(tickers)] * len(tickers))
eq_ret, eq_vol, eq_sr = portfolio_performance(equal_weights, returns[tickers], returns[tickers].cov())

# S&P 500 benchmark (use if available)
spy_ret = returns["SPY"].mean() * 252 if "SPY" in returns.columns else 0.10
spy_vol = returns["SPY"].std() * np.sqrt(252) if "SPY" in returns.columns else 0.15
spy_sr = (spy_ret - risk_free_rate) / spy_vol if spy_vol > 0 else 0

# Tabs
tab1, tab2, tab3 = st.tabs(["Manual Portfolio", "Optimized Portfolios", "Efficient Frontier"])

with tab1:
    st.subheader("Build Your Own Portfolio")
    weights_dict = {}
    cols = st.columns(len(tickers))
    for i, tick in enumerate(tickers):
        with cols[i]:
            weights_dict[tick] = st.slider(tick, 0, 100, 100//len(tickers), key=f"manual_{tick}") / 100
    weights = pd.Series(weights_dict)
    # Normalize to sum to 1
    weights = weights / weights.sum()

    ret, vol, sr = portfolio_performance(weights.values, returns[tickers], returns[tickers].cov())
    cum_ret = (1 + (returns[tickers] * weights).sum(axis=1)).cumprod()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Return", f"{ret:.1%}", f"{ret - spy_ret:+.1%}")
    col2.metric("Volatility", f"{vol:.1%}")
    col3.metric("Sharpe Ratio", f"{sr:.2f}", f"{sr - spy_sr:+.2f}")
    col4.metric("vs S&P 500", "Your Pick" if sr > spy_sr else "Underperform")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Your Portfolio", line=dict(width=4)))
    if "SPY" in prices.columns:
        spy_cum = (1 + returns["SPY"]).cumprod()
        fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500", line=dict(color="gray", dash="dot")))
    fig.update_layout(title="$1 → Final Value", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Donut chart for manual weights
    fig_donut = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, hole=0.5, textinfo='label+percent')])
    fig_donut.update_layout(title="Your Portfolio Allocation", height=400)
    st.plotly_chart(fig_donut, use_container_width=True)

with tab2:
    st.success("Maximum Sharpe Ratio Portfolio Found!")
    
    # Comparison table
    comparison = pd.DataFrame({
        "Strategy": ["Max Sharpe", "Min Volatility", "Equal Weight", "S&P 500"],
        "Return": [max_sharpe_ret, min_vol_ret, eq_ret, spy_ret],
        "Volatility": [max_sharpe_vol, min_vol_vol, eq_vol, spy_vol],
        "Sharpe Ratio": [max_sharpe_sr, min_vol_sr, eq_sr, spy_sr]
    }).round(4)
    comparison["Return"] = (comparison["Return"]*100).round(2).astype(str) + "%"
    comparison["Volatility"] = (comparison["Volatility"]*100).round(2).astype(str) + "%"
    st.dataframe(comparison, use_container_width=True)

    # Show best weights
    best_weights = pd.Series(max_sharpe_weights.round(4), index=tickers)
    best_weights = best_weights[best_weights > 0.01]  # hide tiny weights
    st.bar_chart(best_weights * 100)

    # Download button
    csv = pd.Series(max_sharpe_weights, index=tickers).to_csv(header=["Weight"])
    st.download_button(
        "Download Max Sharpe Portfolio as CSV",
        data=csv.encode(),
        file_name="optimized_portfolio_max_sharpe.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Efficient Frontier — 10,000 Random Portfolios")

    np.random.seed(42)
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        w = np.random.random(len(tickers))
        w /= w.sum()
        weights_record.append(w)
        ret, vol, sr = portfolio_performance(w, returns[tickers], returns[tickers].cov())
        results[0,i] = ret
        results[1,i] = vol
        results[2,i] = sr

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[1,:], y=results[0,:],
        mode='markers',
        marker=dict(color=results[2,:], colorscale='Viridis', size=6,
                    colorbar=dict(title="Sharpe Ratio"), showscale=True),
        name="Random Portfolios",
        text=[f"Sharpe: {s:.2f}" for s in results[2,:]],
        hoverinfo="text"
    ))

    fig.add_trace(go.Scatter(x=[max_sharpe_vol], y=[max_sharpe_ret],
                             mode='markers', marker=dict(color='red', size=16, symbol='star'),
                             name=f"Max Sharpe (Best)"))
    fig.add_trace(go.Scatter(x=[min_vol_vol], y=[min_vol_ret],
                             mode='markers', marker=dict(color='lime', size=14, symbol='circle'),
                             name="Min Volatility"))
    if "SPY" in prices.columns:
        fig.add_trace(go.Scatter(x=[spy_vol], y=[spy_ret],
                                 mode='markers', marker=dict(color='white', size=12, symbol='x'),
                                 name="S&P 500"))

    fig.update_layout(
        title="Efficient Frontier — Higher Sharpe = Better",
        xaxis_title="Annual Risk (Volatility)",
        yaxis_title="Annual Return",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built by Braden Bourgeois • Masters in Analytics")
