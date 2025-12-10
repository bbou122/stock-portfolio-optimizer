# streamlit_app.py – FINAL 
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")

# Beautiful banner
st.markdown("""
<div style="background: linear-gradient(90deg, #1E3A8A, #3B82F6); padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 28px; font-weight: bold; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
Live Stock Portfolio Optimizer — Real Prices • Real Risk • Real Alpha
</div>
""", unsafe_allow_html=True)

st.markdown("**Apple • Tesla • Nvidia • JPMorgan • S&P 500 • Instant analytics**")

# Stock selection
tickers = st.multiselect(
    "Select stocks (or keep defaults)",
    ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "JPM", "V", "META", "BRK-B", "UNH"],
    default=["AAPL", "NVDA", "JPM", "TSLA"]
)

if tickers:
    with st.spinner("Downloading latest market data..."):
        try:
            raw_data = yf.download(tickers, period="3y", interval="1d", progress=False)
            if raw_data.empty:
                raise ValueError("No data returned")
            
            # FIXED: Robust Adj Close extraction
            if len(tickers) == 1:
                data = pd.DataFrame(raw_data["Adj Close"]).rename(columns={"Adj Close": tickers[0]})
            else:
                data = raw_data["Adj Close"]
            data = data.dropna()
        except Exception as e:
            st.error(f"Data download failed: {e}. Try fewer stocks or later.")
            st.stop()
    
    returns = data.pct_change().dropna()
    
    # Allocation sliders
    st.subheader("Set Your Portfolio Weights")
    weights = {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            weights[t] = st.slider(t, 0, 100, 25, key=t) / 100
    
    total = sum(weights.values())
    if total == 0:
        st.error("Total allocation must be >0%")
        st.stop()
    weights = pd.Series(weights) / total
    
    # Portfolio math
    port_returns = (returns * weights).sum(axis=1)
    cum_returns = (1 + port_returns).cumprod()
    
    annual_return = port_returns.mean() * 252
    annual_vol = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    max_drawdown = (cum_returns.cummax() - cum_returns).max()

    # S&P 500 benchmark — SAFE
    try:
        spy_raw = yf.download("SPY", period="3y", interval="1d", progress=False)
        if spy_raw.empty or 'Adj Close' not in spy_raw.columns:
            raise ValueError("No S&P data")
        spy = spy_raw['Adj Close'].pct_change().dropna()
        spy_cum = (1 + spy).cumprod()
        spy_annual_return = spy.mean() * 252
        spy_sharpe = spy_annual_return / (spy.std() * np.sqrt(252))
    except:
        st.warning("S&P 500 data unavailable. Using fallback values.")
        spy_annual_return = 0.10
        spy_sharpe = 0.8
        spy_cum = pd.Series([1.0], index=[data.index[-1]] if not data.empty else [datetime.now()])

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return", f"{annual_return:.1%}", f"{annual_return - spy_annual_return:+.1%}")
    col2.metric("Volatility", f"{annual_vol:.1%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}", f"{sharpe - spy_sharpe:+.2f}")
    col4.metric("Max Drawdown", f"-{max_drawdown:.1%}")

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name="Your Portfolio", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Portfolio vs S&P 500 ($1 → ?)", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Allocation donut
    fig_donut = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, hole=.5, textinfo='label+percent')])
    fig_donut.update_layout(title="Portfolio Allocation", height=450)
    st.plotly_chart(fig_donut, use_container_width=True)

    # PDF Report
    @st.cache_data
    def make_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 10, "Stock Portfolio Report", ln=1, align='C')
        pdf.ln(10)
        pdf.image(fig.to_image(format="png"), x=15, w=180)
        pdf.image(fig_donut.to_image(format="png"), x=15, y=130, w=180)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            return f.read()

    st.download_button(
        "Download Full PDF Report",
        make_pdf(),
        "stock_portfolio_report.pdf",
        "application/pdf"
    )

else:
    st.info("Pick at least one stock to begin")
    st.balloons()

st.markdown("---")
st.caption("Built quickly by Braden Bourgeois • Master’s in Analytics")

