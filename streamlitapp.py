# streamlit_app.py 
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #1E3A8A, #3B82F6); padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 28px; font-weight: bold; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
Live Stock Portfolio Optimizer — Real Prices • Real Risk • Real Alpha
</div>
""", unsafe_allow_html=True)

st.markdown("**Apple • Tesla • Nvidia • JPMorgan • S&P 500 • Instant analytics**")

tickers = st.multiselect(
    "Select stocks (or keep defaults)",
    ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "JPM", "V", "META", "BRK-B", "UNH"],
    default=["AAPL", "NVDA", "JPM"]
)

if tickers:
    with st.spinner("Fetching live market data..."):
        try:
            # Download with max safety
            raw = yf.download(tickers, period="3y", interval="1d", progress=False, threads=True, auto_adjust=False)
            
            if raw.empty:
                st.error("No data returned from Yahoo Finance. Try again in a minute.")
                st.stop()

            # Extract Adj Close — works for single or multiple tickers
            if len(tickers) == 1:
                if "Adj Close" in raw.columns:
                    prices = raw["Adj Close"].to_frame(name=tickers[0])
                elif "Close" in raw.columns:
                    prices = raw["Close"].to_frame(name=tickers[0])
                else:
                    st.error("No price data found for this stock.")
                    st.stop()
            else:
                if "Adj Close" in raw.columns:
                    prices = raw["Adj Close"]
                elif "Close" in raw.columns:
                    prices = raw["Close"]
                else:
                    st.error("No price data available.")
                    st.stop()

            prices = prices.dropna()
            if prices.empty:
                st.error("All data missing after cleaning.")
                st.stop()

        except Exception as e:
            st.error("Yahoo Finance temporarily unavailable. Please wait 30 seconds and refresh.")
            st.stop()

    # Now we KNOW we have clean price data
    returns = prices.pct_change().dropna()

    # Allocation
    st.subheader("Set Your Portfolio Weights")
    weights = {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            weights[t] = st.slider(t, 0, 100, round(100/len(tickers)), key=t) / 100

    total_w = sum(weights.values())
    weights = pd.Series(weights) / total_w if total_w > 0 else pd.Series({t: 1/len(tickers) for t in tickers})

    # Portfolio math
    port_returns = (returns * weights).sum(axis=1)
    cum_returns = (1 + port_returns).cumprod()

    annual_return = port_returns.mean() * 252
    annual_vol = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    max_drawdown = (cum_returns.cummax() - cum_returns).max()

    # S&P 500 benchmark (safe fallback)
    try:
        spy_raw = yf.download("SPY", period="3y", interval="1d", progress=False)
        spy_price = spy_raw["Adj Close" if "Adj Close" in spy_raw.columns else spy_raw["Close"]
        spy_returns = spy_price.pct_change().dropna()
        spy_cum = (1 + spy_returns).cumprod()
        spy_annual = spy_returns.mean() * 252
        spy_sharpe = spy_annual / (spy_returns.std() * np.sqrt(252))
    except:
        spy_annual = 0.10
        spy_sharpe = 0.8
        spy_cum = pd.Series([1.0], index=[prices.index[-1]])

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return", f"{annual_return:.1%}", f"{annual_return - spy_annual:+.1%}")
    col2.metric("Volatility", f"{annual_vol:.1%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}", f"{sharpe - spy_sharpe:+.2f}")
    col4.metric("Max Drawdown", f"-{max_drawdown:.1%}")

    # Charts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name="Your Portfolio", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Portfolio vs S&P 500 ($1 to ?)", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    fig_donut = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, hole=.5, textinfo='label+percent')])
    fig_donut.update_layout(title="Portfolio Allocation", height=450)
    st.plotly_chart(fig_donut, use_container_width=True)

    # PDF
    @st.cache_data
    def make_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 10, "Stock Portfolio Report", ln=1, align='C')
        pdf.image(fig.to_image(format="png"), x=15, w=180)
        pdf.image(fig_donut.to_image(format="png"), x=15, y=130, w=180)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            return f.read()

    st.download_button("Download PDF Report", make_pdf(), "stock_portfolio_report.pdf", "application/pdf")

else:
    st.info("Pick at least one stock to begin")
    st.balloons()

st.markdown("---")
st.caption("Built quickly by Braden Bourgeois • Master’s in Analytics")

