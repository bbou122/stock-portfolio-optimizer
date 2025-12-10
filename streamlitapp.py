# streamlit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF

st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #1E3A8A, #3B82F6); padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 28px; font-weight: bold; margin-bottom: 30px; box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
Live Stock Portfolio Optimizer — 3 Years Real Market Data
</div>
""", unsafe_allow_html=True)

# Load data from your CSV 
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/bbou122/stock-portfolio-optimizer/main/stock_data.csv"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    return df

prices = load_data()
returns = prices.pct_change().dropna()

st.markdown("**Select any stocks from 10 major companies + S&P 500**")
tickers = st.multiselect("Choose your stocks", prices.columns.tolist(), default=["AAPL", "NVDA", "JPM"])

if tickers:
    # Allocation
    st.subheader("Portfolio Allocation")
    weights = {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            weights[t] = st.slider(t, 0, 100, round(100/len(tickers)), key=t) / 100

    total = sum(weights.values())
    weights = pd.Series(weights) / total if total > 0 else pd.Series({t:1/len(tickers) for t in tickers})

    # Portfolio math
    port_returns = (returns[tickers] * weights).sum(axis=1)
    cum_returns = (1 + port_returns).cumprod()

    annual_return = port_returns.mean() * 252
    annual_vol = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    max_drawdown = (cum_returns.cummax() - cum_returns).max()

    # S&P 500
    spy_cum = (1 + returns["SPY"]).cumprod()
    spy_annual = returns["SPY"].mean() * 252
    spy_sharpe = spy_annual / (returns["SPY"].std() * np.sqrt(252))

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return", f"{annual_return:.1%}", f"{annual_return - spy_annual:+.1%}")
    col2.metric("Volatility", f"{annual_vol:.1%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}", f"{sharpe - spy_sharpe:+.2f}")
    col4.metric("Max Drawdown", f"-{max_drawdown:.1%}")

    # Equity curve (Plotly)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, name="Your Portfolio", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Portfolio vs S&P 500", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Allocation pie (Plotly)
    fig_pie = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, hole=.5, textinfo='label+percent')])
    fig_pie.update_layout(title="Portfolio Allocation", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    # PDF Report — uses matplotlib 
    @st.cache_data
    def make_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Stock Portfolio Report", ln=1, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        # Save Plotly charts as PNG via matplotlib 
        fig.write_image("chart1.png")
        fig_pie.write_image("chart2.png")

        pdf.image("chart1.png", x=15, w=180)
        pdf.image("chart2.png", x=15, y=120, w=180)

        # Clean up
        for f in ["chart1.png", "chart2.png"]:
            if os.path.exists(f):
                os.remove(f)

        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            return f.read()

    st.download_button("Download PDF Report", make_pdf(), "stock_portfolio_report.pdf", "application/pdf")

else:
    st.info("Select stocks to begin")
    st.balloons()

st.markdown("---")
st.caption("Built quickly by Braden Bourgeois • Master’s in Analytics")

