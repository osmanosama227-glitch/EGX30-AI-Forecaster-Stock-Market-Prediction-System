import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="EGX30 AI Forecaster",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# MODERN UI
# =========================================================
st.markdown("""
<style>

.stApp {
    background: #050816;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0b1023;
    border-right: 1px solid #1f2937;
}

/* HERO */
.hero {
    background: linear-gradient(145deg,#0f172a,#111827);
    border: 1px solid #1f2937;
    padding: 35px;
    border-radius: 24px;
    margin-bottom: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.hero-sub {
    color: #94a3b8;
    font-size: 17px;
    margin-top: 8px;
}

/* BUTTONS */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* METRICS */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 18px;
    border-radius: 18px;
}

/* CHARTS */
.js-plotly-plot {
    border-radius: 20px;
    overflow: hidden;
}

/* TABLE */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #1f2937;
}

/* SELECTBOX */
div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    border-radius: 12px !important;
    border: 1px solid #374151 !important;
}

/* TABS */
button[data-baseweb="tab"] {
    background: #111827 !important;
    border-radius: 12px !important;
    margin-right: 10px;
}

button[aria-selected="true"] {
    background: linear-gradient(90deg,#2563eb,#06b6d4) !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">
        📈 EGX30 AI FORECASTER
    </div>

    <div class="hero-sub">
        12-Month Egyptian Stock Market Prediction System
        using Monte Carlo Simulation & GBM
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# UPDATED EGX30
# =========================================================
EGX30 = {

    "COMI.CA": "Commercial International Bank",
    "TMGH.CA": "Talaat Moustafa Group",
    "EAST.CA": "Eastern Company",
    "FWRY.CA": "Fawry",
    "HRHO.CA": "EFG Holding",
    "EFIH.CA": "eFinance Investment Group",
    "ETEL.CA": "Telecom Egypt",
    "BTFH.CA": "Beltone Holding",
    "ABUK.CA": "Abu Qir Fertilizers",
    "PHDC.CA": "Palm Hills Development",
    "OCDI.CA": "Orascom Development Egypt",
    "HELI.CA": "Heliopolis Housing",
    "MNHD.CA": "Madinet Masr Housing",
    "ESRS.CA": "Ezz Steel",
    "SWDY.CA": "Elsewedy Electric",
    "AMOC.CA": "Alexandria Mineral Oils",
    "ALCN.CA": "Alexandria Containers",
    "EKHO.CA": "Egypt Kuwait Holding",
    "JUFO.CA": "Juhayna Food Industries",
    "ORWE.CA": "Oriental Weavers",
    "AUTO.CA": "GB Corp",
    "CNFN.CA": "Contact Financial",
    "CICH.CA": "CI Capital Holding",
    "EFID.CA": "Edita Food Industries",
    "EGAL.CA": "Egypt Aluminum"

}

# =========================================================
# FILTER WORKING TICKERS
# =========================================================
@st.cache_data
def get_working_tickers():

    working = []

    for ticker in EGX30.keys():

        try:
            test = yf.download(
                ticker,
                period="5d",
                progress=False
            )

            if not test.empty:
                working.append(ticker)

        except:
            pass

    return working

working_tickers = get_working_tickers()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.selectbox(
    "Choose EGX30 Stock",
    options=working_tickers,
    format_func=lambda x: f"{x} — {EGX30[x]}"
)

start_date = st.sidebar.date_input(
    "Training Start",
    date(2022, 1, 1)
)

end_date = st.sidebar.date_input(
    "Training End",
    date.today()
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Trading Days)",
    30,
    252,
    252
)

paths = st.sidebar.select_slider(
    "Monte Carlo Paths",
    options=[100, 500, 1000, 2000],
    value=1000
)

run = st.sidebar.button("🚀 Run Forecast")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(ticker, start, end):

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    return close.dropna()

# =========================================================
# GBM MODEL
# =========================================================
def gbm(prices, horizon=252, n_paths=1000):

    returns = np.log(prices / prices.shift(1)).dropna()

    mu = returns.mean()
    sigma = returns.std()

    S0 = prices.iloc[-1]

    simulations = np.zeros((horizon, n_paths))

    for i in range(n_paths):

        path = [S0]

        for _ in range(horizon - 1):

            shock = np.random.normal(mu, sigma)

            next_price = path[-1] * np.exp(shock)

            path.append(next_price)

        simulations[:, i] = path

    return {
        "mean": simulations.mean(axis=1),
        "p5": np.percentile(simulations, 5, axis=1),
        "p95": np.percentile(simulations, 95, axis=1),
        "paths": simulations,
        "mu": mu,
        "sigma": sigma
    }

# =========================================================
# MAIN
# =========================================================
if run:

    with st.spinner("Loading stock data..."):

        prices = load_data(
            ticker,
            start_date,
            end_date
        )

    if len(prices) < 50:
        st.error("Not enough data available.")
        st.stop()

    result = gbm(
        prices,
        forecast_days,
        paths
    )

    # =====================================================
    # FUTURE DATES
    # =====================================================
    future_dates = pd.bdate_range(
        start=prices.index[-1] + pd.offsets.BDay(1),
        periods=forecast_days
    )

    # =====================================================
    # METRICS
    # =====================================================
    current_price = prices.iloc[-1]

    expected_price = result["mean"][-1]

    expected_return = (
        (expected_price - current_price)
        / current_price
    ) * 100

    volatility = result["sigma"] * np.sqrt(252) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"{current_price:.2f} EGP"
    )

    c2.metric(
        "Forecast Price",
        f"{expected_price:.2f} EGP"
    )

    c3.metric(
        "Expected Return",
        f"{expected_return:.2f}%"
    )

    c4.metric(
        "Annual Volatility",
        f"{volatility:.2f}%"
    )

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "📈 Forecast",
        "📊 Statistics",
        "📋 Forecast Table"
    ])

    # =====================================================
    # FORECAST TAB
    # =====================================================
    with tab1:

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            name="Historical",
            line=dict(
                width=2
            )
        ))

        # Forecast Mean
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=result["mean"],
            name="Forecast",
            line=dict(
                width=3
            )
        ))

        # Upper Band
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=result["p95"],
            line=dict(width=0),
            showlegend=False
        ))

        # Lower Band
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=result["p5"],
            fill="tonexty",
            name="95% Confidence Interval"
        ))

        fig.update_layout(

            template="plotly_dark",

            title=f"{ticker} — {EGX30[ticker]}",

            paper_bgcolor="#050816",
            plot_bgcolor="#050816",

            height=700,

            xaxis=dict(
                gridcolor="rgba(255,255,255,0.05)"
            ),

            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # STATISTICS TAB
    # =====================================================
    with tab2:

        returns = np.log(
            prices / prices.shift(1)
        ).dropna()

        st.subheader("📊 Training Statistics")

        stats_df = pd.DataFrame({

            "Metric": [
                "Trading Days",
                "Daily Mean Return",
                "Daily Volatility",
                "Annualized Return",
                "Annualized Volatility"
            ],

            "Value": [
                len(prices),
                f"{result['mu']:.5f}",
                f"{result['sigma']:.5f}",
                f"{result['mu'] * 252:.2%}",
                f"{result['sigma'] * np.sqrt(252):.2%}"
            ]
        })

        st.dataframe(
            stats_df,
            use_container_width=True
        )

        st.subheader("📉 Return Distribution")

        hist_fig = go.Figure()

        hist_fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50
        ))

        hist_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#050816",
            plot_bgcolor="#050816",
            height=450
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

    # =====================================================
    # TABLE TAB
    # =====================================================
    with tab3:

        df = pd.DataFrame({

            "Date": future_dates,

            "Forecast Price": result["mean"],

            "Low Estimate": result["p5"],

            "High Estimate": result["p95"]

        })

        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )

else:

    st.info(
        "Choose settings from the sidebar and click Run Forecast 🚀"
    )