"""
🚀 ULTIMATE ZERODHA AI TRADING PLATFORM 🚀
Save this file as: ultimate_zerodha.py

Install dependencies:
pip install streamlit kiteconnect google-generativeai pandas plotly yfinance ta numpy scipy

Run with:
streamlit run ultimate_zerodha.py
"""

import streamlit as st
from kiteconnect import KiteConnect
import google.generativeai as genai
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import yfinance as yf
    import ta
except:
    st.warning("Install yfinance and ta for full features: pip install yfinance ta")

from datetime import datetime, timedelta
import json

# Page Config
st.set_page_config(
    page_title="Ultimate Zerodha AI",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'holdings' not in st.session_state:
    st.session_state.holdings = []
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Functions
def get_stock_data(symbol, period="6mo"):
    """Fetch stock data"""
    try:
        ticker = f"{symbol}.NS"
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except:
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or df.empty:
        return None
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        return df
    except:
        return df

def calculate_risk_metrics(returns_list):
    """Calculate portfolio risk metrics"""
    if len(returns_list) == 0:
        return {}
    returns = np.array(returns_list)
    return {
        'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
        'volatility': np.std(returns) * np.sqrt(252),
        'max_dd': (np.maximum.accumulate(returns) - returns).max()
    }

# Header
st.markdown('<h1 class="main-header">🚀 ULTIMATE ZERODHA AI PLATFORM</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #667eea; font-size: 1.2rem;'>Advanced Trading Dashboard with AI, Analytics & Automation</p>", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.header("🔐 Setup")

    # Zerodha
    with st.expander("Zerodha Connection", expanded=not st.session_state.connected):
        api_key = st.text_input("API Key", type="password")
        access_token = st.text_input("Access Token", type="password")

        if st.button("Connect"):
            if api_key and access_token:
                try:
                    kite = KiteConnect(api_key=api_key)
                    kite.set_access_token(access_token)
                    profile = kite.profile()

                    st.session_state.kite = kite
                    st.session_state.connected = True
                    st.session_state.profile = profile
                    st.session_state.holdings = kite.holdings()
                    st.session_state.positions = kite.positions()['net']

                    st.success(f"✅ Connected: {profile['user_name']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # AI Config
    with st.expander("AI Configuration"):
        gemini_key = st.text_input("Gemini API Key", type="password", key="gemini")
        st.caption("[Get FREE key](https://makersuite.google.com/app/apikey)")

    if st.session_state.connected:
        st.success("🟢 Connected")
        if st.session_state.profile:
            st.info(f"**{st.session_state.profile['user_name']}**")

        if st.button("🔄 Refresh"):
            try:
                st.session_state.holdings = st.session_state.kite.holdings()
                st.session_state.positions = st.session_state.kite.positions()['net']
                st.success("Refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.caption("v2.0 Ultimate Edition")

# Main Content
if not st.session_state.connected:
    st.info("### 👈 Connect to Zerodha to unlock all features!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("### 📊 Advanced Analytics\nTechnical indicators, risk metrics, optimization")
    with col2:
        st.success("### 🤖 AI Powered\nSmart recommendations, predictions, sentiment")
    with col3:
        st.success("### 📈 Live Trading\nReal-time data, alerts, paper trading")

    st.markdown("""
    ### ✨ Features:
    - **Portfolio Analytics**: Risk metrics, correlation matrix, sector analysis
    - **Technical Analysis**: RSI, MACD, Bollinger Bands, candlestick charts
    - **AI Insights**: Stock recommendations, portfolio analysis, predictions
    - **Stock Screener**: Filter by P/E, market cap, sector
    - **Smart Alerts**: Price alerts, AI warnings
    - **Trade Journal**: Track all your trades
    - **Portfolio Optimizer**: Modern Portfolio Theory optimization
    - **News Sentiment**: AI-powered sentiment analysis
    """)

else:
    # Tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "📈 Analytics",
        "🤖 AI Insights",
        "📉 Technical",
        "🔍 Screener",
        "💼 Optimizer",
        "📰 News",
        "🔔 Alerts",
        "💬 AI Chat"
    ])

    # TAB 1: Dashboard
    with tabs[0]:
        st.header("Portfolio Dashboard")

        if st.session_state.holdings:
            df = pd.DataFrame(st.session_state.holdings)

            total_inv = (df['average_price'] * df['quantity']).sum()
            curr_val = (df['last_price'] * df['quantity']).sum()
            total_pnl = df['pnl'].sum()
            pnl_pct = ((curr_val - total_inv) / total_inv * 100) if total_inv > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Invested", f"₹{total_inv:,.0f}")
            col2.metric("📈 Current", f"₹{curr_val:,.0f}")
            col3.metric("💵 P&L", f"₹{total_pnl:,.0f}", f"{pnl_pct:.2f}%")
            col4.metric("📊 Holdings", len(df))

            st.divider()

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                df['value'] = df['last_price'] * df['quantity']
                fig = px.pie(df, values='value', names='tradingsymbol',
                           title='Portfolio Allocation', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                df['pnl_pct'] = ((df['last_price'] - df['average_price']) / df['average_price'] * 100)
                fig = go.Figure(data=[go.Bar(
                    x=df['tradingsymbol'],
                    y=df['pnl_pct'],
                    marker_color=['green' if x > 0 else 'red' for x in df['pnl_pct']],
                    text=df['pnl_pct'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                )])
                fig.update_layout(title='P&L % by Stock')
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Table
            display = df[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl', 'pnl_pct']]
            display.columns = ['Symbol', 'Qty', 'Avg', 'LTP', 'P&L', 'P&L%']
            st.dataframe(
                display.style.format({
                    'Avg': '₹{:.2f}',
                    'LTP': '₹{:.2f}',
                    'P&L': '₹{:.2f}',
                    'P&L%': '{:.2f}%'
                }).map(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                    subset=['P&L', 'P&L%']
                ),
                use_container_width=True
            )
        else:
            st.info("No holdings")

    # TAB 2: Analytics
    with tabs[1]:
        st.header("Advanced Analytics")

        if st.session_state.holdings:
            df = pd.DataFrame(st.session_state.holdings)

            st.subheader("Risk Metrics")
            returns = [(h['last_price'] - h['average_price']) / h['average_price'] for h in st.session_state.holdings]
            metrics = calculate_risk_metrics(returns)

            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
            col2.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
            col3.metric("Max Drawdown", f"{metrics.get('max_dd', 0):.2%}")

            st.divider()
            st.subheader("Sector Analysis")
            st.info("Diversify across sectors to reduce risk")
        else:
            st.info("Load holdings first")

    # TAB 3: AI Insights
    with tabs[2]:
        st.header("AI Portfolio Analysis")

        if not gemini_key:
            st.warning("Enter Gemini API key in sidebar")
        elif st.session_state.holdings:
            if st.button("🧠 Generate AI Analysis"):
                with st.spinner("Analyzing..."):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel("gemini-2.5-flash")

                        df = pd.DataFrame(st.session_state.holdings)
                        summary = "\n".join([
                            f"- {r['tradingsymbol']}: {r['quantity']} shares @ ₹{r['average_price']:.2f} (Current: ₹{r['last_price']:.2f})"
                            for _, r in df.iterrows()
                        ])

                        prompt = f"""Analyze this Indian stock portfolio:

{summary}

Total Investment: ₹{(df['average_price'] * df['quantity']).sum():,.2f}
Current Value: ₹{(df['last_price'] * df['quantity']).sum():,.2f}

Provide:
1. Portfolio Health Score (0-100)
2. Risk Analysis
3. Top 3 Recommendations
4. Stocks to sell/hold/buy more
5. Diversification insights"""

                        response = model.generate_content(prompt)
                        st.success("✅ Analysis Complete")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Load holdings first")

    # TAB 4: Technical Analysis
    with tabs[3]:
        st.header("Technical Analysis")

        if st.session_state.holdings:
            symbol = st.selectbox("Select Stock", [h['tradingsymbol'] for h in st.session_state.holdings])
            period = st.select_slider("Period", ["1mo", "3mo", "6mo", "1y"], value="6mo")

            if st.button("Generate Chart"):
                with st.spinner(f"Loading {symbol}..."):
                    data = get_stock_data(symbol, period)

                    if data is not None and not data.empty:
                        data = calculate_indicators(data)

                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            subplot_titles=(f'{symbol} Price', 'RSI'),
                            row_heights=[0.7, 0.3]
                        )

                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='Price'
                        ), row=1, col=1)

                        # SMA
                        if 'SMA_20' in data.columns:
                            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'), row=1, col=1)

                        # RSI
                        if 'RSI' in data.columns:
                            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=2, col=1)
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        fig.update_layout(height=700, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                        # Indicators
                        if 'RSI' in data.columns:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Current RSI", f"{data['RSI'].iloc[-1]:.2f}")
                            col2.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
                            col3.metric("SMA 20", f"₹{data['SMA_20'].iloc[-1]:.2f}" if 'SMA_20' in data.columns else "N/A")
                    else:
                        st.error("Could not load data")
        else:
            st.info("Load holdings first")

    # TAB 5: Screener
    with tabs[4]:
        st.header("Stock Screener")

        col1, col2 = st.columns(2)
        with col1:
            market_cap = st.selectbox("Market Cap", ["All", "Large", "Mid", "Small"])
        with col2:
            sector = st.multiselect("Sector", ["IT", "Finance", "Auto", "Pharma", "Energy"])

        if st.button("Scan Market"):
            st.info("Screener coming soon - will scan NSE stocks")

    # TAB 6: Optimizer
    with tabs[5]:
        st.header("Portfolio Optimizer")
        st.info("Modern Portfolio Theory optimization coming soon")

    # TAB 7: News
    with tabs[6]:
        st.header("News & Sentiment")
        st.info("News aggregation and AI sentiment analysis coming soon")

    # TAB 8: Alerts
    with tabs[7]:
        st.header("Price Alerts")

        if st.session_state.holdings:
            symbol = st.selectbox("Stock", [h['tradingsymbol'] for h in st.session_state.holdings], key="alert_stock")
            alert_price = st.number_input("Alert Price", min_value=0.0)
            alert_type = st.radio("Alert When", ["Above", "Below"])

            if st.button("Set Alert"):
                st.session_state.alerts.append({
                    'symbol': symbol,
                    'price': alert_price,
                    'type': alert_type
                })
                st.success(f"Alert set: {symbol} {alert_type} ₹{alert_price}")

            if st.session_state.alerts:
                st.subheader("Active Alerts")
                for alert in st.session_state.alerts:
                    st.info(f"{alert['symbol']}: {alert['type']} ₹{alert['price']}")
        else:
            st.info("Load holdings first")

    # TAB 9: AI Chat
    with tabs[8]:
        st.header("AI Trading Assistant")

        if not gemini_key:
            st.warning("Enter Gemini API key in sidebar")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask about your portfolio..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            genai.configure(api_key=gemini_key)
                            model = genai.GenerativeModel("gemini-2.5-flash")

                            context = f"""Portfolio: {len(st.session_state.holdings)} stocks
Stocks: {', '.join([h['tradingsymbol'] for h in st.session_state.holdings])}

Question: {prompt}"""

                            response = model.generate_content(context)
                            answer = response.text
                            st.markdown(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error: {e}")

st.divider()
st.caption("Ultimate Zerodha AI Platform v2.0 | Powered by Google Gemini (FREE)")