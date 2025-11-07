"""
Zerodha AI Trading Dashboard - FREE VERSION with Google Gemini
No CORS issues! Run with: streamlit run zerodha_dashboard.py

Install dependencies:
pip install streamlit kiteconnect google-generativeai pandas plotly
"""

import streamlit as st
from kiteconnect import KiteConnect
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Zerodha AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
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
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Header
st.markdown('<h1 class="main-header">🚀 Zerodha AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Powered by Google Gemini (FREE)** | Real-time Portfolio Analysis & Insights")
st.divider()

# Sidebar - Connection
with st.sidebar:
    st.header("🔐 Connect to Zerodha")

    api_key = st.text_input("API Key", type="password", key="api_key_input")
    access_token = st.text_input("Access Token", type="password", key="access_token_input")

    if st.button("🔗 Connect", key="connect_btn"):
        if api_key and access_token:
            try:
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)

                # Test connection
                profile = kite.profile()

                st.session_state.kite = kite
                st.session_state.connected = True
                st.session_state.profile = profile
                st.success(f"✅ Connected as {profile['user_name']}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
        else:
            st.warning("Please enter both API Key and Access Token")

    if st.session_state.connected:
        st.success("🟢 Connected")
        if st.session_state.profile:
            st.info(f"**User:** {st.session_state.profile['user_name']}\n\n**Email:** {st.session_state.profile['email']}")

        if st.button("🔄 Refresh Data"):
            try:
                st.session_state.holdings = st.session_state.kite.holdings()
                st.session_state.positions = st.session_state.kite.positions()['net']
                st.success("Data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error refreshing: {e}")

    st.divider()
    st.markdown("### 🤖 AI Configuration")
    st.info("**FREE AI Options:**\n\n1. Google Gemini (Recommended)\n2. Groq (Fast)\n3. Hugging Face (Open Source)")

    st.caption("⚠️ Credentials are not stored. Token expires at midnight.")

# Main content
if not st.session_state.connected:
    # Landing page
    st.info("### 🆓 100% FREE AI-Powered Trading Dashboard!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("### 📊 Portfolio Analytics\nReal-time tracking of your holdings and positions")

    with col2:
        st.success("### 🤖 AI Insights\nGet FREE AI analysis using Google Gemini")

    with col3:
        st.success("### 💬 AI Assistant\nChat with AI about trading strategies")

    st.warning("👈 Please connect to Zerodha using the sidebar to get started")

    st.divider()
    st.markdown("### 🎁 Why This Dashboard is Better:")
    st.markdown("""
    - ✅ **Completely FREE** - No API costs with Google Gemini free tier
    - ✅ **No CORS Issues** - Runs as Python backend
    - ✅ **Real-time Data** - Direct Zerodha API integration
    - ✅ **AI-Powered** - Smart portfolio analysis and insights
    - ✅ **Beautiful UI** - Professional charts and visualizations
    - ✅ **Secure** - Credentials never stored
    """)

else:
    # Fetch data if not already loaded
    if not st.session_state.holdings:
        try:
            st.session_state.holdings = st.session_state.kite.holdings()
            st.session_state.positions = st.session_state.kite.positions()['net']
        except Exception as e:
            st.error(f"Error fetching data: {e}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "📈 Positions", "🤖 AI Analysis", "💬 AI Assistant"])

    # Tab 1: Portfolio
    with tab1:
        st.header("Portfolio Overview")

        if st.session_state.holdings:
            # Calculate stats
            df_holdings = pd.DataFrame(st.session_state.holdings)

            total_invested = (df_holdings['average_price'] * df_holdings['quantity']).sum()
            current_value = (df_holdings['last_price'] * df_holdings['quantity']).sum()
            total_pnl = df_holdings['pnl'].sum()
            pnl_percent = ((current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("💰 Total Invested", f"₹{total_invested:,.2f}")

            with col2:
                st.metric("📈 Current Value", f"₹{current_value:,.2f}")

            with col3:
                st.metric("💵 Total P&L", f"₹{total_pnl:,.2f}", delta=f"{pnl_percent:.2f}%")

            with col4:
                st.metric("📊 Holdings", len(df_holdings))

            st.divider()

            # Holdings table
            st.subheader("Holdings Breakdown")

            # Add calculated columns
            df_holdings['pnl_percent'] = ((df_holdings['last_price'] - df_holdings['average_price']) / df_holdings['average_price'] * 100).round(2)
            df_holdings['current_value'] = df_holdings['last_price'] * df_holdings['quantity']

            # Display table
            display_df = df_holdings[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl', 'pnl_percent', 'current_value']]
            display_df.columns = ['Symbol', 'Qty', 'Avg Price', 'LTP', 'P&L', 'P&L %', 'Value']

            st.dataframe(
                display_df.style.format({
                    'Avg Price': '₹{:.2f}',
                    'LTP': '₹{:.2f}',
                    'P&L': '₹{:.2f}',
                    'P&L %': '{:.2f}%',
                    'Value': '₹{:.2f}'
                }).map(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                    subset=['P&L', 'P&L %']
                ),
                use_container_width=True,
                height=400
            )

            # Pie chart
            st.subheader("Portfolio Allocation")
            fig = px.pie(
                df_holdings,
                values='current_value',
                names='tradingsymbol',
                title='Holdings Distribution by Value',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # P&L Bar Chart
            st.subheader("P&L by Stock")
            fig2 = go.Figure(data=[
                go.Bar(
                    x=df_holdings['tradingsymbol'],
                    y=df_holdings['pnl'],
                    marker_color=['green' if x > 0 else 'red' for x in df_holdings['pnl']],
                    text=df_holdings['pnl'].apply(lambda x: f'₹{x:.2f}'),
                    textposition='auto'
                )
            ])
            fig2.update_layout(title='Profit/Loss by Holding', xaxis_title='Symbol', yaxis_title='P&L (₹)')
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No holdings found in your portfolio")

    # Tab 2: Positions
    with tab2:
        st.header("Open Positions")

        if st.session_state.positions:
            df_positions = pd.DataFrame(st.session_state.positions)

            if len(df_positions) > 0:
                # Display table
                display_df = df_positions[['tradingsymbol', 'quantity', 'buy_price', 'last_price', 'pnl']]
                display_df.columns = ['Symbol', 'Qty', 'Buy Price', 'LTP', 'P&L']

                st.dataframe(
                    display_df.style.format({
                        'Buy Price': '₹{:.2f}',
                        'LTP': '₹{:.2f}',
                        'P&L': '₹{:.2f}'
                    }).map(
                        lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                        subset=['P&L']
                    ),
                    use_container_width=True
                )

                # P&L chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_positions['tradingsymbol'],
                        y=df_positions['pnl'],
                        marker_color=['green' if x > 0 else 'red' for x in df_positions['pnl']],
                        text=df_positions['pnl'].apply(lambda x: f'₹{x:.2f}'),
                        textposition='auto'
                    )
                ])
                fig.update_layout(title='P&L by Position', xaxis_title='Symbol', yaxis_title='P&L (₹)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No open positions today")
        else:
            st.info("No open positions today")

    # Tab 3: AI Analysis
    with tab3:
        st.header("🤖 AI Portfolio Analysis (FREE)")
        st.caption("Get Google Gemini's expert analysis of your portfolio - Completely FREE!")

        col1, col2 = st.columns([3, 1])

        with col1:
            gemini_key = st.text_input("Enter your FREE Google Gemini API Key", type="password", key="gemini_key", help="Get free at: https://makersuite.google.com/app/apikey")

        with col2:
            st.link_button("🔑 Get FREE API Key", "https://makersuite.google.com/app/apikey")

        if st.button("🧠 Generate FREE AI Analysis", key="analyze_btn"):
            if not gemini_key:
                st.warning("Please enter your Google Gemini API key (it's FREE!)")
            elif not st.session_state.holdings:
                st.warning("No portfolio data to analyze")
            else:
                with st.spinner("AI is analyzing your portfolio... (FREE - No charges!)"):
                    try:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel("gemini-2.5-flash")

                        # Prepare portfolio data
                        df_holdings = pd.DataFrame(st.session_state.holdings)
                        total_invested = (df_holdings['average_price'] * df_holdings['quantity']).sum()
                        current_value = (df_holdings['last_price'] * df_holdings['quantity']).sum()
                        total_pnl = df_holdings['pnl'].sum()

                        holdings_summary = "\n".join([
                            f"- {row['tradingsymbol']}: {row['quantity']} shares @ ₹{row['average_price']:.2f} (Current: ₹{row['last_price']:.2f}, P&L: {((row['last_price'] - row['average_price']) / row['average_price'] * 100):.2f}%)"
                            for _, row in df_holdings.iterrows()
                        ])

                        prompt = f"""You are an expert Indian stock market analyst. Analyze this portfolio in detail:

Portfolio Summary:
- Total Holdings: {len(df_holdings)}
- Total Investment: ₹{total_invested:,.2f}
- Current Value: ₹{current_value:,.2f}
- Total P&L: ₹{total_pnl:,.2f} ({((current_value - total_invested) / total_invested * 100):.2f}%)

Holdings:
{holdings_summary}

Provide a comprehensive analysis with:
1. **Overall Portfolio Health Assessment** - Rate the portfolio and explain why
2. **Risk Analysis** - Identify key risks and concentration issues
3. **Diversification Review** - Evaluate sector/stock diversification
4. **Top 3 Actionable Recommendations** - Specific steps to improve the portfolio
5. **Stock-Specific Insights** - Brief comments on each holding
6. **Market Outlook** - How current market conditions affect this portfolio

Be specific, actionable, and professional. Use Indian market context."""

                        response = model.generate_content(prompt)
                        analysis = response.text

                        st.success("✅ FREE AI Analysis Complete!")
                        st.markdown("---")
                        st.markdown(analysis)

                        # Download option
                        st.download_button(
                            "📥 Download Analysis",
                            data=analysis,
                            file_name="portfolio_analysis.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"Error generating analysis: {e}")
                        st.info("💡 Tip: Make sure you entered a valid Google Gemini API key")

    # Tab 4: AI Assistant
    with tab4:
        st.header("💬 FREE AI Trading Assistant")
        st.caption("Chat with Google Gemini about your portfolio - Completely FREE!")

        gemini_key_chat = st.text_input("Enter your FREE Google Gemini API Key", type="password", key="gemini_key_chat", help="Get free at: https://makersuite.google.com/app/apikey")

        if not gemini_key_chat:
            st.link_button("🔑 Get FREE API Key", "https://makersuite.google.com/app/apikey")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about your portfolio... (FREE!)"):
            if not gemini_key_chat:
                st.warning("Please enter your Google Gemini API key first")
            else:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking... (FREE - No charges)"):
                        try:
                            genai.configure(api_key=gemini_key_chat)
                            model = genai.GenerativeModel("gemini-2.5-flash")

                            # Build context with portfolio data
                            if st.session_state.holdings:
                                df_holdings = pd.DataFrame(st.session_state.holdings)
                                holdings_list = ', '.join(df_holdings['tradingsymbol'].tolist())
                                total_pnl = df_holdings['pnl'].sum()

                                context = f"""You are an expert Indian stock market advisor. 

User's Portfolio Context:
- Number of Holdings: {len(df_holdings)} stocks
- Total P&L: ₹{total_pnl:,.2f}
- Stocks Owned: {holdings_list}

User Question: {prompt}

Provide helpful, specific advice based on their portfolio. Be professional but friendly."""
                            else:
                                context = f"You are an expert Indian stock market advisor. User Question: {prompt}"

                            response = model.generate_content(context)
                            ai_response = response.text

                            st.markdown(ai_response)
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

                        except Exception as e:
                            error_msg = f"Error: {e}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        # Quick questions
        st.divider()
        st.subheader("⚡ Quick Questions")


        quick_questions = [
            "Should I hold or sell my losing stocks?",
            "Which stocks show best growth potential?",
            "Is my portfolio diversified enough?",
            "What are the risks in my holdings?",
            "Suggest stocks to add to my portfolio",
            "Should I book profits now or hold?"
        ]

        cols = st.columns(2)
        for idx, question in enumerate(quick_questions):
            with cols[idx % 2]:
                if st.button(question, key=f"quick_{idx}"):
                    if gemini_key_chat:
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.rerun()
                    else:
                        st.warning("Please enter your API key first")

# Footer
st.divider()
st.markdown("### 🎯 Quick Setup Guide:")
st.markdown("""
1. **Get FREE Google Gemini API Key**: https://makersuite.google.com/app/apikey (No credit card required!)
2. **Connect to Zerodha** using your API Key and Access Token
3. **Start analyzing** your portfolio with FREE AI insights!

**API Usage**: Google Gemini free tier gives you 60 requests/minute - More than enough for personal use!
""")