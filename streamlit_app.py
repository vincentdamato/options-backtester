"""
Credit Spread Catalyst Scanner
==============================

Strategic tool for deploying credit spreads around news catalysts.
Focus: High probability income strategies with defined risk.

Strategies:
- Bull Put Spreads (bullish)
- Bear Call Spreads (bearish)  
- Iron Condors (neutral)

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtester import OptionsBacktester
from data_fetcher import DataPreparator

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Credit Spread Scanner",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }
    
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .header p { margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.9rem; }
    
    .strategy-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.2s;
        height: 100%;
    }
    .strategy-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    .strategy-card.selected {
        border-color: #6366f1;
        background: #f5f3ff;
    }
    .strategy-name { font-size: 1.1rem; font-weight: 700; color: #111827; }
    .strategy-desc { font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem; }
    .strategy-bias { 
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .bias-bullish { background: #dcfce7; color: #166534; }
    .bias-bearish { background: #fee2e2; color: #991b1b; }
    .bias-neutral { background: #e0e7ff; color: #3730a3; }
    
    .setup-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .setup-title { font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem; }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-label { color: #6b7280; font-size: 0.85rem; }
    .metric-value { font-weight: 600; color: #111827; font-size: 0.9rem; }
    .metric-value.positive { color: #059669; }
    .metric-value.negative { color: #dc2626; }
    
    .trade-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .trade-card h3 { margin: 0 0 1rem 0; font-size: 1rem; opacity: 0.9; }
    .trade-details { font-size: 1.1rem; font-weight: 600; }
    
    .risk-meter {
        height: 8px;
        background: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .risk-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    
    .catalyst-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .catalyst-earnings { background: #fef3c7; color: #92400e; }
    .catalyst-fda { background: #dbeafe; color: #1e40af; }
    .catalyst-news { background: #f3e8ff; color: #6b21a8; }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.9; }
    
    .prob-circle {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        font-weight: 700;
        color: white;
        margin: 0 auto;
    }
    .prob-high { background: linear-gradient(135deg, #059669, #10b981); }
    .prob-med { background: linear-gradient(135deg, #d97706, #f59e0b); }
    .prob-low { background: linear-gradient(135deg, #dc2626, #ef4444); }
    
    .ticker-input input {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_data(ticker: str):
    try:
        prep = DataPreparator()
        data = prep.prepare_backtest_data(ticker, period="1y")
        if data.empty or len(data) < 50:
            return None
        return data
    except:
        return None


def calculate_credit_spread(data, spread_type, width=5, dte=30, delta=0.16):
    """Calculate credit spread parameters"""
    if data is None:
        return None
    
    price = data['close'].iloc[-1]
    iv = data['iv'].iloc[-1]
    vix = data['vix'].iloc[-1]
    
    # IV Rank
    iv_rank = (data['iv'] < iv).mean() * 100
    
    # Historical volatility
    hv_20 = data['close'].pct_change().tail(20).std() * np.sqrt(252)
    
    # RSI
    delta_price = data['close'].diff()
    gain = delta_price.where(delta_price > 0, 0).rolling(14).mean()
    loss = (-delta_price.where(delta_price < 0, 0)).rolling(14).mean()
    rsi = (100 - (100 / (1 + gain/loss))).iloc[-1]
    
    # Calculate strikes based on delta
    # Approximate: delta 0.16 ≈ 1 standard deviation move
    std_move = price * iv * np.sqrt(dte / 365)
    
    if spread_type == "bull_put":
        # Bull Put Spread: Sell put, buy lower put
        short_strike = round((price - std_move) / 5) * 5  # Round to $5
        long_strike = short_strike - width
        
        # Estimate credit (simplified)
        credit = width * 0.30 * (iv_rank / 100 + 0.5)  # Higher IV = more credit
        
    elif spread_type == "bear_call":
        # Bear Call Spread: Sell call, buy higher call
        short_strike = round((price + std_move) / 5) * 5
        long_strike = short_strike + width
        
        credit = width * 0.30 * (iv_rank / 100 + 0.5)
        
    elif spread_type == "iron_condor":
        # Iron Condor: Both spreads
        put_short = round((price - std_move) / 5) * 5
        put_long = put_short - width
        call_short = round((price + std_move) / 5) * 5
        call_long = call_short + width
        
        credit = width * 0.50 * (iv_rank / 100 + 0.5)
        
        return {
            'type': 'iron_condor',
            'price': price,
            'put_short': put_short,
            'put_long': put_long,
            'call_short': call_short,
            'call_long': call_long,
            'width': width,
            'credit': credit,
            'max_loss': width - credit,
            'breakeven_low': put_short - credit,
            'breakeven_high': call_short + credit,
            'iv_rank': iv_rank,
            'iv': iv * 100,
            'hv_20': hv_20 * 100,
            'rsi': rsi,
            'vix': vix,
            'prob_profit': min(85, 50 + iv_rank * 0.3 + (50 - abs(rsi - 50)) * 0.2),
            'dte': dte
        }
    
    # For single spreads
    max_loss = width - credit
    prob_profit = min(85, 50 + iv_rank * 0.3)
    
    return {
        'type': spread_type,
        'price': price,
        'short_strike': short_strike,
        'long_strike': long_strike,
        'width': width,
        'credit': credit,
        'max_loss': max_loss,
        'breakeven': short_strike - credit if spread_type == "bull_put" else short_strike + credit,
        'risk_reward': credit / max_loss if max_loss > 0 else 0,
        'iv_rank': iv_rank,
        'iv': iv * 100,
        'hv_20': hv_20 * 100,
        'rsi': rsi,
        'vix': vix,
        'prob_profit': prob_profit,
        'dte': dte
    }


def run_backtest(data, spread_type):
    """Run backtest for the strategy"""
    if data is None:
        return None
    
    try:
        bt = OptionsBacktester(data)
        
        if spread_type == "bull_put":
            result = bt.backtest_short_put(delta_target=-0.16, dte_target=30, profit_target=0.5, stop_loss=2.0)
        elif spread_type == "bear_call":
            result = bt.backtest_short_put(delta_target=-0.16, dte_target=30, profit_target=0.5, stop_loss=2.0)
        else:  # iron_condor
            result = bt.backtest_iron_condor(put_delta=-0.16, call_delta=0.16, dte_target=30, profit_target=0.5, stop_loss=2.0)
        
        return {
            'win_rate': result.win_rate * 100,
            'profit_factor': result.profit_factor,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'num_trades': result.num_trades,
            'total_return': result.total_return * 100
        }
    except:
        return None


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="header">
    <h1>💰 Credit Spread Catalyst Scanner</h1>
    <p>Deploy high-probability credit spreads around news catalysts • Defined risk, consistent income</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# STRATEGY SELECTION
# =============================================================================

st.markdown("### 1️⃣ Select Your Strategy")

col1, col2, col3 = st.columns(3)

with col1:
    bull_put = st.button("🐂 Bull Put Spread", use_container_width=True, key="bull")
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-name">Bull Put Spread</div>
        <div class="strategy-desc">Sell put spread below support. Profit if stock stays above short strike.</div>
        <div class="strategy-bias bias-bullish">BULLISH</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    bear_call = st.button("🐻 Bear Call Spread", use_container_width=True, key="bear")
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-name">Bear Call Spread</div>
        <div class="strategy-desc">Sell call spread above resistance. Profit if stock stays below short strike.</div>
        <div class="strategy-bias bias-bearish">BEARISH</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    iron_condor = st.button("🦅 Iron Condor", use_container_width=True, key="condor")
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-name">Iron Condor</div>
        <div class="strategy-desc">Sell both spreads. Profit if stock stays in range. Double premium.</div>
        <div class="strategy-bias bias-neutral">NEUTRAL</div>
    </div>
    """, unsafe_allow_html=True)

# Track selected strategy
if 'strategy' not in st.session_state:
    st.session_state.strategy = 'bull_put'

if bull_put:
    st.session_state.strategy = 'bull_put'
elif bear_call:
    st.session_state.strategy = 'bear_call'
elif iron_condor:
    st.session_state.strategy = 'iron_condor'

strategy = st.session_state.strategy


# =============================================================================
# TICKER & PARAMETERS
# =============================================================================

st.markdown("---")
st.markdown("### 2️⃣ Enter Ticker & Parameters")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    ticker = st.text_input("Ticker Symbol", value="SPY", key="ticker").upper()

with col2:
    dte = st.selectbox("Days to Expiration", [7, 14, 21, 30, 45, 60], index=3)

with col3:
    width = st.selectbox("Spread Width", [2.5, 5, 10, 15, 20], index=1, format_func=lambda x: f"${x:.0f}" if x >= 1 else f"${x}")

with col4:
    contracts = st.number_input("# Contracts", min_value=1, max_value=100, value=1)


# =============================================================================
# ANALYSIS
# =============================================================================

analyze_btn = st.button("🔍 Analyze Trade", type="primary", use_container_width=True)

if analyze_btn or 'last_ticker' in st.session_state:
    st.session_state.last_ticker = ticker
    
    with st.spinner(f"Analyzing {ticker}..."):
        data = get_stock_data(ticker)
    
    if data is None:
        st.error(f"Could not fetch data for {ticker}")
    else:
        spread = calculate_credit_spread(data, strategy, width=width, dte=dte)
        backtest = run_backtest(data, strategy)
        
        st.markdown("---")
        st.markdown("### 3️⃣ Trade Setup")
        
        # Main layout
        left, right = st.columns([1.5, 1])
        
        with left:
            # Current conditions
            st.markdown('<div class="setup-box">', unsafe_allow_html=True)
            st.markdown('<div class="setup-title">📊 Current Market Conditions</div>', unsafe_allow_html=True)
            
            cond1, cond2, cond3, cond4 = st.columns(4)
            with cond1:
                st.metric("Price", f"${spread['price']:.2f}")
            with cond2:
                iv_color = "🟢" if spread['iv_rank'] > 50 else "🟡" if spread['iv_rank'] > 30 else "🔴"
                st.metric("IV Rank", f"{spread['iv_rank']:.0f}% {iv_color}")
            with cond3:
                st.metric("RSI", f"{spread['rsi']:.0f}")
            with cond4:
                st.metric("VIX", f"{spread['vix']:.1f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # The Trade
            st.markdown('<div class="setup-box">', unsafe_allow_html=True)
            st.markdown('<div class="setup-title">📝 Your Trade</div>', unsafe_allow_html=True)
            
            strategy_names = {
                'bull_put': '🐂 Bull Put Spread',
                'bear_call': '🐻 Bear Call Spread',
                'iron_condor': '🦅 Iron Condor'
            }
            
            st.markdown(f"**Strategy:** {strategy_names[strategy]}")
            st.markdown(f"**Expiration:** {dte} DTE")
            
            if strategy == 'iron_condor':
                st.markdown(f"""
                **Put Side:** SELL ${spread['put_short']:.0f} / BUY ${spread['put_long']:.0f}
                
                **Call Side:** SELL ${spread['call_short']:.0f} / BUY ${spread['call_long']:.0f}
                """)
            else:
                action = "SELL" if "short" in strategy else "BUY"
                st.markdown(f"""
                **SELL:** ${spread['short_strike']:.0f} strike
                
                **BUY:** ${spread['long_strike']:.0f} strike
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # P&L Analysis
            st.markdown('<div class="setup-box">', unsafe_allow_html=True)
            st.markdown('<div class="setup-title">💵 P&L Analysis</div>', unsafe_allow_html=True)
            
            credit_total = spread['credit'] * contracts * 100
            max_loss_total = spread['max_loss'] * contracts * 100
            
            st.markdown(f"""
            <div class="metric-row">
                <span class="metric-label">Credit Received</span>
                <span class="metric-value positive">${credit_total:,.0f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Max Loss</span>
                <span class="metric-value negative">${max_loss_total:,.0f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Risk/Reward</span>
                <span class="metric-value">{spread['credit']/spread['max_loss']:.1%}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Breakeven</span>
                <span class="metric-value">${spread.get('breakeven', spread.get('breakeven_low', 0)):.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with right:
            # Probability
            prob = spread['prob_profit']
            prob_class = "prob-high" if prob >= 70 else "prob-med" if prob >= 55 else "prob-low"
            
            st.markdown(f"""
            <div class="setup-box" style="text-align: center;">
                <div class="setup-title">🎯 Probability of Profit</div>
                <div class="prob-circle {prob_class}">{prob:.0f}%</div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #6b7280;">
                    Based on IV rank and historical patterns
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Backtest Results
            if backtest:
                st.markdown(f"""
                <div class="setup-box">
                    <div class="setup-title">📈 Historical Performance</div>
                    <div class="metric-row">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value">{backtest['win_rate']:.0f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Profit Factor</span>
                        <span class="metric-value">{backtest['profit_factor']:.2f}x</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Trades</span>
                        <span class="metric-value">{backtest['num_trades']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # IV Analysis
            iv_premium = "RICH ✓" if spread['iv_rank'] > 50 else "FAIR" if spread['iv_rank'] > 30 else "CHEAP"
            iv_color = "#059669" if spread['iv_rank'] > 50 else "#d97706" if spread['iv_rank'] > 30 else "#dc2626"
            
            st.markdown(f"""
            <div class="setup-box">
                <div class="setup-title">📊 IV Analysis</div>
                <div class="metric-row">
                    <span class="metric-label">IV Rank</span>
                    <span class="metric-value" style="color: {iv_color}">{spread['iv_rank']:.0f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Current IV</span>
                    <span class="metric-value">{spread['iv']:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">20D HV</span>
                    <span class="metric-value">{spread['hv_20']:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Premium</span>
                    <span class="metric-value" style="color: {iv_color}">{iv_premium}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Trade Summary Card
        st.markdown(f"""
        <div class="trade-card">
            <h3>📋 TRADE SUMMARY</h3>
            <div class="trade-details">
                {ticker} {strategy_names[strategy]} | {dte} DTE | {contracts} contract(s)
                <br><br>
                Credit: <strong>${credit_total:,.0f}</strong> &nbsp;|&nbsp; 
                Max Risk: <strong>${max_loss_total:,.0f}</strong> &nbsp;|&nbsp;
                Prob. Profit: <strong>{prob:.0f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Price Chart
        st.markdown("---")
        st.markdown("### 📈 Price Chart with Strike Levels")
        
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Candlestick(
            x=data.index[-60:],
            open=data['open'].iloc[-60:],
            high=data['high'].iloc[-60:],
            low=data['low'].iloc[-60:],
            close=data['close'].iloc[-60:],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))
        
        # Strike levels
        if strategy == 'iron_condor':
            fig.add_hline(y=spread['put_short'], line_dash="dash", line_color="#ef4444", 
                         annotation_text=f"Put Short ${spread['put_short']:.0f}")
            fig.add_hline(y=spread['call_short'], line_dash="dash", line_color="#ef4444",
                         annotation_text=f"Call Short ${spread['call_short']:.0f}")
        else:
            fig.add_hline(y=spread['short_strike'], line_dash="dash", line_color="#ef4444",
                         annotation_text=f"Short ${spread['short_strike']:.0f}")
            fig.add_hline(y=spread['long_strike'], line_dash="dot", line_color="#6b7280",
                         annotation_text=f"Long ${spread['long_strike']:.0f}")
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# QUICK SCAN SECTION
# =============================================================================

st.markdown("---")
st.markdown("### 🔥 Quick Scan: High IV Opportunities")

quick_tickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "META"]

if st.button("Scan for High IV Setups", use_container_width=True):
    results = []
    progress = st.progress(0)
    
    for i, t in enumerate(quick_tickers):
        progress.progress((i + 1) / len(quick_tickers), f"Scanning {t}...")
        data = get_stock_data(t)
        if data is not None:
            spread = calculate_credit_spread(data, 'bull_put')
            if spread and spread['iv_rank'] > 40:
                results.append({
                    'Ticker': t,
                    'Price': f"${spread['price']:.2f}",
                    'IV Rank': f"{spread['iv_rank']:.0f}%",
                    'RSI': f"{spread['rsi']:.0f}",
                    'Premium': "RICH ✓" if spread['iv_rank'] > 50 else "FAIR",
                    'Suggested': "Bull Put" if spread['rsi'] < 50 else "Bear Call" if spread['rsi'] > 50 else "Iron Condor"
                })
    
    progress.empty()
    
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    else:
        st.info("No high IV setups found. Market volatility is low.")


# =============================================================================
# EDUCATION
# =============================================================================

st.markdown("---")
with st.expander("📚 Credit Spread Quick Guide"):
    st.markdown("""
    ### Why Credit Spreads?
    
    **Defined Risk** - You know your max loss before entering
    
    **High Probability** - Sell OTM options with 70-85% win rates
    
    **Time Decay** - Theta works FOR you, not against
    
    **IV Crush** - Profit when volatility drops after news
    
    ---
    
    ### When to Use Each Strategy
    
    | Strategy | Use When | IV Rank | RSI |
    |----------|----------|---------|-----|
    | **Bull Put** | Bullish bias, support held | > 50% | < 40 |
    | **Bear Call** | Bearish bias, resistance held | > 50% | > 60 |
    | **Iron Condor** | Neutral, range-bound | > 40% | 40-60 |
    
    ---
    
    ### Position Sizing Rule
    
    **Never risk more than 2-5% of account on a single trade**
    
    Example: $50,000 account → Max risk per trade = $1,000-2,500
    
    ---
    
    ### Exit Rules
    
    - **Take profit at 50%** of max credit
    - **Stop loss at 2x** credit received
    - **Close at 21 DTE** if not already closed
    """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.75rem;">
    Credit Spread Scanner • For educational purposes only • Not financial advice
</div>
""", unsafe_allow_html=True)