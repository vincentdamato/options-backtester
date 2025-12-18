"""
Options Strategy Backtester - Professional Portal
==================================================

Clean, modern single-page interface for options backtesting.
Real market data only. No synthetic data.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtester import OptionsBacktester, BacktestResult, StrategyType
from patterns import PatternAnalyzer
from data_fetcher import DataPreparator

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Options Backtester",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# PROFESSIONAL CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Import clean fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.25rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
    }
    
    .card-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a2e;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    .metric-positive { color: #10b981; }
    .metric-negative { color: #ef4444; }
    
    /* Button styling - cushioned/pill style */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary button */
    .secondary-btn > button {
        background: #f3f4f6;
        color: #374151;
        box-shadow: none;
    }
    
    .secondary-btn > button:hover {
        background: #e5e7eb;
        box-shadow: none;
    }
    
    /* Input styling */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #667eea;
    }
    
    /* Table styling */
    .dataframe {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem;
    }
    
    .dataframe th {
        background: #f9fafb !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }
    
    /* Signal badges */
    .signal-active {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .signal-inactive {
        background: #f3f4f6;
        color: #6b7280;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Results table */
    .results-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .results-table th {
        background: #f9fafb;
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        color: #374151;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .results-table td {
        padding: 1rem;
        border-top: 1px solid #f0f0f0;
        color: #1f2937;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
        color: #374151;
        background: #f9fafb;
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f3f4f6;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #6b7280;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #1a1a2e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Status indicators */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-green { background: #10b981; }
    .status-red { background: #ef4444; }
    .status-yellow { background: #f59e0b; }
    
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<p class="main-title">◉ Options Strategy Backtester</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quantitative analysis for options traders • Real market data</p>', unsafe_allow_html=True)


# =============================================================================
# TOP CONTROLS - Single Row
# =============================================================================

col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

with col1:
    symbol = st.text_input("Symbol", value="SPY", label_visibility="collapsed", placeholder="Enter symbol (e.g., SPY)")

with col2:
    years = st.selectbox("History", [1, 2, 3, 5, 10], index=2, format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}")

with col3:
    strategy_type = st.selectbox("Strategy", ["Short Put", "Iron Condor", "Long Straddle", "Compare All"])

with col4:
    dte_target = st.selectbox("DTE", [21, 30, 45, 60, 90], index=2, format_func=lambda x: f"{x} DTE")

with col5:
    run_backtest = st.button("Run Analysis", type="primary", use_container_width=True)


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_real_data(sym: str, yrs: int):
    """Load real market data from Yahoo Finance"""
    try:
        prep = DataPreparator()
        data = prep.prepare_backtest_data(sym, period=f"{yrs}y")
        if data.empty:
            return None, "No data available"
        return data, None
    except Exception as e:
        return None, str(e)


# Load data
with st.spinner(""):
    data, error = load_real_data(symbol.upper(), years)

if error:
    st.error(f"⚠️ Could not load data for {symbol}: {error}")
    st.stop()


# =============================================================================
# MARKET OVERVIEW CARDS
# =============================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

current_price = data['close'].iloc[-1]
price_change = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100
current_vix = data['vix'].iloc[-1]

# Calculate RSI
delta = data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
current_rsi = (100 - (100 / (1 + rs))).iloc[-1]

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">Current Price</div>
        <div class="metric-value">${current_price:.2f}</div>
        <div class="metric-label {'metric-positive' if price_change >= 0 else 'metric-negative'}">
            {'▲' if price_change >= 0 else '▼'} {abs(price_change):.2f}% today
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">VIX Level</div>
        <div class="metric-value">{current_vix:.1f}</div>
        <div class="metric-label">{'High' if current_vix > 25 else 'Normal' if current_vix > 15 else 'Low'} volatility</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">RSI (14)</div>
        <div class="metric-value">{current_rsi:.1f}</div>
        <div class="metric-label">{'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    iv_proxy = data['iv'].iloc[-1] * 100
    st.markdown(f"""
    <div class="card">
        <div class="card-header">IV Estimate</div>
        <div class="metric-value">{iv_proxy:.1f}%</div>
        <div class="metric-label">Implied volatility</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">Data Range</div>
        <div class="metric-value">{len(data):,}</div>
        <div class="metric-label">Trading days</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Two column layout: Settings + Results
left_col, right_col = st.columns([1, 2.5])

with left_col:
    st.markdown("#### ⚙️ Parameters")
    
    with st.container():
        delta_target = st.slider(
            "Delta Target",
            min_value=0.05,
            max_value=0.50,
            value=0.16,
            step=0.01,
            help="Strike selection: 0.16 = ~84% OTM"
        )
        
        profit_target = st.slider(
            "Profit Target",
            min_value=25,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
            help="Exit at this % of max profit"
        )
        
        stop_loss = st.slider(
            "Stop Loss",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            format="%.1fx",
            help="Exit at this multiple of credit"
        )
        
        signal_filter = st.selectbox(
            "Entry Signal",
            ["Premium Sell", "VIX Spike", "BB Squeeze", "RSI Reversal", "No Filter"],
            help="Only enter when signal is active"
        )
        
        # Map display names to internal names
        signal_map = {
            "Premium Sell": "premium_sell",
            "VIX Spike": "vix_spike", 
            "BB Squeeze": "bb_squeeze",
            "RSI Reversal": "rsi_reversal",
            "No Filter": "none"
        }

with right_col:
    # Run backtest when button clicked or on first load
    if run_backtest or 'backtest_results' not in st.session_state:
        with st.spinner("Analyzing..."):
            bt = OptionsBacktester(data)
            results = []
            
            signal_key = signal_map.get(signal_filter, "premium_sell")
            
            if strategy_type in ["Short Put", "Compare All"]:
                result = bt.backtest_short_put(
                    delta_target=-delta_target,
                    dte_target=dte_target,
                    profit_target=profit_target/100,
                    stop_loss=stop_loss,
                    signal_filter=signal_key if signal_key != "none" else "premium_sell",
                    signal_threshold=0.5 if signal_key != "none" else 0.0
                )
                results.append(("Short Put", result))
            
            if strategy_type in ["Iron Condor", "Compare All"]:
                result = bt.backtest_iron_condor(
                    put_delta=-delta_target,
                    call_delta=delta_target,
                    dte_target=dte_target,
                    profit_target=profit_target/100,
                    stop_loss=stop_loss,
                    signal_filter=signal_key if signal_key != "none" else "premium_sell",
                    signal_threshold=0.5 if signal_key != "none" else 0.0
                )
                results.append(("Iron Condor", result))
            
            if strategy_type in ["Long Straddle", "Compare All"]:
                result = bt.backtest_long_straddle(
                    dte_target=dte_target,
                    profit_target=(profit_target/100) * 2,
                    stop_loss=profit_target/100,
                    signal_filter="bb_squeeze" if signal_key == "none" else signal_key,
                    signal_threshold=0.5
                )
                results.append(("Long Straddle", result))
            
            st.session_state['backtest_results'] = results
    
    # Display results
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        # Best strategy callout
        best = max(results, key=lambda x: x[1].profit_factor)
        
        if best[1].profit_factor > 1.0:
            st.success(f"✓ **{best[0]}** shows positive edge with {best[1].profit_factor:.2f}x profit factor")
        else:
            st.warning(f"⚠️ No strategy shows consistent edge in this period")
        
        # Results table
        st.markdown("#### 📊 Backtest Results")
        
        results_df = pd.DataFrame([{
            'Strategy': name,
            'Trades': r.num_trades,
            'Win Rate': f"{r.win_rate*100:.1f}%",
            'Profit Factor': r.profit_factor,
            'Avg Win': f"${r.avg_win:.0f}",
            'Avg Loss': f"${r.avg_loss:.0f}",
            'Total Return': f"{r.total_return*100:.1f}%",
            'Max DD': f"{r.max_drawdown*100:.1f}%",
            'Sharpe': f"{r.sharpe_ratio:.2f}"
        } for name, r in results])
        
        # Style the dataframe
        def style_profit_factor(val):
            if isinstance(val, float):
                color = '#10b981' if val > 1.0 else '#ef4444'
                return f'color: {color}; font-weight: 600'
            return ''
        
        styled_df = results_df.style.applymap(
            style_profit_factor, 
            subset=['Profit Factor']
        ).format({'Profit Factor': '{:.2f}'})
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Equity curves
        st.markdown("#### 📈 Equity Curves")
        
        fig = go.Figure()
        colors = ['#667eea', '#10b981', '#f59e0b']
        
        for i, (name, r) in enumerate(results):
            fig.add_trace(go.Scatter(
                y=r.equity_curve.values,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2.5)
            ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickformat='$,.0f')
        )
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PATTERN ANALYSIS & SIGNALS
# =============================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("#### 🎯 Current Market Signals")

analyzer = PatternAnalyzer(data)
predictions = analyzer.predict_next_opportunity()

signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)

# Premium Sell Signal
premium_active = current_vix > 15 and 35 < current_rsi < 65
with signal_col1:
    status = "signal-active" if premium_active else "signal-inactive"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <div class="{status}">{'ACTIVE' if premium_active else 'WAIT'}</div>
        <div style="margin-top: 0.75rem; font-weight: 600; color: #374151;">Premium Sell</div>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.25rem;">VIX + RSI favorable</div>
    </div>
    """, unsafe_allow_html=True)

# VIX Spike Signal
vix_spike = current_vix > 25
with signal_col2:
    status = "signal-active" if vix_spike else "signal-inactive"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <div class="{status}">{'ACTIVE' if vix_spike else 'WAIT'}</div>
        <div style="margin-top: 0.75rem; font-weight: 600; color: #374151;">VIX Spike</div>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.25rem;">Mean reversion setup</div>
    </div>
    """, unsafe_allow_html=True)

# RSI Signal
rsi_signal = current_rsi < 30 or current_rsi > 70
with signal_col3:
    status = "signal-active" if rsi_signal else "signal-inactive"
    label = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <div class="{status}">{'ACTIVE' if rsi_signal else 'WAIT'}</div>
        <div style="margin-top: 0.75rem; font-weight: 600; color: #374151;">RSI Reversal</div>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.25rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# BB Squeeze
bb_squeeze = 'bb_squeeze' in predictions and predictions['bb_squeeze'].get('probability', 0) > 0.5
with signal_col4:
    status = "signal-active" if bb_squeeze else "signal-inactive"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <div class="{status}">{'ACTIVE' if bb_squeeze else 'WAIT'}</div>
        <div style="margin-top: 0.75rem; font-weight: 600; color: #374151;">BB Squeeze</div>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.25rem;">Breakout pending</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PRICE CHART
# =============================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("#### 📉 Price History")

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=(f'{symbol.upper()} Price', 'RSI (14)')
)

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=data.index[-252:],  # Last year
    open=data['open'].iloc[-252:],
    high=data['high'].iloc[-252:],
    low=data['low'].iloc[-252:],
    close=data['close'].iloc[-252:],
    name='Price',
    increasing_line_color='#10b981',
    decreasing_line_color='#ef4444'
), row=1, col=1)

# RSI
rsi_series = 100 - (100 / (1 + rs))
fig.add_trace(go.Scatter(
    x=data.index[-252:],
    y=rsi_series.iloc[-252:],
    mode='lines',
    name='RSI',
    line=dict(color='#667eea', width=1.5)
), row=2, col=1)

# RSI levels
fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1, row=2, col=1)

fig.update_layout(
    height=500,
    margin=dict(l=0, r=0, t=30, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter'),
    showlegend=False,
    xaxis_rangeslider_visible=False
)

fig.update_xaxes(showgrid=True, gridcolor='#f5f5f5')
fig.update_yaxes(showgrid=True, gridcolor='#f5f5f5')

st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 1rem 0;">
    Options Strategy Backtester • Built for VINCENT DAMATO <br>
    <span style="color: #d1d5db;">Data provided by Yahoo Finance • For educational purposes only</span>
</div>
""", unsafe_allow_html=True)