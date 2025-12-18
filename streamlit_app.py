"""
Options Strategy Backtester - Web Interface
============================================

A Streamlit web app for:
1. Running backtests with visual results
2. Pattern detection and signal analysis
3. Strategy comparison
4. Real-time opportunity scanning

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
from data_fetcher import DataPreparator, generate_synthetic_data

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Options Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - Settings
# =============================================================================

st.sidebar.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
st.sidebar.title("⚙️ Settings")

# Data Source
st.sidebar.subheader("📊 Data Source")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Synthetic Data", "Yahoo Finance (Real)"],
    help="Synthetic data is faster for testing. Real data requires internet."
)

if data_source == "Yahoo Finance (Real)":
    symbol = st.sidebar.text_input("Symbol", value="SPY")
    years = st.sidebar.slider("Years of History", 1, 10, 5)
else:
    symbol = "SYNTHETIC"
    years = st.sidebar.slider("Years of Synthetic Data", 1, 10, 5)

# Strategy Parameters
st.sidebar.subheader("🎯 Strategy Settings")
strategy_type = st.sidebar.selectbox(
    "Strategy",
    ["Short Put", "Iron Condor", "Long Straddle", "Compare All"]
)

delta_target = st.sidebar.slider(
    "Delta Target (absolute)", 
    0.05, 0.50, 0.16, 0.01,
    help="16 delta = ~84% probability OTM"
)

dte_target = st.sidebar.slider(
    "Days to Expiration (DTE)", 
    7, 90, 45, 1,
    help="45 DTE is optimal for theta decay"
)

profit_target = st.sidebar.slider(
    "Profit Target (%)", 
    25, 100, 50, 5,
    help="Exit when this % of max profit is reached"
) / 100

stop_loss = st.sidebar.slider(
    "Stop Loss (x credit)", 
    1.0, 5.0, 2.0, 0.5,
    help="Exit when loss = this multiple of credit received"
)

# Signal Filter
st.sidebar.subheader("📡 Entry Signals")
signal_filter = st.sidebar.selectbox(
    "Entry Signal",
    ["premium_sell", "vix_spike", "bb_squeeze", "rsi_reversal", "trend", "none"],
    help="Only enter when this signal is active"
)

signal_threshold = st.sidebar.slider(
    "Signal Strength Threshold",
    0.0, 1.0, 0.5, 0.1
)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<h1 class="main-header">📈 Options Strategy Backtester</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data(ttl=3600)
def load_data(source, sym, yrs):
    if source == "Yahoo Finance (Real)":
        try:
            prep = DataPreparator()
            data = prep.prepare_backtest_data(sym, period=f"{yrs}y")
            if data.empty:
                raise Exception("Empty data")
            return data, "real"
        except Exception as e:
            st.warning(f"Failed to fetch real data: {e}. Using synthetic data.")
            return generate_synthetic_data(days=yrs * 252), "synthetic"
    else:
        return generate_synthetic_data(days=yrs * 252), "synthetic"


with st.spinner("Loading data..."):
    data, data_type = load_data(data_source, symbol, years)

# Data info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Points", f"{len(data):,}")
with col2:
    st.metric("Date Range", f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
with col3:
    st.metric("Price Range", f"${data['close'].min():.2f} - ${data['close'].max():.2f}")
with col4:
    st.metric("VIX Range", f"{data['vix'].min():.1f} - {data['vix'].max():.1f}")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Backtest Results", "🔍 Pattern Analysis", "📈 Charts", "🎯 Live Signals"])


# =============================================================================
# TAB 1: Backtest Results
# =============================================================================

with tab1:
    st.subheader("🚀 Run Backtest")
    
    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            bt = OptionsBacktester(data)
            
            results = []
            
            if strategy_type == "Short Put" or strategy_type == "Compare All":
                result = bt.backtest_short_put(
                    delta_target=-delta_target,
                    dte_target=dte_target,
                    profit_target=profit_target,
                    stop_loss=stop_loss,
                    signal_filter=signal_filter if signal_filter != "none" else "premium_sell",
                    signal_threshold=signal_threshold if signal_filter != "none" else 0.0
                )
                results.append(("Short Put", result))
            
            if strategy_type == "Iron Condor" or strategy_type == "Compare All":
                result = bt.backtest_iron_condor(
                    put_delta=-delta_target,
                    call_delta=delta_target,
                    dte_target=dte_target,
                    profit_target=profit_target,
                    stop_loss=stop_loss,
                    signal_filter=signal_filter if signal_filter != "none" else "premium_sell",
                    signal_threshold=signal_threshold if signal_filter != "none" else 0.0
                )
                results.append(("Iron Condor", result))
            
            if strategy_type == "Long Straddle" or strategy_type == "Compare All":
                result = bt.backtest_long_straddle(
                    dte_target=dte_target,
                    profit_target=profit_target * 2,  # Higher target for long vol
                    stop_loss=profit_target,
                    signal_filter="bb_squeeze" if signal_filter == "none" else signal_filter,
                    signal_threshold=signal_threshold
                )
                results.append(("Long Straddle", result))
            
            # Store in session state
            st.session_state['backtest_results'] = results
    
    # Display results
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        # Summary metrics
        st.subheader("📋 Results Summary")
        
        comparison_data = []
        for name, r in results:
            comparison_data.append({
                'Strategy': name,
                'Trades': r.num_trades,
                'Win Rate': f"{r.win_rate*100:.1f}%",
                'Profit Factor': f"{r.profit_factor:.2f}",
                'EV/Trade': f"${r.expected_value:.2f}",
                'Sharpe': f"{r.sharpe_ratio:.2f}",
                'Max DD': f"{r.max_drawdown*100:.1f}%",
                'Total Return': f"{r.total_return*100:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best strategy highlight
        best = max(results, key=lambda x: x[1].profit_factor)
        
        st.success(f"🏆 **Best Strategy: {best[0]}** with Profit Factor {best[1].profit_factor:.2f}")
        
        # Detailed metrics for each strategy
        for name, r in results:
            with st.expander(f"📊 {name} - Detailed Results"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total P&L", f"${r.total_pnl:,.2f}")
                    st.metric("# Trades", r.num_trades)
                
                with col2:
                    st.metric("Win Rate", f"{r.win_rate*100:.1f}%")
                    st.metric("Profit Factor", f"{r.profit_factor:.2f}")
                
                with col3:
                    st.metric("Avg Win", f"${r.avg_win:.2f}")
                    st.metric("Avg Loss", f"${r.avg_loss:.2f}")
                
                with col4:
                    st.metric("Sharpe Ratio", f"{r.sharpe_ratio:.2f}")
                    st.metric("Max Drawdown", f"{r.max_drawdown*100:.1f}%")
                
                # Equity curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=r.equity_curve.values,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title=f"{name} - Equity Curve",
                    xaxis_title="Time",
                    yaxis_title="Portfolio Value ($)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade distribution
                if r.trades:
                    trade_pnls = [t.pnl for t in r.trades]
                    fig_hist = px.histogram(
                        x=trade_pnls,
                        nbins=30,
                        title=f"{name} - P&L Distribution",
                        labels={'x': 'P&L ($)', 'y': 'Count'}
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB 2: Pattern Analysis
# =============================================================================

with tab2:
    st.subheader("🔍 Pattern Detection")
    
    analyzer = PatternAnalyzer(data)
    
    # Pattern summary
    pattern_summary = analyzer.get_pattern_summary()
    
    if not pattern_summary.empty:
        st.dataframe(pattern_summary, use_container_width=True)
        
        # Pattern performance chart
        fig = px.bar(
            pattern_summary,
            x='Pattern',
            y='Occurrences',
            color='Win Rate',
            title="Pattern Occurrences and Win Rates",
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Next opportunity predictions
    st.subheader("🎯 Upcoming Opportunities")
    
    predictions = analyzer.predict_next_opportunity()
    
    if predictions:
        for pattern, pred in predictions.items():
            prob = pred.get('probability', 0)
            if prob > 0.3:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(pattern, f"{prob*100:.0f}%")
                with col2:
                    details = ", ".join([f"{k}: {v}" for k, v in pred.items() if k != 'probability'])
                    st.info(details)
    else:
        st.info("No strong signals detected currently.")


# =============================================================================
# TAB 3: Charts
# =============================================================================

with tab3:
    st.subheader("📈 Market Analysis Charts")
    
    # Price chart with indicators
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=('Price & Bollinger Bands', 'RSI', 'VIX', 'IV Percentile')
    )
    
    # Price with Bollinger Bands
    analyzer = PatternAnalyzer(data)
    df = analyzer.data
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_upper'],
        mode='lines', name='BB Upper',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_lower'],
        mode='lines', name='BB Lower',
        line=dict(color='gray', dash='dash'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi'],
        mode='lines', name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # VIX
    if 'vix' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['vix'],
            mode='lines', name='VIX',
            line=dict(color='orange')
        ), row=3, col=1)
    
    # IV Percentile
    if 'iv_percentile' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['iv_percentile'],
            mode='lines', name='IV %ile',
            line=dict(color='blue')
        ), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility analysis
    st.subheader("📊 Volatility Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # VIX distribution
        fig_vix = px.histogram(
            df, x='vix', nbins=50,
            title="VIX Distribution",
            labels={'vix': 'VIX Level', 'count': 'Frequency'}
        )
        current_vix = df['vix'].iloc[-1]
        fig_vix.add_vline(x=current_vix, line_dash="dash", line_color="red",
                         annotation_text=f"Current: {current_vix:.1f}")
        st.plotly_chart(fig_vix, use_container_width=True)
    
    with col2:
        # Returns distribution
        returns = df['close'].pct_change().dropna()
        fig_ret = px.histogram(
            x=returns * 100, nbins=50,
            title="Daily Returns Distribution",
            labels={'x': 'Return (%)', 'y': 'Frequency'}
        )
        fig_ret.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_ret, use_container_width=True)


# =============================================================================
# TAB 4: Live Signals
# =============================================================================

with tab4:
    st.subheader("🎯 Current Market Signals")
    
    analyzer = PatternAnalyzer(data)
    signals = analyzer.signal_generator.get_all_signals() if hasattr(analyzer, 'signal_generator') else None
    
    # Get current values
    current = data.iloc[-1]
    df = analyzer.data
    
    # Signal dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📉 Price Indicators")
        rsi = df['rsi'].iloc[-1]
        rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "⚪"
        st.metric("RSI (14)", f"{rsi:.1f} {rsi_color}")
        
        bb_pct = df['bb_pct'].iloc[-1]
        bb_color = "🔴" if bb_pct > 0.8 else "🟢" if bb_pct < 0.2 else "⚪"
        st.metric("BB Position", f"{bb_pct*100:.1f}% {bb_color}")
    
    with col2:
        st.markdown("### 📊 Volatility Indicators")
        vix = current['vix']
        vix_color = "🔴" if vix > 25 else "🟢" if vix < 15 else "⚪"
        st.metric("VIX", f"{vix:.1f} {vix_color}")
        
        if 'iv_percentile' in df.columns:
            iv_pct = df['iv_percentile'].iloc[-1]
            iv_color = "🔴" if iv_pct > 80 else "🟢" if iv_pct < 20 else "⚪"
            st.metric("IV Percentile", f"{iv_pct:.1f}% {iv_color}")
    
    with col3:
        st.markdown("### 🎯 Trade Signals")
        
        # Premium selling signal
        premium_signal = "✅ ACTIVE" if (vix > 15 and rsi > 35 and rsi < 65) else "❌ WAIT"
        st.metric("Premium Sell", premium_signal)
        
        # Long vol signal
        bb_width = df['bb_width'].iloc[-1]
        bb_squeeze = bb_width < df['bb_width'].quantile(0.2)
        vol_signal = "✅ ACTIVE" if bb_squeeze else "❌ WAIT"
        st.metric("Long Volatility", vol_signal)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Current Recommendations")
    
    recommendations = []
    
    if vix > 20 and rsi > 35 and rsi < 65:
        recommendations.append("🟢 **VIX elevated** - Good time for premium selling (short puts, iron condors)")
    
    if bb_squeeze:
        recommendations.append("🟢 **BB Squeeze detected** - Consider long straddle/strangle for breakout")
    
    if rsi < 30:
        recommendations.append("🟢 **RSI oversold** - Potential bounce, consider bullish strategies")
    
    if rsi > 70:
        recommendations.append("🔴 **RSI overbought** - Potential pullback, be cautious with bullish positions")
    
    if not recommendations:
        recommendations.append("⚪ **No strong signals** - Consider waiting for better setup")
    
    for rec in recommendations:
        st.markdown(rec)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    Options Strategy Backtester | Built with Streamlit<br>
    ⚠️ For educational purposes only. Options trading involves significant risk.
</div>
""", unsafe_allow_html=True)
