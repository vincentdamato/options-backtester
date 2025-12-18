"""
Options Play Screener - Find Exact Trades
==========================================

Scans the market to find the best options plays RIGHT NOW.
No more guessing - the screener tells you exactly what to trade.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtester import OptionsBacktester
from patterns import PatternAnalyzer
from data_fetcher import DataPreparator

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Options Screener",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
    
    .main-title { font-size: 2.25rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.25rem; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-bottom: 2rem; }
    
    .card {
        background: #ffffff; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #f0f0f0; margin-bottom: 1rem;
    }
    .card-header { font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
    .metric-value { font-size: 1.5rem; font-weight: 600; color: #1a1a2e; }
    .metric-label { font-size: 0.8rem; color: #9ca3af; margin-top: 0.25rem; }
    .metric-positive { color: #10b981; }
    .metric-negative { color: #ef4444; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 12px; padding: 0.75rem 2rem;
        font-weight: 600; font-size: 0.95rem; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5); }
    
    .play-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px; padding: 1.75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem; transition: all 0.3s ease;
    }
    .play-card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }
    .play-ticker { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
    .play-strategy { font-size: 0.9rem; font-weight: 600; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px; }
    .play-details { display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap; }
    .play-stat { text-align: center; }
    .play-stat-value { font-size: 1.25rem; font-weight: 600; color: #1a1a2e; }
    .play-stat-label { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; }
    
    .signal-badge { display: inline-block; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-right: 0.5rem; margin-bottom: 0.5rem; }
    .signal-bullish { background: linear-gradient(135deg, #10b981, #059669); color: white; }
    .signal-bearish { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    .signal-neutral { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    
    .score-excellent { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 700; font-size: 1.1rem; }
    .score-good { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 700; font-size: 1.1rem; }
    .score-moderate { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 700; font-size: 1.1rem; }
    
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent, #e5e7eb, transparent); margin: 2rem 0; }
    .filter-section { background: #f8fafc; border-radius: 16px; padding: 1.25rem; margin-bottom: 1.5rem; }
    .no-plays { text-align: center; padding: 3rem; color: #6b7280; }
    .scanning { display: inline-block; animation: pulse 1.5s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# WATCHLISTS
# =============================================================================

WATCHLISTS = {
    "Popular ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "XLF", "XLE", "XLK"],
    "Mega Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "CRM"],
    "High IV Names": ["TSLA", "NVDA", "AMD", "COIN", "RIVN", "PLTR", "SOFI", "MARA", "SQ", "SHOP"],
    "Dividend Stocks": ["JNJ", "PG", "KO", "PEP", "MCD", "WMT", "HD", "V", "MA", "UNH"],
    "Custom": []
}


# =============================================================================
# SCREENER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ticker_data(ticker: str, years: int = 2):
    try:
        prep = DataPreparator()
        data = prep.prepare_backtest_data(ticker, period=f"{years}y")
        return data if not data.empty and len(data) >= 100 else None
    except:
        return None


def analyze_ticker(ticker: str, data: pd.DataFrame) -> dict:
    if data is None or len(data) < 100:
        return None
    
    try:
        current_price = data['close'].iloc[-1]
        current_vix = data['vix'].iloc[-1]
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain/loss))).iloc[-1]
        
        # IV percentile
        iv = data['iv'].iloc[-1]
        iv_percentile = (data['iv'] < iv).mean() * 100
        
        # BB
        sma20 = data['close'].rolling(20).mean().iloc[-1]
        std20 = data['close'].rolling(20).std().iloc[-1]
        bb_pct = (current_price - (sma20 - 2*std20)) / (4*std20)
        
        # Price changes
        price_change_1d = (current_price / data['close'].iloc[-2] - 1) * 100
        
        # Signals and scoring
        signals = []
        score = 50
        
        if current_vix > 25:
            signals.append(("VIX SPIKE", "bullish")); score += 15
        elif current_vix > 20:
            signals.append(("Elevated VIX", "neutral")); score += 8
        
        if iv_percentile > 80:
            signals.append(("HIGH IV", "bullish")); score += 20
        elif iv_percentile > 60:
            signals.append(("Above Avg IV", "neutral")); score += 10
        
        if rsi < 30:
            signals.append(("OVERSOLD", "bullish")); score += 15
        elif rsi > 70:
            signals.append(("OVERBOUGHT", "bearish")); score += 10
        
        if bb_pct < 0.1:
            signals.append(("BELOW BB", "bullish")); score += 10
        elif bb_pct > 0.9:
            signals.append(("ABOVE BB", "bearish")); score += 8
        
        # Determine strategy
        if iv_percentile > 60 and current_vix > 18:
            if rsi < 40:
                best_strategy, strategy_reason = "Short Put", "High IV + Oversold = Sell puts"
            else:
                best_strategy, strategy_reason = "Iron Condor", "High IV + Neutral = Sell condors"
        elif rsi < 30:
            best_strategy, strategy_reason = "Short Put", "Oversold bounce play"
        else:
            best_strategy, strategy_reason = "Iron Condor", "Neutral conditions"
        
        # Quick backtest
        bt = OptionsBacktester(data)
        if best_strategy == "Short Put":
            result = bt.backtest_short_put(delta_target=-0.16, dte_target=45, profit_target=0.5, stop_loss=2.0)
        else:
            result = bt.backtest_iron_condor(put_delta=-0.16, call_delta=0.16, dte_target=45, profit_target=0.5, stop_loss=2.0)
        
        if result.profit_factor > 1.5: score += 20
        elif result.profit_factor > 1.0: score += 10
        elif result.profit_factor < 0.8: score -= 15
        
        score = min(100, max(0, score))
        
        return {
            'ticker': ticker, 'price': current_price, 'price_change_1d': price_change_1d,
            'vix': current_vix, 'rsi': rsi, 'iv_percentile': iv_percentile,
            'signals': signals, 'score': score, 'best_strategy': best_strategy,
            'strategy_reason': strategy_reason, 'win_rate': result.win_rate,
            'profit_factor': result.profit_factor, 'num_trades': result.num_trades, 'data': data
        }
    except:
        return None


def scan_watchlist(tickers: list, progress_callback=None) -> list:
    opportunities = []
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i / len(tickers), f"Scanning {ticker}...")
        data = fetch_ticker_data(ticker)
        if data is not None:
            analysis = analyze_ticker(ticker, data)
            if analysis and analysis['score'] >= 40:
                opportunities.append(analysis)
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<p class="main-title">◎ Options Play Screener</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find the best options trades right now • Real-time market scanning</p>', unsafe_allow_html=True)


# =============================================================================
# FILTERS
# =============================================================================

st.markdown('<div class="filter-section">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])

with col1:
    watchlist_choice = st.selectbox("Watchlist", list(WATCHLISTS.keys()), index=0)
    if watchlist_choice == "Custom":
        custom_tickers = st.text_input("Enter tickers (comma separated)", placeholder="AAPL, MSFT, GOOGL")
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    else:
        tickers = WATCHLISTS[watchlist_choice]

with col2:
    strategy_filter = st.selectbox("Strategy", ["All", "Premium Selling", "Directional"])

with col3:
    min_score = st.selectbox("Min Score", [40, 50, 60, 70, 80], index=1, format_func=lambda x: f"{x}+")

with col4:
    scan_btn = st.button("🔍 Scan", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# SCANNING
# =============================================================================

if scan_btn or 'scan_results' not in st.session_state:
    if tickers:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct, text):
            progress_bar.progress(pct)
            status_text.markdown(f'<span class="scanning">⟳ {text}</span>', unsafe_allow_html=True)
        
        opportunities = scan_watchlist(tickers, update_progress)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state['scan_results'] = opportunities
        st.session_state['scan_time'] = datetime.now()


# =============================================================================
# RESULTS
# =============================================================================

if 'scan_results' in st.session_state:
    opportunities = st.session_state['scan_results']
    scan_time = st.session_state.get('scan_time', datetime.now())
    
    # Filter
    if strategy_filter == "Premium Selling":
        opportunities = [o for o in opportunities if o['best_strategy'] in ["Short Put", "Iron Condor"]]
    elif strategy_filter == "Directional":
        opportunities = [o for o in opportunities if o['best_strategy'] == "Short Put"]
    
    opportunities = [o for o in opportunities if o['score'] >= min_score]
    
    # Summary
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card"><div class="card-header">Found</div><div class="metric-value">{len(opportunities)}</div><div class="metric-label">opportunities</div></div>', unsafe_allow_html=True)
    with col2:
        avg_score = np.mean([o['score'] for o in opportunities]) if opportunities else 0
        st.markdown(f'<div class="card"><div class="card-header">Avg Score</div><div class="metric-value">{avg_score:.0f}</div><div class="metric-label">quality</div></div>', unsafe_allow_html=True)
    with col3:
        high_iv = len([o for o in opportunities if o['iv_percentile'] > 70])
        st.markdown(f'<div class="card"><div class="card-header">High IV</div><div class="metric-value">{high_iv}</div><div class="metric-label">setups</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="card-header">Scanned</div><div class="metric-value">{scan_time.strftime("%H:%M")}</div><div class="metric-label">{scan_time.strftime("%b %d")}</div></div>', unsafe_allow_html=True)
    
    # Play Cards
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Top Plays")
    
    if not opportunities:
        st.markdown('<div class="no-plays"><h3>No plays found</h3><p>Try lowering minimum score or changing watchlist</p></div>', unsafe_allow_html=True)
    else:
        for opp in opportunities[:10]:
            score_class = "score-excellent" if opp['score'] >= 75 else "score-good" if opp['score'] >= 60 else "score-moderate"
            signals_html = "".join([f'<span class="signal-badge signal-{t}">{n}</span>' for n, t in opp['signals']])
            price_color = "metric-positive" if opp['price_change_1d'] >= 0 else "metric-negative"
            arrow = "▲" if opp['price_change_1d'] >= 0 else "▼"
            
            st.markdown(f"""
            <div class="play-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="play-ticker">{opp['ticker']}</div>
                        <div class="play-strategy">{opp['best_strategy']}</div>
                        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #6b7280;">{opp['strategy_reason']}</div>
                    </div>
                    <div class="{score_class}">{opp['score']}</div>
                </div>
                <div style="margin-top: 1rem;">{signals_html}</div>
                <div class="play-details">
                    <div class="play-stat"><div class="play-stat-value">${opp['price']:.2f}</div><div class="play-stat-label">Price</div></div>
                    <div class="play-stat"><div class="play-stat-value {price_color}">{arrow} {abs(opp['price_change_1d']):.1f}%</div><div class="play-stat-label">1D Chg</div></div>
                    <div class="play-stat"><div class="play-stat-value">{opp['iv_percentile']:.0f}%</div><div class="play-stat-label">IV Rank</div></div>
                    <div class="play-stat"><div class="play-stat-value">{opp['rsi']:.0f}</div><div class="play-stat-label">RSI</div></div>
                    <div class="play-stat"><div class="play-stat-value">{opp['win_rate']*100:.0f}%</div><div class="play-stat-label">Win Rate</div></div>
                    <div class="play-stat"><div class="play-stat-value">{opp['profit_factor']:.2f}x</div><div class="play-stat-label">Profit Factor</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📊 {opp['ticker']} Chart & Details"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = go.Figure(go.Candlestick(
                        x=opp['data'].index[-60:], open=opp['data']['open'].iloc[-60:],
                        high=opp['data']['high'].iloc[-60:], low=opp['data']['low'].iloc[-60:],
                        close=opp['data']['close'].iloc[-60:],
                        increasing_line_color='#10b981', decreasing_line_color='#ef4444'
                    ))
                    fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), xaxis_rangeslider_visible=False, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.markdown(f"""
                    **Suggested Setup:**
                    - Strategy: **{opp['best_strategy']}**
                    - Delta: **16Δ** (~84% OTM)
                    - DTE: **45 days**
                    - Take Profit: **50%**
                    - Stop Loss: **2x credit**
                    
                    **Backtest Stats:**
                    - Trades: {opp['num_trades']}
                    - Win Rate: {opp['win_rate']*100:.0f}%
                    - Profit Factor: {opp['profit_factor']:.2f}x
                    """)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#9ca3af;font-size:0.8rem;">Options Play Screener • Data by Yahoo Finance • Educational purposes only</div>', unsafe_allow_html=True)