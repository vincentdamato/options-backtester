"""
Options Screener - Clean Table-Based Design
============================================

Inspired by Finviz & Barchart:
- Clean table layout with sortable columns
- Compact filter row
- Color-coded signals (green/red)
- IV Rank, RSI, signals prominently displayed
- One-click to expand details

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtester import OptionsBacktester
from patterns import PatternAnalyzer
from data_fetcher import DataPreparator

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Options Screener",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CLEAN CSS - Finviz/Barchart inspired
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .main .block-container { padding: 1rem 2rem; max-width: 1600px; }
    
    /* Header */
    .header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .logo { font-size: 1.5rem; font-weight: 700; color: #111827; }
    .logo span { color: #6366f1; }
    .timestamp { font-size: 0.8rem; color: #6b7280; }
    
    /* Filter bar */
    .filter-bar {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Summary stats row */
    .stats-row {
        display: flex;
        gap: 2rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .stat-item { text-align: center; }
    .stat-value { font-size: 1.25rem; font-weight: 600; color: #111827; }
    .stat-label { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Results table */
    .results-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .results-table th {
        background: #f3f4f6;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        color: #374151;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #e5e7eb;
        cursor: pointer;
    }
    .results-table th:hover { background: #e5e7eb; }
    .results-table td {
        padding: 0.875rem 1rem;
        border-bottom: 1px solid #f3f4f6;
        color: #111827;
    }
    .results-table tr:hover { background: #f9fafb; }
    
    /* Ticker column */
    .ticker { font-weight: 700; font-size: 0.95rem; color: #111827; }
    .ticker-name { font-size: 0.75rem; color: #6b7280; }
    
    /* Price change colors */
    .positive { color: #059669; font-weight: 600; }
    .negative { color: #dc2626; font-weight: 600; }
    
    /* Signal badges - compact */
    .signal {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    .signal-buy { background: #dcfce7; color: #166534; }
    .signal-sell { background: #fee2e2; color: #991b1b; }
    .signal-neutral { background: #e0e7ff; color: #3730a3; }
    
    /* Strategy badge */
    .strategy-badge {
        background: #111827;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* IV Rank bar */
    .iv-bar-container {
        width: 60px;
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
        margin-left: 0.5rem;
    }
    .iv-bar {
        height: 100%;
        border-radius: 3px;
    }
    .iv-high { background: #059669; }
    .iv-medium { background: #f59e0b; }
    .iv-low { background: #6b7280; }
    
    /* Score circle */
    .score-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        color: white;
    }
    .score-high { background: linear-gradient(135deg, #059669, #10b981); }
    .score-med { background: linear-gradient(135deg, #2563eb, #3b82f6); }
    .score-low { background: linear-gradient(135deg, #d97706, #f59e0b); }
    
    /* Button */
    .stButton > button {
        background: #111827;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stButton > button:hover { background: #1f2937; }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 500;
        background: #f9fafb;
        border-radius: 6px;
    }
    
    /* No results */
    .no-results {
        text-align: center;
        padding: 3rem;
        color: #6b7280;
    }
    
    /* Hide default metrics styling */
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# WATCHLISTS
# =============================================================================

WATCHLISTS = {
    "S&P 500 ETFs": ["SPY", "QQQ", "IWM", "DIA", "VOO"],
    "Mega Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "High Beta": ["TSLA", "NVDA", "AMD", "COIN", "MARA", "SQ", "SHOP", "PLTR"],
    "Sector ETFs": ["XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU"],
    "Volatility": ["VXX", "UVXY", "SVXY"],
}


# =============================================================================
# SCANNER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(ticker: str):
    try:
        prep = DataPreparator()
        data = prep.prepare_backtest_data(ticker, period="2y")
        return data if not data.empty and len(data) >= 100 else None
    except:
        return None


def analyze(ticker: str, data: pd.DataFrame) -> dict:
    if data is None or len(data) < 100:
        return None
    
    try:
        price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        change_pct = (price / prev_price - 1) * 100
        vix = data['vix'].iloc[-1]
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain/loss))).iloc[-1]
        
        # IV Rank
        iv = data['iv'].iloc[-1]
        iv_rank = (data['iv'] < iv).mean() * 100
        
        # 20-day volatility
        vol_20d = data['close'].pct_change().tail(20).std() * np.sqrt(252) * 100
        
        # Signals
        signals = []
        score = 50
        
        # IV signals
        if iv_rank > 80:
            signals.append(("HIGH IV", "buy"))
            score += 20
        elif iv_rank > 60:
            signals.append(("IV↑", "neutral"))
            score += 10
        elif iv_rank < 20:
            signals.append(("LOW IV", "sell"))
            score += 5
        
        # RSI signals
        if rsi < 30:
            signals.append(("OVERSOLD", "buy"))
            score += 15
        elif rsi > 70:
            signals.append(("OVERBOUGHT", "sell"))
            score += 10
        
        # VIX signal
        if vix > 25:
            signals.append(("VIX SPIKE", "buy"))
            score += 15
        
        # Determine strategy
        if iv_rank > 60:
            if rsi < 40:
                strategy = "SELL PUT"
            elif rsi > 60:
                strategy = "IRON CONDOR"
            else:
                strategy = "STRANGLE"
        elif iv_rank < 30:
            strategy = "BUY STRADDLE"
        else:
            strategy = "IRON CONDOR"
        
        # Backtest
        bt = OptionsBacktester(data)
        if "PUT" in strategy:
            result = bt.backtest_short_put(delta_target=-0.16, dte_target=45, profit_target=0.5, stop_loss=2.0)
        else:
            result = bt.backtest_iron_condor(put_delta=-0.16, call_delta=0.16, dte_target=45, profit_target=0.5, stop_loss=2.0)
        
        if result.profit_factor > 1.5:
            score += 15
        elif result.profit_factor > 1.0:
            score += 8
        
        score = min(99, max(1, score))
        
        return {
            'ticker': ticker,
            'price': price,
            'change': change_pct,
            'iv_rank': iv_rank,
            'rsi': rsi,
            'vix': vix,
            'vol_20d': vol_20d,
            'signals': signals,
            'strategy': strategy,
            'win_rate': result.win_rate * 100,
            'profit_factor': result.profit_factor,
            'score': score,
            'data': data
        }
    except:
        return None


def scan(tickers: list, progress=None) -> list:
    results = []
    for i, t in enumerate(tickers):
        if progress:
            progress.progress((i + 1) / len(tickers), f"Scanning {t}...")
        data = fetch_data(t)
        if data is not None:
            analysis = analyze(t, data)
            if analysis:
                results.append(analysis)
    return sorted(results, key=lambda x: x['score'], reverse=True)


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="header">
    <div class="logo">◎ Options<span>Screener</span></div>
    <div class="timestamp">Last updated: """ + datetime.now().strftime("%b %d, %Y %H:%M") + """</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# FILTER BAR
# =============================================================================

st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])

with c1:
    watchlist = st.selectbox("Watchlist", list(WATCHLISTS.keys()), label_visibility="collapsed")
    tickers = WATCHLISTS[watchlist]

with c2:
    iv_filter = st.selectbox("IV Rank", ["Any", "> 70%", "> 50%", "< 30%"], label_visibility="collapsed")

with c3:
    rsi_filter = st.selectbox("RSI", ["Any", "Oversold (<30)", "Overbought (>70)"], label_visibility="collapsed")

with c4:
    strategy_filter = st.selectbox("Strategy", ["All", "Sell Premium", "Buy Premium"], label_visibility="collapsed")

with c5:
    sort_by = st.selectbox("Sort", ["Score", "IV Rank", "Win Rate"], label_visibility="collapsed")

with c6:
    scan_btn = st.button("🔍 Scan", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# SCAN & RESULTS
# =============================================================================

if scan_btn or 'results' not in st.session_state:
    progress = st.progress(0, "Starting scan...")
    results = scan(tickers, progress)
    progress.empty()
    st.session_state['results'] = results

if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Apply filters
    if iv_filter == "> 70%":
        results = [r for r in results if r['iv_rank'] > 70]
    elif iv_filter == "> 50%":
        results = [r for r in results if r['iv_rank'] > 50]
    elif iv_filter == "< 30%":
        results = [r for r in results if r['iv_rank'] < 30]
    
    if rsi_filter == "Oversold (<30)":
        results = [r for r in results if r['rsi'] < 30]
    elif rsi_filter == "Overbought (>70)":
        results = [r for r in results if r['rsi'] > 70]
    
    if strategy_filter == "Sell Premium":
        results = [r for r in results if r['strategy'] in ["SELL PUT", "IRON CONDOR", "STRANGLE"]]
    elif strategy_filter == "Buy Premium":
        results = [r for r in results if r['strategy'] in ["BUY STRADDLE"]]
    
    # Sort
    if sort_by == "IV Rank":
        results = sorted(results, key=lambda x: x['iv_rank'], reverse=True)
    elif sort_by == "Win Rate":
        results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    # Stats row
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-value">{len(results)}</div>
            <div class="stat-label">Results</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{len([r for r in results if r['iv_rank'] > 70])}</div>
            <div class="stat-label">High IV</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{len([r for r in results if r['rsi'] < 30])}</div>
            <div class="stat-label">Oversold</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{len([r for r in results if r['rsi'] > 70])}</div>
            <div class="stat-label">Overbought</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{np.mean([r['score'] for r in results]):.0f}</div>
            <div class="stat-label">Avg Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Results table
    if not results:
        st.markdown('<div class="no-results"><h3>No results match your filters</h3></div>', unsafe_allow_html=True)
    else:
        # Table header
        st.markdown("""
        <table class="results-table">
            <thead>
                <tr>
                    <th style="width: 50px;">Score</th>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>Change</th>
                    <th>IV Rank</th>
                    <th>RSI</th>
                    <th>Signals</th>
                    <th>Strategy</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                </tr>
            </thead>
        </table>
        """, unsafe_allow_html=True)
        
        # Table rows (using columns for interactivity)
        for r in results:
            # Score color
            if r['score'] >= 70:
                score_class = "score-high"
            elif r['score'] >= 50:
                score_class = "score-med"
            else:
                score_class = "score-low"
            
            # Price change color
            change_class = "positive" if r['change'] >= 0 else "negative"
            change_arrow = "▲" if r['change'] >= 0 else "▼"
            
            # IV bar color
            if r['iv_rank'] > 70:
                iv_class = "iv-high"
            elif r['iv_rank'] > 40:
                iv_class = "iv-medium"
            else:
                iv_class = "iv-low"
            
            # Signals HTML
            signals_html = ""
            for sig_name, sig_type in r['signals']:
                sig_class = f"signal-{sig_type}"
                signals_html += f'<span class="signal {sig_class}">{sig_name}</span>'
            
            # Create expandable row
            with st.expander(f"**{r['ticker']}** — ${r['price']:.2f} — Score: {r['score']}"):
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    **Price:** ${r['price']:.2f} <span class="{change_class}">{change_arrow} {abs(r['change']):.2f}%</span>
                    
                    **IV Rank:** {r['iv_rank']:.0f}%
                    
                    **RSI:** {r['rsi']:.1f}
                    
                    **20D Vol:** {r['vol_20d']:.1f}%
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    **Recommended Strategy:**
                    
                    ### {r['strategy']}
                    
                    **Setup:**
                    - Delta: 16Δ (~84% OTM)
                    - DTE: 45 days
                    - Take Profit: 50%
                    - Stop Loss: 2x credit
                    """)
                
                with col3:
                    st.markdown(f"""
                    **Backtest Results:**
                    
                    - Win Rate: **{r['win_rate']:.0f}%**
                    - Profit Factor: **{r['profit_factor']:.2f}x**
                    
                    **Active Signals:**
                    """)
                    for sig_name, sig_type in r['signals']:
                        color = "🟢" if sig_type == "buy" else "🔴" if sig_type == "sell" else "🔵"
                        st.write(f"{color} {sig_name}")
                
                # Mini chart
                st.markdown("**60-Day Price Chart:**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=r['data']['close'].iloc[-60:].values,
                    mode='lines',
                    line=dict(color='#6366f1', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(99, 102, 241, 0.1)'
                ))
                fig.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.75rem;">
    Options Screener • Data by Yahoo Finance • For educational purposes only
</div>
""", unsafe_allow_html=True)