# Options Strategy Backtester

A professional-grade backtesting framework for options strategies with pattern detection and web interface.

## 🌐 Live Demo

Deploy your own version to [Streamlit Community Cloud](https://streamlit.io/cloud) for free!

## Features

- **Web Interface**: Beautiful Streamlit dashboard with interactive charts
- **Multiple Strategy Support**: Short puts, iron condors, long straddles, and more
- **Pattern Detection**: RSI reversals, VIX spikes, Bollinger squeezes, IV crush setups
- **Signal-Based Entries**: Only enter trades when conditions are favorable
- **Complete Metrics**: Win rate, profit factor, Sharpe ratio, max drawdown
- **Next Opportunity Prediction**: Know when the next pattern is likely to occur

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/options-backtester.git
cd options-backtester

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run streamlit_app.py
```

### Option 2: Run CLI (no web interface)

```bash
python main.py                    # Run with synthetic data
python main.py --real             # Use real market data
python main.py --symbol QQQ       # Different symbol
```

### Option 3: Deploy to Streamlit Cloud (Free!)

1. Push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repo and `streamlit_app.py`
5. Click Deploy!

## 📁 Project Structure

```
options_backtester/
├── streamlit_app.py     # Web interface (run with: streamlit run streamlit_app.py)
├── main.py              # CLI runner
├── requirements.txt     # Python dependencies
├── src/
│   ├── backtester.py    # Core backtesting engine
│   ├── patterns.py      # Pattern detection
│   └── data_fetcher.py  # Data sources
└── tests/
```

## 📊 Screenshots

The web app includes:
- **Backtest Results**: Run strategies with visual equity curves
- **Pattern Analysis**: See which patterns are most profitable
- **Interactive Charts**: Price, RSI, VIX, IV percentile with Bollinger Bands
- **Live Signals**: Current market conditions and recommendations

## Key Concepts

### Win Rate vs Profit Factor

**Win rate alone is meaningless!** A 90% win rate can lose money if losses are too large.

```
Profit Factor = (Win Rate × Avg Win) / (Loss Rate × Avg Loss)
Must be > 1.0 to be profitable
```

### Pattern-Based Entry

Instead of entering randomly, we wait for favorable conditions:

| Signal | Description | Best For |
|--------|-------------|----------|
| `premium_sell` | High IV + elevated VIX + neutral RSI | Short puts, iron condors |
| `vix_spike` | VIX z-score > 1.5 | Selling premium on spike |
| `bb_squeeze` | Bollinger Bands contract | Long straddles |
| `rsi_reversal` | RSI bounces off 30/70 | Directional trades |

### Optimal DTE (Tenor)

| DTE | Best For | Why |
|-----|----------|-----|
| 45 | Premium selling | Optimal theta/gamma balance |
| 30 | Aggressive premium | Higher gamma risk |
| 21 | Roll/exit point | Avoid gamma explosion |

## 🔧 Configuration

All parameters can be adjusted in the sidebar:
- **Delta Target**: Strike selection (0.16 = ~84% OTM)
- **DTE**: Days to expiration
- **Profit Target**: When to take profits
- **Stop Loss**: When to cut losses
- **Signal Filter**: Which pattern to use for entry

## 📈 Example Output

```
STRATEGY COMPARISON
====================
Strategy                    Win Rate  Profit Factor  EV/Trade
Short Put (VIX spike)       92.9%     4.75           $1,129
Long Straddle (BB squeeze)  76.5%     2.77           $2,595
Iron Condor (16Δ)           81.7%     1.56           $213

🏆 Best: Short Put with VIX spike signal
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## ⚠️ Disclaimer

This software is for educational purposes only. Options trading involves significant risk of loss. Past performance does not guarantee future results. Always do your own research.

## License

MIT License
