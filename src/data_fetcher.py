"""
Data Fetcher - Get Real Market Data
====================================

Fetches historical data from free sources:
1. Yahoo Finance - OHLCV, VIX
2. FRED - Economic data
3. CBOE - VIX term structure

For Schwab API integration, see schwab_client.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
import time


class YahooDataFetcher:
    """
    Fetch data from Yahoo Finance
    
    Free, no API key required
    """
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_historical_data(self, symbol: str, period: str = "5y",
                           interval: str = "1d") -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Ticker symbol (SPY, QQQ, ^VIX, etc.)
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            
        Returns:
            DataFrame with date, open, high, low, close, volume
        """
        url = f"{self.BASE_URL}/{symbol}"
        params = {
            'period1': 0,  # Start from beginning
            'period2': int(datetime.now().timestamp()),
            'interval': interval,
            'range': period
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            data = response.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            timestamps = result.get('timestamp', [])
            quote = result.get('indicators', {}).get('quote', [{}])[0]
            
            df = pd.DataFrame({
                'date': pd.to_datetime(timestamps, unit='s'),
                'open': quote.get('open'),
                'high': quote.get('high'),
                'low': quote.get('low'),
                'close': quote.get('close'),
                'volume': quote.get('volume')
            })
            
            df.set_index('date', inplace=True)
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_vix_data(self, period: str = "5y") -> pd.DataFrame:
        """Get VIX historical data"""
        return self.get_historical_data("^VIX", period=period)
    
    def get_spy_data(self, period: str = "5y") -> pd.DataFrame:
        """Get SPY historical data"""
        return self.get_historical_data("SPY", period=period)
    
    def get_multiple_symbols(self, symbols: List[str], 
                             period: str = "5y") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        result = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            result[symbol] = self.get_historical_data(symbol, period=period)
            time.sleep(0.5)  # Rate limiting
        return result


class DataPreparator:
    """
    Prepare data for backtesting
    
    Merges price data with VIX and calculates IV proxy
    """
    
    def __init__(self):
        self.yahoo = YahooDataFetcher()
    
    def prepare_backtest_data(self, symbol: str = "SPY", 
                              period: str = "5y") -> pd.DataFrame:
        """
        Prepare complete dataset for backtesting
        
        Includes:
        - OHLCV data
        - VIX
        - IV proxy (VIX-based)
        - Basic indicators
        """
        # Fetch price data
        price_data = self.yahoo.get_historical_data(symbol, period=period)
        
        if price_data.empty:
            print("Failed to fetch price data")
            return pd.DataFrame()
        
        # Fetch VIX
        vix_data = self.yahoo.get_vix_data(period=period)
        
        # Merge
        if not vix_data.empty:
            price_data['vix'] = vix_data['close'].reindex(price_data.index, method='ffill')
        else:
            # Estimate VIX from realized volatility
            returns = price_data['close'].pct_change()
            price_data['vix'] = returns.rolling(21).std() * np.sqrt(252) * 100
        
        # IV proxy (slightly higher than VIX for SPY options)
        price_data['iv'] = price_data['vix'] / 100 * 1.1  # 10% premium over VIX
        
        # Fill missing values
        price_data = price_data.ffill().dropna()
        
        print(f"Prepared {len(price_data)} rows of data for {symbol}")
        print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
        
        return price_data
    
    def prepare_multi_asset_data(self, symbols: List[str] = None,
                                  period: str = "5y") -> Dict[str, pd.DataFrame]:
        """Prepare data for multiple assets"""
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
        
        result = {}
        for symbol in symbols:
            print(f"\nPreparing {symbol}...")
            result[symbol] = self.prepare_backtest_data(symbol, period)
        
        return result


def generate_synthetic_data(days: int = 1260, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic data for testing
    
    Useful when API is unavailable
    
    Args:
        days: Number of trading days (1260 = ~5 years)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    dates = pd.bdate_range(end=datetime.now(), periods=days)
    
    # Generate realistic returns with volatility clustering
    base_vol = 0.01
    vol = np.zeros(days)
    vol[0] = base_vol
    
    for i in range(1, days):
        # GARCH-like volatility
        vol[i] = 0.94 * vol[i-1] + 0.06 * base_vol + 0.1 * abs(np.random.randn()) * 0.01
    
    returns = np.random.randn(days) * vol
    
    # Add some drift
    returns += 0.0003
    
    # Add occasional jumps (earnings, news)
    jump_days = np.random.choice(days, size=int(days * 0.02), replace=False)
    returns[jump_days] *= 3
    
    # Generate price series
    prices = 400 * np.cumprod(1 + returns)
    
    # Generate VIX (inverse correlation with returns)
    vix = 15 + np.zeros(days)
    for i in range(1, days):
        vix[i] = 0.95 * vix[i-1] + 0.05 * 15 - 500 * returns[i] + np.random.randn() * 0.5
    vix = np.clip(vix, 9, 80)
    
    # IV (slightly higher than VIX)
    iv = vix / 100 * 1.15
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.002),
        'high': prices * (1 + np.abs(np.random.randn(days)) * 0.008),
        'low': prices * (1 - np.abs(np.random.randn(days)) * 0.008),
        'close': prices,
        'volume': np.random.randint(50000000, 150000000, days),
        'vix': vix,
        'iv': iv
    }, index=dates)
    
    # Ensure high > open, close and low < open, close
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA FETCHER - DEMO")
    print("=" * 70)
    
    # Try to fetch real data
    print("\n1. Attempting to fetch real data...")
    
    prep = DataPreparator()
    
    try:
        data = prep.prepare_backtest_data("SPY", period="2y")
        if not data.empty:
            print("\nReal data fetched successfully!")
            print(data.tail())
        else:
            raise Exception("Empty data")
    except Exception as e:
        print(f"\nCouldn't fetch real data: {e}")
        print("\n2. Generating synthetic data instead...")
        
        data = generate_synthetic_data(days=504)  # ~2 years
        print("\nSynthetic data generated!")
        print(data.tail())
    
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nSummary statistics:")
    print(data.describe())
