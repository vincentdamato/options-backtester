"""
Pattern Analyzer - Find Repeating Trading Opportunities
========================================================

This module identifies:
1. Repeating patterns like RSI bounces off 30/70
2. VIX mean reversion opportunities
3. IV crush setups
4. Bollinger Band squeeze breakouts
5. Seasonal patterns
6. Time-of-day patterns (for 0DTE)

Think of this as finding "when the next opportunity will arise"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternMatch:
    """A detected pattern instance"""
    date: datetime
    pattern_name: str
    signal_strength: float  # 0-1
    forward_return: float  # Actual return after pattern
    holding_period: int  # Days held
    success: bool  # Did it work?
    details: Dict


@dataclass 
class PatternStats:
    """Statistics for a pattern"""
    pattern_name: str
    occurrences: int
    win_rate: float
    avg_return: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_days_to_profit: float
    best_holding_period: int
    description: str


class PatternAnalyzer:
    """
    Finds repeating patterns in market data
    
    Similar to how you'd look for RSI bouncing off 30/70,
    this finds systematic patterns that precede profitable moves.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with date index and columns:
                  close, high, low, open, volume, vix, iv
        """
        self.data = data.copy()
        self._precompute_indicators()
    
    def _precompute_indicators(self):
        """Calculate all indicators upfront"""
        df = self.data
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving Averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['realized_vol'] = df['returns'].rolling(21).std() * np.sqrt(252)
        
        if 'vix' in df.columns:
            df['vix_sma'] = df['vix'].rolling(20).mean()
            df['vix_zscore'] = (df['vix'] - df['vix'].rolling(252).mean()) / df['vix'].rolling(252).std()
            df['vrp'] = df['vix'] / 100 - df['realized_vol']  # Vol risk premium
        
        if 'iv' in df.columns:
            df['iv_percentile'] = df['iv'].rolling(252).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x) * 100 if len(x) > 1 else 50
            )
        
        # Forward returns for analysis
        for days in [1, 5, 10, 21, 45]:
            df[f'fwd_{days}d'] = df['close'].pct_change(days).shift(-days)
        
        # Day of week
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['day_of_month'] = df.index.day
        
        self.data = df
    
    # =========================================================================
    # PATTERN DETECTORS
    # =========================================================================
    
    def find_rsi_reversals(self, oversold: float = 30, overbought: float = 70,
                           holding_period: int = 10) -> List[PatternMatch]:
        """
        Find RSI reversal patterns
        
        Like the classic bounce off 30/70 levels
        """
        df = self.data
        patterns = []
        
        for i in range(1, len(df) - holding_period):
            rsi_prev = df['rsi'].iloc[i-1]
            rsi_curr = df['rsi'].iloc[i]
            
            # Bullish reversal: crosses above oversold
            if rsi_prev < oversold and rsi_curr >= oversold:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='rsi_bullish_reversal',
                        signal_strength=1 - (rsi_curr / 100),  # Lower RSI = stronger
                        forward_return=fwd_return,
                        holding_period=holding_period,
                        success=fwd_return > 0,
                        details={'rsi': rsi_curr, 'type': 'bullish'}
                    ))
            
            # Bearish reversal: crosses below overbought
            if rsi_prev > overbought and rsi_curr <= overbought:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='rsi_bearish_reversal',
                        signal_strength=rsi_curr / 100,  # Higher RSI = stronger
                        forward_return=fwd_return,
                        holding_period=holding_period,
                        success=fwd_return < 0,
                        details={'rsi': rsi_curr, 'type': 'bearish'}
                    ))
        
        return patterns
    
    def find_vix_spikes(self, zscore_threshold: float = 1.5,
                        holding_period: int = 21) -> List[PatternMatch]:
        """
        Find VIX spike mean reversion opportunities
        
        When VIX spikes above normal, it tends to revert.
        Great for selling premium.
        """
        if 'vix_zscore' not in self.data.columns:
            return []
        
        df = self.data
        patterns = []
        
        for i in range(1, len(df) - holding_period):
            zscore_prev = df['vix_zscore'].iloc[i-1]
            zscore_curr = df['vix_zscore'].iloc[i]
            
            # VIX spike detected
            if zscore_prev < zscore_threshold and zscore_curr >= zscore_threshold:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                vix_change = (df['vix'].iloc[i + holding_period] - df['vix'].iloc[i]) / df['vix'].iloc[i] \
                             if i + holding_period < len(df) else np.nan
                
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='vix_spike',
                        signal_strength=min(1, zscore_curr / 3),  # Cap at z=3
                        forward_return=fwd_return,
                        holding_period=holding_period,
                        success=fwd_return > 0,  # Market usually rallies after VIX spike
                        details={
                            'vix': df['vix'].iloc[i],
                            'vix_zscore': zscore_curr,
                            'vix_change': vix_change
                        }
                    ))
        
        return patterns
    
    def find_bollinger_squeezes(self, squeeze_percentile: float = 20,
                                holding_period: int = 21) -> List[PatternMatch]:
        """
        Find Bollinger Band squeeze setups
        
        Low volatility (narrow bands) often precedes big moves.
        Great for long straddles/strangles.
        """
        df = self.data
        patterns = []
        
        # Calculate squeeze threshold
        squeeze_threshold = df['bb_width'].rolling(100).quantile(squeeze_percentile / 100)
        
        for i in range(100, len(df) - holding_period):
            width_prev = df['bb_width'].iloc[i-1]
            width_curr = df['bb_width'].iloc[i]
            threshold = squeeze_threshold.iloc[i]
            
            # Squeeze detected: width drops below threshold
            if width_prev >= threshold and width_curr < threshold:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                
                # For straddle, we want absolute move
                abs_return = abs(fwd_return) if pd.notna(fwd_return) else np.nan
                
                if pd.notna(abs_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='bb_squeeze',
                        signal_strength=1 - (width_curr / threshold),  # Tighter = stronger
                        forward_return=abs_return,  # Absolute move matters
                        holding_period=holding_period,
                        success=abs_return > 0.03,  # 3% move is success for straddle
                        details={
                            'bb_width': width_curr,
                            'threshold': threshold,
                            'direction': 'up' if fwd_return > 0 else 'down'
                        }
                    ))
        
        return patterns
    
    def find_iv_crush_setups(self, iv_percentile_threshold: float = 80,
                             holding_period: int = 10) -> List[PatternMatch]:
        """
        Find IV crush opportunities
        
        When IV is very high, it tends to crush.
        Great for selling premium before events.
        """
        if 'iv_percentile' not in self.data.columns:
            return []
        
        df = self.data
        patterns = []
        
        for i in range(1, len(df) - holding_period):
            iv_pct = df['iv_percentile'].iloc[i]
            
            if iv_pct >= iv_percentile_threshold:
                # Check if IV actually crushed
                iv_start = df['iv'].iloc[i]
                iv_end = df['iv'].iloc[i + holding_period] if i + holding_period < len(df) else iv_start
                iv_change = (iv_end - iv_start) / iv_start
                
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='iv_crush_setup',
                        signal_strength=iv_pct / 100,
                        forward_return=-iv_change,  # Premium seller profits from IV drop
                        holding_period=holding_period,
                        success=iv_change < -0.10,  # 10% IV drop = success
                        details={
                            'iv_percentile': iv_pct,
                            'iv_start': iv_start,
                            'iv_end': iv_end,
                            'iv_change': iv_change
                        }
                    ))
        
        return patterns
    
    def find_mean_reversion_setups(self, std_threshold: float = 2.0,
                                   holding_period: int = 5) -> List[PatternMatch]:
        """
        Find mean reversion opportunities
        
        Price extended beyond 2 standard deviations tends to revert.
        """
        df = self.data
        patterns = []
        
        for i in range(20, len(df) - holding_period):
            bb_pct = df['bb_pct'].iloc[i]
            
            # Oversold (below lower band)
            if bb_pct < 0:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='mean_reversion_long',
                        signal_strength=abs(bb_pct),
                        forward_return=fwd_return,
                        holding_period=holding_period,
                        success=fwd_return > 0,
                        details={'bb_pct': bb_pct, 'type': 'oversold'}
                    ))
            
            # Overbought (above upper band)
            elif bb_pct > 1:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='mean_reversion_short',
                        signal_strength=bb_pct - 1,
                        forward_return=-fwd_return,  # Short profits from down move
                        holding_period=holding_period,
                        success=fwd_return < 0,
                        details={'bb_pct': bb_pct, 'type': 'overbought'}
                    ))
        
        return patterns
    
    def find_trend_continuation(self, holding_period: int = 21) -> List[PatternMatch]:
        """
        Find trend continuation setups
        
        Price above all MAs in uptrend, below all in downtrend.
        """
        df = self.data
        patterns = []
        
        for i in range(200, len(df) - holding_period):
            close = df['close'].iloc[i]
            sma_10 = df['sma_10'].iloc[i]
            sma_20 = df['sma_20'].iloc[i]
            sma_50 = df['sma_50'].iloc[i]
            sma_200 = df['sma_200'].iloc[i]
            
            # Strong uptrend
            if close > sma_10 > sma_20 > sma_50 > sma_200:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='trend_continuation_bull',
                        signal_strength=0.8,
                        forward_return=fwd_return,
                        holding_period=holding_period,
                        success=fwd_return > 0,
                        details={'trend': 'bullish', 'ma_aligned': True}
                    ))
            
            # Strong downtrend
            elif close < sma_10 < sma_20 < sma_50 < sma_200:
                fwd_return = df[f'fwd_{holding_period}d'].iloc[i]
                if pd.notna(fwd_return):
                    patterns.append(PatternMatch(
                        date=df.index[i],
                        pattern_name='trend_continuation_bear',
                        signal_strength=0.8,
                        forward_return=-fwd_return,
                        holding_period=holding_period,
                        success=fwd_return < 0,
                        details={'trend': 'bearish', 'ma_aligned': True}
                    ))
        
        return patterns
    
    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    
    def analyze_pattern(self, patterns: List[PatternMatch]) -> PatternStats:
        """Analyze a list of pattern matches"""
        if not patterns:
            return None
        
        pattern_name = patterns[0].pattern_name
        
        returns = [p.forward_return for p in patterns if pd.notna(p.forward_return)]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        win_rate = len(wins) / len(returns) if returns else 0
        avg_return = np.mean(returns) if returns else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss
        
        # Find days to first profit
        days_to_profit = [p.holding_period for p in patterns if p.success]
        avg_days = np.mean(days_to_profit) if days_to_profit else 0
        
        return PatternStats(
            pattern_name=pattern_name,
            occurrences=len(patterns),
            win_rate=win_rate,
            avg_return=avg_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_days_to_profit=avg_days,
            best_holding_period=patterns[0].holding_period,
            description=self._get_pattern_description(pattern_name)
        )
    
    def _get_pattern_description(self, name: str) -> str:
        descriptions = {
            'rsi_bullish_reversal': 'RSI crosses above 30 (oversold bounce)',
            'rsi_bearish_reversal': 'RSI crosses below 70 (overbought reversal)',
            'vix_spike': 'VIX spikes above normal (mean reversion opportunity)',
            'bb_squeeze': 'Bollinger Bands contract (breakout imminent)',
            'iv_crush_setup': 'IV at extreme high (crush likely)',
            'mean_reversion_long': 'Price below lower BB (bounce expected)',
            'mean_reversion_short': 'Price above upper BB (pullback expected)',
            'trend_continuation_bull': 'All MAs aligned bullish (trend continuation)',
            'trend_continuation_bear': 'All MAs aligned bearish (trend continuation)'
        }
        return descriptions.get(name, 'Pattern')
    
    def find_all_patterns(self) -> Dict[str, List[PatternMatch]]:
        """Find all patterns in the data"""
        return {
            'rsi_bullish': self.find_rsi_reversals(holding_period=10),
            'vix_spike': self.find_vix_spikes(holding_period=21),
            'bb_squeeze': self.find_bollinger_squeezes(holding_period=21),
            'iv_crush': self.find_iv_crush_setups(holding_period=10),
            'mean_reversion': self.find_mean_reversion_setups(holding_period=5),
            'trend': self.find_trend_continuation(holding_period=21)
        }
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary statistics for all patterns"""
        all_patterns = self.find_all_patterns()
        
        summaries = []
        for name, patterns in all_patterns.items():
            stats = self.analyze_pattern(patterns)
            if stats:
                summaries.append({
                    'Pattern': stats.pattern_name,
                    'Occurrences': stats.occurrences,
                    'Win Rate': f"{stats.win_rate*100:.1f}%",
                    'Avg Return': f"{stats.avg_return*100:.2f}%",
                    'Profit Factor': f"{stats.profit_factor:.2f}",
                    'Avg Days': f"{stats.avg_days_to_profit:.1f}",
                    'Description': stats.description
                })
        
        return pd.DataFrame(summaries)
    
    def predict_next_opportunity(self) -> Dict:
        """
        Based on current market state, predict next pattern opportunity
        
        Returns likelihood of each pattern occurring soon
        """
        df = self.data
        current = df.iloc[-1]
        
        predictions = {}
        
        # RSI approaching oversold
        rsi = current['rsi']
        if rsi < 40:
            predictions['rsi_bullish_reversal'] = {
                'probability': (40 - rsi) / 10,  # Higher as RSI drops
                'current_value': rsi,
                'trigger_level': 30,
                'days_away_estimate': max(1, int((rsi - 30) / 2))
            }
        
        # RSI approaching overbought
        if rsi > 60:
            predictions['rsi_bearish_reversal'] = {
                'probability': (rsi - 60) / 10,
                'current_value': rsi,
                'trigger_level': 70,
                'days_away_estimate': max(1, int((70 - rsi) / 2))
            }
        
        # VIX spike potential
        if 'vix_zscore' in df.columns:
            vix_z = current['vix_zscore']
            if vix_z > 0.5:
                predictions['vix_spike'] = {
                    'probability': min(1, vix_z / 1.5),
                    'current_value': current['vix'],
                    'zscore': vix_z,
                    'trigger_level': 1.5
                }
        
        # BB squeeze forming
        bb_width = current['bb_width']
        bb_width_pct = (df['bb_width'] < bb_width).mean()
        if bb_width_pct < 0.3:  # Width in bottom 30%
            predictions['bb_squeeze'] = {
                'probability': 1 - bb_width_pct,
                'current_width': bb_width,
                'percentile': bb_width_pct * 100,
                'signal': 'Squeeze forming - expect breakout'
            }
        
        # IV crush setup
        if 'iv_percentile' in df.columns:
            iv_pct = current['iv_percentile']
            if iv_pct > 60:
                predictions['iv_crush'] = {
                    'probability': iv_pct / 100,
                    'iv_percentile': iv_pct,
                    'signal': 'High IV - crush likely'
                }
        
        return predictions


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PATTERN ANALYZER - DEMO")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    returns = np.random.normal(0.0003, 0.012, n)
    prices = 300 * np.cumprod(1 + returns)
    vix = 15 + 10 * np.random.randn(n).cumsum() * 0.01
    vix = np.clip(vix + (returns < -0.02) * 5, 10, 80)
    iv = 0.15 + 0.05 * (vix - 15) / 10
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.002),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': prices,
        'vix': vix,
        'iv': iv
    }, index=dates)
    
    print(f"\nAnalyzing {len(data)} days of data...")
    
    # Initialize analyzer
    analyzer = PatternAnalyzer(data)
    
    # Get pattern summary
    print("\n" + "-" * 70)
    print("PATTERN SUMMARY")
    print("-" * 70)
    
    summary = analyzer.get_pattern_summary()
    print("\n" + summary.to_string(index=False))
    
    # Predict next opportunities
    print("\n" + "-" * 70)
    print("NEXT OPPORTUNITY PREDICTIONS")
    print("-" * 70)
    
    predictions = analyzer.predict_next_opportunity()
    for pattern, pred in predictions.items():
        print(f"\n{pattern}:")
        for k, v in pred.items():
            print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("Pattern Analyzer ready!")
    print("=" * 70)
