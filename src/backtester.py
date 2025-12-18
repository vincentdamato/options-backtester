"""
Options Strategy Backtester with Pattern Detection
===================================================

Core backtesting engine that:
1. Tests options strategies across historical data
2. Detects repeating patterns (like RSI bounces)
3. Calculates profit factor, EV, and all key metrics
4. Identifies optimal entry/exit conditions

Author: Options Backtesting Project
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyType(Enum):
    """Options strategy types"""
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    SHORT_STRANGLE = "short_strangle"
    SHORT_STRADDLE = "short_straddle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"


@dataclass
class Trade:
    """Single trade record"""
    entry_date: datetime
    exit_date: datetime
    strategy: StrategyType
    entry_price: float
    exit_price: float
    underlying_entry: float
    underlying_exit: float
    pnl: float
    pnl_pct: float
    days_held: int
    entry_signal: str
    entry_iv: float = 0.0
    entry_vix: float = 0.0
    entry_rsi: float = 0.0
    max_drawdown: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Complete backtest results"""
    strategy: StrategyType
    trades: List[Trade]
    total_pnl: float
    total_return: float
    win_rate: float
    profit_factor: float
    expected_value: float
    sharpe_ratio: float
    max_drawdown: float
    avg_days_held: float
    num_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    win_streak_max: int
    loss_streak_max: int
    equity_curve: pd.Series
    
    def summary(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKTEST RESULTS: {self.strategy.value.upper():^42} ║
╠══════════════════════════════════════════════════════════════════╣
║  Total P&L:        ${self.total_pnl:>12,.2f}                      
║  Total Return:      {self.total_return*100:>12.2f}%                
║  # Trades:          {self.num_trades:>12}                         
║  Win Rate:          {self.win_rate*100:>12.1f}%                    
║  Profit Factor:     {self.profit_factor:>12.2f}                    
║  Expected Value:   ${self.expected_value:>12.2f} per trade         
║  Sharpe Ratio:      {self.sharpe_ratio:>12.2f}                     
║  Max Drawdown:      {self.max_drawdown*100:>12.1f}%                
║  Avg Days Held:     {self.avg_days_held:>12.1f}                    
╠══════════════════════════════════════════════════════════════════╣
║  Avg Win:          ${self.avg_win:>12.2f}                          
║  Avg Loss:         ${self.avg_loss:>12.2f}                         
║  Largest Win:      ${self.largest_win:>12.2f}                      
║  Largest Loss:     ${self.largest_loss:>12.2f}                     
║  Max Win Streak:    {self.win_streak_max:>12}                      
║  Max Loss Streak:   {self.loss_streak_max:>12}                     
╚══════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# TECHNICAL INDICATORS FOR PATTERN DETECTION
# =============================================================================

class Indicators:
    """
    Technical indicators for finding entry/exit patterns
    
    Similar to how RSI shows overbought/oversold bounces,
    we look for repeating patterns in volatility and price.
    """
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        RSI > 70 = Overbought (potential reversal down)
        RSI < 30 = Oversold (potential reversal up)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Price near upper band = extended, potential mean reversion
        Price near lower band = oversold, potential bounce
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def iv_percentile(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
        """
        IV Percentile (IV Rank)
        
        Shows where current IV sits relative to past year
        High IV% = good for selling premium
        Low IV% = good for buying premium
        """
        def calc_percentile(x):
            if len(x) < 2:
                return 50
            return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100
        
        return iv_series.rolling(window=lookback, min_periods=20).apply(calc_percentile)
    
    @staticmethod
    def vix_mean_reversion(vix: pd.Series, lookback: int = 252) -> pd.Series:
        """
        VIX Mean Reversion Signal
        
        VIX tends to mean-revert. Extreme readings signal opportunities.
        Z-score > 2 = VIX spike, good for selling premium
        Z-score < -1 = VIX crushed, be careful selling premium
        """
        mean = vix.rolling(window=lookback).mean()
        std = vix.rolling(window=lookback).std()
        z_score = (vix - mean) / std
        return z_score
    
    @staticmethod
    def put_call_ratio(put_volume: pd.Series, call_volume: pd.Series,
                       period: int = 5) -> pd.Series:
        """
        Put/Call Ratio (smoothed)
        
        High P/C = Fear, potential bottom
        Low P/C = Complacency, potential top
        """
        ratio = put_volume / call_volume
        return ratio.rolling(window=period).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Average True Range - volatility measure
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def moving_average_crossover(prices: pd.Series, 
                                  fast: int = 10, slow: int = 30) -> pd.Series:
        """
        MA Crossover Signal
        
        1 = Fast above slow (bullish)
        -1 = Fast below slow (bearish)
        """
        fast_ma = prices.rolling(window=fast).mean()
        slow_ma = prices.rolling(window=slow).mean()
        return np.sign(fast_ma - slow_ma)
    
    @staticmethod
    def volatility_regime(vix: pd.Series, 
                          low_threshold: float = 15,
                          high_threshold: float = 25) -> pd.Series:
        """
        Classify volatility regime
        
        0 = Low vol (VIX < 15)
        1 = Normal vol (15 <= VIX < 25)
        2 = High vol (VIX >= 25)
        """
        regime = pd.Series(index=vix.index, dtype=int)
        regime[vix < low_threshold] = 0
        regime[(vix >= low_threshold) & (vix < high_threshold)] = 1
        regime[vix >= high_threshold] = 2
        return regime


# =============================================================================
# SIGNAL GENERATORS (Pattern Detection)
# =============================================================================

class SignalGenerator:
    """
    Generate trading signals based on patterns
    
    These are the "repeating patterns" you're looking for -
    conditions that historically precede profitable trades.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with columns: date, close, high, low, vix, iv
        """
        self.data = data.copy()
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """Pre-calculate all indicators"""
        df = self.data
        
        # Price-based indicators
        df['rsi'] = Indicators.rsi(df['close'])
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = Indicators.bollinger_bands(df['close'])
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility indicators
        if 'vix' in df.columns:
            df['vix_zscore'] = Indicators.vix_mean_reversion(df['vix'])
            df['vol_regime'] = Indicators.volatility_regime(df['vix'])
        
        if 'iv' in df.columns:
            df['iv_percentile'] = Indicators.iv_percentile(df['iv'])
        
        # Trend indicators
        df['ma_signal'] = Indicators.moving_average_crossover(df['close'])
        
        if 'high' in df.columns and 'low' in df.columns:
            df['atr'] = Indicators.atr(df['high'], df['low'], df['close'])
            df['atr_pct'] = df['atr'] / df['close'] * 100
        
        self.data = df
    
    def premium_selling_signal(self) -> pd.Series:
        """
        Signal for premium selling strategies (short puts, iron condors, etc.)
        
        Best conditions for selling premium:
        1. High IV (IV percentile > 50)
        2. VIX elevated but not spiking (zscore between 0.5 and 2)
        3. RSI not extreme (between 35-65 for neutral strategies)
        4. Price within Bollinger Bands
        
        Returns:
            Series with values: 1 (strong signal), 0.5 (weak signal), 0 (no signal)
        """
        df = self.data
        signal = pd.Series(0.0, index=df.index)
        
        # Condition 1: IV is elevated
        if 'iv_percentile' in df.columns:
            iv_condition = df['iv_percentile'] > 50
        else:
            iv_condition = pd.Series(True, index=df.index)
        
        # Condition 2: VIX is favorable
        if 'vix_zscore' in df.columns:
            vix_condition = (df['vix_zscore'] > 0) & (df['vix_zscore'] < 2.5)
        else:
            vix_condition = pd.Series(True, index=df.index)
        
        # Condition 3: RSI is neutral (for non-directional strategies)
        rsi_neutral = (df['rsi'] > 35) & (df['rsi'] < 65)
        
        # Condition 4: Price not at extremes
        bb_condition = (df['bb_pct'] > 0.2) & (df['bb_pct'] < 0.8)
        
        # Combine conditions
        strong_signal = iv_condition & vix_condition & rsi_neutral & bb_condition
        weak_signal = iv_condition & (vix_condition | rsi_neutral)
        
        signal[weak_signal] = 0.5
        signal[strong_signal] = 1.0
        
        return signal
    
    def vix_spike_signal(self) -> pd.Series:
        """
        Signal for VIX spike mean reversion
        
        When VIX spikes (z-score > 1.5), it tends to revert.
        This is a great time to sell premium.
        
        Returns:
            Series: 1 = spike detected, 0 = no spike
        """
        df = self.data
        if 'vix_zscore' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # VIX spike: z-score > 1.5
        spike = (df['vix_zscore'] > 1.5).astype(int)
        return spike
    
    def rsi_reversal_signal(self, oversold: float = 30, 
                            overbought: float = 70) -> pd.Series:
        """
        RSI Reversal Signal
        
        Like the classic RSI bounce off 30/70:
        - RSI crosses above 30 = bullish reversal
        - RSI crosses below 70 = bearish reversal
        
        Returns:
            Series: 1 = bullish, -1 = bearish, 0 = neutral
        """
        df = self.data
        rsi = df['rsi']
        
        signal = pd.Series(0, index=df.index)
        
        # Bullish: RSI was below 30, now crossing above
        bullish = (rsi.shift(1) < oversold) & (rsi >= oversold)
        
        # Bearish: RSI was above 70, now crossing below
        bearish = (rsi.shift(1) > overbought) & (rsi <= overbought)
        
        signal[bullish] = 1
        signal[bearish] = -1
        
        return signal
    
    def bollinger_squeeze_signal(self) -> pd.Series:
        """
        Bollinger Band Squeeze Signal
        
        When bands contract (low volatility), a breakout often follows.
        Good for long straddle/strangle entries.
        
        Returns:
            Series: 1 = squeeze detected, 0 = no squeeze
        """
        df = self.data
        
        # Band width as percentage
        band_width = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # Squeeze: band width in lowest 20% of recent history
        squeeze_threshold = band_width.rolling(window=100).quantile(0.2)
        squeeze = (band_width < squeeze_threshold).astype(int)
        
        return squeeze
    
    def trend_following_signal(self) -> pd.Series:
        """
        Trend Following Signal
        
        For directional strategies (credit spreads, etc.)
        
        Returns:
            Series: 1 = bullish trend, -1 = bearish trend, 0 = no trend
        """
        df = self.data
        
        # Combine MA crossover with RSI confirmation
        ma_signal = df['ma_signal']
        
        # RSI confirms trend
        rsi_bullish = df['rsi'] > 50
        rsi_bearish = df['rsi'] < 50
        
        signal = pd.Series(0, index=df.index)
        signal[(ma_signal == 1) & rsi_bullish] = 1
        signal[(ma_signal == -1) & rsi_bearish] = -1
        
        return signal
    
    def iv_crush_setup_signal(self, days_before_earnings: int = 5) -> pd.Series:
        """
        IV Crush Setup Signal
        
        IV tends to rise before events and crush after.
        This signal identifies high IV that's likely to crush.
        
        Note: Requires earnings dates in data
        
        Returns:
            Series: 1 = IV crush likely, 0 = not detected
        """
        df = self.data
        if 'iv_percentile' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # High IV (top 20%)
        high_iv = df['iv_percentile'] > 80
        
        # Recent IV expansion (IV increased over last 5 days)
        iv_expanding = df['iv'].diff(5) > 0 if 'iv' in df.columns else pd.Series(True, index=df.index)
        
        signal = (high_iv & iv_expanding).astype(int)
        return signal
    
    def get_all_signals(self) -> pd.DataFrame:
        """Get all signals in a single DataFrame"""
        signals = pd.DataFrame(index=self.data.index)
        signals['premium_sell'] = self.premium_selling_signal()
        signals['vix_spike'] = self.vix_spike_signal()
        signals['rsi_reversal'] = self.rsi_reversal_signal()
        signals['bb_squeeze'] = self.bollinger_squeeze_signal()
        signals['trend'] = self.trend_following_signal()
        signals['iv_crush'] = self.iv_crush_setup_signal()
        return signals


# =============================================================================
# OPTIONS PRICING (Simplified for Backtesting)
# =============================================================================

class OptionPricer:
    """
    Simplified option pricing for backtesting
    
    Uses Black-Scholes approximations for speed
    """
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str = 'put') -> float:
        """Quick BS price calculation"""
        from scipy.stats import norm
        
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            return max(0, K - S)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, 
              sigma: float, option_type: str = 'put') -> float:
        """Calculate delta"""
        from scipy.stats import norm
        
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        return norm.cdf(d1) - 1
    
    @staticmethod
    def find_strike_by_delta(S: float, T: float, r: float, sigma: float,
                             target_delta: float, option_type: str = 'put') -> float:
        """Find strike price for a given delta"""
        from scipy.optimize import brentq
        
        def objective(K):
            return OptionPricer.delta(S, K, T, r, sigma, option_type) - target_delta
        
        # Search bounds
        if option_type == 'put':
            low, high = S * 0.7, S * 1.0
        else:
            low, high = S * 1.0, S * 1.3
        
        try:
            return brentq(objective, low, high)
        except:
            return S * (1 + target_delta) if option_type == 'call' else S * (1 + target_delta)


# =============================================================================
# MAIN BACKTESTER
# =============================================================================

class OptionsBacktester:
    """
    Main backtesting engine
    
    Runs strategy backtests with pattern-based entry signals
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        """
        Args:
            data: DataFrame with columns: date, open, high, low, close, vix, iv
            initial_capital: Starting capital
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.signal_generator = SignalGenerator(data)
        self.pricer = OptionPricer()
        
    def backtest_short_put(self,
                           delta_target: float = -0.16,
                           dte_target: int = 45,
                           profit_target: float = 0.50,
                           stop_loss: float = 2.0,
                           signal_filter: str = 'premium_sell',
                           signal_threshold: float = 0.5) -> BacktestResult:
        """
        Backtest short put strategy
        
        Args:
            delta_target: Target delta for put (negative)
            dte_target: Days to expiration at entry
            profit_target: Exit at this % of max profit
            stop_loss: Exit at this multiple of credit received
            signal_filter: Which signal to use for entry
            signal_threshold: Minimum signal strength to enter
        """
        df = self.data.copy()
        signals = self.signal_generator.get_all_signals()
        
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        
        i = 0
        while i < len(df) - dte_target - 1:
            row = df.iloc[i]
            
            # Check entry signal
            if signal_filter in signals.columns:
                signal_value = signals[signal_filter].iloc[i]
                if signal_value < signal_threshold:
                    i += 1
                    equity_curve.append(capital)
                    continue
            
            # Entry conditions
            S = row['close']
            iv = row.get('iv', 0.20)
            vix = row.get('vix', 15)
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns else 50
            
            T = dte_target / 365
            r = 0.05
            
            # Find strike at target delta
            K = self.pricer.find_strike_by_delta(S, T, r, iv, delta_target, 'put')
            
            # Entry premium (credit received)
            entry_premium = self.pricer.black_scholes_price(S, K, T, r, iv, 'put')
            
            if entry_premium <= 0.01:
                i += 1
                equity_curve.append(capital)
                continue
            
            # Simulate trade through expiration or exit
            max_profit = entry_premium
            exit_price = None
            exit_reason = ""
            exit_idx = i
            max_dd = 0
            
            for j in range(1, dte_target + 1):
                if i + j >= len(df):
                    break
                
                future_row = df.iloc[i + j]
                S_now = future_row['close']
                iv_now = future_row.get('iv', iv)
                T_now = (dte_target - j) / 365
                
                # Current option price
                current_price = self.pricer.black_scholes_price(S_now, K, T_now, r, iv_now, 'put')
                
                # P&L (short position: profit when price decreases)
                current_pnl = entry_premium - current_price
                
                # Track max drawdown
                if current_pnl < 0:
                    max_dd = min(max_dd, current_pnl)
                
                # Check profit target
                if current_pnl >= max_profit * profit_target:
                    exit_price = current_price
                    exit_reason = "profit_target"
                    exit_idx = i + j
                    break
                
                # Check stop loss
                if current_pnl <= -entry_premium * stop_loss:
                    exit_price = current_price
                    exit_reason = "stop_loss"
                    exit_idx = i + j
                    break
            
            # If no exit triggered, close at expiration
            if exit_price is None:
                final_row = df.iloc[min(i + dte_target, len(df) - 1)]
                S_final = final_row['close']
                exit_price = max(0, K - S_final)  # Intrinsic value at expiration
                exit_reason = "expiration"
                exit_idx = min(i + dte_target, len(df) - 1)
            
            # Calculate final P&L
            pnl = (entry_premium - exit_price) * 100  # Per contract
            pnl_pct = pnl / (K * 100)  # % of notional
            
            # Record trade
            trade = Trade(
                entry_date=df.index[i] if isinstance(df.index[i], datetime) else datetime.now(),
                exit_date=df.index[exit_idx] if isinstance(df.index[exit_idx], datetime) else datetime.now(),
                strategy=StrategyType.SHORT_PUT,
                entry_price=entry_premium,
                exit_price=exit_price,
                underlying_entry=S,
                underlying_exit=df.iloc[exit_idx]['close'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                days_held=exit_idx - i,
                entry_signal=signal_filter,
                entry_iv=iv,
                entry_vix=vix,
                entry_rsi=rsi,
                max_drawdown=max_dd,
                exit_reason=exit_reason
            )
            trades.append(trade)
            
            capital += pnl
            equity_curve.append(capital)
            
            # Move to next potential entry (after this trade exits)
            i = exit_idx + 1
        
        # Pad equity curve
        while len(equity_curve) < len(df):
            equity_curve.append(capital)
        
        return self._compile_results(trades, equity_curve, StrategyType.SHORT_PUT)
    
    def backtest_iron_condor(self,
                             put_delta: float = -0.16,
                             call_delta: float = 0.16,
                             wing_width: float = 0.05,
                             dte_target: int = 45,
                             profit_target: float = 0.50,
                             stop_loss: float = 2.0,
                             signal_filter: str = 'premium_sell',
                             signal_threshold: float = 0.5) -> BacktestResult:
        """
        Backtest iron condor strategy
        
        Short put spread + short call spread
        """
        df = self.data.copy()
        signals = self.signal_generator.get_all_signals()
        
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        
        i = 0
        while i < len(df) - dte_target - 1:
            row = df.iloc[i]
            
            # Check entry signal
            if signal_filter in signals.columns:
                signal_value = signals[signal_filter].iloc[i]
                if signal_value < signal_threshold:
                    i += 1
                    equity_curve.append(capital)
                    continue
            
            S = row['close']
            iv = row.get('iv', 0.20)
            vix = row.get('vix', 15)
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns else 50
            
            T = dte_target / 365
            r = 0.05
            
            # Find strikes
            short_put_K = self.pricer.find_strike_by_delta(S, T, r, iv, put_delta, 'put')
            short_call_K = self.pricer.find_strike_by_delta(S, T, r, iv, call_delta, 'call')
            long_put_K = short_put_K * (1 - wing_width)
            long_call_K = short_call_K * (1 + wing_width)
            
            # Entry premium (net credit)
            short_put_premium = self.pricer.black_scholes_price(S, short_put_K, T, r, iv, 'put')
            long_put_premium = self.pricer.black_scholes_price(S, long_put_K, T, r, iv, 'put')
            short_call_premium = self.pricer.black_scholes_price(S, short_call_K, T, r, iv, 'call')
            long_call_premium = self.pricer.black_scholes_price(S, long_call_K, T, r, iv, 'call')
            
            entry_credit = (short_put_premium - long_put_premium + 
                           short_call_premium - long_call_premium)
            
            if entry_credit <= 0.05:
                i += 1
                equity_curve.append(capital)
                continue
            
            # Max risk is wing width minus credit
            max_risk = max(short_put_K - long_put_K, long_call_K - short_call_K) - entry_credit
            
            # Simulate trade
            exit_value = None
            exit_reason = ""
            exit_idx = i
            max_dd = 0
            
            for j in range(1, dte_target + 1):
                if i + j >= len(df):
                    break
                
                future_row = df.iloc[i + j]
                S_now = future_row['close']
                iv_now = future_row.get('iv', iv)
                T_now = (dte_target - j) / 365
                
                # Current spread values
                sp_now = self.pricer.black_scholes_price(S_now, short_put_K, T_now, r, iv_now, 'put')
                lp_now = self.pricer.black_scholes_price(S_now, long_put_K, T_now, r, iv_now, 'put')
                sc_now = self.pricer.black_scholes_price(S_now, short_call_K, T_now, r, iv_now, 'call')
                lc_now = self.pricer.black_scholes_price(S_now, long_call_K, T_now, r, iv_now, 'call')
                
                current_value = sp_now - lp_now + sc_now - lc_now
                current_pnl = (entry_credit - current_value)
                
                if current_pnl < 0:
                    max_dd = min(max_dd, current_pnl)
                
                # Check profit target
                if current_pnl >= entry_credit * profit_target:
                    exit_value = current_value
                    exit_reason = "profit_target"
                    exit_idx = i + j
                    break
                
                # Check stop loss
                if current_pnl <= -entry_credit * stop_loss:
                    exit_value = current_value
                    exit_reason = "stop_loss"
                    exit_idx = i + j
                    break
            
            # Expiration
            if exit_value is None:
                final_S = df.iloc[min(i + dte_target, len(df) - 1)]['close']
                # Calculate expiration value
                put_spread_value = max(0, short_put_K - final_S) - max(0, long_put_K - final_S)
                call_spread_value = max(0, final_S - short_call_K) - max(0, final_S - long_call_K)
                exit_value = put_spread_value + call_spread_value
                exit_reason = "expiration"
                exit_idx = min(i + dte_target, len(df) - 1)
            
            pnl = (entry_credit - exit_value) * 100
            pnl_pct = pnl / (max_risk * 100)
            
            trade = Trade(
                entry_date=df.index[i] if isinstance(df.index[i], datetime) else datetime.now(),
                exit_date=df.index[exit_idx] if isinstance(df.index[exit_idx], datetime) else datetime.now(),
                strategy=StrategyType.IRON_CONDOR,
                entry_price=entry_credit,
                exit_price=exit_value,
                underlying_entry=S,
                underlying_exit=df.iloc[exit_idx]['close'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                days_held=exit_idx - i,
                entry_signal=signal_filter,
                entry_iv=iv,
                entry_vix=vix,
                entry_rsi=rsi,
                max_drawdown=max_dd,
                exit_reason=exit_reason
            )
            trades.append(trade)
            
            capital += pnl
            equity_curve.append(capital)
            
            i = exit_idx + 1
        
        while len(equity_curve) < len(df):
            equity_curve.append(capital)
        
        return self._compile_results(trades, equity_curve, StrategyType.IRON_CONDOR)
    
    def backtest_long_straddle(self,
                               dte_target: int = 45,
                               profit_target: float = 1.0,
                               stop_loss: float = 0.50,
                               signal_filter: str = 'bb_squeeze',
                               signal_threshold: float = 1.0) -> BacktestResult:
        """
        Backtest long straddle (buy ATM call + put)
        
        Best entered during low volatility (BB squeeze)
        """
        df = self.data.copy()
        signals = self.signal_generator.get_all_signals()
        
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        
        i = 0
        while i < len(df) - dte_target - 1:
            row = df.iloc[i]
            
            # Check entry signal (BB squeeze for long vol)
            if signal_filter in signals.columns:
                signal_value = signals[signal_filter].iloc[i]
                if signal_value < signal_threshold:
                    i += 1
                    equity_curve.append(capital)
                    continue
            
            S = row['close']
            K = S  # ATM
            iv = row.get('iv', 0.20)
            vix = row.get('vix', 15)
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns else 50
            
            T = dte_target / 365
            r = 0.05
            
            # Entry cost (debit)
            call_price = self.pricer.black_scholes_price(S, K, T, r, iv, 'call')
            put_price = self.pricer.black_scholes_price(S, K, T, r, iv, 'put')
            entry_cost = call_price + put_price
            
            if entry_cost <= 0.10:
                i += 1
                equity_curve.append(capital)
                continue
            
            # Simulate trade
            exit_value = None
            exit_reason = ""
            exit_idx = i
            max_dd = 0
            
            for j in range(1, dte_target + 1):
                if i + j >= len(df):
                    break
                
                future_row = df.iloc[i + j]
                S_now = future_row['close']
                iv_now = future_row.get('iv', iv)
                T_now = (dte_target - j) / 365
                
                call_now = self.pricer.black_scholes_price(S_now, K, T_now, r, iv_now, 'call')
                put_now = self.pricer.black_scholes_price(S_now, K, T_now, r, iv_now, 'put')
                current_value = call_now + put_now
                
                current_pnl = current_value - entry_cost
                
                if current_pnl < 0:
                    max_dd = min(max_dd, current_pnl)
                
                # Profit target (100% gain)
                if current_pnl >= entry_cost * profit_target:
                    exit_value = current_value
                    exit_reason = "profit_target"
                    exit_idx = i + j
                    break
                
                # Stop loss (50% of premium paid)
                if current_pnl <= -entry_cost * stop_loss:
                    exit_value = current_value
                    exit_reason = "stop_loss"
                    exit_idx = i + j
                    break
            
            # Expiration
            if exit_value is None:
                final_S = df.iloc[min(i + dte_target, len(df) - 1)]['close']
                exit_value = abs(final_S - K)  # Intrinsic value
                exit_reason = "expiration"
                exit_idx = min(i + dte_target, len(df) - 1)
            
            pnl = (exit_value - entry_cost) * 100
            pnl_pct = pnl / (entry_cost * 100)
            
            trade = Trade(
                entry_date=df.index[i] if isinstance(df.index[i], datetime) else datetime.now(),
                exit_date=df.index[exit_idx] if isinstance(df.index[exit_idx], datetime) else datetime.now(),
                strategy=StrategyType.LONG_STRADDLE,
                entry_price=entry_cost,
                exit_price=exit_value,
                underlying_entry=S,
                underlying_exit=df.iloc[exit_idx]['close'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                days_held=exit_idx - i,
                entry_signal=signal_filter,
                entry_iv=iv,
                entry_vix=vix,
                entry_rsi=rsi,
                max_drawdown=max_dd,
                exit_reason=exit_reason
            )
            trades.append(trade)
            
            capital += pnl
            equity_curve.append(capital)
            
            i = exit_idx + 1
        
        while len(equity_curve) < len(df):
            equity_curve.append(capital)
        
        return self._compile_results(trades, equity_curve, StrategyType.LONG_STRADDLE)
    
    def _compile_results(self, trades: List[Trade], equity_curve: List[float],
                         strategy: StrategyType) -> BacktestResult:
        """Compile trade list into BacktestResult"""
        if not trades:
            return BacktestResult(
                strategy=strategy,
                trades=[],
                total_pnl=0,
                total_return=0,
                win_rate=0,
                profit_factor=0,
                expected_value=0,
                sharpe_ratio=0,
                max_drawdown=0,
                avg_days_held=0,
                num_trades=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                win_streak_max=0,
                loss_streak_max=0,
                equity_curve=pd.Series(equity_curve)
            )
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total_pnl = sum(pnls)
        total_return = total_pnl / self.initial_capital
        win_rate = len(wins) / len(trades) if trades else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss
        
        expected_value = np.mean(pnls) if pnls else 0
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / 45)  # Annualized
        else:
            sharpe = 0
        
        # Max drawdown
        equity = pd.Series(equity_curve)
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_dd = abs(drawdowns.min())
        
        # Win/loss streaks
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        return BacktestResult(
            strategy=strategy,
            trades=trades,
            total_pnl=total_pnl,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expected_value=expected_value,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_days_held=np.mean([t.days_held for t in trades]),
            num_trades=len(trades),
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            win_streak_max=max_win_streak,
            loss_streak_max=max_loss_streak,
            equity_curve=pd.Series(equity_curve)
        )
    
    def analyze_patterns(self) -> pd.DataFrame:
        """
        Analyze which patterns are most predictive
        
        Returns correlation between signals and future returns
        """
        signals = self.signal_generator.get_all_signals()
        df = self.data.copy()
        
        # Calculate forward returns at various horizons
        for horizon in [5, 10, 21, 45]:
            df[f'fwd_ret_{horizon}d'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Merge signals with forward returns
        analysis = pd.concat([signals, df[[c for c in df.columns if 'fwd_ret' in c]]], axis=1)
        
        # Calculate correlation
        correlations = analysis.corr()
        
        return correlations


# =============================================================================
# MAIN - DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OPTIONS BACKTESTER - DEMO")
    print("=" * 70)
    
    # Generate synthetic data for demo
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    # Simulate SPY-like price action
    returns = np.random.normal(0.0003, 0.012, n)
    prices = 300 * np.cumprod(1 + returns)
    
    # Simulate VIX (inverse correlation with prices)
    vix = 15 + 10 * np.random.randn(n).cumsum() * 0.01
    vix = np.clip(vix + (returns < -0.02) * 5, 10, 80)  # Spike on big down days
    
    # Simulate IV
    iv = 0.15 + 0.05 * (vix - 15) / 10
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n) * 0.002),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': prices,
        'vix': vix,
        'iv': iv
    })
    data.set_index('date', inplace=True)
    
    print(f"\nGenerated {len(data)} days of synthetic data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"VIX range: {data['vix'].min():.1f} - {data['vix'].max():.1f}")
    
    # Initialize backtester
    bt = OptionsBacktester(data)
    
    # Run backtests
    print("\n" + "-" * 70)
    print("RUNNING BACKTESTS...")
    print("-" * 70)
    
    # 1. Short Put with premium selling signal
    print("\n1. Short Put (16 delta, 45 DTE, premium_sell signal)")
    result_sp = bt.backtest_short_put(
        delta_target=-0.16,
        dte_target=45,
        profit_target=0.50,
        stop_loss=2.0,
        signal_filter='premium_sell',
        signal_threshold=0.5
    )
    print(result_sp.summary())
    
    # 2. Iron Condor
    print("\n2. Iron Condor (16 delta wings, 45 DTE)")
    result_ic = bt.backtest_iron_condor(
        put_delta=-0.16,
        call_delta=0.16,
        dte_target=45,
        profit_target=0.50,
        stop_loss=2.0,
        signal_filter='premium_sell',
        signal_threshold=0.5
    )
    print(result_ic.summary())
    
    # 3. Long Straddle with BB squeeze signal
    print("\n3. Long Straddle (ATM, BB squeeze signal)")
    result_ls = bt.backtest_long_straddle(
        dte_target=45,
        profit_target=1.0,
        stop_loss=0.50,
        signal_filter='bb_squeeze',
        signal_threshold=1.0
    )
    print(result_ls.summary())
    
    # Compare strategies
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Strategy': [r.strategy.value for r in [result_sp, result_ic, result_ls]],
        'Win Rate': [f"{r.win_rate*100:.1f}%" for r in [result_sp, result_ic, result_ls]],
        'Profit Factor': [f"{r.profit_factor:.2f}" for r in [result_sp, result_ic, result_ls]],
        'EV/Trade': [f"${r.expected_value:.2f}" for r in [result_sp, result_ic, result_ls]],
        'Sharpe': [f"{r.sharpe_ratio:.2f}" for r in [result_sp, result_ic, result_ls]],
        'Max DD': [f"{r.max_drawdown*100:.1f}%" for r in [result_sp, result_ic, result_ls]],
        '# Trades': [r.num_trades for r in [result_sp, result_ic, result_ls]]
    })
    print("\n" + comparison.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Backtester ready for your data!")
    print("=" * 70)
