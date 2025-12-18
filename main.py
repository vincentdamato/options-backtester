#!/usr/bin/env python3
"""
Options Strategy Backtester - Main Runner
==========================================

Run this script to:
1. Fetch or generate market data
2. Find repeating patterns
3. Backtest multiple strategies
4. Compare results and find the best setup

Usage:
    python main.py                    # Run with synthetic data
    python main.py --real             # Try to fetch real data
    python main.py --symbol QQQ       # Use different symbol
    python main.py --years 3          # Change lookback period
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtester import OptionsBacktester, BacktestResult, StrategyType
from patterns import PatternAnalyzer
from data_fetcher import DataPreparator, generate_synthetic_data

import pandas as pd
import numpy as np


def run_full_analysis(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run complete analysis pipeline
    
    1. Find patterns
    2. Backtest strategies
    3. Compare results
    """
    results = {}
    
    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: PATTERN ANALYSIS")
        print("=" * 70)
    
    analyzer = PatternAnalyzer(data)
    pattern_summary = analyzer.get_pattern_summary()
    
    if verbose:
        print("\nDetected Patterns:")
        print(pattern_summary.to_string(index=False))
    
    results['patterns'] = pattern_summary
    
    # Predict next opportunities
    predictions = analyzer.predict_next_opportunity()
    results['predictions'] = predictions
    
    if verbose and predictions:
        print("\n" + "-" * 70)
        print("UPCOMING OPPORTUNITIES:")
        for pattern, pred in predictions.items():
            prob = pred.get('probability', 0)
            if prob > 0.3:
                print(f"\n  {pattern}: {prob*100:.0f}% likely")
                for k, v in pred.items():
                    if k != 'probability':
                        print(f"    {k}: {v}")
    
    # =========================================================================
    # STRATEGY BACKTESTING
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: STRATEGY BACKTESTING")
        print("=" * 70)
    
    bt = OptionsBacktester(data)
    
    # Test multiple strategies with different parameters
    strategies_to_test = [
        # Short Put variations
        ('Short Put (16Δ, premium signal)', 'short_put', {
            'delta_target': -0.16, 'dte_target': 45,
            'profit_target': 0.50, 'stop_loss': 2.0,
            'signal_filter': 'premium_sell', 'signal_threshold': 0.5
        }),
        ('Short Put (16Δ, VIX spike)', 'short_put', {
            'delta_target': -0.16, 'dte_target': 45,
            'profit_target': 0.50, 'stop_loss': 2.0,
            'signal_filter': 'vix_spike', 'signal_threshold': 1.0
        }),
        ('Short Put (30Δ, aggressive)', 'short_put', {
            'delta_target': -0.30, 'dte_target': 30,
            'profit_target': 0.50, 'stop_loss': 1.5,
            'signal_filter': 'premium_sell', 'signal_threshold': 0.5
        }),
        
        # Iron Condor variations
        ('Iron Condor (16Δ wings)', 'iron_condor', {
            'put_delta': -0.16, 'call_delta': 0.16,
            'dte_target': 45, 'profit_target': 0.50, 'stop_loss': 2.0,
            'signal_filter': 'premium_sell', 'signal_threshold': 0.5
        }),
        ('Iron Condor (10Δ wings, wide)', 'iron_condor', {
            'put_delta': -0.10, 'call_delta': 0.10,
            'dte_target': 45, 'profit_target': 0.50, 'stop_loss': 2.0,
            'signal_filter': 'premium_sell', 'signal_threshold': 0.5
        }),
        
        # Long Straddle (volatility buying)
        ('Long Straddle (BB squeeze)', 'long_straddle', {
            'dte_target': 45, 'profit_target': 1.0, 'stop_loss': 0.50,
            'signal_filter': 'bb_squeeze', 'signal_threshold': 1.0
        }),
    ]
    
    backtest_results = []
    
    for name, strategy_type, params in strategies_to_test:
        if verbose:
            print(f"\nTesting: {name}...")
        
        try:
            if strategy_type == 'short_put':
                result = bt.backtest_short_put(**params)
            elif strategy_type == 'iron_condor':
                result = bt.backtest_iron_condor(**params)
            elif strategy_type == 'long_straddle':
                result = bt.backtest_long_straddle(**params)
            else:
                continue
            
            backtest_results.append({
                'name': name,
                'result': result
            })
            
            if verbose:
                print(f"  Trades: {result.num_trades}, Win Rate: {result.win_rate*100:.1f}%, "
                      f"PF: {result.profit_factor:.2f}, Sharpe: {result.sharpe_ratio:.2f}")
        
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
    
    results['backtests'] = backtest_results
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: STRATEGY COMPARISON")
        print("=" * 70)
    
    comparison_data = []
    for bt_result in backtest_results:
        r = bt_result['result']
        comparison_data.append({
            'Strategy': bt_result['name'],
            'Trades': r.num_trades,
            'Win Rate': f"{r.win_rate*100:.1f}%",
            'Profit Factor': f"{r.profit_factor:.2f}",
            'EV/Trade': f"${r.expected_value:.2f}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Max DD': f"{r.max_drawdown*100:.1f}%",
            'Total Return': f"{r.total_return*100:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    results['comparison'] = comparison_df
    
    if verbose:
        print("\n" + comparison_df.to_string(index=False))
    
    # Find best strategy
    if backtest_results:
        # Rank by profit factor (primary) and Sharpe (secondary)
        sorted_results = sorted(backtest_results, 
                               key=lambda x: (x['result'].profit_factor, x['result'].sharpe_ratio),
                               reverse=True)
        
        best = sorted_results[0]
        results['best_strategy'] = best
        
        if verbose:
            print("\n" + "-" * 70)
            print(f"BEST STRATEGY: {best['name']}")
            print("-" * 70)
            print(best['result'].summary())
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Options Strategy Backtester')
    parser.add_argument('--real', action='store_true', help='Fetch real market data')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to analyze')
    parser.add_argument('--years', type=int, default=5, help='Years of history')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTIONS STRATEGY BACKTESTER")
    print("=" * 70)
    
    # Get data
    if args.real:
        print(f"\nFetching real data for {args.symbol}...")
        prep = DataPreparator()
        try:
            data = prep.prepare_backtest_data(args.symbol, period=f"{args.years}y")
            if data.empty:
                raise Exception("Empty data returned")
            print(f"Loaded {len(data)} days of real data")
        except Exception as e:
            print(f"Failed to fetch real data: {e}")
            print("Falling back to synthetic data...")
            data = generate_synthetic_data(days=args.years * 252)
    else:
        print("\nGenerating synthetic data...")
        data = generate_synthetic_data(days=args.years * 252)
        print(f"Generated {len(data)} days of synthetic data")
    
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"VIX range: {data['vix'].min():.1f} - {data['vix'].max():.1f}")
    
    # Run analysis
    results = run_full_analysis(data, verbose=not args.quiet)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    if 'predictions' in results and results['predictions']:
        print("\n📊 ACTIONABLE SIGNALS:")
        for pattern, pred in results['predictions'].items():
            prob = pred.get('probability', 0)
            if prob > 0.5:
                print(f"  ⚡ {pattern}: {prob*100:.0f}% probability - consider entering soon")
    
    if 'best_strategy' in results:
        best = results['best_strategy']
        print(f"\n🏆 RECOMMENDED STRATEGY: {best['name']}")
        print(f"   Expected value: ${best['result'].expected_value:.2f} per trade")
        print(f"   Profit factor: {best['result'].profit_factor:.2f}")
    
    print("\n✅ Results saved. Run with --real to use live market data.")
    
    return results


if __name__ == "__main__":
    main()
