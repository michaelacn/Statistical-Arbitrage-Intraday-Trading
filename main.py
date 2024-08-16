import sys
import argparse

import numpy as np
import pandas as pd

from lib.dash_app.dash_app import *
from lib.models.models import *
from lib.data.data import *
from lib.backtest.backtest import *
from lib.utils.utils import *

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/basic.yaml", help="Path to the config file.")
    opts = parser.parse_args()
    return opts


def get_trading_config(args):
    """
    Identifies trading days and parameters when intraday log prices are suitable for trading based on two-stage cointegration analysis.
    """
    trading_dates = []
    trading_params = {'spread_mean': [], 'spread_std': [], 'beta': [], 'const': [], 'entry_factor': [], 'sl_factor': [], 'warmup_period': []}

    print('----------------------------------------')
    print('######### FETCH TRADING CONFIG #########')
    print('----------------------------------------')

    print(f'[INFO DATA] Retrieving common timestamps for {args.symbols}')
    hlocv_data, prices, volumes = get_data_and_preprocess(args.csv_file, args.symbols)
    logprices = np.log(prices)
    dates = pd.Series(hlocv_data.index.date).unique()

    for current_date in dates:
        print(f'----------------{current_date}----------------')
        current_date_logprices = filter_current_date(logprices, current_date)

        data_train = current_date_logprices[:args.train_size]
        if len(data_train) < args.train_size:
            print('[INFO DATA] Not enough data, skipping.')
            continue

        check_stationary = {col: is_stationary(data_train[col]) for col in data_train.columns}
        if any(check_stationary.values()) is True:
            print('[INFO TEST] One or more series are stationary, skipping.')
            continue            

        # Find the optimal lag for the VECM
        var_model, optimal_lag = fit_var_model(data_train)
        
        # Performs two-stage cointegration analysis
        vecm_model, model_type = perform_cointegration_analysis(data_train, optimal_lag)
            
        # Check if any cointegration relationship was found
        if not vecm_model:
            print('[INFO TEST] No cointegration relationship, skipping.')
            continue  

        # Calculate the VECM spread
        beta = vecm_model.beta[1][0]
        const = vecm_model.det_coef_coint[0][0] if vecm_model.det_coef_coint.size > 0 else 0
        spread = data_train @ np.array([1.0, beta]) + const  
        
        print(f'[INFO MODEL] {model_type}.')
        trading_dates.append(current_date)
        trading_params['spread_mean'].append(spread.mean())
        trading_params['spread_std'].append(spread.std())
        trading_params['beta'].append(beta)
        trading_params['const'].append(const)
        trading_params['entry_factor'].append(args.entry_factor)
        trading_params['sl_factor'].append(args.sl_factor)
        trading_params['warmup_period'].append(args.warmup_period)

    trading_config = pd.DataFrame(trading_params, index=trading_dates)
    return trading_config, hlocv_data


def backtest_trading_config(args, trading_config: pd.DataFrame, hlocv_data: pd.DataFrame) -> tuple:
    """
    Runs a trading simulation over specified periods to obtain a complete trade history.
    """
    # Dynamically retrieve strategy and analyzer classes from names in the configuration.
    strategy = getattr(sys.modules[__name__], args.strategy)
    current_cash = args.cash
    analyzers_config = {}
    for analyzer_name, analyzer_params in args.analyzers_config.items():
        analyzers_config[getattr(sys.modules[__name__], analyzer_name)] = analyzer_params

    trades_history = pd.DataFrame()
    pnl_history = pd.Series(dtype=float)

    print('--------------------------------------------')
    print('######### START TRADING SIMULATION #########')
    print('--------------------------------------------')

    for current_date, params in trading_config.iterrows():
        print(f'----------------{current_date}----------------')
        hlocv_current_date = filter_current_date(hlocv_data, current_date)

        # Skip if not enough trading data
        if len(hlocv_current_date) < args.warmup_period * 2:
            print(f"[INFO] {current_date}: Not enough trading data, skipping.")
            continue

        # Convert parameters to a dictionary and update them for the current day
        updated_params = params.to_dict()
        updated_params['warmup_period'] = int(updated_params['warmup_period'])

        # Run the strategy with the updated parameters
        strat, pf_val = run_bt(hlocv_current_date, updated_params, args.symbols, strategy, current_cash, args.comm_pct, analyzers_config)
        analyzers_results = get_analyzers_results(strat)
        orders = analyzers_results['orders']
        trades = analyzers_results['trades']

        # Skip if no trade activity for the day
        if orders.empty:
            print(f"[INFO] {current_date}: No trade activity, skipping.")
            continue
        
        # Process and accumulate the day's orders, trades, and P&L
        trades_history = pd.concat([trades_history, trades])
        pnl_history[current_date] = pf_val - current_cash
        current_cash = pf_val

        # Log P&L and current portfolio value
        print(f'[INFO PNL] {current_date} | P&L: {pnl_history[current_date]:.2f} | Portfolio Value: {current_cash:.2f}')

    return trades_history.reset_index(drop=True), pnl_history


if __name__ == "__main__": 
    opts = parse_args()
    args = get_config(opts.config)
    trading_config, hlocv_data = get_trading_config(args)
    trades_history, pnl_history = backtest_trading_config(args, trading_config, hlocv_data)
    performance_metrics, returns_history = analyze_strategy(args, pnl_history)
    run_dash_app(args, trades_history, pnl_history, returns_history, performance_metrics, hlocv_data)
