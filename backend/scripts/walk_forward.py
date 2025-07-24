# scripts/walk_forward.py

import pandas as pd
import numpy as np
import logging
import random
from typing import Dict, Any
from deap import base, creator, tools, algorithms
import yfinance as yf
import sys
import os

# --- Add the project root (backend) to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import the integrated components from your scalping platform ---
from app.core.engines.backtester import Backtester, BacktestConfig, DataProcessor
from app.core.config import TradingConfig

# --- A GLOBAL TOOLBOX FOR THE GENETIC ALGORITHM ---
toolbox = base.Toolbox()

def setup_deap_toolbox():
    """Sets up the DEAP toolbox with genes and operators for the EnhancedScalpingEngine."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # --- Define the "Genes" to be optimized for YOUR engine ---
    toolbox.register("attr_momentum_weight", random.uniform, 1.5, 2.5)
    toolbox.register("attr_volume_weight", random.uniform, 1.5, 2.5)
    toolbox.register("attr_candle_weight", random.uniform, 1.0, 2.0)
    
    # --- THIS IS THE CHANGE: Allow the GA to test lower, more sensitive thresholds ---
    toolbox.register("attr_bull_thresh", random.uniform, 0.8, 2.5)
    toolbox.register("attr_strong_bull_thresh", random.uniform, 3.0, 5.0)

    attributes = (
        toolbox.attr_momentum_weight,
        toolbox.attr_volume_weight,
        toolbox.attr_candle_weight,
        toolbox.attr_bull_thresh,
        toolbox.attr_strong_bull_thresh
    )
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_fitness(individual, prepared_training_data: pd.DataFrame):
    """The fitness function for the genetic algorithm."""
    mom_weight, vol_weight, candle_weight, bull_thresh, strong_bull_thresh = individual
    
    dynamic_trading_config = TradingConfig()
    dynamic_trading_config.base_weights['momentum'] = mom_weight
    dynamic_trading_config.base_weights['volume'] = vol_weight
    dynamic_trading_config.base_weights['candle_analysis'] = candle_weight
    dynamic_trading_config.base_thresholds = {
        'strong_bull': strong_bull_thresh, 'bull': bull_thresh,
        'bear': -bull_thresh, 'strong_bear': -strong_bull_thresh
    }
    
    backtester = Backtester(trading_config=dynamic_trading_config)
    results = backtester.run(prepared_training_data, data_is_prepared=True)
    
    return (results.get('sharpe_ratio', -10.0),)

def optimize_parameters(prepared_training_data: pd.DataFrame) -> Dict[str, float]:
    """Runs the Genetic Algorithm to find the best parameters."""
    toolbox.register("evaluate", evaluate_fitness, prepared_training_data=prepared_training_data)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=15, halloffame=hof, verbose=False)
    
    best_ind = hof[0]
    return {
        "momentum_weight": best_ind[0], 
        "volume_weight": best_ind[1],
        "candle_weight": best_ind[2],
        "bull_thresh": best_ind[3], 
        "strong_bull_thresh": best_ind[4],
        "sharpe": best_ind.fitness.values[0]
    }

def load_historical_data(ticker: str = 'SPY', period: str = '60d', interval: str = '5m') -> pd.DataFrame:
    """
    Loads and cleans high-frequency historical data using yfinance.
    """
    print(f"Downloading {interval} data for {ticker} over the last {period}...")
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("No data downloaded.")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).lower() for col in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Downloaded data is missing one of the required columns: {required_cols}")
        
        print(f"Successfully downloaded and cleaned {len(df)} data points.")
        return df
    except Exception as e:
        print(f"Error downloading or cleaning data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    full_data_df = load_historical_data()
    if full_data_df.empty:
        print("Could not load data. Exiting.")
        exit()

    opt_window = 21 * 78
    val_window = 7 * 78
    step = val_window

    all_validation_results = []
    
    setup_deap_toolbox()
    
    start_index = opt_window
    while start_index + val_window < len(full_data_df):
        train_end = start_index
        val_end = start_index + val_window
        
        train_data = full_data_df.iloc[train_end - opt_window : train_end]
        val_data = full_data_df.iloc[train_end : val_end]
        
        period_start = val_data.index[0].strftime('%Y-%m-%d')
        period_end = val_data.index[-1].strftime('%Y-%m-%d')
        print(f"\n--- Processing Period: {period_start} to {period_end} ---")

        print(f"Preparing {len(train_data)} bars of training data...")
        data_processor = DataProcessor(config=BacktestConfig()) 
        prepared_train_data = data_processor.prepare_data(train_data)
        if prepared_train_data.empty:
            print("  > Not enough data to prepare for this training period. Skipping.")
            start_index += step
            continue
        
        print("Optimizing parameters with Genetic Algorithm...")
        best_params = optimize_parameters(prepared_train_data)
        print(f"  > Best params found with Sharpe: {best_params['sharpe']:.2f}")

        print(f"Validating on {len(val_data)} bars of unseen data...")
        
        optimized_config = TradingConfig()
        optimized_config.base_weights['momentum'] = best_params['momentum_weight']
        optimized_config.base_weights['volume'] = best_params['volume_weight']
        optimized_config.base_weights['candle_analysis'] = best_params['candle_weight']
        optimized_config.base_thresholds = {
            'strong_bull': best_params['strong_bull_thresh'], 'bull': best_params['bull_thresh'],
            'bear': -best_params['bull_thresh'], 'strong_bear': -best_params['strong_bull_thresh']
        }
        
        validation_backtester = Backtester(trading_config=optimized_config)
        validation_results = validation_backtester.run(val_data)
        all_validation_results.append(validation_results)
        
        pnl = validation_results.get('total_pnl', 0)
        sharpe = validation_results.get('sharpe_ratio', 0)
        drawdown = validation_results.get('max_drawdown_pct', 0)
        trades = validation_results.get('total_trades', 0)
        print(f"  > Validation Results: PnL=${pnl:.2f}, Sharpe={sharpe:.2f}, DD={drawdown:.2f}%, Trades={trades}")

        start_index += step

    print("\n--- Walk-Forward Analysis Complete ---")
    total_pnl = sum(res.get('total_pnl', 0) for res in all_validation_results if res)
    num_trades = sum(res.get('total_trades', 0) for res in all_validation_results if res)
    print(f"Combined PnL across all validation periods: ${total_pnl:,.2f}")
    print(f"Total number of trades: {num_trades}")
