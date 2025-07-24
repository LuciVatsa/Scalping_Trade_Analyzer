# scripts/walk_forward.py

import pandas as pd
import numpy as np
import logging
import random
from typing import Dict, Any
from deap import base, creator, tools, algorithms

# --- Import the integrated components from your scalping platform ---
from backend.app.core.engines.backtester import Backtester, BacktestConfig, DataProcessor
from backend.app.core.config import TradingConfig

# --- A GLOBAL TOOLBOX FOR THE GENETIC ALGORITHM ---
toolbox = base.Toolbox()

def setup_deap_toolbox():
    """Sets up the DEAP toolbox with genes and operators for the EnhancedScalpingEngine."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # We want to maximize the fitness value
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # --- Define the "Genes" to be optimized for YOUR engine ---
    # These will control the weights and thresholds in TradingConfig.
    toolbox.register("attr_momentum_weight", random.uniform, 1.5, 2.5)
    toolbox.register("attr_volume_weight", random.uniform, 1.5, 2.5)
    toolbox.register("attr_candle_weight", random.uniform, 1.0, 2.0)
    toolbox.register("attr_bull_thresh", random.uniform, 1.5, 3.5)
    toolbox.register("attr_strong_bull_thresh", random.uniform, 4.0, 7.0)

    # Create an individual with these genes
    attributes = (
        toolbox.attr_momentum_weight,
        toolbox.attr_volume_weight,
        toolbox.attr_candle_weight,
        toolbox.attr_bull_thresh,
        toolbox.attr_strong_bull_thresh
    )
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register GA operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_fitness(individual, prepared_training_data: pd.DataFrame):
    """
    The fitness function. It runs a backtest with the parameters from the individual
    and returns a fitness score (e.g., Sharpe Ratio).
    """
    # Unpack the genes from the individual
    mom_weight, vol_weight, candle_weight, bull_thresh, strong_bull_thresh = individual
    
    # --- Dynamically create a TradingConfig for this specific run ---
    dynamic_trading_config = TradingConfig()
    dynamic_trading_config.base_weights['momentum'] = mom_weight
    dynamic_trading_config.base_weights['volume'] = vol_weight
    dynamic_trading_config.base_weights['candle_analysis'] = candle_weight # Using the proxy strategy
    dynamic_trading_config.base_thresholds = {
        'strong_bull': strong_bull_thresh, 'bull': bull_thresh,
        'bear': -bull_thresh, 'strong_bear': -strong_bull_thresh
    }
    
    # Initialize the backtester with this dynamic config
    backtester = Backtester(trading_config=dynamic_trading_config)
    
    # Run the backtest on the pre-prepared training data
    results = backtester.run(prepared_training_data, data_is_prepared=True)
    
    # The fitness is the Sharpe Ratio. Return as a tuple.
    return (results.get('sharpe_ratio', -10.0),)

def optimize_parameters(prepared_training_data: pd.DataFrame) -> Dict[str, float]:
    """
    Runs the Genetic Algorithm to find the best parameters for the given data.
    """
    # Register the fitness function with the data it needs
    toolbox.register("evaluate", evaluate_fitness, prepared_training_data=prepared_training_data)

    pop = toolbox.population(n=30) # Population size
    hof = tools.HallOfFame(1) # Hall of Fame to store the best individual
    
    # Run the genetic algorithm
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=20, halloffame=hof, verbose=False)
    
    best_ind = hof[0]
    return {
        "momentum_weight": best_ind[0], 
        "volume_weight": best_ind[1],
        "candle_weight": best_ind[2],
        "bull_thresh": best_ind[3], 
        "strong_bull_thresh": best_ind[4],
        "sharpe": best_ind.fitness.values[0]
    }

def load_historical_data() -> pd.DataFrame:
    """
    Placeholder function for loading high-frequency historical data.
    Replace this with your Schwab API client logic.
    """
    print("Loading historical data... (Using mock data for now)")
    # In a real scenario, you would fetch this from Schwab, save it to a file,
    # and load it here to avoid re-downloading every time.
    # The data should be 1-minute or 5-minute OHLCV data.
    
    # Creating a sample DataFrame for demonstration
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10000, freq="5min"))
    price = 150 + np.random.randn(10000).cumsum() * 0.1
    data = {
        'open': price,
        'high': price + np.random.rand(10000) * 0.2,
        'low': price - np.random.rand(10000) * 0.2,
        'close': price + (np.random.randn(10000) * 0.1),
        'volume': np.random.randint(1000, 5000, 10000)
    }
    df = pd.DataFrame(data, index=dates)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load the full historical dataset (replace with your data source)
    full_data_df = load_historical_data()

    # 2. Setup Walk-Forward Parameters
    opt_window = 252 * 78 # Approx. 2 years of 5-min bars (252 days * 78 5-min bars/day)
    val_window = 63 * 78  # Approx. 3 months
    step = val_window

    all_validation_results = []
    
    # 3. Initialize DEAP and run the Walk-Forward Loop
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

        # Prepare training data ONCE per fold
        print(f"Preparing {len(train_data)} bars of training data...")
        data_processor = DataProcessor(config=BacktestConfig()) 
        prepared_train_data = data_processor.prepare_data(train_data)
        
        # Optimize on training data
        print("Optimizing parameters with Genetic Algorithm...")
        best_params = optimize_parameters(prepared_train_data)
        print(f"  > Best params found with Sharpe: {best_params['sharpe']:.2f}")

        # Test on validation data with the best parameters found
        print(f"Validating on {len(val_data)} bars of unseen data...")
        
        # Create a TradingConfig with the optimized parameters
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
        
        # Print results for this fold
        pnl = validation_results.get('total_pnl', 0)
        sharpe = validation_results.get('sharpe_ratio', 0)
        drawdown = validation_results.get('max_drawdown_pct', 0)
        trades = validation_results.get('total_trades', 0)
        print(f"  > Validation Results: PnL=${pnl:.2f}, Sharpe={sharpe:.2f}, DD={drawdown:.2f}%, Trades={trades}")

        start_index += step

    # 4. Analyze overall results
    print("\n--- Walk-Forward Analysis Complete ---")
    total_pnl = sum(res.get('total_pnl', 0) for res in all_validation_results if res)
    num_trades = sum(res.get('total_trades', 0) for res in all_validation_results if res)
    print(f"Combined PnL across all validation periods: ${total_pnl:,.2f}")
    print(f"Total number of trades: {num_trades}")
