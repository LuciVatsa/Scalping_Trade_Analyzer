# backend/app/core/engines/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import logging

# --- Integration with the Scalping Platform's Core Components ---
from app.core.engines.scalping_engine import EnhancedScalpingEngine
from app.core.engines.risk_engine import EnhancedRiskManager
from app.core.config import TradingConfig
from app.core.models.signals import ScalpingInputs, SignalResult
from app.core.models.enums import SignalDirection

# --- Configuration and Data Structures for Backtesting ---

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005   # 0.05% slippage
    risk_per_trade: float = 0.02    # 2% of account per trade
    lookback_period: int = 50       # Min historical data for indicators
    take_profit_risk_ratio: float = 1.5 # Default Risk/Reward ratio

@dataclass
class Trade:
    """Represents a single trade during the backtest."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    direction: str = 'long'
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    signal_strength: str = ''
    exit_reason: str = ''
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

# --- Data Preparation ---

class DataProcessor:
    """Handles data preparation and indicator calculation for backtesting."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares OHLCV data with all indicators required by the scalping strategies.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Calculate all required technical indicators
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['atr'] = self._calculate_atr(df)
        df['vwap'] = self._calculate_daily_vwap(df)
        
        # Ensure there are no NaN values in critical columns after calculations
        df.dropna(inplace=True)
        return df

    def create_trading_inputs(self, df: pd.DataFrame, index: int) -> Dict:
        """
        Creates the input dictionary that matches the ScalpingInputs model
        for the EnhancedScalpingEngine at a specific point in time.
        """
        lookback = self.config.lookback_period
        start_idx = max(0, index - lookback)
        
        current_bar = df.iloc[index]
        price_data = df.iloc[start_idx:index+1]
        
        # Map historical data and calculated indicators to ScalpingInputs fields
        inputs = {
            # Core Market Data
            'current_price': current_bar['close'],
            'price_close_history': price_data['close'].tolist(),
            'price_high_history': price_data['high'].tolist(),
            'price_low_history': price_data['low'].tolist(),
            'volume': int(current_bar['volume']),
            'volume_history': price_data['volume'].tolist(),
            'vwap_value': current_bar['vwap'],

            # Microstructure (Mocked for backtesting)
            'bid_price': current_bar['close'] - (current_bar['atr'] * 0.05),
            'ask_price': current_bar['close'] + (current_bar['atr'] * 0.05),
            'bid_ask_spread': current_bar['atr'] * 0.1,
            'last_trade_size': 10,
            'uptick_volume': current_bar['volume'] * 0.6,
            'downtick_volume': current_bar['volume'] * 0.4,

            # Options Data (Mocked for backtesting - requires dedicated options data source for accuracy)
            'strike_price': round(current_bar['close']),
            'dte': 5,
            'iv_percentile': 0.5,
            'implied_volatility': 0.25,
            'option_price': 2.50,
            'option_delta': 0.5,
            'option_gamma': 0.05,
            'option_theta': -0.10,
            'option_volume': 100,
            'open_interest': 500,

            # Context and Indicators
            'current_time': current_bar.name.strftime('%H:%M'),
            'vix_level': 20.0, # Using a stable VIX for backtesting consistency
            'atr_value': current_bar['atr'],
            'bb_top': current_bar['bb_upper'],
            'bb_mid': current_bar['bb_middle'],
            'bb_bot': current_bar['bb_lower'],

            # User/Account Inputs (Capital is updated dynamically)
            'account_size': self.config.initial_capital,
            'risk_percent': self.config.risk_per_trade * 100,
        }
        
        return inputs
    
    # --- Helper methods for indicator calculations ---
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std * std_dev), sma, sma - (std * std_dev)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_daily_vwap(self, df: pd.DataFrame) -> pd.Series:
        df['typical_price_vol'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
        day_starts = df.index.date != df.index.to_series().shift(1).dt.date
        daily_cumulative_vol = df['volume'].groupby(day_starts.cumsum()).cumsum()
        daily_cumulative_tpv = df['typical_price_vol'].groupby(day_starts.cumsum()).cumsum()
        vwap = daily_cumulative_tpv / (daily_cumulative_vol + 1e-9) # Add epsilon to avoid division by zero
        df.drop(columns=['typical_price_vol'], inplace=True)
        return vwap

# --- Main Backtester Class ---

class Backtester:
    """
    Simulates trading using the EnhancedScalpingEngine and EnhancedRiskManager
    to evaluate performance on historical data.
    """
    
    def __init__(self, config: BacktestConfig = None, trading_config: TradingConfig = None):
        self.config = config or BacktestConfig()
        self.engine = EnhancedScalpingEngine(config=trading_config)
        self.risk_manager = EnhancedRiskManager()
        self.data_processor = DataProcessor(self.config)
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
        self.capital = self.config.initial_capital
        self.equity_curve = []
        
    def run(self, df: pd.DataFrame, data_is_prepared: bool = False) -> Dict:
        """Runs the backtest on the provided historical data."""
        self.logger.info(f"Starting backtest with {len(df)} data points...")
        
        if not data_is_prepared:
            df = self.data_processor.prepare_data(df)
            self.logger.info("Data preparation complete.")
        
        self._reset_state()
        
        for i in range(self.config.lookback_period, len(df)):
            self._process_bar(df, i)
            self._update_equity_curve(df.iloc[i])
        
        if self.current_position:
            self._close_position(df.iloc[-1], "End of backtest")
        
        return self._generate_results()
    
    def _reset_state(self):
        """Resets the backtester's state for a new run."""
        self.trades = []
        self.current_position = None
        self.capital = self.config.initial_capital
        self.equity_curve = []

    def _process_bar(self, df: pd.DataFrame, index: int):
        """Processes a single bar of data, gets a signal, and acts on it."""
        try:
            inputs = self.data_processor.create_trading_inputs(df, index)
            inputs['account_size'] = self.capital
            
            result: SignalResult = self.engine.analyze_signal(inputs)
            
            self._update_position(df.iloc[index], result)
            
        except Exception as e:
            self.logger.error(f"Error processing bar {index}: {e}", exc_info=True)
    
    def _update_position(self, bar_data: pd.Series, result: SignalResult):
        """Updates the current position based on the trading signal."""
        signal = result.direction
        
        if self.current_position is None:
            if signal in [SignalDirection.BULLISH, SignalDirection.STRONG_BULLISH]:
                self._enter_position(bar_data, 'long', result)
            elif signal in [SignalDirection.BEARISH, SignalDirection.STRONG_BEARISH]:
                self._enter_position(bar_data, 'short', result)
        else:
            current_price = bar_data['close']
            exit_reason = self._check_exit_conditions(current_price, signal)
            if exit_reason:
                self._close_position(bar_data, exit_reason)

    def _enter_position(self, bar_data: pd.Series, direction: str, result: SignalResult):
        """Enters a new position using the EnhancedRiskManager."""
        entry_price = bar_data['close']
        
        stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
            current_price=entry_price, atr=bar_data.get('atr', 0),
            trend_direction=direction, vix_level=20.0
        )
        
        confidence_map = {"High": 1.0, "Moderate-High": 0.85, "Moderate": 0.75, "Moderate-Low": 0.6, "Low": 0.5, "N/A": 0.5}
        signal_confidence = confidence_map.get(result.confidence, 0.5)

        position_data = self.risk_manager.calculate_enhanced_position_size(
            account_value=self.capital, risk_percent=self.config.risk_per_trade * 100,
            current_price=entry_price, stop_loss_price=stop_loss,
            signal_confidence=signal_confidence
        )
        
        if not position_data or position_data['recommended_size'] <= 0:
            return

        quantity = int(position_data['recommended_size'])
        take_profit = entry_price + (abs(entry_price - stop_loss) * self.config.take_profit_risk_ratio) if direction == 'long' else entry_price - (abs(entry_price - stop_loss) * self.config.take_profit_risk_ratio)

        if quantity > 0:
            slippage = entry_price * self.config.slippage_pct
            commission = entry_price * quantity * self.config.commission_rate
            adjusted_entry_price = entry_price + slippage if direction == 'long' else entry_price - slippage
            
            self.current_position = Trade(
                entry_time=bar_data.name, entry_price=adjusted_entry_price,
                quantity=quantity, direction=direction, signal_strength=result.strength,
                commission=commission, slippage=slippage, stop_loss=stop_loss,
                take_profit=take_profit
            )
            self.capital -= commission
            self.logger.debug(f"Entered {direction} position: {quantity} shares at {adjusted_entry_price:.2f}")

    def _check_exit_conditions(self, current_price: float, signal: SignalDirection) -> Optional[str]:
        """Checks all exit conditions for the current position."""
        if not self.current_position:
            return None

        pos = self.current_position
        
        # Check Stop-Loss and Take-Profit
        if pos.direction == 'long':
            if pos.stop_loss and current_price <= pos.stop_loss: return "Stop-Loss Hit"
            if pos.take_profit and current_price >= pos.take_profit: return "Take-Profit Hit"
            if signal in [SignalDirection.BEARISH, SignalDirection.STRONG_BEARISH]: return "Opposing Signal"
        else: # Short position
            if pos.stop_loss and current_price >= pos.stop_loss: return "Stop-Loss Hit"
            if pos.take_profit and current_price <= pos.take_profit: return "Take-Profit Hit"
            if signal in [SignalDirection.BULLISH, SignalDirection.STRONG_BULLISH]: return "Opposing Signal"
            
        return None

    def _close_position(self, bar_data: pd.Series, reason: str):
        """Closes the current active position."""
        if not self.current_position: return

        exit_price = bar_data['close']
        pos = self.current_position
        
        slippage = exit_price * self.config.slippage_pct
        commission = exit_price * pos.quantity * self.config.commission_rate
        adjusted_exit_price = exit_price - slippage if pos.direction == 'long' else exit_price + slippage
        
        pnl = (adjusted_exit_price - pos.entry_price) * pos.quantity if pos.direction == 'long' else (pos.entry_price - adjusted_exit_price) * pos.quantity
        net_pnl = pnl - (pos.commission + commission)
        
        self.capital += net_pnl
        
        pos.exit_time = bar_data.name
        pos.exit_price = adjusted_exit_price
        pos.pnl = net_pnl
        pos.exit_reason = reason
        
        self.trades.append(pos)
        self.current_position = None
        self.logger.debug(f"Closed position: P&L {net_pnl:.2f}, Reason: {reason}")

    def _update_equity_curve(self, bar_data: pd.Series):
        """Updates the equity curve with the current portfolio value."""
        equity = self.capital
        if self.current_position:
            unrealized_pnl = (bar_data['close'] - self.current_position.entry_price) * self.current_position.quantity if self.current_position.direction == 'long' else (self.current_position.entry_price - bar_data['close']) * self.current_position.quantity
            equity += unrealized_pnl
        self.equity_curve.append({'timestamp': bar_data.name, 'equity': equity})

    def _generate_results(self) -> Dict:
        """Generates a dictionary of comprehensive backtest results."""
        if not self.trades: return {"message": "No trades were executed."}
        
        pnls = [t.pnl for t in self.trades]
        total_return = ((self.capital - self.config.initial_capital) / self.config.initial_capital) * 100
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len([p for p in pnls if p > 0]),
            'losing_trades': len([p for p in pnls if p <= 0]),
            'win_rate': (len([p for p in pnls if p > 0]) / len(pnls) * 100) if pnls else 0,
            'total_pnl': sum(pnls),
            'average_pnl': np.mean(pnls),
            'profit_factor': abs(sum(p for p in pnls if p > 0) / (sum(p for p in pnls if p < 0) + 1e-9)),
            'total_return_pct': total_return,
            'final_equity': self.capital,
            'max_drawdown_pct': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'trades': [t.__dict__ for t in self.trades],
            'equity_curve': self.equity_curve
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculates the maximum drawdown of the equity curve."""
        if not self.equity_curve: return 0.0
        equity = pd.Series([p['equity'] for p in self.equity_curve]).cummax()
        drawdown = (equity - pd.Series([p['equity'] for p in self.equity_curve])) / equity
        return drawdown.max() * 100

    def _calculate_sharpe_ratio(self) -> float:
        """Calculates the annualized Sharpe ratio."""
        if not self.equity_curve: return 0.0
        returns = pd.Series([p['equity'] for p in self.equity_curve]).pct_change().dropna()
        if returns.std() == 0: return 0.0
        # Assuming 252 trading days in a year, adapt if using different timeframes
        return (returns.mean() / returns.std()) * np.sqrt(252)
