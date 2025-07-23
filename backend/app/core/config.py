# backend/app/core/config.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ScalpingConstants:
    """
    Central repository for fixed constants and thresholds used across strategies.
    Fine-tuning these values can alter the sensitivity and behavior of the engine.
    """
    # Signal Quality & Confirmation
    MIN_CONFIRMATION_COUNT: int = 2
    STRONG_SIGNAL_THRESHOLD: float = 1.2
    CONFIRMATION_DECAY_FACTOR: float = 0.8

    # Volume Analysis
    UNUSUAL_VOLUME_SPIKE_FACTOR: float = 2.5
    HIGH_VOLUME_FACTOR: float = 1.6
    VOLUME_CONFIRMATION_WINDOW: int = 5

    # Risk & Options Filtering
    DEFAULT_RISK_PER_TRADE: float = 50.0
    MAX_BID_ASK_SPREAD_PCT: float = 0.04
    MIN_OPTION_PRICE: float = 0.05
    MAX_THETA_DECAY_PER_HOUR: float = 0.10
    CONSERVATIVE_MODE_MINUTES_TO_EXP: int = 120
    FORCE_EXIT_MINUTES_TO_CLOSE: int = 30
    
    # Event-Based Risk
    AVOID_TRADE_MINUTES_BEFORE_FOMC: int = 30
    
    # Price Movement Thresholds
    MIN_PRICE_MOVE_BPS: int = 5  # Basis points
    STRONG_MOVE_BPS: int = 15 # Basis points
    
    # Systemic Risk
    MAX_GAMMA_EXPOSURE_THRESHOLD: float = 1e9


@dataclass
class TradingConfig:
    """
    The main configuration for the trading engine. This class defines the weights
    of each strategy, allowing you to control the engine's overall behavior by
    adjusting how much influence each analytical component has.
    """
    # --- Primary Control Panel: Strategy Weights ---
    # Adjust these values to change the importance of each strategy.
    # Higher values give a strategy more influence on the final score.
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 2.0,          # Strongest driver for short-term direction
        'volume': 1.8,            # High importance for confirming moves
        'microstructure': 1.6,    # Captures immediate order flow pressure
        'levels': 1.7,            # Key support/resistance is critical
        'vwap': 1.5,              # A standard institutional benchmark
        'bollinger_bands': 1.3,   # Good for volatility and breakout patterns
        'gamma_levels': 1.4,      # Important for identifying market turning points
        'market_structure': 1.2,  # Provides broader market context
        'time_session': 1.0,      # Context for volatility and liquidity
        'dte_risk': 1.1,          # Manages risk associated with time decay
        'options_filter': 1.9     # A critical filter to avoid bad trades
    })

    # --- Dynamic Adjustments ---
    # These weights can be dynamically adjusted based on market conditions.
    volatility_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'high_vol': {'momentum': 1.2, 'volume': 1.3, 'bollinger_bands': 1.4},
        'low_vol': {'microstructure': 1.3, 'vwap': 1.2, 'levels': 1.2}
    })

    # --- Signal Thresholds ---
    # Defines the total score needed to trigger a specific signal.
    base_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'strong_bull': 7.0,
        'bull': 3.0,
        'bear': -3.0,
        'strong_bear': -7.0
    })
