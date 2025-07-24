# backend/app/core/models/signals.py
from typing import List, Tuple, Optional, Any, Literal
from pydantic import BaseModel, field_validator
from dataclasses import dataclass
from .enums import SignalDirection

@dataclass
class SignalResult:
    """Data structure to hold the final signal result."""
    direction: SignalDirection
    strength: str
    confidence: str
    total_score: float
    key_factors: List[str]
    exit_signals: List[str]
    risk_advice: List[str]

class ScalpingInputs(BaseModel):
    """
    Enhanced inputs with microstructure and gamma data. This model serves as the
    single source of truth for all data required by the trading strategies.
    """
    # Core Market Data
    current_price: float
    price_close_history: List[float]
    price_high_history: List[float] = []
    price_low_history: List[float] = []
    volume: int 
    volume_history: List[float]
    vwap_value: float
    strike_price: Optional[float] = None 

    # Enhanced Market Microstructure
    bid_price: float
    ask_price: float
    last_trade_size: int
    uptick_volume: float
    downtick_volume: float
    large_trade_imbalance: float = 0.0
    
    # Options Flow Data
    call_put_ratio: float = 1.0
    unusual_options_activity: bool = False
    
    # --- NEW: Expanded Gamma Exposure Data ---
    dealer_gamma_exposure: float = 0.0
    gamma_flip_level: Optional[float] = None
    zero_gamma_level: Optional[float] = None
    call_wall_level: Optional[float] = None
    put_wall_level: Optional[float] = None
    net_gamma_exposure: Optional[float] = None
    
    # Context & Risk Inputs
    market_regime: Literal['trending_up', 'trending_down', 'ranging', 'volatile'] = 'ranging'
    vix_level: float = 20.0
    current_time: str # Expected format: "HH:MM"
    
    # --- NEW & ENHANCED: Options Contract Data ---
    dte: int
    iv_percentile: float
    implied_volatility: Optional[float] = None # Added for DTEStrategy
    option_volume: Optional[int] = None # ADDED: For liquidity checks
    open_interest: Optional[int] = None # ADDED: For liquidity checks
    is_weekly_expiration: bool = False # Added for DTEStrategy
    bid_ask_spread: float
    option_price: float
    option_delta: float
    option_gamma: float
    option_theta: float
    option_vega: Optional[float] = None # Added for DTEStrategy
    
    # Risk Events
    news_event_imminent: bool = False
    fomc_meeting_today: bool = False

    # Indicator Inputs
    bb_top: Optional[float] = None
    bb_mid: Optional[float] = None
    bb_bot: Optional[float] = None
    atr_value: Optional[float] = None
    
    # User/Account Inputs
    account_size: Optional[float] = None
    risk_percent: float = 1.5

    @field_validator(
        'current_price', 'bid_price', 'ask_price', 'vwap_value', 
        'option_price', 'account_size', 'atr_value', 'vix_level'
    )
    def validate_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Price/value must be positive")
        return v
