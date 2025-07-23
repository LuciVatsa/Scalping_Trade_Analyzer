# backend/tests/unit/strategies/test_momentum_strategy.py
from app.core.models.signals import ScalpingInputs
from app.core.strategies.momentum import MomentumStrategy
from app.core.config import TradingConfig, ScalpingConstants

config = TradingConfig()
constants = ScalpingConstants()

def test_strong_bullish_momentum():
    # Price history now has 21 elements to pass the length check
    price_history = [100.0] * 15 + [100.1, 100.2, 100.4, 100.7, 101.1, 101.6]
    
    inputs = ScalpingInputs(
        current_price=101.6,
        price_close_history=price_history,
        volume=1000,
        volume_history=[1000] * 21,
        vwap_value=100.0,
        bid_price=101.59,
        ask_price=101.61,
        last_trade_size=10,
        uptick_volume=500,
        downtick_volume=500,
        current_time="10:00",
        dte=1,
        iv_percentile=0.5,
        bid_ask_spread=0.02,
        option_price=2.5,
        option_delta=0.5,
        option_gamma=0.04,
        option_theta=-0.1
    )

    strategy = MomentumStrategy(config, constants)
    score, factors = strategy.calculate_score(inputs)

    assert score > 2.0
    factors_string = "".join(factors)
    assert "🚀 Strong 1-period momentum" in factors_string
    assert "⚡ Momentum accelerating" in factors_string