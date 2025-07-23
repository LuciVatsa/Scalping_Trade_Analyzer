# backend/tests/unit/strategies/test_options_filter_strategy.py
from app.core.models.signals import ScalpingInputs
from app.core.strategies.options_filter import EnhancedOptionsFilterStrategy
from app.core.config import TradingConfig, ScalpingConstants

config = TradingConfig()
constants = ScalpingConstants()

def test_filter_rejects_excessive_spread():
    inputs = ScalpingInputs(
        option_price=2.00,
        bid_ask_spread=0.25,
        current_price=150.0,
        price_close_history=[150.0] * 21,
        volume=1000,
        volume_history=[1000] * 21,
        vwap_value=150.0,
        bid_price=149.9,
        ask_price=150.1,
        last_trade_size=10,
        uptick_volume=500,
        downtick_volume=500,
        current_time="10:00",
        dte=1,
        iv_percentile=0.5,
        option_delta=0.5,
        option_gamma=0.04,
        option_theta=-0.1,
        option_volume=100,
        open_interest=500
    )

    strategy = EnhancedOptionsFilterStrategy(config, constants)
    score, factors = strategy.calculate_score(inputs)

    assert score <= -5.0
    assert "❌ Excessive spread (12.5% vs 4.0% max)" in factors