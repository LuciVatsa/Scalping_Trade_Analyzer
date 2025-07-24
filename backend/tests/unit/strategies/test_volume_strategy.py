# tests/strategies/test_volume_strategy.py

from app.core.models.signals import ScalpingInputs
from app.core.strategies.volume import VolumeStrategy
from app.core.config import TradingConfig, ScalpingConstants

# You'll need instances of your config classes
config = TradingConfig()
constants = ScalpingConstants()

def test_low_volume_scenario():
    """
    Tests that a low volume scenario generates a negative score and the correct factor.
    """
    # 1. ARRANGE
    volume_history = [10000] * 20
    volume_history.append(4000) # Current volume is low
    price_history = [150.0] * 21 # Stable price history

    # Create a valid ScalpingInputs object with all required fields
    inputs = ScalpingInputs(
        current_price=150.0,
        price_close_history=price_history,
        price_high_history=price_history,
        price_low_history=price_history,
        volume=4000, # ADDED: The most recent volume bar
        volume_history=volume_history,
        vwap_value=150.0,
        bid_price=149.99,
        ask_price=150.01,
        last_trade_size=50,
        uptick_volume=2000,
        downtick_volume=2000,
        current_time="11:00",
        dte=5,
        iv_percentile=0.45,
        bid_ask_spread=0.02,
        option_price=3.15,
        option_delta=0.5,
        option_gamma=0.05,
        option_theta=-0.08,
        strike_price=150.0, # ADDED: Required for some strategies
        option_volume=100,
        open_interest=500
    )

    # 2. ACT
    strategy = VolumeStrategy(config, constants)
    score, factors = strategy.calculate_score(inputs)

    # 3. ASSERT
    assert score < 0
    assert "📉 Low volume (0.4x average) - weak conviction" in factors