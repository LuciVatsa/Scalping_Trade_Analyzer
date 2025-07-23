# backend/tests/integration/test_engine.py
from app.core.models.signals import ScalpingInputs, SignalDirection
from app.core.engines.scalping_engine import EnhancedScalpingEngine

def test_engine_produces_strong_bullish_signal():
    price_history = [150.0, 150.1, 150.3, 150.6, 151.0, 151.5]
    volume_history = [10000] * 5
    volume_history.append(15000)

    inputs = ScalpingInputs(
        current_price=151.5,
        price_close_history=price_history,
        price_high_history=price_history,
        price_low_history=price_history,
        volume=15000,
        volume_history=volume_history,
        vwap_value=150.5,
        bid_price=151.48,
        ask_price=151.52,
        bid_ask_spread=0.04,
        last_trade_size=100,
        uptick_volume=12000,
        downtick_volume=3000,
        current_time="10:30",
        dte=3,
        iv_percentile=0.5,
        option_price=4.50,
        option_delta=0.6,
        option_gamma=0.08,
        option_theta=-0.12,
        vix_level=18.0,
        option_volume=500,
        open_interest=1200
    )

    engine = EnhancedScalpingEngine()
    result = engine.analyze_signal(inputs.model_dump())

    assert result is not None
    assert result.direction == SignalDirection.STRONG_BULLISH
    key_factors_string = "".join(result.key_factors)
    assert "momentum" in key_factors_string.lower()
    assert "volume" in key_factors_string.lower()
    assert "above vwap" in key_factors_string.lower()