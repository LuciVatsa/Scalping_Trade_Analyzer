# backend/tests/unit/test_risk_engine.py

from app.core.engines.risk_engine import EnhancedRiskManager

def test_risk_manager_provides_comprehensive_advice():
    """
    Tests that the risk manager returns a list of advice strings,
    including a calculated stop-loss and position size.
    """
    # 1. ARRANGE
    risk_manager = EnhancedRiskManager()
    inputs = {
        "current_price": 200.0,
        "atr_value": 1.5,
        "vix_level": 22.0,
        "dte": 2,
        "current_time": "14:00",
        "account_size": 50000.0,
        "risk_percent": 1.0,
    }

    # 2. ACT
    advice = risk_manager.get_comprehensive_risk_advice(
        inputs, 
        signal_direction="STRONG_BULLISH", 
        signal_confidence=0.8
    )

    # 3. ASSERT
    assert isinstance(advice, list)
    assert len(advice) > 2

    # Check that key pieces of advice are present and correctly formatted
    assert any("🛡️ Dynamic Stop-Loss:" in s for s in advice)
    assert any("💰 Recommended Size:" in s for s in advice)
    assert any("Confidence Adjusted:" in s for s in advice)