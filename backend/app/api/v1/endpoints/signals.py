from fastapi import APIRouter, HTTPException
from app.core.models.signals import ScalpingInputs, SignalResult
from app.core.engines.scalping_engine import EnhancedScalpingEngine

# Create the router for this endpoint
router = APIRouter()

# Instantiate a single, reusable instance of the scalping engine
engine = EnhancedScalpingEngine()

@router.post("/analyze", response_model=SignalResult, summary="Analyze Market Data for a Scalping Signal")
def analyze_signal(inputs: ScalpingInputs) -> SignalResult:
    """
    Receives market and options data, processes it through the full analytical engine,
    and returns a comprehensive trading signal.
    """
    # The engine's analyze_signal method expects a dictionary
    # We use model_dump() to convert the Pydantic model
    result = engine.analyze_signal(inputs.model_dump())

    if result is None:
        # This can happen if input validation fails or trading is avoided
        raise HTTPException(
            status_code=400,
            detail="Signal analysis failed. This could be due to invalid input data or unfavorable market conditions for trading."
        )

    return result