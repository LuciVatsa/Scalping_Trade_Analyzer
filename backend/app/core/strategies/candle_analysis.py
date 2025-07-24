# backend/app/core/strategies/candle_analysis.py

from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class CandleAnalysisStrategy(ScalpingStrategy):
    """
    Analyzes recent candlestick patterns to infer buying or selling pressure,
    acting as a proxy for tick-level microstructure analysis.
    """
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        
        # Requires history of high and low prices
        if len(inputs.price_high_history) < 1 or len(inputs.price_low_history) < 1:
            return 0, ["Candle analysis requires high/low history."]

        # Analyze the most recent completed candle
        last_high = inputs.price_high_history[-1]
        last_low = inputs.price_low_history[-1]
        last_close = inputs.price_close_history[-1]
        candle_range = last_high - last_low

        if candle_range > 0:
            position_in_range = (last_close - last_low) / candle_range
            
            # Candle closed in the top 25% -> buying pressure
            if position_in_range > 0.75:
                score += 2.0
                factors.append(f"🕯️ Strong buying pressure (candle closed at {position_in_range:.0%} of range)")
            # Candle closed in the bottom 25% -> selling pressure
            elif position_in_range < 0.25:
                score -= 2.0
                factors.append(f"🕯️ Strong selling pressure (candle closed at {position_in_range:.0%} of range)")

        return score, factors