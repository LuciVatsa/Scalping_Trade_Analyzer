# backend/app/core/strategies/open_interest.py

from typing import Tuple, List
import numpy as np
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class OpenInterestStrategy(ScalpingStrategy):
    """
    Analyzes Open Interest (OI) at nearby strikes to identify potential
    support and resistance, acting as a proxy for gamma walls.
    """
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        
        # This strategy requires the full options chain data, which is not yet in ScalpingInputs.
        # For now, we'll assume a simplified input. We'll need to expand this later.
        # Placeholder: Assume we have a list of (strike, open_interest) tuples.
        # In a real implementation, you'd fetch the chain and pass it in.
        
        # MOCK DATA for demonstration - you will replace this with real data later
        # Let's pretend the options chain for the underlying is passed in a new field.
        # We will add 'options_chain' to the ScalpingInputs model later if needed.
        mock_chain = [
            {'strike': inputs.current_price - 2, 'open_interest': 1500},
            {'strike': inputs.current_price - 1, 'open_interest': 8000}, # High OI support
            {'strike': inputs.current_price + 1, 'open_interest': 4000},
            {'strike': inputs.current_price + 2, 'open_interest': 9500}, # High OI resistance
        ]

        # Find the highest OI levels below and above the current price
        try:
            support_level = max([c for c in mock_chain if c['strike'] < inputs.current_price], key=lambda x: x['open_interest'])
            resistance_level = max([c for c in mock_chain if c['strike'] > inputs.current_price], key=lambda x: x['open_interest'])
            
            # Check if price is near the high OI support level
            if abs(inputs.current_price - support_level['strike']) < 0.5:
                score += 1.5
                factors.append(f"📈 Near high OI support at {support_level['strike']} ({support_level['open_interest']} OI)")

            # Check if price is near the high OI resistance level
            if abs(inputs.current_price - resistance_level['strike']) < 0.5:
                score -= 1.5
                factors.append(f"📉 Near high OI resistance at {resistance_level['strike']} ({resistance_level['open_interest']} OI)")

        except (ValueError, IndexError):
            # This can happen if there's no strike above/below the current price in the chain
            factors.append("↔️ Price is outside of significant OI levels.")

        return score, factors