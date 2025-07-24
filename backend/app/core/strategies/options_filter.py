from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs
import math
import numpy as np

class EnhancedOptionsFilterStrategy(ScalpingStrategy):
    """
    A critical safety filter to avoid trading options with poor liquidity,
    excessive spreads, or unfavorable risk characteristics.
    """
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        
        # Basic validation
        if inputs.option_price <= 0:
            return -5.0, ["❌ Invalid option price"]
        
        # 1. Enhanced Spread Analysis
        spread_pct = inputs.bid_ask_spread / inputs.option_price
        
        if spread_pct > self.constants.MAX_BID_ASK_SPREAD_PCT:
            penalty = min(-5.0, -2.0 * (spread_pct / self.constants.MAX_BID_ASK_SPREAD_PCT))
            score += penalty
            factors.append(f"❌ Excessive spread ({spread_pct:.1%} vs {self.constants.MAX_BID_ASK_SPREAD_PCT:.1%} max)")
        elif spread_pct < 0.02:  # Very tight spread
            score += 1.0
            factors.append(f"✅ Excellent spread ({spread_pct:.1%})")
        
        # 2. Dynamic DTE and Moneyness Analysis
        if inputs.dte == 0:
            delta_threshold = 0.25
            if abs(inputs.option_delta) < delta_threshold:
                score -= 4.0
                factors.append(f"❌ Far OTM on 0DTE (delta: {inputs.option_delta:.3f})")
        
        # 3. Gamma Risk Assessment
        # --- FIX: Changed inputs.underlying_price to inputs.current_price ---
        if inputs.option_gamma > 0 and inputs.current_price > 0:
            gamma_risk = inputs.option_gamma * (inputs.current_price ** 2) / 100
            if inputs.dte <= 1 and gamma_risk > 50:
                score -= 1.5
                factors.append(f"⚠️ High gamma risk ({gamma_risk:.1f})")
        
        # 4. Theta Decay Analysis
        if inputs.option_theta != 0:
            daily_theta_pct = abs(inputs.option_theta) / inputs.option_price
            if inputs.dte <= 1 and daily_theta_pct > 0.15:
                score -= 2.0
                factors.append(f"❌ Excessive theta decay ({daily_theta_pct:.1%}/day)")

        # 5. Implied Volatility Analysis
        if inputs.implied_volatility and len(inputs.price_close_history) >= 20:
            # Calculate 20-day historical volatility (HV)
            daily_returns = np.log(np.array(inputs.price_close_history[-20:]) / np.array(inputs.price_close_history[-21:-1]))
            hv = np.std(daily_returns) * np.sqrt(252) # Annualized
            
            iv = inputs.implied_volatility
            
            # Compare IV to HV
            if iv > hv * 1.5: # If IV is 50% higher than recent volatility
                score -= 1.0
                factors.append(f"⚠️ High IV ({iv:.1%}) vs HV ({hv:.1%}) - Expensive premium, risk of IV crush")
            elif iv < hv * 0.8: # If IV is lower than recent volatility
                score += 0.5
                factors.append(f"✅ Low IV ({iv:.1%}) vs HV ({hv:.1%}) - Cheaper premium")

        # 6. Volume and Open Interest Quality
        if inputs.option_volume is not None and inputs.open_interest is not None:
            if inputs.option_volume < 10:
                score -= 1.5
                factors.append(f"⚠️ Low volume ({inputs.option_volume}) - liquidity risk")
            if inputs.open_interest < 50:
                score -= 1.0
                factors.append(f"⚠️ Low open interest ({inputs.open_interest})")

        # 7. Pin Risk Assessment
        if inputs.dte <= 1 and inputs.strike_price is not None:
            distance_from_strike = abs(inputs.current_price - inputs.strike_price) / inputs.strike_price
            if distance_from_strike < 0.005: # Within 0.5% of strike
                score -= 1.5
                factors.append(f"⚠️ Pin risk (underlying {distance_from_strike:.1%} from strike)")

        return max(min(score, 2.0), -5.0), factors
