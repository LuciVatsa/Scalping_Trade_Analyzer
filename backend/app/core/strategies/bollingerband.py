from typing import Tuple, List
import numpy as np
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class BollingerBandsStrategy(ScalpingStrategy):
    """Enhanced Bollinger Bands analysis for squeeze, breakouts, and mean reversion."""
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        top, mid, bot, price = inputs.bb_top, inputs.bb_mid, inputs.bb_bot, inputs.current_price

        if any(v is None for v in [top, mid, bot, price]):
            return 0, ["Bollinger Bands data unavailable."]

        if mid <= 0: 
            return 0, ["Invalid BB middle band value."]

        # Calculate key BB metrics
        band_width = (top - bot) / mid
        bb_position = (price - bot) / (top - bot) if (top - bot) > 0 else 0.5
        distance_from_mid = abs(price - mid) / mid
        
        # Analyze price history for context
        price_history = inputs.price_close_history
        if len(price_history) >= 5:
            recent_volatility = np.std(price_history[-5:]) / np.mean(price_history[-5:])
            price_momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
        else:
            recent_volatility = 0.01
            price_momentum = 0

        # 1. BB Squeeze Analysis (Low Volatility -> High Volatility Expected)
        squeeze_threshold = 0.012 if recent_volatility < 0.015 else 0.018
        if band_width < squeeze_threshold:
            squeeze_strength = (squeeze_threshold - band_width) / squeeze_threshold
            score += 1.5 + (squeeze_strength * 1.0)
            factors.append(f"⚡ BB Squeeze detected (width: {band_width:.3f}, strength: {squeeze_strength:.2f})")
            
            # Enhanced squeeze: Look for decreasing volatility trend
            if len(price_history) >= 10:
                older_volatility = np.std(price_history[-10:-5]) / np.mean(price_history[-10:-5])
                if recent_volatility < older_volatility * 0.8:
                    score += 0.5
                    factors.append("📉 Volatility compression trend confirmed")

        # 2. BB Breakout/Breakdown Analysis
        if price > top:
            breakout_strength = (price - top) / (top - mid)
            base_score = 2.0 + min(breakout_strength * 2, 1.5)
            
            # Confirm with volume and momentum
            if hasattr(inputs, 'volume_current') and hasattr(inputs, 'volume_avg'):
                if inputs.volume_current > inputs.volume_avg * 1.2:
                    base_score += 0.5
                    factors.append("📈 Volume-confirmed BB breakout")
            
            if price_momentum > 0.002:
                base_score += 0.5
                factors.append("🚀 Momentum-confirmed BB breakout")
                
            score += base_score
            factors.append(f"🚀 BB Breakout: Price {(price-top)/top*100:.2f}% above upper band")
            
        elif price < bot:
            breakdown_strength = (bot - price) / (mid - bot)
            base_score = -(2.0 + min(breakdown_strength * 2, 1.5))
            
            # Confirm with volume and momentum
            if hasattr(inputs, 'volume_current') and hasattr(inputs, 'volume_avg'):
                if inputs.volume_current > inputs.volume_avg * 1.2:
                    base_score -= 0.5
                    factors.append("📉 Volume-confirmed BB breakdown")
            
            if price_momentum < -0.002:
                base_score -= 0.5
                factors.append("💥 Momentum-confirmed BB breakdown")
                
            score += base_score
            factors.append(f"💥 BB Breakdown: Price {(bot-price)/price*100:.2f}% below lower band")

        # 3. Mean Reversion Signals
        elif bb_position > 0.8:  # Near upper band but not breaking out
            if price_momentum < 0.001:  # Losing momentum
                score -= 1.0
                factors.append(f"⚠️ Near BB upper band ({bb_position:.2f}) with weak momentum - potential reversal")
        elif bb_position < 0.2:  # Near lower band but not breaking down
            if price_momentum > -0.001:  # Momentum stabilizing
                score += 1.0
                factors.append(f"🔄 Near BB lower band ({bb_position:.2f}) with stabilizing momentum - potential bounce")

        # 4. BB Walk Analysis (Trending conditions)
        if 0.85 <= bb_position <= 1.0 and len(price_history) >= 3:
            # Check if price has been walking the upper band
            upper_walk_count = sum(1 for i in range(-3, 0) if len(price_history) > abs(i) and 
                                 (price_history[i] - top) / top > -0.002)
            if upper_walk_count >= 2:
                score += 1.5
                factors.append("🚶‍♂️ BB Upper Band Walk - strong uptrend")
                
        elif 0.0 <= bb_position <= 0.15 and len(price_history) >= 3:
            # Check if price has been walking the lower band
            lower_walk_count = sum(1 for i in range(-3, 0) if len(price_history) > abs(i) and 
                                 (bot - price_history[i]) / price_history[i] > -0.002)
            if lower_walk_count >= 2:
                score -= 1.5
                factors.append("🚶‍♀️ BB Lower Band Walk - strong downtrend")

        # 5. Volatility Expansion Signal
        if band_width > 0.025:  # High volatility
            if distance_from_mid > 0.015:  # Price far from middle
                volatility_score = min((band_width - 0.025) * 10, 1.0)
                if bb_position > 0.5:
                    score += volatility_score
                    factors.append(f"🌪️ High volatility expansion (upside bias)")
                else:
                    score -= volatility_score
                    factors.append(f"🌪️ High volatility expansion (downside bias)")

        # 6. BB Middle Band Interaction
        if abs(distance_from_mid) < 0.003:  # Very close to middle band
            if price_momentum > 0.001:
                score += 0.8
                factors.append("🎯 Bullish middle band test with momentum")
            elif price_momentum < -0.001:
                score -= 0.8
                factors.append("🎯 Bearish middle band test with momentum")

        return score, factors