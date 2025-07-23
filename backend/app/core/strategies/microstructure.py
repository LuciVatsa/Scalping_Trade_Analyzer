from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs
import statistics

class MicrostructureStrategy(ScalpingStrategy):
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        
        # Core microstructure analysis
        mid_price = (inputs.bid_price + inputs.ask_price) / 2
        spread_normalized = inputs.bid_ask_spread / inputs.current_price if inputs.current_price > 0 else 0
        
        # 1. Enhanced Price Position Analysis
        if inputs.bid_ask_spread > 0:
            price_position = (inputs.current_price - inputs.bid_price) / inputs.bid_ask_spread
            
            if price_position >= 0.8:
                score += 2.5
                factors.append(f"🔥 Aggressive buying (price at {price_position:.1%} of spread)")
            elif price_position >= 0.6:
                score += 1.5
                factors.append(f"📈 Strong buying pressure (price at {price_position:.1%} of spread)")
            elif price_position <= 0.2:
                score -= 2.5
                factors.append(f"🔥 Aggressive selling (price at {price_position:.1%} of spread)")
            elif price_position <= 0.4:
                score -= 1.5
                factors.append(f"📉 Strong selling pressure (price at {price_position:.1%} of spread)")
        
        # 2. Sophisticated Tick Volume Analysis
        if inputs.uptick_volume > 0 and inputs.downtick_volume > 0:
            total_tick_volume = inputs.uptick_volume + inputs.downtick_volume
            tick_imbalance = (inputs.uptick_volume - inputs.downtick_volume) / total_tick_volume
            tick_ratio = inputs.uptick_volume / total_tick_volume
            
            # Volume-weighted tick analysis
            if total_tick_volume > inputs.volume * 0.5:  # Significant tick data
                if tick_imbalance >= 0.4:
                    score += 2.5
                    factors.append(f"💪 Strong tick imbalance ({tick_imbalance:.1%} net buying)")
                elif tick_imbalance >= 0.2:
                    score += 1.5
                    factors.append(f"↗️ Moderate tick imbalance ({tick_imbalance:.1%} net buying)")
                elif tick_imbalance <= -0.4:
                    score -= 2.5
                    factors.append(f"💪 Strong tick imbalance ({abs(tick_imbalance):.1%} net selling)")
                elif tick_imbalance <= -0.2:
                    score -= 1.5
                    factors.append(f"↘️ Moderate tick imbalance ({abs(tick_imbalance):.1%} net selling)")
            
            # Tick velocity (intensity of directional flow)
            if tick_ratio > 0.75 or tick_ratio < 0.25:
                velocity_bonus = 1.0 if tick_ratio > 0.75 else -1.0
                score += velocity_bonus
                direction = "bullish" if velocity_bonus > 0 else "bearish"
                factors.append(f"⚡ High tick velocity ({direction} flow)")
        
        # 3. Spread Quality Assessment
        if spread_normalized > 0:
            if spread_normalized < 0.001:  # Very tight spread
                score += 1.0
                factors.append("✅ Excellent liquidity (tight spread)")
            elif spread_normalized > 0.005:  # Wide spread
                score -= 1.5
                factors.append(f"⚠️ Poor liquidity (wide spread: {spread_normalized:.3%})")
        
        # 4. Order Flow Momentum (using recent price history if available)
        if hasattr(inputs, 'price_history') and len(inputs.price_history) >= 5:
            recent_prices = inputs.price_history[-5:]
            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(price_momentum) > 0.002:  # Significant momentum
                momentum_score = min(max(price_momentum * 1000, -2.0), 2.0)  # Cap at ±2.0
                score += momentum_score
                direction = "upward" if momentum_score > 0 else "downward"
                factors.append(f"🎯 Price momentum ({direction}: {abs(price_momentum):.3%})")
        
        # 5. Market Structure Analysis
        if hasattr(inputs, 'market_structure'):
            if inputs.market_structure == "BULLISH_TREND":
                score += 0.5
                factors.append("📊 Bullish market structure support")
            elif inputs.market_structure == "BEARISH_TREND":
                score -= 0.5
                factors.append("📊 Bearish market structure pressure")
        
        # 6. Volume-Price Confirmation
        if inputs.volume > 0:
            # Check if volume supports price action
            avg_volume = getattr(inputs, 'avg_volume', inputs.volume)
            volume_ratio = inputs.volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.5 and abs(score) > 1.0:  # High volume confirms signal
                volume_boost = 0.5 * (1 if score > 0 else -1)
                score += volume_boost
                factors.append(f"🔊 Volume confirmation ({volume_ratio:.1f}x avg)")
            elif volume_ratio < 0.5 and abs(score) > 1.0:  # Low volume weakens signal
                score *= 0.7  # Reduce confidence
                factors.append("🔇 Low volume - reduced confidence")
        
        # 7. Risk Adjustment for Extreme Conditions
        if spread_normalized > 0.01:  # Very wide spreads
            score *= 0.5  # Reduce all signals in poor liquidity
            factors.append("⚠️ Liquidity risk - signal strength reduced")
        
        # Ensure score bounds
        score = max(min(score, 5.0), -5.0)
        
        return score, factors