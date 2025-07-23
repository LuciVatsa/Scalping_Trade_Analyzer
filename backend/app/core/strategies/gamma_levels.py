import numpy as np
from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class GammaLevelsStrategy(ScalpingStrategy):
    """Enhanced gamma exposure analysis for key market turning points."""
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        
        # Core gamma data
        gamma_flip = inputs.gamma_flip_level
        current_price = inputs.current_price
        price_history = inputs.price_close_history
        
        if gamma_flip is None:
            return 0, ["Gamma exposure data unavailable"]
            
        # Additional gamma levels if available
        zero_gamma = getattr(inputs, 'zero_gamma_level', None)
        call_wall = getattr(inputs, 'call_wall_level', None)
        put_wall = getattr(inputs, 'put_wall_level', None)
        net_gamma = getattr(inputs, 'net_gamma_exposure', None)
        
        # Calculate distance from gamma flip
        dist_from_flip = (current_price - gamma_flip) / gamma_flip
        
        # 1. Gamma Flip Level Analysis
        if abs(dist_from_flip) < 0.001:  # Very close to flip level
            score += 2.0
            factors.append(f"🎯 At gamma flip level ({gamma_flip:.0f}) - Critical inflection point")
            
            # Check for recent breakthrough
            if len(price_history) >= 3:
                recent_avg = np.mean(price_history[-3:])
                if recent_avg < gamma_flip and current_price > gamma_flip:
                    score += 1.5
                    factors.append("🚀 Fresh break above gamma flip - bullish acceleration expected")
                elif recent_avg > gamma_flip and current_price < gamma_flip:
                    score -= 1.5
                    factors.append("💥 Fresh break below gamma flip - bearish acceleration expected")
                    
        elif dist_from_flip > 0.002:  # Above gamma flip
            proximity_score = max(0, 2.0 - abs(dist_from_flip) * 100)
            score += proximity_score
            factors.append(f"📈 Above gamma flip ({gamma_flip:.0f}) by {dist_from_flip*100:.2f}%")
            
            # Check for sustained move above flip
            if len(price_history) >= 5:
                above_flip_count = sum(1 for p in price_history[-5:] if p > gamma_flip)
                if above_flip_count >= 4:
                    score += 1.0
                    factors.append("🔥 Sustained move above gamma flip - positive gamma environment")
                elif above_flip_count == 5:
                    # Check if we're getting extended
                    if dist_from_flip > 0.01:
                        score -= 0.5
                        factors.append("⚠️ Extended above gamma flip - potential mean reversion")
                        
        elif dist_from_flip < -0.002:  # Below gamma flip
            proximity_penalty = min(0, -1.0 + abs(dist_from_flip) * 50)
            score += proximity_penalty
            factors.append(f"📉 Below gamma flip ({gamma_flip:.0f}) by {abs(dist_from_flip)*100:.2f}%")
            
            # Check for sustained move below flip
            if len(price_history) >= 5:
                below_flip_count = sum(1 for p in price_history[-5:] if p < gamma_flip)
                if below_flip_count >= 4:
                    score -= 0.8
                    factors.append("🔻 Sustained move below gamma flip - negative gamma environment")
                    
        # 2. Zero Gamma Level Analysis
        if zero_gamma is not None:
            dist_from_zero = abs(current_price - zero_gamma) / zero_gamma
            if dist_from_zero < 0.005:  # Within 0.5% of zero gamma
                score += 1.5
                factors.append(f"⚖️ Near zero gamma level ({zero_gamma:.0f}) - Maximum gamma impact zone")
                
        # 3. Gamma Walls Analysis
        if call_wall is not None and put_wall is not None:
            # Check if we're between the walls (compression zone)
            if put_wall < current_price < call_wall:
                wall_range = call_wall - put_wall
                position_in_range = (current_price - put_wall) / wall_range
                
                # Compression effect - price tends to stay range-bound
                compression_strength = max(0, 1.0 - wall_range / current_price * 100)
                if compression_strength > 0.5:
                    score += 0.5
                    factors.append(f"🏢 Between gamma walls ({put_wall:.0f}-{call_wall:.0f}) - Range compression")
                    
                    # Check position within range for bias
                    if position_in_range > 0.7:
                        score -= 0.3
                        factors.append("📊 Near call wall - potential resistance")
                    elif position_in_range < 0.3:
                        score += 0.3
                        factors.append("📊 Near put wall - potential support")
                        
            elif current_price > call_wall:
                # Above call wall - potential for gamma squeeze
                distance_above = (current_price - call_wall) / call_wall
                if distance_above < 0.01:  # Just above
                    score += 2.0
                    factors.append(f"🚀 Breaking call wall ({call_wall:.0f}) - Potential gamma squeeze")
                else:
                    # Check momentum
                    if len(price_history) >= 3:
                        momentum = (current_price - price_history[-3]) / price_history[-3]
                        if momentum > 0.005:
                            score += 1.5
                            factors.append("⚡ Above call wall with momentum - Gamma squeeze active")
                        else:
                            score -= 0.5
                            factors.append("⚠️ Above call wall losing momentum - Potential reversal")
                            
            elif current_price < put_wall:
                # Below put wall - potential for negative gamma acceleration
                distance_below = (put_wall - current_price) / current_price
                if distance_below < 0.01:  # Just below
                    score -= 2.0
                    factors.append(f"💥 Breaking put wall ({put_wall:.0f}) - Negative gamma acceleration")
                else:
                    # Check if selling is accelerating
                    if len(price_history) >= 3:
                        momentum = (current_price - price_history[-3]) / price_history[-3]
                        if momentum < -0.005:
                            score -= 1.5
                            factors.append("📉 Below put wall with negative momentum - Accelerating decline")
                        else:
                            score += 0.5
                            factors.append("🔄 Below put wall stabilizing - Potential bounce")
                            
        # 4. Net Gamma Exposure Analysis
        if net_gamma is not None:
            if net_gamma > 0:
                # Positive net gamma - market makers long gamma (stabilizing)
                gamma_strength = min(net_gamma / 1000000, 2.0)  # Normalize by 1M
                score += gamma_strength * 0.5
                factors.append(f"🛡️ Positive net gamma exposure - Market stabilizing force")
                
                # In positive gamma, dips get bought, rallies get sold
                if len(price_history) >= 2:
                    recent_change = (current_price - price_history[-2]) / price_history[-2]
                    if recent_change < -0.005:  # Recent dip
                        score += 0.8
                        factors.append("🔄 Dip in positive gamma environment - Buying opportunity")
                    elif recent_change > 0.005:  # Recent rally
                        score -= 0.3
                        factors.append("⚠️ Rally in positive gamma environment - Potential fade")
                        
            else:
                # Negative net gamma - market makers short gamma (destabilizing)
                gamma_strength = min(abs(net_gamma) / 1000000, 2.0)
                score -= gamma_strength * 0.3
                factors.append(f"⚡ Negative net gamma exposure - Market destabilizing force")
                
                # In negative gamma, moves tend to accelerate
                if len(price_history) >= 2:
                    recent_change = (current_price - price_history[-2]) / price_history[-2]
                    if abs(recent_change) > 0.003:
                        accel_score = min(abs(recent_change) * 100, 1.0)
                        if recent_change > 0:
                            score += accel_score
                            factors.append("🚀 Upward acceleration in negative gamma - Momentum trade")
                        else:
                            score -= accel_score
                            factors.append("💥 Downward acceleration in negative gamma - Momentum trade")
                            
        # 5. Time-Based Gamma Effects
        if hasattr(inputs, 'current_time'):
            hour = int(inputs.current_time.split(':')[0])
            
            # Gamma effects are stronger during high-volume periods
            if 9 <= hour <= 11 or 14 <= hour <= 16:  # Market open and close
                if abs(dist_from_flip) < 0.005:
                    score += 0.5
                    factors.append("⏰ High-volume period near gamma levels - Amplified effects")
                    
        # 6. Volume Confirmation
        if hasattr(inputs, 'volume_current') and hasattr(inputs, 'volume_avg'):
            if inputs.volume_current > inputs.volume_avg * 1.5:
                # High volume enhances gamma effects
                if abs(dist_from_flip) < 0.01:
                    score += 0.7
                    factors.append("📊 High volume near gamma levels - Strong directional signal")
                    
        return score, factors