import numpy as np
from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class MomentumStrategy(ScalpingStrategy):
    """Advanced momentum analysis with multiple timeframes and momentum quality assessment."""
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        history = inputs.price_close_history
        if len(history) < 20:
            return 0, ["Insufficient price history for momentum analysis"]
        
        score, factors = 0, []
        current_price = inputs.current_price
        
        # 1. Multi-timeframe Momentum Analysis
        momentum_1m = self._calculate_momentum(history, 1)  # 1-period momentum
        momentum_3m = self._calculate_momentum(history, 3)  # 3-period momentum  
        momentum_5m = self._calculate_momentum(history, 5)  # 5-period momentum
        momentum_10m = self._calculate_momentum(history, 10) # 10-period momentum
        
        # 2. Momentum Acceleration
        acceleration = self._calculate_acceleration(history)
        
        # 3. Momentum Persistence Score
        persistence = self._calculate_momentum_persistence(history)
        
        # 4. Rate of Change (ROC) Analysis
        roc_fast = self._calculate_roc(history, 3)
        roc_medium = self._calculate_roc(history, 7)
        
        # 5. Momentum Strength Assessment
        strength = self._assess_momentum_strength([momentum_1m, momentum_3m, momentum_5m, momentum_10m])
        
        # Scoring Logic
        base_momentum_score = 0
        
        # Short-term momentum (most important for scalping)
        if momentum_1m > 0.003:  # Strong bullish 1-period momentum
            base_score = min(4.0, momentum_1m * 1000)  # Cap at 4.0
            base_momentum_score += base_score
            factors.append(f"🚀 Strong 1-period momentum: {momentum_1m:.3%}")
            
        elif momentum_1m < -0.003:  # Strong bearish 1-period momentum
            base_score = max(-4.0, momentum_1m * 1000)  # Cap at -4.0
            base_momentum_score += base_score
            factors.append(f"💥 Strong bearish 1-period momentum: {momentum_1m:.3%}")
            
        elif abs(momentum_1m) > 0.001:  # Moderate momentum
            base_score = momentum_1m * 500
            base_momentum_score += base_score
            direction = "bullish" if momentum_1m > 0 else "bearish"
            factors.append(f"📈 Moderate {direction} momentum: {momentum_1m:.3%}")
        
        # Medium-term momentum confirmation
        if momentum_5m > 0.005:
            base_momentum_score += 1.5
            factors.append(f"✅ 5-period momentum confirmation: {momentum_5m:.3%}")
        elif momentum_5m < -0.005:
            base_momentum_score -= 1.5
            factors.append(f"✅ 5-period bearish momentum: {momentum_5m:.3%}")
        
        # Momentum Alignment Bonus
        alignment_score = self._calculate_momentum_alignment(
            [momentum_1m, momentum_3m, momentum_5m, momentum_10m]
        )
        if alignment_score > 0.8:
            base_momentum_score *= 1.3  # 30% bonus for aligned momentum
            factors.append("🎯 Strong momentum alignment across timeframes")
        elif alignment_score < -0.8:
            base_momentum_score *= 1.3  # 30% bonus for aligned bearish momentum
            factors.append("🎯 Strong bearish momentum alignment")
        
        # Acceleration Factor
        if acceleration > 0.002 and base_momentum_score > 0:
            base_momentum_score *= 1.4
            factors.append(f"⚡ Momentum accelerating: {acceleration:.3%}")
        elif acceleration < -0.002 and base_momentum_score < 0:
            base_momentum_score *= 1.4
            factors.append(f"⚡ Bearish acceleration: {acceleration:.3%}")
        elif abs(acceleration) > 0.001:
            accel_direction = "positive" if acceleration > 0 else "negative"
            factors.append(f"📊 Momentum acceleration: {accel_direction}")
        
        # Persistence Factor
        if persistence > 0.7:
            base_momentum_score *= 1.2
            factors.append("🔄 Persistent momentum pattern")
        elif persistence < 0.3:
            base_momentum_score *= 0.8
            factors.append("⚠️ Inconsistent momentum")
        
        # Momentum Exhaustion Detection
        exhaustion_score = self._detect_momentum_exhaustion(history)
        if exhaustion_score != 0:
            base_momentum_score += exhaustion_score
            if exhaustion_score > 0:
                factors.append("🔄 Potential momentum reversal setup")
            else:
                factors.append("⚠️ Momentum showing exhaustion signs")
        
        # ROC Confirmation
        if roc_fast > 0.004 and base_momentum_score > 0:
            base_momentum_score += 0.5
            factors.append("📈 ROC confirming momentum")
        elif roc_fast < -0.004 and base_momentum_score < 0:
            base_momentum_score -= 0.5
            factors.append("📉 ROC confirming bearish momentum")
        
        score = base_momentum_score
        
        return score, factors
    
    def _calculate_momentum(self, history: List[float], periods: int) -> float:
        """Calculate momentum over specified periods."""
        if len(history) < periods + 1:
            return 0
        
        current = history[-1]
        previous = history[-(periods + 1)]
        
        if previous == 0:
            return 0
            
        return (current - previous) / previous
    
    def _calculate_acceleration(self, history: List[float]) -> float:
        """Calculate momentum acceleration (momentum of momentum)."""
        if len(history) < 6:
            return 0
        
        # Calculate momentum for two different periods
        recent_momentum = self._calculate_momentum(history, 2)
        earlier_momentum = self._calculate_momentum(history[:-2], 2)
        
        return recent_momentum - earlier_momentum
    
    def _calculate_momentum_persistence(self, history: List[float]) -> float:
        """Calculate how persistent the momentum has been."""
        if len(history) < 10:
            return 0.5
        
        # Check last 8 periods for consistent direction
        momentums = []
        for i in range(2, min(9, len(history))):
            mom = self._calculate_momentum(history[:-i+1], 1)
            momentums.append(mom)
        
        if not momentums:
            return 0.5
        
        # Count how many have the same sign as most recent
        recent_momentum = self._calculate_momentum(history, 1)
        if recent_momentum == 0:
            return 0.5
        
        same_direction = sum(1 for mom in momentums if (mom > 0) == (recent_momentum > 0))
        persistence = same_direction / len(momentums)
        
        return persistence
    
    def _calculate_roc(self, history: List[float], periods: int) -> float:
        """Calculate Rate of Change."""
        if len(history) < periods + 1:
            return 0
        
        current = history[-1]
        previous = history[-(periods + 1)]
        
        if previous == 0:
            return 0
            
        return (current - previous) / previous
    
    def _assess_momentum_strength(self, momentums: List[float]) -> str:
        """Assess overall momentum strength."""
        avg_momentum = np.mean([abs(m) for m in momentums if m != 0])
        
        if avg_momentum > 0.01:
            return "STRONG"
        elif avg_momentum > 0.005:
            return "MODERATE"
        elif avg_momentum > 0.002:
            return "WEAK"
        else:
            return "NEGLIGIBLE"
    
    def _calculate_momentum_alignment(self, momentums: List[float]) -> float:
        """Calculate how aligned different timeframe momentums are."""
        if not momentums or all(m == 0 for m in momentums):
            return 0
        
        # Filter out zero momentums
        non_zero_momentums = [m for m in momentums if m != 0]
        if not non_zero_momentums:
            return 0
        
        # Check if they're all positive or all negative
        positive_count = sum(1 for m in non_zero_momentums if m > 0)
        negative_count = len(non_zero_momentums) - positive_count
        
        # Calculate alignment score
        if positive_count == len(non_zero_momentums):
            return 1.0  # Perfect bullish alignment
        elif negative_count == len(non_zero_momentums):
            return -1.0  # Perfect bearish alignment
        else:
            # Mixed signals - calculate dominance
            dominance = abs(positive_count - negative_count) / len(non_zero_momentums)
            if positive_count > negative_count:
                return dominance
            else:
                return -dominance
    
    def _detect_momentum_exhaustion(self, history: List[float]) -> float:
        """Detect potential momentum exhaustion or divergence."""
        if len(history) < 15:
            return 0
        
        # Look for momentum divergence
        recent_prices = history[-5:]
        earlier_prices = history[-10:-5]
        
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        earlier_high = max(earlier_prices)
        earlier_low = min(earlier_prices)
        
        recent_momentum = self._calculate_momentum(history, 3)
        earlier_momentum = self._calculate_momentum(history[:-3], 3)
        
        # Bullish exhaustion: higher highs but weakening momentum
        if recent_high > earlier_high and recent_momentum < earlier_momentum and recent_momentum > 0:
            return -0.5  # Potential bullish exhaustion
        
        # Bearish exhaustion: lower lows but weakening bearish momentum
        elif recent_low < earlier_low and recent_momentum > earlier_momentum and recent_momentum < 0:
            return 0.5  # Potential bearish exhaustion (bullish reversal)
        
        return 0