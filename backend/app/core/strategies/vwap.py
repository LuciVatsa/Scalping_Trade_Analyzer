import numpy as np
from typing import Tuple, List, Optional
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class VWAPStrategy(ScalpingStrategy):
    """Advanced VWAP analysis with deviation bands, multi-timeframe, and mean reversion patterns."""
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        if inputs.vwap_value is None or inputs.vwap_value <= 0:
            return 0, ["VWAP data unavailable"]
        
        score, factors = 0, []
        current_price = inputs.current_price
        vwap = inputs.vwap_value
        
        # 1. Calculate VWAP deviation bands (if we have volume data)
        vwap_bands = self._calculate_vwap_bands(inputs)
        
        # 2. Basic VWAP position scoring
        price_vwap_ratio = (current_price - vwap) / vwap
        distance_score = self._calculate_distance_score(price_vwap_ratio)
        
        # 3. VWAP reclaim/rejection patterns
        reclaim_score, reclaim_factor = self._analyze_vwap_reclaim_rejection(inputs)
        
        # 4. VWAP trend analysis
        vwap_trend_score, trend_factor = self._analyze_vwap_trend(inputs)
        
        # 5. Mean reversion opportunities
        mean_reversion_score, reversion_factor = self._identify_mean_reversion_setup(inputs, vwap_bands)
        
        # 6. VWAP momentum (price moving towards/away from VWAP)
        momentum_score, momentum_factor = self._analyze_vwap_momentum(inputs)
        
        # Combine all scores
        base_score = distance_score + reclaim_score + vwap_trend_score + mean_reversion_score + momentum_score
        
        # Add factors
        if abs(price_vwap_ratio) > 0.002:
            direction = "above" if price_vwap_ratio > 0 else "below"
            factors.append(f"📍 Price {direction} VWAP by {abs(price_vwap_ratio):.2%}")
        
        if reclaim_factor:
            factors.append(reclaim_factor)
        if trend_factor:
            factors.append(trend_factor)
        if reversion_factor:
            factors.append(reversion_factor)
        if momentum_factor:
            factors.append(momentum_factor)
        
        # VWAP band analysis
        if vwap_bands:
            band_score, band_factors = self._analyze_vwap_bands(current_price, vwap_bands)
            base_score += band_score
            factors.extend(band_factors)
        
        return base_score, factors
    
    def _calculate_vwap_bands(self, inputs: ScalpingInputs) -> Optional[dict]:
        """Calculate VWAP standard deviation bands."""
        if len(inputs.price_close_history) < 20 or len(inputs.volume_history) < 20:
            return None
        
        prices = np.array(inputs.price_close_history[-20:])
        volumes = np.array(inputs.volume_history[-20:])
        vwap = inputs.vwap_value
        
        if np.sum(volumes) == 0:
            return None
        
        # Calculate VWAP standard deviation
        price_volume_product = prices * volumes
        squared_price_volume = (prices ** 2) * volumes
        
        vwap_variance = (np.sum(squared_price_volume) / np.sum(volumes)) - (vwap ** 2)
        vwap_std = np.sqrt(max(0, vwap_variance))
        
        return {
            'upper_1': vwap + vwap_std,
            'upper_2': vwap + 2 * vwap_std,
            'lower_1': vwap - vwap_std,
            'lower_2': vwap - 2 * vwap_std,
            'std_dev': vwap_std
        }
    
    def _calculate_distance_score(self, price_vwap_ratio: float) -> float:
        """Calculate score based on distance from VWAP."""
        abs_ratio = abs(price_vwap_ratio)
        
        # Close to VWAP - neutral to slightly positive for mean reversion
        if abs_ratio < 0.001:
            return 0.5  # Neutral, but slight positive for potential breakout
        
        # Moderate distance - directional bias
        elif abs_ratio < 0.003:
            return 1.5 if price_vwap_ratio > 0 else -1.0
        
        # Significant distance - strong directional bias but watch for exhaustion
        elif abs_ratio < 0.006:
            return 2.0 if price_vwap_ratio > 0 else -1.5
        
        # Very far from VWAP - potential mean reversion setup
        elif abs_ratio > 0.01:
            # Extreme deviation - potential reversal
            return -1.0 if price_vwap_ratio > 0 else 1.0
        
        # Far but not extreme
        else:
            return 1.0 if price_vwap_ratio > 0 else -0.5
    
    def _analyze_vwap_reclaim_rejection(self, inputs: ScalpingInputs) -> Tuple[float, str]:
        """Analyze VWAP reclaim/rejection patterns."""
        if len(inputs.price_close_history) < 10:
            return 0, ""
        
        current_price = inputs.current_price
        vwap = inputs.vwap_value
        recent_prices = inputs.price_close_history[-5:]
        
        # Check if we recently crossed VWAP
        above_vwap = [p > vwap for p in recent_prices]
        current_above = current_price > vwap
        
        # VWAP reclaim (was below, now above)
        if not above_vwap[0] and current_above and any(above_vwap[-3:]):
            return 2.0, "🚀 VWAP reclaim in progress"
        
        # VWAP rejection (was above, now below)
        elif above_vwap[0] and not current_above and not any(above_vwap[-3:]):
            return -1.5, "💥 VWAP rejection pattern"
        
        # Failed reclaim (tried to get above but failed)
        elif any(above_vwap[-3:]) and not current_above:
            return -1.0, "❌ Failed VWAP reclaim"
        
        # Failed rejection (tried to go below but bounced)
        elif not any(above_vwap[-3:]) and current_above:
            return 1.5, "✅ VWAP bounce/hold"
        
        return 0, ""
    
    def _analyze_vwap_trend(self, inputs: ScalpingInputs) -> Tuple[float, str]:
        """Analyze the trend of VWAP itself if we have historical VWAP data."""
        # For now, we'll analyze price trend relative to VWAP
        if len(inputs.price_close_history) < 10:
            return 0, ""
        
        recent_prices = inputs.price_close_history[-10:]
        vwap = inputs.vwap_value
        
        # Calculate how price has been trending relative to VWAP
        above_count = sum(1 for p in recent_prices if p > vwap)
        
        if above_count >= 8:
            return 1.0, "📈 Sustained price action above VWAP"
        elif above_count <= 2:
            return -0.5, "📉 Sustained price action below VWAP"
        elif 6 <= above_count <= 7:
            return 0.5, "📊 Price mostly above VWAP"
        elif 3 <= above_count <= 4:
            return -0.3, "📊 Price mostly below VWAP"
        
        return 0, ""
    
    def _identify_mean_reversion_setup(self, inputs: ScalpingInputs, vwap_bands: Optional[dict]) -> Tuple[float, str]:
        """Identify mean reversion opportunities."""
        if not vwap_bands:
            return 0, ""
        
        current_price = inputs.current_price
        vwap = inputs.vwap_value
        
        # Price at extreme bands - mean reversion candidate
        if current_price > vwap_bands['upper_2']:
            return -2.0, "🔄 Extreme high - mean reversion candidate"
        elif current_price < vwap_bands['lower_2']:
            return 2.0, "🔄 Extreme low - mean reversion candidate"
        elif current_price > vwap_bands['upper_1']:
            return -1.0, "⚠️ Above 1-std band - watch for reversal"
        elif current_price < vwap_bands['lower_1']:
            return 1.0, "⚠️ Below 1-std band - bounce candidate"
        
        return 0, ""
    
    def _analyze_vwap_momentum(self, inputs: ScalpingInputs) -> Tuple[float, str]:
        """Analyze momentum towards/away from VWAP."""
        if len(inputs.price_close_history) < 5:
            return 0, ""
        
        current_price = inputs.current_price
        vwap = inputs.vwap_value
        recent_prices = inputs.price_close_history[-3:]
        
        # Calculate if we're moving towards or away from VWAP
        current_distance = abs(current_price - vwap)
        prev_distance = abs(recent_prices[-1] - vwap) if recent_prices else current_distance
        
        if len(recent_prices) < 2:
            return 0, ""
        
        distance_change = current_distance - prev_distance
        
        # Moving towards VWAP (mean reversion)
        if distance_change < -0.0005:  # Getting closer to VWAP
            if current_price > vwap:
                return -0.5, "🎯 Moving towards VWAP from above"
            else:
                return 0.5, "🎯 Moving towards VWAP from below"
        
        # Moving away from VWAP (trend continuation)
        elif distance_change > 0.0005:  # Getting further from VWAP
            if current_price > vwap:
                return 1.0, "🚀 Breaking away from VWAP upward"
            else:
                return -1.0, "💥 Breaking away from VWAP downward"
        
        return 0, ""
    
    def _analyze_vwap_bands(self, current_price: float, vwap_bands: dict) -> Tuple[float, List[str]]:
        """Analyze price action relative to VWAP bands."""
        score = 0
        factors = []
        
        # Band position scoring
        if current_price > vwap_bands['upper_2']:
            score -= 1.5
            factors.append(f"🔴 Price above 2-std VWAP band ({vwap_bands['upper_2']:.2f})")
        elif current_price > vwap_bands['upper_1']:
            score += 0.5  # Bullish but watch for reversal
            factors.append(f"🟡 Price above 1-std VWAP band ({vwap_bands['upper_1']:.2f})")
        elif current_price < vwap_bands['lower_2']:
            score += 1.5
            factors.append(f"🟢 Price below 2-std VWAP band ({vwap_bands['lower_2']:.2f}) - oversold")
        elif current_price < vwap_bands['lower_1']:
            score += 0.5
            factors.append(f"🟡 Price below 1-std VWAP band ({vwap_bands['lower_1']:.2f})")
        else:
            factors.append("📊 Price within VWAP standard deviation bands")
        
        # Band compression analysis
        band_width = (vwap_bands['upper_1'] - vwap_bands['lower_1']) / (vwap_bands['std_dev'] + 1e-9)
        if band_width < 0.5:
            score += 1.0
            factors.append("⚡ VWAP band compression - potential breakout")
        
        return score, factors