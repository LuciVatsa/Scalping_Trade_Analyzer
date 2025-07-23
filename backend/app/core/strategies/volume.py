import numpy as np
from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class VolumeStrategy(ScalpingStrategy):
    """Advanced volume analysis for scalping with multiple confirmation signals."""
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        if len(inputs.volume_history) < 20:
            return 0, ["Insufficient volume history for analysis"]
        
        score, factors = 0, []
        volume_hist = inputs.volume_history
        price_hist = inputs.price_close_history
        current_vol = volume_hist[-1]
        current_price = inputs.current_price
        
        # 1. Volume Moving Averages
        vol_sma_20 = np.mean(volume_hist[-20:])
        vol_sma_10 = np.mean(volume_hist[-10:])
        vol_sma_5 = np.mean(volume_hist[-5:])
        
        # 2. Volume Spike Detection
        vol_spike_factor = current_vol / vol_sma_20 if vol_sma_20 > 0 else 0
        
        if vol_spike_factor > 2.5:
            # Massive volume spike - check price confirmation
            price_change = (current_price - price_hist[-2]) / price_hist[-2] if len(price_hist) > 1 else 0
            
            if abs(price_change) > 0.001:  # Price moving with volume
                direction_score = 3.0 if price_change > 0 else -3.0
                score += direction_score
                factors.append(f"🔥 MASSIVE volume spike ({vol_spike_factor:.1f}x avg) with price confirmation")
            else:
                # High volume but no price movement - potential absorption
                score += 0.5
                factors.append(f"⚠️ High volume ({vol_spike_factor:.1f}x) but limited price movement - absorption?")
        
        elif vol_spike_factor > 1.8:
            score += 1.5
            factors.append(f"📈 Strong volume ({vol_spike_factor:.1f}x average)")
        
        elif vol_spike_factor < 0.5:
            score -= 0.5
            factors.append(f"📉 Low volume ({vol_spike_factor:.1f}x average) - weak conviction")
        
        # 3. Volume Trend Analysis
        vol_trend_score = self._analyze_volume_trend(volume_hist[-10:])
        score += vol_trend_score
        if vol_trend_score > 0:
            factors.append("📊 Volume trend increasing")
        elif vol_trend_score < 0:
            factors.append("📊 Volume trend decreasing")
        
        # 4. Volume-Price Divergence
        if len(price_hist) >= 10:
            divergence_score, divergence_factor = self._check_volume_price_divergence(
                price_hist[-10:], volume_hist[-10:]
            )
            score += divergence_score
            if divergence_factor:
                factors.append(divergence_factor)
        
        # 5. Volume at Key Levels (if we have support/resistance data)
        if hasattr(inputs, 'support_levels') and hasattr(inputs, 'resistance_levels'):
            level_volume_score, level_factor = self._analyze_volume_at_levels(inputs)
            score += level_volume_score
            if level_factor:
                factors.append(level_factor)
        
        # 6. Volume Confirmation Quality
        if len(volume_hist) >= 5:
            quality_score = self._assess_volume_quality(volume_hist[-5:])
            score *= quality_score  # Multiply by quality factor
            if quality_score < 1.0:
                factors.append("⚠️ Volume quality concerns")
        
        # 7. Unusual Volume Detection (compared to time-of-day average)
        unusual_score = self._detect_unusual_volume(current_vol, vol_sma_20)
        score += unusual_score
        if unusual_score > 0:
            factors.append(f"🚨 Unusual volume for this time period")
        
        return score, factors
    
    def _analyze_volume_trend(self, recent_volumes: List[float]) -> float:
        """Analyze if volume is trending up or down."""
        if len(recent_volumes) < 5:
            return 0
        
        # Simple linear regression on volume
        x = np.arange(len(recent_volumes))
        y = np.array(recent_volumes)
        
        if len(x) < 2:
            return 0
            
        slope = np.polyfit(x, y, 1)[0]
        avg_volume = np.mean(y)
        
        if avg_volume == 0:
            return 0
            
        # Normalize slope by average volume
        normalized_slope = slope / avg_volume
        
        if normalized_slope > 0.1:
            return 1.0  # Strong increasing volume trend
        elif normalized_slope > 0.05:
            return 0.5  # Moderate increasing volume trend
        elif normalized_slope < -0.1:
            return -0.5  # Decreasing volume trend
        
        return 0
    
    def _check_volume_price_divergence(self, prices: List[float], volumes: List[float]) -> Tuple[float, str]:
        """Check for volume-price divergence patterns."""
        if len(prices) < 5 or len(volumes) < 5:
            return 0, ""
        
        # Price trend (last 5 periods)
        price_trend = np.polyfit(range(5), prices[-5:], 1)[0]
        volume_trend = np.polyfit(range(5), volumes[-5:], 1)[0]
        
        avg_price = np.mean(prices[-5:])
        avg_volume = np.mean(volumes[-5:])
        
        if avg_price == 0 or avg_volume == 0:
            return 0, ""
        
        # Normalize trends
        price_slope = price_trend / avg_price
        volume_slope = volume_trend / avg_volume
        
        # Bullish divergence: price falling, volume increasing
        if price_slope < -0.001 and volume_slope > 0.05:
            return 1.5, "🔄 Bullish volume-price divergence (price down, volume up)"
        
        # Bearish divergence: price rising, volume decreasing
        elif price_slope > 0.001 and volume_slope < -0.05:
            return -1.0, "🔄 Bearish volume-price divergence (price up, volume down)"
        
        # Confirmation: both price and volume trending same direction
        elif price_slope > 0.001 and volume_slope > 0.05:
            return 0.5, "✅ Volume confirming price uptrend"
        elif price_slope < -0.001 and volume_slope > 0.05:
            return -0.5, "✅ Volume confirming price downtrend"
        
        return 0, ""
    
    def _analyze_volume_at_levels(self, inputs: ScalpingInputs) -> Tuple[float, str]:
        """Analyze volume behavior near support/resistance levels."""
        # This would require support/resistance level data
        # Placeholder for future implementation
        return 0, ""
    
    def _assess_volume_quality(self, recent_volumes: List[float]) -> float:
        """Assess the quality/consistency of recent volume."""
        if len(recent_volumes) < 3:
            return 1.0
        
        # Calculate coefficient of variation
        mean_vol = np.mean(recent_volumes)
        if mean_vol == 0:
            return 0.5
        
        std_vol = np.std(recent_volumes)
        cv = std_vol / mean_vol
        
        # Lower CV = more consistent volume = higher quality
        if cv < 0.3:
            return 1.2  # High quality, boost signal
        elif cv < 0.6:
            return 1.0  # Normal quality
        else:
            return 0.8  # Poor quality, reduce signal
    
    def _detect_unusual_volume(self, current_vol: float, avg_vol: float) -> float:
        """Detect if current volume is unusual for the time period."""
        if avg_vol == 0:
            return 0
        
        vol_ratio = current_vol / avg_vol
        
        # Extremely unusual volume
        if vol_ratio > 5.0:
            return 2.0
        elif vol_ratio > 3.0:
            return 1.0
        elif vol_ratio < 0.2:
            return -1.0  # Unusually low volume
        
        return 0