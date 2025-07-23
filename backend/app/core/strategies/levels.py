from typing import Dict, Tuple, List, Optional
import numpy as np
from collections import defaultdict
from .base import ScalpingStrategy
from scipy.signal import find_peaks
from app.core.models.signals import ScalpingInputs
from app.core.config import TradingConfig, ScalpingConstants

class EnhancedLevelAnalyzer:
    """Advanced support and resistance level identification with strength analysis."""
    
    def __init__(self, min_touches: int = 2, max_levels: int = 8):
        self.min_touches = min_touches
        self.max_levels = max_levels
    
    def find_enhanced_levels(self, highs: List[float], lows: List[float], 
                           close_prices: List[float], volumes: List[float] = None,
                           lookback: int = 15) -> Dict[str, List[Dict]]:
        """Find support/resistance levels with strength, touch count, and recency analysis."""
        
        if len(highs) < lookback * 2 or len(lows) < lookback * 2:
            return {"support": [], "resistance": []}
        
        # Find resistance levels from highs
        resistance_indices, properties = find_peaks(
            highs, 
            distance=max(3, lookback // 3),
            prominence=np.std(highs) * 0.3,
            width=1
        )
        
        # Find support levels from inverted lows
        support_indices, support_properties = find_peaks(
            [-x for x in lows], 
            distance=max(3, lookback // 3),
            prominence=np.std(lows) * 0.3,
            width=1
        )
        
        # Build resistance levels with metadata
        resistance_levels = []
        for i, idx in enumerate(resistance_indices):
            if idx < len(highs):
                level_data = {
                    'price': highs[idx],
                    'index': idx,
                    'strength': self._calculate_level_strength(highs[idx], highs, lows, close_prices, volumes),
                    'touches': self._count_touches(highs[idx], highs + lows, tolerance_pct=0.1),
                    'recency_score': self._calculate_recency_score(idx, len(highs)),
                    'volume_confirmation': self._get_volume_confirmation(idx, volumes) if volumes else 1.0
                }
                resistance_levels.append(level_data)
        
        # Build support levels with metadata
        support_levels = []
        for i, idx in enumerate(support_indices):
            if idx < len(lows):
                level_data = {
                    'price': lows[idx],
                    'index': idx,
                    'strength': self._calculate_level_strength(lows[idx], highs, lows, close_prices, volumes),
                    'touches': self._count_touches(lows[idx], highs + lows, tolerance_pct=0.1),
                    'recency_score': self._calculate_recency_score(idx, len(lows)),
                    'volume_confirmation': self._get_volume_confirmation(idx, volumes) if volumes else 1.0
                }
                support_levels.append(level_data)
        
        # Filter and sort by composite strength
        resistance_levels = self._filter_and_rank_levels(resistance_levels)
        support_levels = self._filter_and_rank_levels(support_levels)
        
        return {
            "support": support_levels[:self.max_levels],
            "resistance": resistance_levels[:self.max_levels]
        }
    
    def _calculate_level_strength(self, level_price: float, highs: List[float], 
                                lows: List[float], closes: List[float], 
                                volumes: List[float] = None) -> float:
        """Calculate level strength based on multiple factors."""
        
        touches = self._count_touches(level_price, highs + lows, tolerance_pct=0.1)
        if touches < self.min_touches:
            return 0.0
        
        # Base strength from touch count
        strength = min(touches / 5.0, 1.0)  # Cap at 1.0 for 5+ touches
        
        # Age factor - recent levels are stronger
        recent_touches = self._count_touches(level_price, (highs + lows)[-50:], tolerance_pct=0.1)
        if recent_touches > 0:
            strength *= 1.2
        
        # Volume confirmation if available
        if volumes:
            vol_confirmation = self._get_volume_at_level(level_price, closes, volumes)
            strength *= (0.8 + 0.4 * vol_confirmation)  # Range: 0.8 to 1.2
        
        return min(strength, 2.0)  # Cap maximum strength
    
    def _count_touches(self, level: float, prices: List[float], tolerance_pct: float = 0.1) -> int:
        """Count how many times price touched this level within tolerance."""
        tolerance = level * (tolerance_pct / 100)
        return sum(1 for price in prices if abs(price - level) <= tolerance)
    
    def _calculate_recency_score(self, index: int, total_length: int) -> float:
        """Calculate how recent this level is (1.0 = most recent, 0.0 = oldest)."""
        return index / max(total_length - 1, 1)
    
    def _get_volume_confirmation(self, index: int, volumes: List[float]) -> float:
        """Get volume confirmation for this level."""
        if not volumes or index >= len(volumes):
            return 1.0
        
        avg_volume = np.mean(volumes)
        level_volume = volumes[index]
        
        return min(level_volume / max(avg_volume, 1), 2.0)
    
    def _get_volume_at_level(self, level: float, closes: List[float], volumes: List[float]) -> float:
        """Get average volume when price was near this level."""
        if not volumes or len(volumes) != len(closes):
            return 1.0
        
        tolerance = level * 0.001  # 0.1% tolerance
        relevant_volumes = [volumes[i] for i, close in enumerate(closes) 
                           if abs(close - level) <= tolerance]
        
        if not relevant_volumes:
            return 1.0
        
        avg_volume = np.mean(volumes)
        level_avg_volume = np.mean(relevant_volumes)
        
        return min(level_avg_volume / max(avg_volume, 1), 2.0)
    
    def _filter_and_rank_levels(self, levels: List[Dict]) -> List[Dict]:
        """Filter levels by minimum criteria and rank by composite strength."""
        
        # Filter by minimum touches and strength
        filtered = [level for level in levels 
                   if level['touches'] >= self.min_touches and level['strength'] > 0.3]
        
        # Calculate composite score
        for level in filtered:
            composite_score = (
                level['strength'] * 0.4 +
                min(level['touches'] / 3.0, 1.0) * 0.3 +
                level['recency_score'] * 0.2 +
                (level['volume_confirmation'] - 1.0) * 0.1  # Bonus for high volume
            )
            level['composite_score'] = composite_score
        
        # Sort by composite score
        return sorted(filtered, key=lambda x: x['composite_score'], reverse=True)


class LevelStrategy(ScalpingStrategy):
    """Enhanced level strategy with sophisticated S/R analysis and dynamic scoring."""
    
    def __init__(self, config: 'TradingConfig', constants: 'ScalpingConstants'):
        super().__init__(config, constants)
        self.analyzer = EnhancedLevelAnalyzer(min_touches=2, max_levels=6)
        
        # Enhanced thresholds
        self.proximity_threshold = 0.75  # ATR multiplier for "near" level
        self.strong_level_threshold = 1.5  # Minimum strength for strong levels
        self.breakout_confirmation = 0.3  # ATR multiplier for breakout confirmation
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score = 0.0
        factors = []
        
        # Get enhanced levels
        levels = self.analyzer.find_enhanced_levels(
            inputs.price_high_history, 
            inputs.price_low_history,
            inputs.price_close_history,
            getattr(inputs, 'volume_history', None)  # Optional volume data
        )
        
        support_levels = levels.get("support", [])
        resistance_levels = levels.get("resistance", [])
        
        price = inputs.current_price
        atr = inputs.atr_value if inputs.atr_value else price * 0.002
        
        if not support_levels and not resistance_levels:
            return 0.0, ["No significant S/R levels identified"]
        
        # Analyze support levels
        support_score, support_factors = self._analyze_support_levels(
            support_levels, price, atr
        )
        
        # Analyze resistance levels
        resistance_score, resistance_factors = self._analyze_resistance_levels(
            resistance_levels, price, atr
        )
        
        # Analyze potential breakouts/breakdowns
        breakout_score, breakout_factors = self._analyze_breakouts(
            support_levels, resistance_levels, price, atr, inputs.price_close_history
        )
        
        # Combine scores
        score = support_score + resistance_score + breakout_score
        factors.extend(support_factors + resistance_factors + breakout_factors)
        
        # Add level context information
        if support_levels:
            strongest_support = max(support_levels, key=lambda x: x['composite_score'])
            factors.append(f"Strongest support: {strongest_support['price']:.4f} "
                         f"(strength: {strongest_support['strength']:.1f}, "
                         f"touches: {strongest_support['touches']})")
        
        if resistance_levels:
            strongest_resistance = max(resistance_levels, key=lambda x: x['composite_score'])
            factors.append(f"Strongest resistance: {strongest_resistance['price']:.4f} "
                         f"(strength: {strongest_resistance['strength']:.1f}, "
                         f"touches: {strongest_resistance['touches']})")
        
        return max(min(score, 5.0), -5.0), factors  # Cap score between -5 and 5
    
    def _analyze_support_levels(self, support_levels: List[Dict], 
                               price: float, atr: float) -> Tuple[float, List[str]]:
        """Analyze support levels and return score and factors."""
        score = 0.0
        factors = []
        
        for level in support_levels:
            level_price = level['price']
            distance_atr = abs(price - level_price) / atr
            strength = level['strength']
            
            if distance_atr <= self.proximity_threshold:
                # Price is near support
                base_score = 1.0 + (strength * 0.5)
                
                # Bonus for very strong levels
                if strength >= self.strong_level_threshold:
                    base_score *= 1.3
                    factors.append(f"Near STRONG support at {level_price:.4f} "
                                 f"(dist: {distance_atr:.1f} ATR)")
                else:
                    factors.append(f"Near support at {level_price:.4f} "
                                 f"(dist: {distance_atr:.1f} ATR)")
                
                # Proximity bonus - closer = stronger signal
                proximity_bonus = max(0, (self.proximity_threshold - distance_atr) / self.proximity_threshold)
                score += base_score * (0.7 + 0.3 * proximity_bonus)
                
                break  # Only score the nearest level to avoid double-counting
        
        return score, factors
    
    def _analyze_resistance_levels(self, resistance_levels: List[Dict], 
                                  price: float, atr: float) -> Tuple[float, List[str]]:
        """Analyze resistance levels and return score and factors."""
        score = 0.0
        factors = []
        
        for level in resistance_levels:
            level_price = level['price']
            distance_atr = abs(price - level_price) / atr
            strength = level['strength']
            
            if distance_atr <= self.proximity_threshold:
                # Price is near resistance
                base_score = -(1.0 + (strength * 0.5))
                
                # Penalty for very strong levels
                if strength >= self.strong_level_threshold:
                    base_score *= 1.3
                    factors.append(f"Near STRONG resistance at {level_price:.4f} "
                                 f"(dist: {distance_atr:.1f} ATR)")
                else:
                    factors.append(f"Near resistance at {level_price:.4f} "
                                 f"(dist: {distance_atr:.1f} ATR)")
                
                # Proximity penalty - closer = stronger negative signal
                proximity_penalty = max(0, (self.proximity_threshold - distance_atr) / self.proximity_threshold)
                score += base_score * (0.7 + 0.3 * proximity_penalty)
                
                break  # Only score the nearest level
        
        return score, factors
    
    def _analyze_breakouts(self, support_levels: List[Dict], resistance_levels: List[Dict],
                          price: float, atr: float, close_history: List[float]) -> Tuple[float, List[str]]:
        """Analyze potential breakouts and breakdowns."""
        score = 0.0
        factors = []
        
        if len(close_history) < 5:
            return score, factors
        
        recent_high = max(close_history[-5:])
        recent_low = min(close_history[-5:])
        
        # Check for resistance breakouts
        for level in resistance_levels:
            level_price = level['price']
            if recent_high > level_price + (atr * self.breakout_confirmation):
                breakout_strength = 1.5 + (level['strength'] * 0.3)
                score += breakout_strength
                factors.append(f"BREAKOUT above resistance {level_price:.4f} "
                             f"(strength: {level['strength']:.1f})")
                break
        
        # Check for support breakdowns
        for level in support_levels:
            level_price = level['price']
            if recent_low < level_price - (atr * self.breakout_confirmation):
                breakdown_strength = -(1.5 + (level['strength'] * 0.3))
                score += breakdown_strength
                factors.append(f"BREAKDOWN below support {level_price:.4f} "
                             f"(strength: {level['strength']:.1f})")
                break
        
        return score, factors