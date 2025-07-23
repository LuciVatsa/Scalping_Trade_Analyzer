from typing import Tuple, List, Dict, Optional
import numpy as np
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs
from app.core.models.enums import MarketStructure
from app.core.config import TradingConfig, ScalpingConstants

class MarketStructureAnalyzer:
    """Advanced market structure analysis with multiple timeframe context."""
    
    def __init__(self):
        self.structure_memory = {}  # Cache for structure analysis
    
    def analyze_comprehensive_structure(self, close_history: List[float], 
                                      high_history: List[float], 
                                      low_history: List[float],
                                      volume_history: Optional[List[float]] = None) -> Dict:
        """Comprehensive market structure analysis with multiple perspectives."""
        
        if len(close_history) < 30:
            return self._get_default_structure()
        
        # Multi-timeframe analysis
        short_term = self._analyze_timeframe(close_history[-10:], high_history[-10:], low_history[-10:])
        medium_term = self._analyze_timeframe(close_history[-20:], high_history[-20:], low_history[-20:])
        long_term = self._analyze_timeframe(close_history[-50:], high_history[-50:], low_history[-50:])
        
        # Trend strength analysis
        trend_strength = self._calculate_trend_strength(close_history)
        
        # Volatility context
        volatility_regime = self._analyze_volatility_regime(close_history)
        
        # Volume confirmation (if available)
        volume_profile = self._analyze_volume_profile(close_history, volume_history) if volume_history else {}
        
        # Higher highs / Lower lows analysis
        swing_analysis = self._analyze_swing_structure(high_history, low_history)
        
        # Consolidation detection
        consolidation_info = self._detect_consolidation(high_history, low_history, close_history)
        
        return {
            'primary_structure': self._determine_primary_structure(short_term, medium_term, long_term),
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'trend_strength': trend_strength,
            'volatility_regime': volatility_regime,
            'volume_profile': volume_profile,
            'swing_analysis': swing_analysis,
            'consolidation': consolidation_info,
            'structure_quality': self._assess_structure_quality(swing_analysis, trend_strength)
        }
    
    def _analyze_timeframe(self, closes: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Analyze structure for a specific timeframe."""
        if len(closes) < 6:
            return {'structure': MarketStructure.RANGE_BOUND, 'confidence': 0.0}
        
        # Calculate recent and previous ranges
        mid_point = len(closes) // 2
        recent_high = max(highs[mid_point:])
        recent_low = min(lows[mid_point:])
        prev_high = max(highs[:mid_point])
        prev_low = min(lows[:mid_point])
        
        # Price momentum
        price_change = (closes[-1] - closes[0]) / closes[0] * 100
        
        # Determine structure
        structure = MarketStructure.RANGE_BOUND
        confidence = 0.5
        
        # Breakout conditions
        if recent_high > prev_high * 1.001 and recent_low >= prev_low * 0.999:
            if price_change > 0.05:  # 0.05% minimum momentum
                structure = MarketStructure.BREAKOUT
                confidence = min(abs(price_change) * 10, 1.0)
        
        # Breakdown conditions
        elif recent_low < prev_low * 0.999 and recent_high <= prev_high * 1.001:
            if price_change < -0.05:
                structure = MarketStructure.BREAKDOWN
                confidence = min(abs(price_change) * 10, 1.0)
        
        # Range-bound with slight bias
        else:
            if abs(price_change) < 0.02:  # Very small movement
                structure = MarketStructure.RANGE_BOUND
                confidence = 0.8
            elif price_change > 0:
                structure = MarketStructure.RANGE_BOUND  # Could add BULLISH_BIAS if enum exists
                confidence = 0.6
            else:
                structure = MarketStructure.RANGE_BOUND  # Could add BEARISH_BIAS if enum exists
                confidence = 0.6
        
        return {
            'structure': structure,
            'confidence': confidence,
            'price_change_pct': price_change,
            'range_expansion': (recent_high - recent_low) / (prev_high - prev_low) if prev_high != prev_low else 1.0
        }
    
    def _calculate_trend_strength(self, closes: List[float]) -> Dict:
        """Calculate trend strength using multiple indicators."""
        if len(closes) < 20:
            return {'strength': 0.0, 'direction': 'neutral'}
        
        # Moving average slopes
        short_ma = np.mean(closes[-5:])
        medium_ma = np.mean(closes[-10:])
        long_ma = np.mean(closes[-20:])
        
        # Trend alignment
        bullish_alignment = short_ma > medium_ma > long_ma
        bearish_alignment = short_ma < medium_ma < long_ma
        
        # Price vs MA position
        current_price = closes[-1]
        ma_position = (current_price - long_ma) / long_ma * 100
        
        # Consecutive higher/lower closes
        consecutive_direction = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i-1]:
                consecutive_direction += 1
            elif closes[i] < closes[i-1]:
                consecutive_direction -= 1
            else:
                break
        
        # Calculate strength
        strength = 0.0
        direction = 'neutral'
        
        if bullish_alignment:
            strength += 0.3
            direction = 'bullish'
        elif bearish_alignment:
            strength += 0.3
            direction = 'bearish'
        
        # Add consecutive momentum
        strength += min(abs(consecutive_direction) * 0.1, 0.4)
        
        # Add MA position component
        strength += min(abs(ma_position) * 0.02, 0.3)
        
        if direction == 'bearish':
            strength = -strength
        
        return {
            'strength': max(min(strength, 1.0), -1.0),
            'direction': direction,
            'ma_alignment': bullish_alignment or bearish_alignment,
            'consecutive_moves': consecutive_direction,
            'ma_position_pct': ma_position
        }
    
    def _analyze_volatility_regime(self, closes: List[float]) -> Dict:
        """Analyze current volatility regime."""
        if len(closes) < 20:
            return {'regime': 'normal', 'percentile': 0.5}
        
        # Calculate recent volatility
        returns = [abs((closes[i] - closes[i-1]) / closes[i-1]) for i in range(1, len(closes))]
        recent_vol = np.mean(returns[-10:])
        historical_vol = np.mean(returns)
        
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio > 1.5:
            regime = 'high_volatility'
        elif vol_ratio < 0.7:
            regime = 'low_volatility'
        else:
            regime = 'normal'
        
        # Calculate percentile rank
        vol_percentile = sum(1 for vol in returns if vol <= recent_vol) / len(returns)
        
        return {
            'regime': regime,
            'ratio': vol_ratio,
            'percentile': vol_percentile,
            'recent_vol': recent_vol,
            'historical_vol': historical_vol
        }
    
    def _analyze_volume_profile(self, closes: List[float], volumes: List[float]) -> Dict:
        """Analyze volume profile for structure confirmation."""
        if not volumes or len(volumes) != len(closes):
            return {}
        
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-5:])
        
        # Volume trend
        volume_trend = 'increasing' if recent_volume > avg_volume * 1.2 else \
                      'decreasing' if recent_volume < avg_volume * 0.8 else 'stable'
        
        # Price-volume relationship
        price_changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, min(len(closes), 10))]
        volume_changes = [(volumes[i] - volumes[i-1]) / volumes[i-1] for i in range(1, min(len(volumes), 10))]
        
        correlation = np.corrcoef(price_changes, volume_changes)[0, 1] if len(price_changes) > 1 else 0
        
        return {
            'trend': volume_trend,
            'ratio': recent_volume / avg_volume,
            'price_volume_correlation': correlation,
            'confirmation': abs(correlation) > 0.3  # Volume confirms price movement
        }
    
    def _analyze_swing_structure(self, highs: List[float], lows: List[float]) -> Dict:
        """Analyze swing highs and lows for structure determination."""
        if len(highs) < 10 or len(lows) < 10:
            return {'pattern': 'insufficient_data'}
        
        # Find recent swing points
        recent_highs = highs[-6:]
        recent_lows = lows[-6:]
        prev_highs = highs[-12:-6]
        prev_lows = lows[-12:-6]
        
        current_high = max(recent_highs)
        current_low = min(recent_lows)
        prev_high = max(prev_highs)
        prev_low = min(prev_lows)
        
        # Determine pattern
        higher_high = current_high > prev_high
        higher_low = current_low > prev_low
        lower_high = current_high < prev_high
        lower_low = current_low < prev_low
        
        if higher_high and higher_low:
            pattern = 'higher_highs_higher_lows'
            bias = 'bullish'
        elif lower_high and lower_low:
            pattern = 'lower_highs_lower_lows'
            bias = 'bearish'
        elif higher_high and lower_low:
            pattern = 'expanding_range'
            bias = 'neutral'
        elif lower_high and higher_low:
            pattern = 'contracting_range'
            bias = 'neutral'
        else:
            pattern = 'mixed_signals'
            bias = 'neutral'
        
        return {
            'pattern': pattern,
            'bias': bias,
            'current_high': current_high,
            'current_low': current_low,
            'prev_high': prev_high,
            'prev_low': prev_low
        }
    
    def _detect_consolidation(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict:
        """Detect consolidation patterns."""
        if len(closes) < 15:
            return {'in_consolidation': False}
        
        recent_range = max(highs[-10:]) - min(lows[-10:])
        historical_range = max(highs[-20:]) - min(lows[-20:])
        
        # Price compression ratio
        compression_ratio = recent_range / historical_range if historical_range > 0 else 1.0
        
        # Volatility compression
        recent_volatility = np.std(closes[-10:])
        historical_volatility = np.std(closes[-20:])
        vol_compression = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        in_consolidation = compression_ratio < 0.6 and vol_compression < 0.8
        
        return {
            'in_consolidation': in_consolidation,
            'compression_ratio': compression_ratio,
            'volatility_compression': vol_compression,
            'consolidation_strength': 1.0 - compression_ratio if in_consolidation else 0.0
        }
    
    def _determine_primary_structure(self, short_term: Dict, medium_term: Dict, long_term: Dict) -> Dict:
        """Determine primary market structure from multiple timeframes."""
        
        # Weight by confidence and timeframe importance
        structures = [
            (short_term['structure'], short_term['confidence'] * 0.5),  # 50% weight
            (medium_term['structure'], medium_term['confidence'] * 0.3),  # 30% weight
            (long_term['structure'], long_term['confidence'] * 0.2)  # 20% weight
        ]
        
        # Count weighted votes
        structure_scores = {}
        for structure, weight in structures:
            structure_scores[structure] = structure_scores.get(structure, 0) + weight
        
        primary = max(structure_scores, key=structure_scores.get)
        confidence = structure_scores[primary] / sum(structure_scores.values())
        
        return {
            'structure': primary,
            'confidence': confidence,
            'agreement': len(set(s[0] for s in structures)) == 1  # All timeframes agree
        }
    
    def _assess_structure_quality(self, swing_analysis: Dict, trend_strength: Dict) -> Dict:
        """Assess the quality and reliability of the identified structure."""
        
        quality_score = 0.5  # Base score
        
        # Clear swing patterns add quality
        if swing_analysis.get('pattern') in ['higher_highs_higher_lows', 'lower_highs_lower_lows']:
            quality_score += 0.2
        
        # Strong trend adds quality
        trend_str = abs(trend_strength.get('strength', 0))
        quality_score += trend_str * 0.2
        
        # MA alignment adds quality
        if trend_strength.get('ma_alignment', False):
            quality_score += 0.1
        
        quality_score = max(min(quality_score, 1.0), 0.0)
        
        return {
            'score': quality_score,
            'reliability': 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.4 else 'low'
        }
    
    def _get_default_structure(self) -> Dict:
        """Return default structure when insufficient data."""
        return {
            'primary_structure': {
                'structure': MarketStructure.RANGE_BOUND,
                'confidence': 0.3,
                'agreement': False
            },
            'trend_strength': {'strength': 0.0, 'direction': 'neutral'},
            'structure_quality': {'score': 0.3, 'reliability': 'low'}
        }


class MarketStructureStrategy(ScalpingStrategy):
    """Enhanced market structure strategy with comprehensive analysis."""
    
    def __init__(self, config: 'TradingConfig', constants: 'ScalpingConstants'):
        super().__init__(config, constants)
        self.analyzer = MarketStructureAnalyzer()
        
        # Scoring parameters
        self.breakout_score = 2.0
        self.breakdown_score = -2.0
        self.trend_multiplier = 1.5
        self.quality_threshold = 0.6
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        """Calculate market structure score with comprehensive analysis."""
        score = 0.0
        factors = []
        
        # Get comprehensive structure analysis
        structure_analysis = self.analyzer.analyze_comprehensive_structure(
            inputs.price_close_history,
            inputs.price_high_history,
            inputs.price_low_history,
            getattr(inputs, 'volume_history', None)
        )
        
        primary = structure_analysis['primary_structure']
        trend_strength = structure_analysis['trend_strength']
        quality = structure_analysis['structure_quality']
        
        # Base structure scoring
        base_score, structure_factors = self._score_primary_structure(primary)
        score += base_score
        factors.extend(structure_factors)
        
        # Trend strength enhancement
        trend_score, trend_factors = self._score_trend_strength(trend_strength)
        score += trend_score
        factors.extend(trend_factors)
        
        # Structure quality adjustment
        quality_multiplier = self._get_quality_multiplier(quality)
        score *= quality_multiplier
        
        # Add contextual factors
        context_factors = self._get_contextual_factors(structure_analysis)
        factors.extend(context_factors)
        
        # Consolidation detection bonus/penalty
        consolidation = structure_analysis.get('consolidation', {})
        if consolidation.get('in_consolidation', False):
            consolidation_strength = consolidation.get('consolidation_strength', 0)
            if consolidation_strength > 0.7:
                factors.append(f"Strong consolidation detected (compression: {consolidation_strength:.2f})")
                score *= 0.5  # Reduce signal strength in tight consolidation
            else:
                factors.append(f"Mild consolidation detected")
                score *= 0.8
        
        # Volume confirmation
        volume_profile = structure_analysis.get('volume_profile', {})
        if volume_profile and volume_profile.get('confirmation', False):
            score *= 1.2
            factors.append(f"Volume confirms structure (correlation: {volume_profile.get('price_volume_correlation', 0):.2f})")
        
        return max(min(score, 4.0), -4.0), factors
    
    def _score_primary_structure(self, primary: Dict) -> Tuple[float, List[str]]:
        """Score the primary market structure."""
        structure = primary['structure']
        confidence = primary['confidence']
        agreement = primary['agreement']
        
        score = 0.0
        factors = []
        
        if structure == MarketStructure.BREAKOUT:
            score = self.breakout_score * confidence
            factors.append(f"Market Structure: BREAKOUT (confidence: {confidence:.2f})")
            if agreement:
                score *= 1.2
                factors.append("All timeframes confirm breakout")
        
        elif structure == MarketStructure.BREAKDOWN:
            score = self.breakdown_score * confidence
            factors.append(f"Market Structure: BREAKDOWN (confidence: {confidence:.2f})")
            if agreement:
                score *= 1.2
                factors.append("All timeframes confirm breakdown")
        
        else:  # RANGE_BOUND
            factors.append(f"Market Structure: RANGE_BOUND (confidence: {confidence:.2f})")
            if confidence > 0.8:
                factors.append("Strong range-bound structure - low breakout probability")
        
        return score, factors
    
    def _score_trend_strength(self, trend_strength: Dict) -> Tuple[float, List[str]]:
        """Score trend strength component."""
        strength = trend_strength.get('strength', 0)
        direction = trend_strength.get('direction', 'neutral')
        ma_alignment = trend_strength.get('ma_alignment', False)
        consecutive_moves = trend_strength.get('consecutive_moves', 0)
        
        score = strength * self.trend_multiplier
        factors = []
        
        if abs(strength) > 0.5:
            factors.append(f"Strong {direction} trend (strength: {strength:.2f})")
            
            if ma_alignment:
                score *= 1.1
                factors.append("Moving averages aligned with trend")
            
            if abs(consecutive_moves) >= 3:
                factors.append(f"{abs(consecutive_moves)} consecutive {direction} closes")
                score *= 1.05
        
        elif abs(strength) > 0.2:
            factors.append(f"Moderate {direction} bias (strength: {strength:.2f})")
        
        return score, factors
    
    def _get_quality_multiplier(self, quality: Dict) -> float:
        """Get quality multiplier for score adjustment."""
        quality_score = quality.get('score', 0.5)
        
        if quality_score > 0.8:
            return 1.2  # High quality structure
        elif quality_score > self.quality_threshold:
            return 1.0  # Normal quality
        else:
            return 0.7  # Low quality structure
    
    def _get_contextual_factors(self, analysis: Dict) -> List[str]:
        """Extract contextual information for factors."""
        factors = []
        
        # Volatility context
        volatility = analysis.get('volatility_regime', {})
        regime = volatility.get('regime', 'normal')
        if regime != 'normal':
            vol_ratio = volatility.get('ratio', 1.0)
            factors.append(f"Volatility regime: {regime.replace('_', ' ')} (ratio: {vol_ratio:.2f})")
        
        # Swing analysis
        swing = analysis.get('swing_analysis', {})
        pattern = swing.get('pattern', '')
        if pattern in ['higher_highs_higher_lows', 'lower_highs_lower_lows']:
            bias = swing.get('bias', 'neutral')
            factors.append(f"Swing pattern: {pattern.replace('_', ' ')} ({bias} bias)")
        elif pattern == 'expanding_range':
            factors.append("Range expansion detected - increased volatility expected")
        elif pattern == 'contracting_range':
            factors.append("Range contraction - potential breakout setup")
        
        # Structure quality
        quality = analysis.get('structure_quality', {})
        reliability = quality.get('reliability', 'medium')
        if reliability == 'high':
            factors.append("High structure reliability")
        elif reliability == 'low':
            factors.append("Low structure reliability - signals may be unreliable")
        
        return factors