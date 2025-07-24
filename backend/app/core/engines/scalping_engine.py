import logging
import numpy as np
from datetime import datetime, time
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict

from app.core.config import TradingConfig, ScalpingConstants
from app.core.models.signals import ScalpingInputs, SignalResult
from app.core.models.enums import SignalDirection
from app.core.engines.risk_engine import EnhancedRiskManager

# Import all strategies
from app.core.strategies.base import ScalpingStrategy
from app.core.strategies.momentum import MomentumStrategy
from app.core.strategies.bollingerband import BollingerBandsStrategy
from app.core.strategies.volume import VolumeStrategy
from app.core.strategies.microstructure import MicrostructureStrategy
from app.core.strategies.gamma_levels import GammaLevelsStrategy
from app.core.strategies.vwap import VWAPStrategy
from app.core.strategies.market_structure import MarketStructureStrategy
from app.core.strategies.options_filter import EnhancedOptionsFilterStrategy
from app.core.strategies.dte_risk import DTEStrategy
from app.core.strategies.time_session import TimeSessionStrategy
from app.core.strategies.levels import LevelStrategy
from app.core.strategies.candle_analysis import CandleAnalysisStrategy
from app.core.strategies.open_interest import OpenInterestStrategy

class EnhancedScalpingEngine:
    """Advanced scalping engine with adaptive strategies and market regime detection."""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.constants = ScalpingConstants()
        self.logger = logging.getLogger(__name__)
        
        # Enhanced risk manager
        self.risk_manager = EnhancedRiskManager()
        
        # Initialize strategies
        self.strategies: Dict[str, ScalpingStrategy] = self._init_strategies()
        
        # Performance tracking
        self.strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.strategy_confidence: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Market regime tracking
        self.market_regime = "RANGING"  # TRENDING, RANGING, VOLATILE
        self.regime_confidence = 0.5
        
        # Signal history for adaptive thresholds
        self.signal_history: deque = deque(maxlen=50)
        
    def _init_strategies(self) -> Dict[str, ScalpingStrategy]:
        """Initialize all trading strategies."""
        return {
            'momentum': MomentumStrategy(self.config, self.constants),
            'volume': VolumeStrategy(self.config, self.constants),
            'microstructure': MicrostructureStrategy(self.config, self.constants),
            'vwap': VWAPStrategy(self.config, self.constants),
            'gamma_levels': GammaLevelsStrategy(self.config, self.constants),
            'market_structure': MarketStructureStrategy(self.config, self.constants),
            'options_filter': EnhancedOptionsFilterStrategy(self.config, self.constants),
            'dte_risk': DTEStrategy(self.config, self.constants),
            'time_session': TimeSessionStrategy(self.config, self.constants),
            'bollinger_bands': BollingerBandsStrategy(self.config, self.constants),
            'levels': LevelStrategy(self.config, self.constants),
            'open_interest': OpenInterestStrategy(self.config, self.constants),
            'candle_analysis': CandleAnalysisStrategy(self.config, self.constants)
        }

    def analyze_signal(self, inputs: Dict[str, Any]) -> Optional[SignalResult]:
        """Enhanced signal analysis with adaptive features."""
        try:
            validated_inputs = ScalpingInputs(**inputs)
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return None

        # Market condition checks
        if self._should_avoid_trading(validated_inputs):
            return self.generate_neutral_result("Market conditions suggest avoiding trades")

        # ... (rest of the method is unchanged) ...
        # Detect current market regime
        self._update_market_regime(validated_inputs)

        # Get adaptive weights based on market conditions and performance
        adjusted_weights = self._get_adaptive_weights(validated_inputs)

        # Calculate strategy scores with confidence
        strategy_results: Dict[str, Tuple[float, float, List[str]]] = {}
        all_key_factors: List[str] = []

        for name, strategy in self.strategies.items():
            try:
                score, factors = strategy.calculate_score(validated_inputs)
                confidence = self._calculate_strategy_confidence(name, score, validated_inputs)
                strategy_results[name] = (score, confidence, factors)
                all_key_factors.extend(factors)
            except Exception as e:
                self.logger.warning(f"Strategy {name} failed: {e}")
                strategy_results[name] = (0, 0, [])

        # Calculate confidence-weighted total score
        total_score = self._calculate_confidence_weighted_score(strategy_results, adjusted_weights)

        # Determine enhanced signal with adaptive thresholds
        direction, strength, confidence = self._determine_adaptive_signal(
            total_score, strategy_results, validated_inputs
        )

        # Generate dynamic exit signals
        exit_signals = self._get_dynamic_exit_signals(validated_inputs, strategy_results)

        # Get comprehensive risk advice
        overall_confidence = np.mean([conf for _, conf, _ in strategy_results.values()]) if strategy_results else 0.5
        risk_advice = self.risk_manager.get_comprehensive_risk_advice(
            inputs, direction.value, overall_confidence
        )

        # Update signal history for adaptive learning
        self._update_signal_history(total_score, direction, strategy_results)

        return SignalResult(
            direction=direction,
            strength=strength,
            confidence=confidence,
            total_score=total_score,
            key_factors=sorted(list(set(all_key_factors)))[:12],
            exit_signals=exit_signals,
            risk_advice=risk_advice
        )


    def _should_avoid_trading(self, inputs: ScalpingInputs) -> bool:
        """Enhanced trading avoidance logic."""
        current_time = datetime.strptime(inputs.current_time, "%H:%M").time()
        
        # Basic avoidance conditions
        if inputs.news_event_imminent:
            return True
            
        # Market open volatility
        if time(9, 30) <= current_time <= time(9, 35):
            return True
            
        # Extreme VIX levels
        if inputs.vix_level > 40 or inputs.vix_level < 10:
            return True
            
        # Poor liquidity conditions
        if inputs.bid_ask_spread / inputs.current_price > 0.005:  # 0.5% spread
            return True
            
        # --- FIX APPLIED HERE ---
        # Weekend/holiday gaps (if volume is extremely low)
        # Changed 'inputs.volume' to check the last item in 'inputs.volume_history'
        if inputs.volume_history and inputs.volume_history[-1] < 1000:
            return True
            
        return False

    # ... (all other methods remain the same) ...
    def _update_market_regime(self, inputs: ScalpingInputs):
        """Detect and update current market regime."""
        # Simple regime detection based on ATR, volume, and VIX
        atr_ratio = inputs.atr_value / inputs.current_price if inputs.atr_value and inputs.current_price > 0 else 0.01
        
        # Assuming volume_ratio might be added later, for now we check history
        volume_factor = 1.0
        if len(inputs.volume_history) > 20:
            recent_avg_vol = np.mean(inputs.volume_history[-5:])
            historical_avg_vol = np.mean(inputs.volume_history[-20:])
            if historical_avg_vol > 0:
                volume_factor = recent_avg_vol / historical_avg_vol

        if inputs.vix_level > 25 and atr_ratio > 0.02:
            new_regime = "VOLATILE"
        elif atr_ratio > 0.015 and volume_factor > 1.2:
            new_regime = "TRENDING"
        else:
            new_regime = "RANGING"
            
        # Update regime with smoothing
        if new_regime != self.market_regime:
            self.regime_confidence = max(0.1, self.regime_confidence - 0.1)
        else:
            self.regime_confidence = min(1.0, self.regime_confidence + 0.1)
            
        if self.regime_confidence < 0.3: # If confidence drops, switch regime
            self.market_regime = new_regime

    def _get_adaptive_weights(self, inputs: ScalpingInputs) -> Dict[str, float]:
        """Calculate adaptive weights based on market regime and strategy performance."""
        base_weights = self.config.base_weights.copy()
        
        # Market regime adjustments
        regime_adjustments = {
            "TRENDING": {
                'momentum': 1.3,
                'market_structure': 1.2,
                'vwap': 0.9,
                'levels': 0.8
            },
            "RANGING": {
                'levels': 1.4,
                'bollinger_bands': 1.2,
                'momentum': 0.8,
                'market_structure': 0.9
            },
            "VOLATILE": {
                'microstructure': 1.3,
                'volume': 1.2,
                'gamma_levels': 1.1,
                'momentum': 0.9
            }
        }
        
        # Apply regime adjustments
        for strategy, adjustment in regime_adjustments.get(self.market_regime, {}).items():
            if strategy in base_weights:
                base_weights[strategy] *= adjustment
        
        # Performance-based adjustments
        for strategy_name, weight in base_weights.items():
            performance_multiplier = self.strategy_confidence.get(strategy_name, 1.0)
            base_weights[strategy_name] = weight * performance_multiplier
        
        # VIX-based adjustments
        if inputs.vix_level > 25:
            base_weights['momentum'] *= 1.2
            base_weights['volume'] *= 1.1
        elif inputs.vix_level < 15:
            base_weights['momentum'] *= 0.9
            base_weights['vwap'] *= 1.1
        
        # Time-based adjustments
        current_time = datetime.strptime(inputs.current_time, "%H:%M").time()
        if time(15, 30) <= current_time <= time(16, 0):  # Power hour
            base_weights['volume'] *= 1.2
            base_weights['momentum'] *= 1.1
        
        # DTE adjustments
        if inputs.dte == 0:
            base_weights['options_filter'] *= 1.5
            base_weights['gamma_levels'] *= 1.3
            base_weights['dte_risk'] *= 2.0
        
        return base_weights

    def _calculate_strategy_confidence(self, 
                                     strategy_name: str, 
                                     score: float, 
                                     inputs: ScalpingInputs) -> float:
        """Calculate confidence level for individual strategy."""
        base_confidence = 0.5
        
        # Score magnitude confidence
        score_confidence = min(1.0, abs(score) / 3.0)  # Higher scores = higher confidence
        
        # Historical performance confidence
        historical_confidence = self.strategy_confidence.get(strategy_name, 1.0)
        
        # Market condition confidence
        condition_confidence = 1.0
        
        # Reduce confidence in poor conditions
        if inputs.current_price > 0 and inputs.bid_ask_spread / inputs.current_price > 0.002:
            condition_confidence *= 0.8
            
        if inputs.vix_level > 35:
            condition_confidence *= 0.9
            
        # Time-based confidence
        current_time = datetime.strptime(inputs.current_time, "%H:%M").time()
        if current_time < time(9, 45) or current_time > time(15, 45):
            condition_confidence *= 0.9
        
        return min(1.0, base_confidence + 
                  (score_confidence * 0.3) + 
                  (historical_confidence * 0.2) + 
                  (condition_confidence * 0.2))

    def _calculate_confidence_weighted_score(self, 
                                           strategy_results: Dict[str, Tuple[float, float, List[str]]], 
                                           weights: Dict[str, float]) -> float:
        """Calculate total score weighted by both strategy weights and confidence."""
        total_weighted = 0
        total_weight = 0
        
        for name, (score, confidence, _) in strategy_results.items():
            strategy_weight = weights.get(name, 1.0)
            final_weight = strategy_weight * confidence
            
            total_weighted += score * final_weight
            total_weight += final_weight
        
        return total_weighted / total_weight if total_weight > 0 else 0

    def _determine_adaptive_signal(self, 
                                 total_score: float, 
                                 strategy_results: Dict[str, Tuple[float, float, List[str]]],
                                 inputs: ScalpingInputs) -> Tuple[SignalDirection, str, str]:
        """Determine signal with adaptive thresholds based on market conditions."""
        
        # Get adaptive thresholds
        thresholds = self._get_adaptive_thresholds(inputs)
        
        # Count supporting strategies
        strong_bull_count = sum(1 for score, conf, _ in strategy_results.values() 
                               if score > 2.0 and conf > 0.6)
        strong_bear_count = sum(1 for score, conf, _ in strategy_results.values() 
                               if score < -2.0 and conf > 0.6)
        
        # Check for options filter veto
        options_score, options_conf, _ = strategy_results.get('options_filter', (0, 0, []))
        if options_score < -2.5 and options_conf > 0.7: # Adjusted threshold slightly
            return SignalDirection.NEUTRAL, "Neutral", "Low - Poor contract quality"
        
        # Determine signal with consensus requirement
        if total_score >= thresholds['strong_bull'] and strong_bull_count >= 3:
            confidence = "High" if strong_bull_count >= 4 else "Moderate-High"
            return SignalDirection.STRONG_BULLISH, "Strong", confidence
            
        elif total_score >= thresholds['bull'] and strong_bull_count >= 2:
            confidence = "Moderate" if strong_bull_count >= 3 else "Moderate-Low"
            return SignalDirection.BULLISH, "Moderate", confidence
            
        elif total_score <= thresholds['strong_bear'] and strong_bear_count >= 3:
            confidence = "High" if strong_bear_count >= 4 else "Moderate-High"
            return SignalDirection.STRONG_BEARISH, "Strong", confidence
            
        elif total_score <= thresholds['bear'] and strong_bear_count >= 2:
            confidence = "Moderate" if strong_bear_count >= 3 else "Moderate-Low"
            return SignalDirection.BEARISH, "Moderate", confidence
        
        return SignalDirection.NEUTRAL, "Neutral", "Low"

    def _get_adaptive_thresholds(self, inputs: ScalpingInputs) -> Dict[str, float]:
        """Calculate adaptive signal thresholds based on market conditions."""
        base = self.config.base_thresholds.copy()
        
        # Adjust thresholds based on volatility
        vix_adjustment = 1.0
        if inputs.vix_level > 30:
            vix_adjustment = 1.2  # Higher thresholds in volatile markets
        elif inputs.vix_level < 15:
            vix_adjustment = 0.9  # Lower thresholds in calm markets
        
        # Adjust based on market regime
        regime_adjustment = {
            "TRENDING": 0.9,  # Lower thresholds for trending markets
            "RANGING": 1.1,   # Higher thresholds for ranging markets
            "VOLATILE": 1.15  # Highest thresholds for volatile markets
        }.get(self.market_regime, 1.0)
        
        # Time-based adjustments
        current_time = datetime.strptime(inputs.current_time, "%H:%M").time()
        time_adjustment = 1.0
        
        if time(9, 30) <= current_time <= time(10, 0):
            time_adjustment = 1.3  # Higher thresholds during open
        elif time(15, 30) <= current_time <= time(16, 0):
            time_adjustment = 1.1  # Slightly higher during power hour
        
        final_adjustment = vix_adjustment * regime_adjustment * time_adjustment
        
        return {
            'strong_bull': base['strong_bull'] * final_adjustment,
            'bull': base['bull'] * final_adjustment,
            'bear': base['bear'] * final_adjustment,
            'strong_bear': base['strong_bear'] * final_adjustment
        }

    def _get_dynamic_exit_signals(self, 
                                inputs: ScalpingInputs, 
                                strategy_results: Dict[str, Tuple[float, float, List[str]]]) -> List[str]:
        """Generate dynamic exit signals based on multiple factors."""
        signals = []
        
        # Volume-based exits
        volume_score, volume_conf, _ = strategy_results.get('volume', (0, 0, []))
        if volume_score < -1.5 and volume_conf > 0.6:
            signals.append("⚠️ EXIT ALERT: Volume deteriorating - move not supported")
        
        # Momentum divergence exits
        momentum_score, momentum_conf, _ = strategy_results.get('momentum', (0, 0, []))
        if momentum_score < -2.0 and momentum_conf > 0.7:
            signals.append("📉 MOMENTUM EXIT: Strong momentum reversal detected")
        
        # Microstructure deterioration
        micro_score, micro_conf, _ = strategy_results.get('microstructure', (0, 0, []))
        if micro_score < -1.0 and micro_conf > 0.6:
            signals.append("🔍 MICRO EXIT: Order flow turning negative")
        
        # Time-based exits for 0DTE
        if inputs.dte == 0:
            current_time = datetime.strptime(inputs.current_time, "%H:%M").time()
            if current_time >= time(15, 0):
                signals.append("🕒 TIME EXIT: 0DTE final hour - extreme theta decay")
            elif current_time >= time(14, 0):
                signals.append("⏰ TIME WARNING: 0DTE approaching final hour")
        
        # Gamma level exits
        gamma_score, gamma_conf, _ = strategy_results.get('gamma_levels', (0, 0, []))
        if gamma_score < -1.5 and gamma_conf > 0.6:
            signals.append("⚡ GAMMA EXIT: Approaching significant gamma resistance")
        
        # VIX spike exits
        if inputs.vix_level > 35:
            signals.append("📊 VOLATILITY EXIT: VIX spike - consider profit taking")
        
        # VWAP deviation exits
        vwap_score, vwap_conf, _ = strategy_results.get('vwap', (0, 0, []))
        if vwap_score < -2.5 and vwap_conf > 0.6: # Corrected to check for negative score for reversal
            signals.append("📈 VWAP EXIT: Extreme deviation from VWAP - reversion likely")
        
        return signals[:5]  # Limit to most critical exit signals

    def _update_signal_history(self, 
                             total_score: float, 
                             direction: SignalDirection, 
                             strategy_results: Dict[str, Tuple[float, float, List[str]]]):
        """Update signal history for adaptive learning."""
        signal_data = {
            'score': total_score,
            'direction': direction,
            'timestamp': datetime.now(),
            'strategy_scores': {name: score for name, (score, _, _) in strategy_results.items()}
        }
        
        self.signal_history.append(signal_data)

    def update_strategy_performance(self, strategy_name: str, performance: float):
        """Update individual strategy performance tracking."""
        if strategy_name in self.strategies:
            self.strategy_performance[strategy_name].append(performance)
            
            # Calculate rolling performance confidence
            recent_performance = list(self.strategy_performance[strategy_name])
            if len(recent_performance) >= 10:
                # Simple performance-based confidence
                win_rate = sum(1 for p in recent_performance[-20:] if p > 0) / min(20, len(recent_performance)) if len(recent_performance) > 0 else 0
                avg_return = np.mean(recent_performance[-20:])
                
                # Combine win rate and average return for confidence
                self.strategy_confidence[strategy_name] = min(1.5, max(0.5, 
                    (win_rate * 1.2) + (avg_return * 0.3)))

    def get_market_regime_info(self) -> Dict[str, Any]:
        """Get current market regime information."""
        return {
            'market_regime': self.market_regime,
            'confidence': self.regime_confidence,
            'strategy_confidences': dict(self.strategy_confidence),
            'recent_signals': len(self.signal_history)
        }

    def generate_neutral_result(self, reason: str) -> SignalResult:
        """Generate a neutral signal result with reason."""
        return SignalResult(
            direction=SignalDirection.NEUTRAL,
            strength="Neutral",
            confidence="N/A",
            total_score=0.0,
            key_factors=[reason],
            exit_signals=[],
            risk_advice=["🚫 Trading avoided due to unfavorable conditions"]
        )

    def calculate_dynamic_exit_thresholds(self, 
                                        entry_score: float, 
                                        time_elapsed_minutes: int,
                                        current_pnl_pct: float = 0) -> Dict[str, float]:
        """Calculate time-decay adjusted exit thresholds."""
        base_profit_target = abs(entry_score) * 0.6  # 60% of entry conviction
        base_stop_threshold = abs(entry_score) * 0.3   # 30% of entry conviction
        
        # Time decay adjustments
        time_decay_factor = 1.0 - (time_elapsed_minutes / 60) * 0.3  # Reduce targets over time
        time_decay_factor = max(0.4, time_decay_factor)  # Don't go below 40%
        
        # PnL-based adjustments
        pnl_adjustment = 1.0
        if current_pnl_pct > 0.5:  # If already profitable
            pnl_adjustment = 0.8  # Tighten targets
        elif current_pnl_pct < -0.3:  # If losing
            pnl_adjustment = 1.2  # Widen stops slightly
        
        return {
            'profit_target': base_profit_target * time_decay_factor,
            'stop_threshold': base_stop_threshold * pnl_adjustment,
            'breakeven_threshold': 0.1,  # Move to breakeven quickly
            'time_decay_factor': time_decay_factor
        }

    def analyze_strategy_consensus(self, 
                                 strategy_results: Dict[str, Tuple[float, float, List[str]]]) -> Dict[str, Any]:
        """Analyze consensus among strategies."""
        scores = [score for score, _, _ in strategy_results.values()]
        confidences = [conf for _, conf, _ in strategy_results.values()]
        
        bullish_strategies = [name for name, (score, conf, _) in strategy_results.items() 
                            if score > 1.0 and conf > 0.5]
        bearish_strategies = [name for name, (score, conf, _) in strategy_results.items() 
                            if score < -1.0 and conf > 0.5]
        neutral_strategies = [name for name, (score, conf, _) in strategy_results.items() 
                            if abs(score) <= 1.0 or conf <= 0.5]
        
        consensus_strength = 0
        consensus_direction = "MIXED"
        if len(strategy_results) > 0:
            if len(bullish_strategies) > len(bearish_strategies) + len(neutral_strategies):
                consensus_strength = len(bullish_strategies) / len(strategy_results)
                consensus_direction = "BULLISH"
            elif len(bearish_strategies) > len(bullish_strategies) + len(neutral_strategies):
                consensus_strength = len(bearish_strategies) / len(strategy_results)
                consensus_direction = "BEARISH"
        
        return {
            'direction': consensus_direction,
            'strength': consensus_strength,
            'bullish_count': len(bullish_strategies),
            'bearish_count': len(bearish_strategies),
            'neutral_count': len(neutral_strategies),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'score_std': np.std(scores) if len(scores) > 1 else 0
        }

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information for debugging and optimization."""
        return {
            'market_regime': {
                'current': self.market_regime,
                'confidence': self.regime_confidence
            },
            'strategy_health': {
                name: {
                    'confidence': self.strategy_confidence.get(name, 1.0),
                    'recent_signals': len(self.strategy_performance.get(name, [])),
                    'avg_performance': np.mean(list(self.strategy_performance.get(name, [0])))
                }
                for name in self.strategies.keys()
            },
            'signal_history_length': len(self.signal_history),
            'last_signal_time': self.signal_history[-1]['timestamp'].isoformat() if self.signal_history else None
        }
