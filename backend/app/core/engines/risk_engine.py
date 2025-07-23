from typing import Optional, Literal, Dict, List, Tuple
import numpy as np
from datetime import datetime, time
import logging

class EnhancedRiskManager:
    """Advanced risk management with dynamic parameters and multi-factor analysis."""
    
    def __init__(self, 
                 base_atr_multiplier: float = 1.5, 
                 max_position_pct: float = 0.10,
                 max_portfolio_risk: float = 0.02,
                 drawdown_threshold: float = 0.05):
        self.base_atr_multiplier = base_atr_multiplier
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.drawdown_threshold = drawdown_threshold
        self.logger = logging.getLogger(__name__)
        
        # Track portfolio state
        self.current_positions: Dict[str, Dict] = {}
        self.recent_performance: List[float] = []
        
    def calculate_dynamic_stop_loss(self, 
                                  current_price: float, 
                                  atr: float, 
                                  trend_direction: Literal["long", "short"],
                                  vix_level: float = 20.0,
                                  market_session: str = "regular") -> float:
        """Enhanced stop-loss calculation with volatility and session adjustments."""
        
        if atr is None or atr <= 0:
            # Fallback to VIX-adjusted percentage
            base_pct = 0.005 * (1 + (vix_level - 20) / 100)
            return current_price * (1 - base_pct) if trend_direction == "long" else current_price * (1 + base_pct)
        
        # Dynamic ATR multiplier based on market conditions
        dynamic_multiplier = self._calculate_dynamic_atr_multiplier(vix_level, market_session)
        
        if trend_direction == "long":
            return current_price - (atr * dynamic_multiplier)
        else:
            return current_price + (atr * dynamic_multiplier)
    
    def _calculate_dynamic_atr_multiplier(self, vix_level: float, market_session: str) -> float:
        """Calculate ATR multiplier based on market volatility and session."""
        base_multiplier = self.base_atr_multiplier
        
        # VIX-based adjustment
        if vix_level > 30:
            base_multiplier *= 1.3  # Wider stops in volatile markets
        elif vix_level < 15:
            base_multiplier *= 0.8  # Tighter stops in calm markets
        
        # Session-based adjustment
        session_adjustments = {
            "pre_market": 1.2,  # Wider stops due to lower liquidity
            "market_open": 1.4,  # Wider stops during volatile open
            "regular": 1.0,
            "power_hour": 1.1,
            "after_hours": 1.3
        }
        
        return base_multiplier * session_adjustments.get(market_session, 1.0)
    
    def calculate_enhanced_position_size(self, 
                                       account_value: float,
                                       risk_percent: float,
                                       current_price: float,
                                       stop_loss_price: float,
                                       signal_confidence: float = 0.5,
                                       expected_return: float = None,
                                       volatility: float = None,
                                       liquidity_score: float = 1.0) -> Optional[Dict[str, float]]:
        """Enhanced position sizing with multiple risk factors."""
        
        if any(v is None for v in [account_value, risk_percent, current_price, stop_loss_price]):
            return None
            
        risk_per_share = abs(current_price - stop_loss_price)
        if risk_per_share <= 0:
            return None
        
        # Base position size
        capital_to_risk = account_value * (risk_percent / 100)
        base_position_size = capital_to_risk / risk_per_share
        
        # Confidence adjustment
        confidence_adjusted_size = base_position_size * signal_confidence
        
        # Sharpe ratio adjustment (if available)
        if expected_return is not None and volatility is not None:
            sharpe_adjustment = self._calculate_sharpe_adjustment(expected_return, volatility)
            confidence_adjusted_size *= sharpe_adjustment
        
        # Liquidity adjustment
        liquidity_adjusted_size = confidence_adjusted_size * liquidity_score
        
        # Portfolio constraints
        max_shares_by_portfolio = (account_value * self.max_position_pct) / current_price
        
        # Drawdown protection
        drawdown_multiplier = self._calculate_drawdown_multiplier()
        
        final_size = min(liquidity_adjusted_size, max_shares_by_portfolio) * drawdown_multiplier
        
        return {
            "recommended_size": final_size,
            "base_size": base_position_size,
            "confidence_adjusted": confidence_adjusted_size,
            "liquidity_adjusted": liquidity_adjusted_size,
            "max_allowed": max_shares_by_portfolio,
            "drawdown_multiplier": drawdown_multiplier
        }
    
    def _calculate_sharpe_adjustment(self, expected_return: float, volatility: float) -> float:
        """Adjust position size based on risk-adjusted returns."""
        if volatility <= 0:
            return 1.0
        
        sharpe_ratio = expected_return / volatility
        
        # Scale position based on Sharpe ratio
        if sharpe_ratio > 2.0:
            return 1.2  # Increase size for high Sharpe
        elif sharpe_ratio > 1.0:
            return 1.0  # Normal size
        elif sharpe_ratio > 0.5:
            return 0.8  # Reduce size for low Sharpe
        else:
            return 0.6  # Significantly reduce for poor risk-adjusted returns
    
    def _calculate_drawdown_multiplier(self) -> float:
        """Reduce position sizes during drawdown periods."""
        if not self.recent_performance:
            return 1.0
        
        # Calculate recent drawdown
        peak = max(self.recent_performance)
        current = self.recent_performance[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        if drawdown > self.drawdown_threshold * 2:
            return 0.5  # Halve position size during severe drawdown
        elif drawdown > self.drawdown_threshold:
            return 0.75  # Reduce position size during moderate drawdown
        else:
            return 1.0
    
    def calculate_correlation_risk(self, 
                                 current_positions: Dict[str, float], 
                                 new_symbol: str,
                                 correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Assess portfolio correlation risk before adding new position."""
        if not current_positions or new_symbol not in correlation_matrix:
            return 1.0  # No correlation data, assume independent
        
        total_correlation = 0
        weighted_exposure = 0
        
        for symbol, position_size in current_positions.items():
            if symbol in correlation_matrix.get(new_symbol, {}):
                correlation = abs(correlation_matrix[new_symbol][symbol])
                total_correlation += correlation * position_size
                weighted_exposure += position_size
        
        if weighted_exposure == 0:
            return 1.0
        
        avg_correlation = total_correlation / weighted_exposure
        
        # Reduce position size based on correlation
        if avg_correlation > 0.7:
            return 0.6  # High correlation, reduce significantly
        elif avg_correlation > 0.5:
            return 0.8  # Moderate correlation, reduce moderately
        else:
            return 1.0  # Low correlation, no adjustment
    
    def assess_multi_timeframe_risk(self, 
                                  timeframe_signals: Dict[str, float],
                                  current_timeframe: str = "1m") -> Dict[str, str]:
        """Analyze risk across multiple timeframes for better context."""
        risk_assessment = {
            "overall_risk": "MODERATE",
            "timeframe_conflicts": [],
            "risk_factors": []
        }
        
        if not timeframe_signals:
            return risk_assessment
        
        # Check for timeframe conflicts
        signals = list(timeframe_signals.values())
        current_signal = timeframe_signals.get(current_timeframe, 0)
        
        # Detect conflicting signals
        bullish_tf = sum(1 for s in signals if s > 1)
        bearish_tf = sum(1 for s in signals if s < -1)
        
        if bullish_tf > 0 and bearish_tf > 0:
            risk_assessment["timeframe_conflicts"].append("Mixed signals across timeframes")
            risk_assessment["overall_risk"] = "HIGH"
        
        # Check higher timeframe alignment
        higher_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        current_idx = higher_timeframes.index(current_timeframe) if current_timeframe in higher_timeframes else 0
        
        for tf in higher_timeframes[current_idx + 1:]:
            if tf in timeframe_signals:
                higher_signal = timeframe_signals[tf]
                if (current_signal > 0 and higher_signal < -1) or (current_signal < 0 and higher_signal > 1):
                    risk_assessment["risk_factors"].append(f"Against {tf} trend")
                    risk_assessment["overall_risk"] = "HIGH"
        
        return risk_assessment
    
    def calculate_time_based_risk_adjustment(self, current_time: str, dte: int) -> float:
        """Adjust risk based on time of day and days to expiration."""
        try:
            time_obj = datetime.strptime(current_time, "%H:%M").time()
        except:
            return 1.0
        
        risk_multiplier = 1.0
        
        # Time of day adjustments
        if time(9, 30) <= time_obj <= time(10, 0):  # Market open
            risk_multiplier *= 0.7  # Reduce size during volatile open
        elif time(15, 30) <= time_obj <= time(16, 0):  # Power hour
            risk_multiplier *= 0.8  # Slightly reduce during power hour
        elif time_obj < time(9, 30) or time_obj > time(16, 0):  # Extended hours
            risk_multiplier *= 0.6  # Significantly reduce during extended hours
        
        # DTE adjustments for options
        if dte == 0:
            if time_obj >= time(15, 0):  # Final hour of 0DTE
                risk_multiplier *= 0.5  # Halve position size
            else:
                risk_multiplier *= 0.7  # Reduce throughout 0DTE
        elif dte <= 3:
            risk_multiplier *= 0.85  # Slight reduction for short DTE
        
        return risk_multiplier
    
    def update_performance_history(self, pnl: float):
        """Update recent performance for drawdown calculations."""
        self.recent_performance.append(pnl)
        
        # Keep only last 50 trades
        if len(self.recent_performance) > 50:
            self.recent_performance.pop(0)
    
    def get_comprehensive_risk_advice(self, 
                                    inputs: Dict,
                                    signal_direction: str,
                                    signal_confidence: float = 0.5) -> List[str]:
        """Generate comprehensive risk advice based on all factors."""
        advice = []
        
        current_price = inputs.get('current_price', 0)
        atr = inputs.get('atr_value', 0)
        vix = inputs.get('vix_level', 20)
        dte = inputs.get('dte', 1)
        current_time = inputs.get('current_time', '10:00')
        
        trend_direction = "long" if "BULLISH" in signal_direction else "short"
        
        # 1. Dynamic Stop Loss
        stop_loss = self.calculate_dynamic_stop_loss(
            current_price, atr, trend_direction, vix, "regular"
        )
        advice.append(f"🛡️ Dynamic Stop-Loss: ${stop_loss:.2f} (ATR-adjusted for VIX {vix:.1f})")
        
        # 2. Position Sizing
        if inputs.get('account_size'):
            position_data = self.calculate_enhanced_position_size(
                inputs['account_size'],
                inputs.get('risk_percent', 1.0),
                current_price,
                stop_loss,
                signal_confidence
            )
            
            if position_data:
                advice.append(f"💰 Recommended Size: {position_data['recommended_size']:.0f} shares")
                advice.append(f"📊 Confidence Adjusted: {position_data['confidence_adjusted']:.0f} shares")
        
        # 3. Time-based Risk
        time_adjustment = self.calculate_time_based_risk_adjustment(current_time, dte)
        if time_adjustment < 1.0:
            advice.append(f"⏰ Time Risk: Reduce size by {(1-time_adjustment)*100:.0f}% due to timing")
        
        # 4. Market Condition Warnings
        if vix > 30:
            advice.append("⚠️ High Volatility: Consider wider stops and smaller positions")
        elif vix < 15:
            advice.append("📈 Low Volatility: Watch for potential volatility expansion")
        
        # 5. DTE-specific advice
        if dte == 0:
            advice.append("🚨 0DTE Warning: Extreme theta decay - set tight profit targets")
        elif dte <= 3:
            advice.append("⌛ Short DTE: Time decay accelerating - monitor closely")
        
        return advice[:8]  # Limit to most important advice