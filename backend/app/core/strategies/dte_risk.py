from typing import Tuple, List
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs

class DTEStrategy(ScalpingStrategy):
    """Enhanced Days to Expiration analysis for options trading risk and opportunities."""
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        score, factors = 0, []
        dte = inputs.dte
        
        if dte is None:
            return 0, ["DTE data unavailable"]
        
        # Get additional context
        option_delta = getattr(inputs, 'option_delta', None)
        option_gamma = getattr(inputs, 'option_gamma', None)
        option_theta = getattr(inputs, 'option_theta', None)
        option_vega = getattr(inputs, 'option_vega', None)
        iv = getattr(inputs, 'implied_volatility', None)
        
        # 1. 0DTE Analysis - High Risk/High Reward
        if dte == 0:
            base_score = -2.0  # Start with penalty for extreme risk
            factors.append("⚠️ 0DTE - Extreme time decay risk")
            
            # But 0DTE can be profitable with right conditions
            if option_delta and abs(option_delta) > 0.7:
                base_score += 1.5
                factors.append(f"💪 Deep ITM 0DTE (Δ={option_delta:.2f}) - Lower gamma risk")
            elif option_delta and abs(option_delta) > 0.4:
                base_score += 0.5
                factors.append(f"⚡ Moderate ITM 0DTE (Δ={option_delta:.2f}) - High gamma potential")
            else:
                base_score -= 1.0
                factors.append("🎰 Far OTM 0DTE - Lottery ticket territory")
            
            # Time of day matters for 0DTE
            if hasattr(inputs, 'current_time'):
                hour = int(inputs.current_time.split(':')[0])
                if hour < 10:  # Morning - more time for moves
                    base_score += 0.5
                    factors.append("🌅 Early 0DTE - More time for price action")
                elif hour >= 15:  # Power hour - high volatility
                    base_score += 0.8
                    factors.append("⚡ 0DTE Power Hour - High volatility window")
                elif 12 <= hour <= 14:  # Lunch lull
                    base_score -= 0.5
                    factors.append("😴 0DTE Lunch Period - Lower volatility risk")
            
            score += base_score
            
        # 2. 1DTE Analysis - Sweet Spot for Many Scalpers
        elif dte == 1:
            score += 1.0
            factors.append("🎯 1DTE - Optimal theta/gamma balance for scalping")
            
            if option_gamma and option_gamma > 0.05:
                score += 0.8
                factors.append(f"🚀 High gamma 1DTE (Γ={option_gamma:.3f}) - Explosive potential")
            
            if option_theta and option_theta < -0.5:
                score -= 0.3
                factors.append(f"⏰ High theta decay (Θ={option_theta:.2f}) - Time pressure")
                
        # 3. 2-7 DTE Analysis - Moderate Risk Zone
        elif 2 <= dte <= 7:
            base_score = 0.5 - (dte - 2) * 0.1  # Decreasing benefit as DTE increases
            score += base_score
            factors.append(f"✅ {dte}DTE - Good scalping window with manageable decay")
            
            if dte <= 3 and option_gamma and option_gamma > 0.02:
                score += 0.5
                factors.append(f"⚡ Short-term high gamma (Γ={option_gamma:.3f})")
                
        # 4. 8-21 DTE Analysis - Lower Theta, Higher Vega Risk
        elif 8 <= dte <= 21:
            score += 0.2
            factors.append(f"📊 {dte}DTE - Lower theta decay, watch for IV changes")
            
            if option_vega and iv:
                if option_vega > 0.1:
                    score -= 0.3
                    factors.append(f"⚠️ High vega exposure (V={option_vega:.2f}) - IV risk")
                    
                # Check if IV is elevated (simple heuristic)
                if iv > 0.3:  # 30% IV threshold
                    score -= 0.2
                    factors.append(f"📈 Elevated IV ({iv:.1%}) - Volatility crush risk")
                    
        # 5. 22+ DTE Analysis - Too Long for Scalping
        elif dte >= 22:
            score -= 0.5
            factors.append(f"⏳ {dte}DTE - Too long for scalping, lower gamma sensitivity")
            
            if dte >= 45:
                score -= 1.0
                factors.append("🐌 Long-dated option - Minimal scalping advantage")
                
        # 6. Weekly vs Monthly Expiration Analysis
        if hasattr(inputs, 'is_weekly_expiration'):
            if inputs.is_weekly_expiration and dte <= 7:
                score += 0.3
                factors.append("📅 Weekly expiration - Higher retail flow")
                
        # 7. Theta Efficiency Analysis
        if option_theta and option_delta:
            theta_efficiency = abs(option_theta) / abs(option_delta) if option_delta != 0 else 0
            if dte <= 3 and theta_efficiency > 0.1:
                score -= 0.4
                factors.append(f"⚠️ High theta efficiency ({theta_efficiency:.2f}) - Rapid decay")
            elif dte >= 7 and theta_efficiency < 0.05:
                score += 0.2
                factors.append(f"✅ Low theta efficiency ({theta_efficiency:.2f}) - Time cushion")
                
        # 8. Gamma Efficiency for Scalping
        if option_gamma and option_delta and dte <= 7:
            # Higher gamma relative to delta is better for scalping
            gamma_ratio = option_gamma / abs(option_delta) if option_delta != 0 else 0
            if gamma_ratio > 0.1:
                score += 0.6
                factors.append(f"🎯 High gamma efficiency ({gamma_ratio:.2f}) - Great for scalping")
            elif gamma_ratio < 0.02:
                score -= 0.3
                factors.append(f"📉 Low gamma efficiency ({gamma_ratio:.2f}) - Limited scalping edge")
                
        # 9. Weekend Risk for Friday Options
        if dte == 0 and hasattr(inputs, 'current_time'):
            # Friday 0DTE has weekend risk
            import datetime
            try:
                # This would need actual date logic in production
                # For now, just flag the risk
                factors.append("⚠️ Check for weekend risk on Friday 0DTE")
            except:
                pass
                
        return score, factors