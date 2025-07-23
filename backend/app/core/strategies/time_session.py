from typing import Tuple, List, Dict, Optional
from datetime import datetime, time
import pytz
from .base import ScalpingStrategy
from app.core.models.signals import ScalpingInputs
from app.core.config import TradingConfig, ScalpingConstants

class TradingSessionAnalyzer:
    """Advanced trading session analysis with timezone awareness and session characteristics."""
    
    def __init__(self):
        # Define major trading sessions (all times in UTC)
        self.sessions = {
            'sydney': {'start': time(22, 0), 'end': time(7, 0), 'characteristics': 'low_volume'},
            'tokyo': {'start': time(0, 0), 'end': time(9, 0), 'characteristics': 'moderate_volume'},
            'london': {'start': time(8, 0), 'end': time(17, 0), 'characteristics': 'high_volume'},
            'new_york': {'start': time(13, 0), 'end': time(22, 0), 'characteristics': 'high_volume'}
        }
        
        # Session overlap periods (high activity)
        self.overlaps = {
            'london_new_york': {'start': time(13, 0), 'end': time(17, 0), 'activity': 'very_high'},
            'sydney_tokyo': {'start': time(0, 0), 'end': time(7, 0), 'activity': 'moderate'},
            'tokyo_london': {'start': time(8, 0), 'end': time(9, 0), 'activity': 'high'}
        }
        
        # Time-based patterns
        self.intraday_patterns = {
            'market_open': [time(9, 30), time(13, 0)],  # US & London opens
            'lunch_lull': [time(11, 0), time(13, 0)],   # Low activity period
            'power_hour': [time(15, 0), time(16, 0)],   # High volatility
            'close_run': [time(20, 0), time(22, 0)]     # End of day moves
        }
    
    def analyze_session_context(self, current_time_str: str, timezone: str = 'UTC') -> Dict:
        """Comprehensive session analysis including overlaps and patterns."""
        
        try:
            # Parse time - handle different formats
            if ':' in current_time_str and len(current_time_str.split(':')) >= 2:
                time_parts = current_time_str.split(':')
                current_hour = int(time_parts[0])
                current_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            else:
                current_hour = int(current_time_str)
                current_minute = 0
            
            current_time_obj = time(current_hour, current_minute)
            
        except (ValueError, IndexError):
            # Fallback for invalid time format
            return self._get_default_session_analysis()
        
        # Determine active sessions
        active_sessions = self._get_active_sessions(current_time_obj)
        
        # Check for session overlaps
        active_overlaps = self._get_active_overlaps(current_time_obj)
        
        # Identify intraday patterns
        current_patterns = self._identify_patterns(current_time_obj)
        
        # Calculate session characteristics
        session_score = self._calculate_session_score(active_sessions, active_overlaps, current_patterns)
        
        # Determine volatility expectation
        volatility_expectation = self._get_volatility_expectation(active_sessions, active_overlaps, current_patterns)
        
        # Get liquidity assessment
        liquidity_assessment = self._assess_liquidity(active_sessions, active_overlaps)
        
        return {
            'active_sessions': active_sessions,
            'active_overlaps': active_overlaps,
            'current_patterns': current_patterns,
            'session_score': session_score,
            'volatility_expectation': volatility_expectation,
            'liquidity_assessment': liquidity_assessment,
            'recommendation': self._get_session_recommendation(session_score, volatility_expectation, liquidity_assessment),
            'current_time': current_time_obj,
            'risk_adjustment': self._get_risk_adjustment(active_sessions, active_overlaps, current_patterns)
        }
    
    def _get_active_sessions(self, current_time: time) -> List[Dict]:
        """Identify currently active trading sessions."""
        active = []
        current_minutes = current_time.hour * 60 + current_time.minute
        
        for session_name, session_info in self.sessions.items():
            start_minutes = session_info['start'].hour * 60 + session_info['start'].minute
            end_minutes = session_info['end'].hour * 60 + session_info['end'].minute
            
            # Handle sessions that cross midnight
            if start_minutes > end_minutes:  # Crosses midnight
                if current_minutes >= start_minutes or current_minutes <= end_minutes:
                    active.append({
                        'name': session_name,
                        'characteristics': session_info['characteristics'],
                        'time_in_session': self._calculate_time_in_session(current_time, session_info)
                    })
            else:  # Normal session
                if start_minutes <= current_minutes <= end_minutes:
                    active.append({
                        'name': session_name,
                        'characteristics': session_info['characteristics'],
                        'time_in_session': self._calculate_time_in_session(current_time, session_info)
                    })
        
        return active
    
    def _get_active_overlaps(self, current_time: time) -> List[Dict]:
        """Identify active session overlaps."""
        active = []
        current_minutes = current_time.hour * 60 + current_time.minute
        
        for overlap_name, overlap_info in self.overlaps.items():
            start_minutes = overlap_info['start'].hour * 60 + overlap_info['start'].minute
            end_minutes = overlap_info['end'].hour * 60 + overlap_info['end'].minute
            
            if start_minutes <= current_minutes <= end_minutes:
                active.append({
                    'name': overlap_name,
                    'activity': overlap_info['activity']
                })
        
        return active
    
    def _identify_patterns(self, current_time: time) -> List[Dict]:
        """Identify current intraday patterns."""
        patterns = []
        current_minutes = current_time.hour * 60 + current_time.minute
        
        for pattern_name, time_ranges in self.intraday_patterns.items():
            for time_range in time_ranges:
                if isinstance(time_range, list) and len(time_range) == 2:
                    start_minutes = time_range[0].hour * 60 + time_range[0].minute
                    end_minutes = time_range[1].hour * 60 + time_range[1].minute
                    
                    if start_minutes <= current_minutes <= end_minutes:
                        patterns.append({
                            'name': pattern_name,
                            'description': self._get_pattern_description(pattern_name)
                        })
        
        return patterns
    
    def _calculate_session_score(self, active_sessions: List[Dict], 
                               active_overlaps: List[Dict], 
                               current_patterns: List[Dict]) -> float:
        """Calculate overall session favorability score."""
        score = 0.0
        
        # Base score from active sessions
        for session in active_sessions:
            if session['characteristics'] == 'high_volume':
                score += 1.5
            elif session['characteristics'] == 'moderate_volume':
                score += 1.0
            else:  # low_volume
                score += 0.3
        
        # Bonus for overlaps
        for overlap in active_overlaps:
            if overlap['activity'] == 'very_high':
                score += 2.0
            elif overlap['activity'] == 'high':
                score += 1.5
            else:  # moderate
                score += 1.0
        
        # Pattern bonuses/penalties
        for pattern in current_patterns:
            pattern_name = pattern['name']
            if pattern_name == 'power_hour':
                score += 1.5
            elif pattern_name == 'market_open':
                score += 1.2
            elif pattern_name == 'close_run':
                score += 1.0
            elif pattern_name == 'lunch_lull':
                score -= 0.8
        
        return max(min(score, 5.0), 0.0)  # Cap between 0 and 5
    
    def _get_volatility_expectation(self, active_sessions: List[Dict], 
                                   active_overlaps: List[Dict], 
                                   current_patterns: List[Dict]) -> Dict:
        """Determine expected volatility based on session analysis."""
        
        volatility_score = 1.0  # Base volatility
        factors = []
        
        # Session contributions
        high_volume_sessions = sum(1 for s in active_sessions if s['characteristics'] == 'high_volume')
        if high_volume_sessions >= 2:
            volatility_score += 0.5
            factors.append("Multiple high-volume sessions active")
        elif high_volume_sessions == 1:
            volatility_score += 0.2
            factors.append("High-volume session active")
        
        # Overlap contributions
        for overlap in active_overlaps:
            if overlap['activity'] == 'very_high':
                volatility_score += 0.6
                factors.append(f"Very high activity overlap: {overlap['name']}")
            elif overlap['activity'] == 'high':
                volatility_score += 0.3
                factors.append(f"High activity overlap: {overlap['name']}")
        
        # Pattern contributions
        for pattern in current_patterns:
            pattern_name = pattern['name']
            if pattern_name == 'power_hour':
                volatility_score += 0.4
                factors.append("Power hour - increased volatility expected")
            elif pattern_name == 'market_open':
                volatility_score += 0.3
                factors.append("Market open - higher volatility")
            elif pattern_name == 'lunch_lull':
                volatility_score -= 0.3
                factors.append("Lunch lull - reduced volatility")
        
        # Determine expectation level
        if volatility_score >= 2.0:
            level = 'very_high'
        elif volatility_score >= 1.5:
            level = 'high'
        elif volatility_score >= 1.2:
            level = 'elevated'
        elif volatility_score >= 0.8:
            level = 'normal'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': volatility_score,
            'factors': factors
        }
    
    def _assess_liquidity(self, active_sessions: List[Dict], active_overlaps: List[Dict]) -> Dict:
        """Assess current market liquidity."""
        
        liquidity_score = 0.5  # Base liquidity
        
        # Session liquidity
        for session in active_sessions:
            if session['characteristics'] == 'high_volume':
                liquidity_score += 0.4
            elif session['characteristics'] == 'moderate_volume':
                liquidity_score += 0.2
            else:
                liquidity_score += 0.05
        
        # Overlap liquidity bonus
        for overlap in active_overlaps:
            if overlap['activity'] == 'very_high':
                liquidity_score += 0.3
            elif overlap['activity'] == 'high':
                liquidity_score += 0.2
        
        liquidity_score = max(min(liquidity_score, 2.0), 0.1)
        
        if liquidity_score >= 1.5:
            assessment = 'excellent'
        elif liquidity_score >= 1.0:
            assessment = 'good'
        elif liquidity_score >= 0.7:
            assessment = 'fair'
        else:
            assessment = 'poor'
        
        return {
            'assessment': assessment,
            'score': liquidity_score
        }
    
    def _get_session_recommendation(self, session_score: float, 
                                   volatility_expectation: Dict, 
                                   liquidity_assessment: Dict) -> Dict:
        """Generate trading recommendation based on session analysis."""
        
        # Overall favorability
        if session_score >= 3.5 and liquidity_assessment['assessment'] in ['excellent', 'good']:
            recommendation = 'highly_favorable'
            confidence = 0.9
        elif session_score >= 2.5 and liquidity_assessment['assessment'] != 'poor':
            recommendation = 'favorable'
            confidence = 0.7
        elif session_score >= 1.5:
            recommendation = 'neutral'
            confidence = 0.5
        else:
            recommendation = 'unfavorable'
            confidence = 0.3
        
        # Adjust for volatility
        vol_level = volatility_expectation['level']
        if vol_level in ['very_high', 'high'] and recommendation in ['highly_favorable', 'favorable']:
            confidence += 0.1
        elif vol_level == 'low' and recommendation in ['unfavorable']:
            confidence += 0.1
        
        return {
            'recommendation': recommendation,
            'confidence': max(min(confidence, 1.0), 0.0),
            'reasoning': self._get_recommendation_reasoning(session_score, volatility_expectation, liquidity_assessment)
        }
    
    def _get_risk_adjustment(self, active_sessions: List[Dict], 
                           active_overlaps: List[Dict], 
                           current_patterns: List[Dict]) -> Dict:
        """Get risk adjustment factors based on session."""
        
        risk_multiplier = 1.0
        adjustments = []
        
        # Low liquidity periods increase risk
        total_sessions = len(active_sessions)
        if total_sessions == 0:
            risk_multiplier += 0.5
            adjustments.append("No active major sessions - increased spread risk")
        elif total_sessions == 1 and active_sessions[0]['characteristics'] == 'low_volume':
            risk_multiplier += 0.3
            adjustments.append("Only low-volume session active - reduced liquidity")
        
        # Lunch lull adjustments
        in_lunch_lull = any(p['name'] == 'lunch_lull' for p in current_patterns)
        if in_lunch_lull:
            risk_multiplier += 0.2
            adjustments.append("Lunch lull period - wider spreads possible")
        
        # High volatility periods
        very_high_activity = any(o['activity'] == 'very_high' for o in active_overlaps)
        if very_high_activity:
            risk_multiplier += 0.1
            adjustments.append("Very high activity period - increased volatility risk")
        
        return {
            'multiplier': max(risk_multiplier, 0.5),  # Don't go below 0.5
            'adjustments': adjustments
        }
    
    def _calculate_time_in_session(self, current_time: time, session_info: Dict) -> float:
        """Calculate how far into the session we are (0.0 to 1.0)."""
        current_minutes = current_time.hour * 60 + current_time.minute
        start_minutes = session_info['start'].hour * 60 + session_info['start'].minute
        end_minutes = session_info['end'].hour * 60 + session_info['end'].minute
        
        # Handle sessions that cross midnight
        if start_minutes > end_minutes:
            if current_minutes >= start_minutes:
                session_length = (24 * 60 - start_minutes) + end_minutes
                time_elapsed = current_minutes - start_minutes
            else:
                session_length = (24 * 60 - start_minutes) + end_minutes
                time_elapsed = (24 * 60 - start_minutes) + current_minutes
        else:
            session_length = end_minutes - start_minutes
            time_elapsed = current_minutes - start_minutes
        
        return max(min(time_elapsed / max(session_length, 1), 1.0), 0.0)
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """Get description for intraday pattern."""
        descriptions = {
            'market_open': 'High volatility and volume at market open',
            'lunch_lull': 'Reduced activity during lunch hours',
            'power_hour': 'Increased volatility in final trading hour',
            'close_run': 'End-of-day positioning and volume surge'
        }
        return descriptions.get(pattern_name, 'Intraday pattern identified')
    
    def _get_recommendation_reasoning(self, session_score: float, 
                                    volatility_expectation: Dict, 
                                    liquidity_assessment: Dict) -> str:
        """Generate reasoning for the recommendation."""
        vol_level = volatility_expectation['level']
        liquidity = liquidity_assessment['assessment']
        
        if session_score >= 3.5:
            return f"Multiple active sessions with {vol_level} volatility and {liquidity} liquidity"
        elif session_score >= 2.5:
            return f"Active trading sessions with {vol_level} volatility"
        elif session_score >= 1.5:
            return f"Moderate session activity with {liquidity} liquidity"
        else:
            return f"Low session activity with {vol_level} volatility expected"
    
    def _get_default_session_analysis(self) -> Dict:
        """Return default analysis when time parsing fails."""
        return {
            'active_sessions': [],
            'active_overlaps': [],
            'current_patterns': [],
            'session_score': 1.0,
            'volatility_expectation': {'level': 'normal', 'score': 1.0, 'factors': []},
            'liquidity_assessment': {'assessment': 'fair', 'score': 0.7},
            'recommendation': {'recommendation': 'neutral', 'confidence': 0.5, 'reasoning': 'Unable to parse time'},
            'current_time': time(12, 0),
            'risk_adjustment': {'multiplier': 1.0, 'adjustments': ['Time parsing error']}
        }


class TimeSessionStrategy(ScalpingStrategy):
    """Enhanced time-based strategy with comprehensive session analysis."""
    
    def __init__(self, config: 'TradingConfig', constants: 'ScalpingConstants'):
        super().__init__(config, constants)
        self.analyzer = TradingSessionAnalyzer()
        
        # Scoring parameters
        self.max_session_score = 2.5
        self.volatility_bonus = 0.5
        self.liquidity_penalty = -1.0
    
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        """Calculate time session score with comprehensive analysis."""
        
        # Get comprehensive session analysis
        session_analysis = self.analyzer.analyze_session_context(inputs.current_time)
        
        # Base score from session favorability
        base_score = self._calculate_base_score(session_analysis)
        
        # Volatility adjustment
        volatility_adjustment = self._get_volatility_adjustment(session_analysis)
        
        # Liquidity adjustment
        liquidity_adjustment = self._get_liquidity_adjustment(session_analysis)
        
        # Combine scores
        total_score = base_score + volatility_adjustment + liquidity_adjustment
        
        # Generate factors
        factors = self._generate_factors(session_analysis)
        
        return max(min(total_score, 3.0), -2.0), factors
    
    def _calculate_base_score(self, analysis: Dict) -> float:
        """Calculate base score from session analysis."""
        session_score = analysis['session_score']
        recommendation = analysis['recommendation']
        
        # Convert session score to strategy score
        base_score = (session_score / 5.0) * self.max_session_score
        
        # Apply recommendation confidence
        confidence = recommendation['confidence']
        base_score *= confidence
        
        return base_score
    
    def _get_volatility_adjustment(self, analysis: Dict) -> float:
        """Get score adjustment based on volatility expectation."""
        volatility = analysis['volatility_expectation']
        vol_level = volatility['level']
        
        if vol_level == 'very_high':
            return self.volatility_bonus * 1.5
        elif vol_level == 'high':
            return self.volatility_bonus
        elif vol_level == 'elevated':
            return self.volatility_bonus * 0.5
        elif vol_level == 'low':
            return -0.3
        else:  # normal
            return 0.0
    
    def _get_liquidity_adjustment(self, analysis: Dict) -> float:
        """Get score adjustment based on liquidity assessment."""
        liquidity = analysis['liquidity_assessment']
        assessment = liquidity['assessment']
        
        if assessment == 'poor':
            return self.liquidity_penalty
        elif assessment == 'fair':
            return self.liquidity_penalty * 0.3
        elif assessment == 'excellent':
            return 0.3
        else:  # good
            return 0.0
    
    def _generate_factors(self, analysis: Dict) -> List[str]:
        """Generate human-readable factors from session analysis."""
        factors = []
        
        # Active sessions
        active_sessions = analysis['active_sessions']
        if active_sessions:
            session_names = [s['name'].title() for s in active_sessions]
            factors.append(f"Active sessions: {', '.join(session_names)}")
        
        # Session overlaps
        active_overlaps = analysis['active_overlaps']
        if active_overlaps:
            overlap_info = []
            for overlap in active_overlaps:
                overlap_name = overlap['name'].replace('_', ' ').title()
                activity = overlap['activity'].replace('_', ' ')
                overlap_info.append(f"{overlap_name} ({activity})")
            factors.append(f"Session overlaps: {'; '.join(overlap_info)}")
        
        # Current patterns
        current_patterns = analysis['current_patterns']
        if current_patterns:
            pattern_names = [p['name'].replace('_', ' ').title() for p in current_patterns]
            factors.append(f"Intraday patterns: {', '.join(pattern_names)}")
        
        # Volatility expectation
        volatility = analysis['volatility_expectation']
        vol_level = volatility['level'].replace('_', ' ')
        factors.append(f"Expected volatility: {vol_level}")
        
        # Add specific volatility factors
        vol_factors = volatility.get('factors', [])
        factors.extend(vol_factors[:2])  # Limit to top 2 factors
        
        # Liquidity assessment
        liquidity = analysis['liquidity_assessment']
        factors.append(f"Liquidity: {liquidity['assessment']}")
        
        # Overall recommendation
        recommendation = analysis['recommendation']
        rec_text = recommendation['recommendation'].replace('_', ' ')
        confidence = recommendation['confidence']
        factors.append(f"Session recommendation: {rec_text} (confidence: {confidence:.1f})")
        
        # Risk adjustments
        risk_adjustment = analysis['risk_adjustment']
        if risk_adjustment['multiplier'] > 1.2:
            factors.append("Elevated session risk - consider reduced position size")
        elif risk_adjustment['adjustments']:
            factors.extend(risk_adjustment['adjustments'][:1])  # Add top risk factor
        
        return factors