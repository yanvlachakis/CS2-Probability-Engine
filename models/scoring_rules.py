"""
CS2 PrizePicks Scoring Rules Module.

This module implements the official PrizePicks scoring rules for CS2 matches,
including validation and special case handling.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pytz

@dataclass
class CS2Action:
    """Represents a scoreable action in CS2."""
    AWP_KILL = "awp_kill"
    FIRST_BLOOD = "first_blood"
    HEADSHOT = "headshot"
    KILL = "kill"

@dataclass
class CS2Score:
    """Represents a player's score breakdown."""
    awp_kills: int = 0
    first_bloods: int = 0
    headshots: int = 0
    kills: int = 0
    
    @property
    def total_score(self) -> int:
        """Calculate total score based on PrizePicks scoring rules."""
        return (self.awp_kills + self.first_bloods + 
                self.headshots + self.kills)

class CS2ScoringValidator:
    """Validates CS2 match conditions and scoring rules."""
    
    @staticmethod
    def validate_match_completion(player_maps_played: List[str], 
                                required_maps: List[str]) -> bool:
        """
        Validate that player completed all required maps.
        
        Args:
            player_maps_played: List of maps the player participated in
            required_maps: List of maps required for the projection
            
        Returns:
            bool: True if player completed all required maps
        """
        return all(map_name in player_maps_played for map_name in required_maps)
    
    @staticmethod
    def validate_match_time(scheduled_time: datetime,
                          actual_start_time: Optional[datetime] = None) -> bool:
        """
        Validate match timing according to PrizePicks rules.
        
        Args:
            scheduled_time: Originally scheduled match time
            actual_start_time: Actual match start time (if available)
            
        Returns:
            bool: True if match timing is valid
        """
        if actual_start_time is None:
            return True
            
        # Convert times to ET for comparison
        et_tz = pytz.timezone('US/Eastern')
        scheduled_et = scheduled_time.astimezone(et_tz)
        actual_et = actual_start_time.astimezone(et_tz)
        
        # Check if match started within 12 hours of scheduled time
        time_diff = actual_et - scheduled_et
        return time_diff <= timedelta(hours=12)
    
    @staticmethod
    def validate_technical_reset(round_stats: Dict[str, Dict],
                               reset_occurred: bool) -> Dict[str, Dict]:
        """
        Handle technical reset scoring according to rules.
        
        Args:
            round_stats: Dictionary of player stats before reset
            reset_occurred: Whether a technical reset occurred
            
        Returns:
            Dictionary of valid stats to count
        """
        if not reset_occurred:
            return round_stats
        return {}  # If reset occurred, only count stats after reset

class CS2ScoreCalculator:
    """Calculates CS2 scores based on PrizePicks rules."""
    
    def __init__(self):
        """Initialize the score calculator."""
        self.validator = CS2ScoringValidator()
    
    def calculate_score(self, 
                       player_stats: Dict[str, Union[int, float]],
                       match_info: Dict[str, any]) -> Optional[CS2Score]:
        """
        Calculate player's score following PrizePicks rules.
        
        Args:
            player_stats: Dictionary of player statistics
            match_info: Dictionary containing match metadata
            
        Returns:
            CS2Score object or None if player is DNP
        """
        # Validate match completion
        if not self.validator.validate_match_completion(
            match_info.get('player_maps_played', []),
            match_info.get('required_maps', [])
        ):
            return None  # DNP
        
        # Validate match timing
        if not self.validator.validate_match_time(
            match_info.get('scheduled_time'),
            match_info.get('actual_start_time')
        ):
            return None  # DNP
        
        # Calculate score components
        score = CS2Score(
            awp_kills=player_stats.get('awp_kills', 0),
            first_bloods=player_stats.get('first_bloods', 0),
            headshots=player_stats.get('headshots', 0),
            kills=player_stats.get('kills', 0)
        )
        
        # Handle team kills (don't deduct points)
        team_kills = player_stats.get('team_kills', 0)
        if team_kills > 0:
            score.kills += team_kills
        
        return score

    def validate_and_score_round(self,
                                round_stats: Dict[str, Dict],
                                technical_info: Dict[str, any]) -> Dict[str, CS2Score]:
        """
        Validate and score a round considering technical difficulties.
        
        Args:
            round_stats: Dictionary of player stats for the round
            technical_info: Information about technical difficulties
            
        Returns:
            Dictionary mapping player IDs to their scores
        """
        # Handle technical resets
        valid_stats = self.validator.validate_technical_reset(
            round_stats,
            technical_info.get('reset_occurred', False)
        )
        
        # Calculate scores for each player
        scores = {}
        for player_id, stats in valid_stats.items():
            match_info = {
                'player_maps_played': technical_info.get('maps_played', []),
                'required_maps': technical_info.get('required_maps', []),
                'scheduled_time': technical_info.get('scheduled_time'),
                'actual_start_time': technical_info.get('actual_start_time')
            }
            
            score = self.calculate_score(stats, match_info)
            if score is not None:
                scores[player_id] = score
                
        return scores 