"""
CS2 Scraper Interface Module.

This module provides a standardized interface for mapping data from various
CS2 match data scrapers to the PrizePicks probability calculator format.
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class ScraperConfig:
    """Configuration for mapping scraper data fields to calculator fields."""
    # Required field mappings
    player_id_field: str
    map_name_field: str
    kills_field: str
    headshots_field: str
    awp_kills_field: str
    first_bloods_field: str
    team_kills_field: str
    match_id_field: str
    scheduled_time_field: str
    actual_start_time_field: str
    technical_reset_field: str
    
    # Optional field mappings with defaults
    round_number_field: Optional[str] = None
    player_team_field: Optional[str] = None
    opponent_team_field: Optional[str] = None
    match_format_field: Optional[str] = None
    tournament_name_field: Optional[str] = None

class ScraperInterface:
    """Interface for standardizing scraper data for the calculator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the scraper interface.
        
        Args:
            config_path: Path to JSON configuration file for field mappings
        """
        self.config = self._load_config(config_path) if config_path else None
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> ScraperConfig:
        """
        Load field mapping configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ScraperConfig object
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return ScraperConfig(**config_data)
    
    def validate_scraper_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate that all required fields are present in the scraped data.
        
        Args:
            df: DataFrame containing scraped data
            
        Returns:
            List of missing required fields
        """
        required_fields = [
            self.config.player_id_field,
            self.config.map_name_field,
            self.config.kills_field,
            self.config.headshots_field,
            self.config.awp_kills_field,
            self.config.first_bloods_field,
            self.config.team_kills_field,
            self.config.match_id_field,
            self.config.scheduled_time_field,
            self.config.actual_start_time_field,
            self.config.technical_reset_field
        ]
        
        return [field for field in required_fields if field not in df.columns]
    
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert scraper-specific field names to calculator standard format.
        
        Args:
            df: DataFrame with scraper-specific field names
            
        Returns:
            DataFrame with standardized field names
        """
        if self.config is None:
            raise ValueError("Configuration must be loaded before standardizing data")
        
        # Validate input data
        missing_fields = self.validate_scraper_data(df)
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Create field mapping dictionary
        field_mapping = {
            self.config.player_id_field: 'player_id',
            self.config.map_name_field: 'map_name',
            self.config.kills_field: 'kills',
            self.config.headshots_field: 'headshots',
            self.config.awp_kills_field: 'awp_kills',
            self.config.first_bloods_field: 'first_bloods',
            self.config.team_kills_field: 'team_kills',
            self.config.match_id_field: 'match_id',
            self.config.scheduled_time_field: 'scheduled_time',
            self.config.actual_start_time_field: 'actual_start_time',
            self.config.technical_reset_field: 'technical_reset'
        }
        
        # Add optional fields if present
        optional_mappings = {
            self.config.round_number_field: 'round_number',
            self.config.player_team_field: 'player_team',
            self.config.opponent_team_field: 'opponent_team',
            self.config.match_format_field: 'match_format',
            self.config.tournament_name_field: 'tournament_name'
        }
        
        field_mapping.update({
            k: v for k, v in optional_mappings.items()
            if k is not None and k in df.columns
        })
        
        # Rename columns
        standardized_df = df.rename(columns=field_mapping)
        
        # Ensure datetime fields are properly formatted
        for field in ['scheduled_time', 'actual_start_time']:
            if field in standardized_df.columns:
                standardized_df[field] = pd.to_datetime(standardized_df[field])
        
        return standardized_df
    
    def save_standardized_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save standardized data to CSV file.
        
        Args:
            df: Standardized DataFrame
            output_path: Path to save the output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Standardized data saved to: {output_path}")

def create_example_config() -> Dict:
    """
    Create an example configuration for common CS2 data sources.
    
    Returns:
        Dictionary containing example field mappings
    """
    return {
        # HLTV-style field mappings
        "hltv": {
            "player_id_field": "player_id",
            "map_name_field": "map",
            "kills_field": "kills",
            "headshots_field": "hs",
            "awp_kills_field": "awp_kills",
            "first_bloods_field": "entry_kills",
            "team_kills_field": "team_kills",
            "match_id_field": "match_id",
            "scheduled_time_field": "date_scheduled",
            "actual_start_time_field": "date_started",
            "technical_reset_field": "tech_reset",
            "round_number_field": "round",
            "player_team_field": "team",
            "opponent_team_field": "opponent",
            "match_format_field": "format",
            "tournament_name_field": "event"
        },
        # FACEIT-style field mappings
        "faceit": {
            "player_id_field": "player_id",
            "map_name_field": "map_name",
            "kills_field": "kills",
            "headshots_field": "headshots",
            "awp_kills_field": "awp_frags",
            "first_bloods_field": "opening_kills",
            "team_kills_field": "team_damage_kills",
            "match_id_field": "match_id",
            "scheduled_time_field": "match_scheduled",
            "actual_start_time_field": "match_started",
            "technical_reset_field": "technical_pause_reset",
            "round_number_field": "round_num",
            "player_team_field": "faction",
            "opponent_team_field": "opponent_faction",
            "match_format_field": "match_type",
            "tournament_name_field": "competition"
        }
    }

if __name__ == '__main__':
    # Example usage
    try:
        # Create example configuration file
        example_configs = create_example_config()
        config_path = Path('config/scraper_mappings.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(example_configs, f, indent=4)
        
        print(f"Example configuration saved to: {config_path}")
        
        # Example of using the interface
        interface = ScraperInterface(str(config_path))
        
        # Example data (this would come from your scraper)
        example_data = pd.DataFrame({
            'player_id': [1, 2],
            'map': ['dust2', 'inferno'],
            'kills': [20, 15],
            'hs': [10, 8],
            'awp_kills': [5, 3],
            'entry_kills': [2, 1],
            'team_kills': [0, 0],
            'match_id': ['m1', 'm1'],
            'date_scheduled': ['2024-01-01', '2024-01-01'],
            'date_started': ['2024-01-01', '2024-01-01'],
            'tech_reset': [False, False]
        })
        
        # Standardize the data
        standardized_data = interface.standardize_data(example_data)
        print("\nStandardized Data:")
        print(standardized_data.head())
        
    except Exception as e:
        print(f"Error: {str(e)}") 