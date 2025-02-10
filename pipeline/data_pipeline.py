import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
from models.scoring_rules import CS2ScoreCalculator, CS2Score

class CS2DataPipeline:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the CS2 data pipeline.
        
        Args:
            raw_data_path: Path to raw data directory
            processed_data_path: Path to processed data directory
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.score_calculator = CS2ScoreCalculator()
        
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw CS2 player statistics from CSV/JSON file.
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            DataFrame containing raw player statistics
        """
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by handling missing values and inconsistencies.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Ensure required columns exist
        required_columns = [
            'player_id', 'map_name', 'kills', 'headshots', 
            'awp_kills', 'first_bloods', 'team_kills', 'match_id',
            'scheduled_time', 'actual_start_time', 'technical_reset'
        ]
        missing_columns = [col for col in required_columns 
                         if col not in cleaned_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert datetime columns
        cleaned_df['scheduled_time'] = pd.to_datetime(cleaned_df['scheduled_time'])
        cleaned_df['actual_start_time'] = pd.to_datetime(cleaned_df['actual_start_time'])
        
        return cleaned_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy for feature engineering
        featured_df = df.copy()
        
        # Calculate per-map averages
        map_stats = featured_df.groupby(['player_id', 'map_name']).agg({
            'kills': 'mean',
            'headshots': 'mean',
            'awp_kills': 'mean',
            'first_bloods': 'mean'
        }).reset_index()
        
        # Calculate overall player averages
        player_stats = featured_df.groupby('player_id').agg({
            'kills': 'mean',
            'headshots': 'mean',
            'awp_kills': 'mean',
            'first_bloods': 'mean'
        }).reset_index()
        
        # Calculate map bias
        map_stats = map_stats.merge(player_stats, on='player_id', suffixes=('_map', '_overall'))
        map_stats['kills_bias'] = map_stats['kills_map'] - map_stats['kills_overall']
        map_stats['headshots_bias'] = map_stats['headshots_map'] - map_stats['headshots_overall']
        map_stats['awp_kills_bias'] = map_stats['awp_kills_map'] - map_stats['awp_kills_overall']
        map_stats['first_bloods_bias'] = map_stats['first_bloods_map'] - map_stats['first_bloods_overall']
        
        # Calculate headshot percentage
        featured_df['headshot_percentage'] = (featured_df['headshots'] / 
                                            featured_df['kills']).fillna(0) * 100
        
        # Calculate PrizePicks scores
        featured_df['prizepicks_score'] = featured_df.apply(
            lambda row: self._calculate_prizepicks_score(row), axis=1
        )
        
        return featured_df
    
    def _calculate_prizepicks_score(self, row: pd.Series) -> float:
        """
        Calculate PrizePicks score for a player's performance.
        
        Args:
            row: DataFrame row containing player statistics
            
        Returns:
            float: PrizePicks score or 0 if DNP
        """
        # Prepare match info
        match_info = {
            'player_maps_played': [row['map_name']],
            'required_maps': [row['map_name']],  # Assuming single map for this row
            'scheduled_time': row['scheduled_time'],
            'actual_start_time': row['actual_start_time']
        }
        
        # Prepare player stats
        player_stats = {
            'kills': row['kills'],
            'headshots': row['headshots'],
            'awp_kills': row['awp_kills'],
            'first_bloods': row['first_bloods'],
            'team_kills': row.get('team_kills', 0)
        }
        
        # Calculate score using the scoring rules
        score = self.score_calculator.calculate_score(player_stats, match_info)
        return score.total_score if score is not None else 0
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed DataFrame
            filename: Name of the output file
        """
        output_path = self.processed_data_path / filename
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    def process_data(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Args:
            input_file: Name of the input file in raw_data directory
            output_file: Name of the output file for processed data
            
        Returns:
            Processed DataFrame
        """
        # Load raw data
        raw_df = self.load_raw_data(self.raw_data_path / input_file)
        
        # Clean data
        cleaned_df = self.clean_data(raw_df)
        
        # Engineer features
        processed_df = self.engineer_features(cleaned_df)
        
        # Save processed data
        self.save_processed_data(processed_df, output_file)
        
        return processed_df

if __name__ == '__main__':
    # Example usage
    pipeline = CS2DataPipeline(
        raw_data_path='data/raw',
        processed_data_path='data/processed'
    )
    
    try:
        processed_data = pipeline.process_data(
            input_file='player_stats.csv',  # Replace with your input file
            output_file='processed_stats.csv'
        )
        print("Data processing completed successfully!")
    except Exception as e:
        print(f"Error processing data: {str(e)}") 