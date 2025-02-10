"""
Excel-based scraper interface for CS2 match data.

This module provides functionality to:
1. Watch for new Excel files in a directory
2. Load and validate Excel data
3. Generate configuration mappings from Excel templates
4. Stream data in real-time
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, List
import json
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import aiofiles
from datetime import datetime
import logging
from pydantic import BaseModel, Field
import openpyxl
from .scraper_interface import ScraperConfig

class ExcelMapping(BaseModel):
    """Model for Excel column mappings."""
    sheet_name: str = "Sheet1"
    header_row: int = 0
    column_mappings: Dict[str, str]
    data_start_row: int = 1

class ExcelScraper:
    """Excel-based scraper for CS2 match data."""
    
    def __init__(self, 
                 watch_directory: Union[str, Path],
                 processed_directory: Optional[Union[str, Path]] = None,
                 config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the Excel scraper.
        
        Args:
            watch_directory: Directory to watch for new Excel files
            processed_directory: Directory to move processed files to
            config_path: Path to save/load Excel mappings configuration
        """
        self.watch_directory = Path(watch_directory)
        self.processed_directory = Path(processed_directory) if processed_directory else None
        self.config_path = Path(config_path) if config_path else None
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        if self.processed_directory:
            self.processed_directory.mkdir(parents=True, exist_ok=True)
        
        self.observer = Observer()
        self.event_handler = ExcelFileHandler(self)
        
    async def start_watching(self):
        """Start watching for new Excel files."""
        self.observer.schedule(self.event_handler, str(self.watch_directory), recursive=False)
        self.observer.start()
        self.logger.info(f"Started watching directory: {self.watch_directory}")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            self.observer.join()
    
    def generate_config_from_excel(self, template_path: Union[str, Path]) -> ScraperConfig:
        """
        Generate scraper configuration from Excel template.
        
        Args:
            template_path: Path to Excel template file
            
        Returns:
            ScraperConfig object
        """
        template_df = pd.read_excel(template_path, sheet_name="Mapping")
        
        # Extract field mappings from template
        mappings = {}
        for _, row in template_df.iterrows():
            if pd.notna(row['Calculator Field']) and pd.notna(row['Excel Column']):
                mappings[row['Calculator Field']] = row['Excel Column']
        
        # Create ScraperConfig
        config = ScraperConfig(
            player_id_field=mappings.get('player_id', ''),
            map_name_field=mappings.get('map_name', ''),
            kills_field=mappings.get('kills', ''),
            headshots_field=mappings.get('headshots', ''),
            awp_kills_field=mappings.get('awp_kills', ''),
            first_bloods_field=mappings.get('first_bloods', ''),
            team_kills_field=mappings.get('team_kills', ''),
            match_id_field=mappings.get('match_id', ''),
            scheduled_time_field=mappings.get('scheduled_time', ''),
            actual_start_time_field=mappings.get('actual_start_time', ''),
            technical_reset_field=mappings.get('technical_reset', ''),
            round_number_field=mappings.get('round_number', None),
            player_team_field=mappings.get('player_team', None),
            opponent_team_field=mappings.get('opponent_team', None),
            match_format_field=mappings.get('match_format', None),
            tournament_name_field=mappings.get('tournament_name', None)
        )
        
        return config
    
    async def process_excel_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Process an Excel file and convert it to standardized format.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Load Excel file
            df = pd.read_excel(file_path)
            
            # Load configuration
            if self.config_path and self.config_path.exists():
                async with aiofiles.open(self.config_path, 'r') as f:
                    config_data = await f.read()
                    excel_mapping = ExcelMapping.parse_raw(config_data)
            else:
                # Generate configuration from template
                config = self.generate_config_from_excel(file_path)
                excel_mapping = ExcelMapping(
                    column_mappings={v: k for k, v in config.__dict__.items() if v}
                )
            
            # Apply column mappings
            df = df.rename(columns=excel_mapping.column_mappings)
            
            # Convert datetime columns
            datetime_columns = ['scheduled_time', 'actual_start_time']
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Move processed file if directory specified
            if self.processed_directory:
                processed_path = self.processed_directory / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(file_path).name}"
                Path(file_path).rename(processed_path)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            raise

class ExcelFileHandler(FileSystemEventHandler):
    """Handler for Excel file events."""
    
    def __init__(self, scraper: ExcelScraper):
        """
        Initialize the file handler.
        
        Args:
            scraper: ExcelScraper instance
        """
        self.scraper = scraper
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(('.xlsx', '.xls')):
            asyncio.create_task(self.process_new_file(event.src_path))
    
    async def process_new_file(self, file_path: str):
        """Process newly created Excel file."""
        try:
            df = await self.scraper.process_excel_file(file_path)
            self.scraper.logger.info(f"Successfully processed {file_path}")
            # Emit processed data (implement your streaming logic here)
            
        except Exception as e:
            self.scraper.logger.error(f"Error processing {file_path}: {str(e)}")

def create_excel_template(output_path: Union[str, Path]):
    """
    Create an Excel template for data mapping configuration.
    
    Args:
        output_path: Path to save the template
    """
    template_data = {
        'Calculator Field': [
            'player_id', 'map_name', 'kills', 'headshots', 'awp_kills',
            'first_bloods', 'team_kills', 'match_id', 'scheduled_time',
            'actual_start_time', 'technical_reset', 'round_number',
            'player_team', 'opponent_team', 'match_format', 'tournament_name'
        ],
        'Excel Column': [''] * 16,
        'Description': [
            'Unique player identifier',
            'Name of the map played',
            'Total kills in the match',
            'Headshot kills',
            'Kills with AWP',
            'First kills of rounds',
            'Team kills (not deducted)',
            'Unique match identifier',
            'Scheduled match time',
            'Actual match start time',
            'Technical reset occurred',
            'Current round number',
            "Player's team name",
            'Opponent team name',
            'Match format (bo1, bo3, etc.)',
            'Tournament/event name'
        ],
        'Required': [
            'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No'
        ]
    }
    
    df = pd.DataFrame(template_data)
    df.to_excel(output_path, sheet_name='Mapping', index=False)

if __name__ == '__main__':
    # Example usage
    try:
        # Create template
        create_excel_template('config/excel_template.xlsx')
        
        # Initialize scraper
        scraper = ExcelScraper(
            watch_directory='data/excel_input',
            processed_directory='data/excel_processed',
            config_path='config/excel_mapping.json'
        )
        
        # Start watching for files
        asyncio.run(scraper.start_watching())
        
    except Exception as e:
        logging.error(f"Error: {str(e)}") 