"""Tests for Excel scraper functionality."""

import pytest
import pandas as pd
import asyncio
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json

from pipeline.excel_scraper import ExcelScraper, create_excel_template

@pytest.fixture
def temp_directories():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as processed_dir:
        yield watch_dir, processed_dir

@pytest.fixture
def example_data():
    """Create example CS2 match data."""
    return pd.DataFrame({
        'Player ID': [1, 2],
        'Map': ['dust2', 'inferno'],
        'Kills': [20, 15],
        'Headshots': [10, 8],
        'AWP Kills': [5, 3],
        'First Bloods': [2, 1],
        'Team Kills': [0, 0],
        'Match ID': ['m1', 'm1'],
        'Scheduled Time': ['2024-01-01', '2024-01-01'],
        'Start Time': ['2024-01-01', '2024-01-01'],
        'Technical Reset': [False, False]
    })

@pytest.fixture
def example_config():
    """Create example configuration mapping."""
    return {
        "test": {
            "player_id_field": "Player ID",
            "map_name_field": "Map",
            "kills_field": "Kills",
            "headshots_field": "Headshots",
            "awp_kills_field": "AWP Kills",
            "first_bloods_field": "First Bloods",
            "team_kills_field": "Team Kills",
            "match_id_field": "Match ID",
            "scheduled_time_field": "Scheduled Time",
            "actual_start_time_field": "Start Time",
            "technical_reset_field": "Technical Reset"
        }
    }

@pytest.mark.asyncio
async def test_excel_template_creation(temp_directories):
    """Test creation of Excel template."""
    watch_dir, _ = temp_directories
    template_path = Path(watch_dir) / "template.xlsx"
    
    # Create template
    create_excel_template(template_path)
    
    # Verify template
    df = pd.read_excel(template_path, sheet_name='Mapping')
    assert 'Calculator Field' in df.columns
    assert 'Excel Column' in df.columns
    assert 'Description' in df.columns
    assert 'Required' in df.columns
    assert len(df) == 16  # Number of fields

@pytest.mark.asyncio
async def test_excel_processing(temp_directories, example_data, example_config):
    """Test processing of Excel files."""
    watch_dir, processed_dir = temp_directories
    
    # Create config file
    config_path = Path(watch_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(example_config, f)
    
    # Create Excel file
    input_path = Path(watch_dir) / "input.xlsx"
    example_data.to_excel(input_path, index=False)
    
    # Initialize scraper
    scraper = ExcelScraper(
        watch_directory=watch_dir,
        processed_directory=processed_dir,
        config_path=config_path
    )
    
    # Process file
    df = await scraper.process_excel_file(input_path)
    
    # Verify processing
    assert 'player_id' in df.columns
    assert 'map_name' in df.columns
    assert 'kills' in df.columns
    assert len(df) == len(example_data)
    assert df['kills'].sum() == example_data['Kills'].sum()

@pytest.mark.asyncio
async def test_file_watching(temp_directories, example_data):
    """Test file watching functionality."""
    watch_dir, processed_dir = temp_directories
    
    # Initialize scraper
    scraper = ExcelScraper(
        watch_directory=watch_dir,
        processed_directory=processed_dir
    )
    
    # Start watching in background
    watch_task = asyncio.create_task(scraper.start_watching())
    
    # Wait for watcher to start
    await asyncio.sleep(1)
    
    # Create new file
    input_path = Path(watch_dir) / "new_file.xlsx"
    example_data.to_excel(input_path, index=False)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Verify file was processed
    processed_files = list(Path(processed_dir).glob("processed_*.xlsx"))
    assert len(processed_files) == 1
    
    # Clean up
    watch_task.cancel()
    try:
        await watch_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_invalid_excel_handling(temp_directories):
    """Test handling of invalid Excel files."""
    watch_dir, processed_dir = temp_directories
    
    # Create invalid Excel file
    input_path = Path(watch_dir) / "invalid.xlsx"
    with open(input_path, 'w') as f:
        f.write("Not an Excel file")
    
    # Initialize scraper
    scraper = ExcelScraper(
        watch_directory=watch_dir,
        processed_directory=processed_dir
    )
    
    # Verify error handling
    with pytest.raises(Exception):
        await scraper.process_excel_file(input_path)

@pytest.mark.asyncio
async def test_datetime_conversion(temp_directories, example_data):
    """Test datetime field conversion."""
    watch_dir, processed_dir = temp_directories
    
    # Add various datetime formats
    example_data['Scheduled Time'] = [
        '2024-01-01T12:00:00Z',
        1704067200  # Unix timestamp
    ]
    
    # Create Excel file
    input_path = Path(watch_dir) / "datetime_test.xlsx"
    example_data.to_excel(input_path, index=False)
    
    # Initialize scraper
    scraper = ExcelScraper(
        watch_directory=watch_dir,
        processed_directory=processed_dir
    )
    
    # Process file
    df = await scraper.process_excel_file(input_path)
    
    # Verify datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(df['scheduled_time'])
    assert pd.api.types.is_datetime64_any_dtype(df['actual_start_time']) 