# Scraper Configuration Guide

This guide explains how to configure data sources for the CS2 PrizePicks Probability Engine.

## Overview

The system supports three main data ingestion methods:
1. Direct scraper integration via JSON configuration
2. Excel-based data ingestion
3. Real-time streaming via Kafka

## Mapping Methods

### 1. JSON-Based Mapping

The JSON configuration method is ideal for direct scraper integration and programmatic data ingestion. Configuration is stored in `config/scraper_mappings.json`.

#### Usage Example
```python
from pipeline.scraper_interface import ScraperInterface, create_example_config
import json

# Create example configuration
config = create_example_config()
with open('config/scraper_mappings.json', 'w') as f:
    json.dump(config, f, indent=4)

# Initialize interface
interface = ScraperInterface('config/scraper_mappings.json')

# Standardize data
standardized_data = interface.standardize_data(your_scraped_data)
```

#### Supported Data Sources
- HLTV (`hltv` configuration)
- FACEIT (`faceit` configuration)
- ESEA (`esea` configuration)

### 2. Excel-Based Mapping

The Excel mapping method is designed for manual data input and spreadsheet-based workflows. Uses a template file (`config/excel_template.xlsx`).

#### Usage Example
```python
from pipeline.excel_scraper import ExcelScraper, create_excel_template

# Create template
create_excel_template('config/excel_template.xlsx')

# Initialize scraper
scraper = ExcelScraper(
    watch_directory='data/excel_input',
    processed_directory='data/excel_processed'
)

# Start watching for files
await scraper.start_watching()
```

## Required Fields

| Field Name | Description | Example Values | Required | JSON Key | Excel Column |
|------------|-------------|----------------|-----------|-----------|--------------|
| Player ID | Unique identifier | "76561198013591977" | Yes | player_id_field | Player ID |
| Map Name | Map played | "dust2", "inferno" | Yes | map_name_field | Map |
| Kills | Total kills | Integer | Yes | kills_field | Kills |
| Headshots | Headshot kills | Integer | Yes | headshots_field | Headshots |
| AWP Kills | Kills with AWP | Integer | Yes | awp_kills_field | AWP Kills |
| First Bloods | First kills of rounds | Integer | Yes | first_bloods_field | First Bloods |
| Team Kills | Team kills | Integer | Yes | team_kills_field | Team Kills |
| Match ID | Match identifier | String | Yes | match_id_field | Match ID |
| Scheduled Time | Scheduled time | Datetime | Yes | scheduled_time_field | Scheduled Time |
| Actual Start | Start time | Datetime | Yes | actual_start_time_field | Actual Start |
| Technical Reset | Reset occurred | Boolean | Yes | technical_reset_field | Technical Reset |
| Round Number | Current round | Integer | No | round_number_field | Round Number |
| Player Team | Player's team | String | No | player_team_field | Player Team |
| Opponent Team | Opponent team | String | No | opponent_team_field | Opponent Team |
| Match Format | bo1, bo3, etc. | String | No | match_format_field | Match Format |
| Tournament | Event name | String | No | tournament_name_field | Tournament |

## Data Format Requirements

### DateTime Fields
- All datetime fields should be convertible to pandas datetime
- Supported formats:
  - ISO format: "2024-01-01T12:00:00Z"
  - Simple date: "2024-01-01"
  - Unix timestamp: 1704067200

### Numeric Fields
- All numeric fields should contain integers
- Missing values will be converted to 0
- Negative values are allowed but will be handled according to PrizePicks rules

### Boolean Fields
- Technical reset field should be boolean (True/False)
- String values "true"/"false" (case-insensitive) will be converted

## Example Configurations

### 1. JSON Configuration Example (HLTV)
```json
{
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
        "technical_reset_field": "tech_reset"
    }
}
```

### 2. Excel Template Structure
The Excel template includes:
- Mapping sheet with field definitions
- Example data sheet
- Validation rules
- Data type specifications

## Real-time Processing

### Stream Configuration
```json
{
    "kafka": {
        "bootstrap_servers": ["localhost:9092"],
        "input_topic": "cs2_raw_data",
        "output_topic": "cs2_processed_data",
        "batch_size": 100,
        "batch_interval": 1.0
    },
    "websocket": {
        "port": 8000,
        "host": "0.0.0.0"
    }
}
```

### WebSocket API
Connect to the WebSocket server for real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received prediction:', data);
};
```

## Validation & Error Handling

### JSON Validation
```python
# Validate JSON configuration
config = interface.validate_config('config/scraper_mappings.json')
if not config.is_valid():
    print(f"Configuration errors: {config.get_errors()}")
```

### Excel Validation
```python
# Validate Excel data
validation_result = scraper.validate_excel_file('input.xlsx')
if not validation_result.is_valid:
    print(f"Excel validation errors: {validation_result.errors}")
```

## Best Practices

1. Choose the appropriate mapping method:
   - Use JSON for programmatic integration
   - Use Excel for manual data entry/spreadsheet workflows
2. Always validate data before processing
3. Include all optional fields when available
4. Maintain consistent datetime formats
5. Document custom mappings

## Troubleshooting

Common issues and solutions:

1. Missing Fields Error
```python
# Solution: Ensure all required fields are present
required_fields = interface.validate_scraper_data(df)
if required_fields:
    print(f"Missing fields: {required_fields}")
```

2. DateTime Format Error
```python
# Solution: Convert to proper format
df['date_scheduled'] = pd.to_datetime(df['date_scheduled'])
```

3. Invalid Data Types
```python
# Solution: Convert to correct types
df['kills'] = df['kills'].astype(int)
df['tech_reset'] = df['tech_reset'].astype(bool)
``` 