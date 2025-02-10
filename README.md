# CS2 PrizePicks Probability Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-blue.svg)

## Project Overview
This project combines statistical analysis and machine learning to predict CS2 player performance scores for PrizePicks. It utilizes two distinct approaches:
1. A hardcoded regression model using historical player statistics
2. An ML/AI solution leveraging OpenAI's API for intelligent predictions

### Key Features
- Data pipeline for processing raw CS2 player statistics
- Feature engineering including per-map averages, recent performance, and map bias adjustments
- Multiple prediction models for comparison and validation
- Easy-to-use interface for generating predictions
- Comprehensive implementation of official PrizePicks scoring rules
- Excel-based data ingestion with automatic mapping
- Real-time data streaming and processing
- WebSocket support for live updates

## PrizePicks Scoring Rules

### Basic Scoring Chart
| Action       | Points |
|-------------|--------|
| AWP Kill    | 1 pt   |
| First Blood | 1 pt   |
| Headshot    | 1 pt   |
| Kill        | 1 pt   |

### Scoring Details
- **Headshots**: Awarded when the final/killing blow lands as damage to the head (uppermost portion, excluding torso) of an opponent's hitbox
- **First Bloods**: Awarded to the first player to get a kill within a round
- **AWP Kills**: Awarded for one-shot kills using the AWP rifle
- **Team Kills**: Do not result in point deductions for the player performing the action

### Match Requirements
- Players must complete all specified maps to be counted
- Maps may be part of best-of-1, 3, 5, etc. series
- Projections apply only to specified maps in the description

### Technical Rules
- Results are based on the official stream
- Technical difficulties:
  - If reset occurs: Only stats from reset round count
  - If no reset: Player not marked DNP unless substituted
- Match timing:
  - Must start within 12 hours of scheduled time
  - Postponements within 24 hours keep projections live
  - Beyond 24 hours: DNP ruling

## Directory Structure
```
cs2-probability-engine/
├── data/
│   ├── raw/                   # Raw scraper output data
│   ├── processed/             # Cleaned and feature-engineered data
│   ├── excel_input/          # Directory for Excel file ingestion
│   └── excel_processed/      # Processed Excel files
├── models/
│   ├── regression_model.py    # Statistical regression model
│   ├── openai_model.py        # ML/AI solution using OpenAI API
│   └── scoring_rules.py       # PrizePicks scoring implementation
├── pipeline/
│   ├── data_pipeline.py       # Data processing and feature engineering
│   ├── scraper_interface.py   # JSON-based scraper interface
│   ├── excel_scraper.py       # Excel-based data ingestion
│   └── stream_processor.py    # Real-time data streaming
├── config/
│   ├── scraper_mappings.json  # JSON scraper field mappings
│   ├── excel_template.xlsx    # Excel mapping template
│   └── stream_config.json     # Streaming configuration
├── docs/
│   ├── scraper_configuration.md  # Detailed mapping documentation
│   └── model_documentation.md    # Model documentation
├── tests/                     # Test suite
├── app.py                     # Main application entry point
├── pyproject.toml            # Poetry configuration
└── README.md                 # Project documentation
```

## Setup Instructions

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your-openai-api-key
```

4. Set up Kafka (for real-time streaming):
```bash
# Using Docker
docker-compose up -d kafka
```

## Usage

### Data Ingestion Methods

1. JSON-based Scraper Integration:
```python
from pipeline.scraper_interface import ScraperInterface

# Initialize interface
interface = ScraperInterface('config/scraper_mappings.json')
# Process data
standardized_data = interface.standardize_data(your_scraped_data)
```

2. Excel-based Data Input:
```python
from pipeline.excel_scraper import ExcelScraper

# Initialize scraper
scraper = ExcelScraper(
    watch_directory='data/excel_input',
    processed_directory='data/excel_processed'
)
# Start watching for files
await scraper.start_watching()
```

3. Real-time Streaming:
```python
from pipeline.stream_processor import StreamProcessor, StreamConfig

# Configure stream processor
config = StreamConfig(
    kafka_bootstrap_servers=['localhost:9092'],
    kafka_input_topic='cs2_raw_data',
    kafka_output_topic='cs2_processed_data'
)
# Start processor
processor = StreamProcessor(config)
await processor.run()
```

### Development

1. Run tests:
```bash
poetry run pytest
```

2. Run linting:
```bash
poetry run black .
poetry run flake8
```

3. Run security checks:
```bash
poetry run bandit -r .
```

## WebSocket API

Connect to the WebSocket server for real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received prediction:', data);
};
```

## Model Comparison

The system employs two complementary prediction approaches:

### Regression Model
- **Best for:** High-volume, real-time predictions
- **Advantages:** Fast, consistent, cost-effective
- **Limitations:** Less adaptable, requires regular retraining
- **Use cases:** Daily matches, standard scenarios, streaming data

### OpenAI Model
- **Best for:** Complex scenarios, detailed analysis
- **Advantages:** Context-aware, adaptable, detailed explanations
- **Limitations:** Higher cost, API dependency, longer latency
- **Use cases:** Tournament finals, unusual matchups, strategy analysis

### Hybrid Approach
The system automatically chooses the appropriate model based on:
- Prediction confidence thresholds
- Match importance and complexity
- Budget considerations
- Performance requirements

See `docs/model_documentation.md` for detailed comparison and implementation details.

## API Costs & Usage

### OpenAI API Costs
The system uses OpenAI's GPT-4 API for ML/AI predictions. Estimated costs:
- Per-prediction cost: ~$0.021
  - Input: ~500 tokens ($0.015)
  - Output: ~100 tokens ($0.006)

### Monthly Estimates
Based on daily prediction volume:
```
Light:   100 predictions/day  ≈ $63/month
Medium:  500 predictions/day  ≈ $315/month
Heavy:   1000 predictions/day ≈ $630/month
```

### Cost Optimization
1. Enable caching in `.env`:
```
ENABLE_PREDICTION_CACHE=true
CACHE_LIFETIME_HOURS=24
```

2. Use hybrid mode to optimize costs:
```
HYBRID_MODE=true
OPENAI_THRESHOLD=0.8  # Use OpenAI only for high-uncertainty predictions
```

3. Set budget limits:
```
MONTHLY_BUDGET=500.00
DAILY_REQUEST_LIMIT=1000
BUDGET_ALERT_THRESHOLD=0.8
```

## License
[MIT License](LICENSE)
