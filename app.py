import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import json
import asyncio
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import logging
from logging.handlers import RotatingFileHandler

from pipeline.data_pipeline import CS2DataPipeline
from models.regression_model import CS2RegressionModel
from models.openai_model import CS2OpenAIModel
from models.ensemble import ModelEnsemble, PredictionContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration
def load_config():
    """Load configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    config_path = Path('config/dashboard_config.json')
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config[env]

# Configure logging
def setup_logging(config):
    """Set up logging based on environment."""
    if config.get('debug', False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        log_dir = Path('/var/log/cs2prizepicks')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = RotatingFileHandler(
            log_dir / 'dashboard.log',
            maxBytes=10000000,
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        ))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

# Initialize Flask app with configuration
config = load_config()
app = Flask(__name__)
app.config['SECRET_KEY'] = config['secret_key']

# Initialize SocketIO with configuration
socketio = SocketIO(
    app,
    async_mode=config['websocket']['async_mode'],
    cors_allowed_origins=config['websocket']['cors_allowed_origins'],
    ping_timeout=config['websocket']['ping_timeout'],
    ping_interval=config['websocket']['ping_interval']
)

class CS2PrizePicks:
    def __init__(self):
        """Initialize the CS2 PrizePicks prediction system."""
        self.pipeline = CS2DataPipeline(
            raw_data_path='data/raw',
            processed_data_path='data/processed'
        )
        self.ensemble = ModelEnsemble()
        self.prediction_history = []
        self.max_history = 100
        
        # Initialize example prediction if in development
        if config.get('debug', False):
            asyncio.create_task(self.create_example_prediction())
    
    async def create_example_prediction(self):
        """Create an example prediction for development."""
        example_stats = {
            'kills': 20,
            'headshots': 10,
            'awp_kills': 5,
            'first_bloods': 2,
            'headshot_percentage': 50.0,
            'kills_bias': 1.5,
            'headshots_bias': 0.8,
            'awp_kills_bias': 0.3,
            'first_bloods_bias': 0.1
        }
        
        example_match = {
            'match_id': 'example_match',
            'tournament_tier': 1,
            'is_lan': True,
            'stage': 'playoff',
            'prize_pool': 1000000,
            'team_ranking_difference': 2,
            'map_name': 'dust2'
        }
        
        prediction_context = PredictionContext(
            match_id=example_match['match_id'],
            tournament_tier=example_match['tournament_tier'],
            is_lan=example_match['is_lan'],
            stage=example_match['stage'],
            prize_pool=example_match['prize_pool'],
            team_ranking_difference=example_match['team_ranking_difference']
        )
        
        await self.predict(example_stats, example_match, prediction_context)
    
    async def predict(self, 
                     player_stats: Dict[str, float],
                     match_context: Dict[str, any],
                     prediction_context: Optional[PredictionContext] = None) -> Dict:
        """Make a prediction using the ensemble model."""
        try:
            if prediction_context is None:
                prediction_context = PredictionContext(
                    match_id=match_context.get('match_id', 'unknown'),
                    tournament_tier=match_context.get('tournament_tier', 3),
                    is_lan=match_context.get('is_lan', False),
                    stage=match_context.get('stage', 'regular_season')
                )
            
            result = await self.ensemble.predict(
                player_stats,
                match_context,
                prediction_context
            )
            
            result['usage_stats'] = self.ensemble.get_usage_stats()
            result['timestamp'] = datetime.now().isoformat()
            
            self.prediction_history.append(result)
            if len(self.prediction_history) > self.max_history:
                self.prediction_history.pop(0)
            
            socketio.emit('prediction_update', result)
            
            if config.get('debug', False):
                logging.debug(f"Prediction made: {result}")
            
            return result
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise
    
    def get_prediction_history(self) -> list:
        """Get prediction history."""
        return self.prediction_history
    
    def get_model_stats(self) -> Dict:
        """Get model performance statistics."""
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'model_usage': {'regression': 0, 'openai': 0, 'hybrid': 0},
                'average_confidence': 0,
                'cost_stats': {'total_cost': 0, 'last_24h_cost': 0}
            }
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'model_usage': {
                'regression': 0,
                'openai': 0,
                'hybrid': 0
            },
            'average_confidence': 0,
            'cost_stats': {
                'total_cost': 0,
                'last_24h_cost': 0
            }
        }
        
        confidence_sum = 0
        last_24h = datetime.now() - timedelta(days=1)
        
        for pred in self.prediction_history:
            stats['model_usage'][pred['model_used']] += 1
            confidence_sum += pred['confidence']
            
            pred_time = datetime.fromisoformat(pred['timestamp'])
            pred_cost = pred['usage_stats']['daily_requests'].get('total_cost', 0)
            stats['cost_stats']['total_cost'] += pred_cost
            
            if pred_time > last_24h:
                stats['cost_stats']['last_24h_cost'] += pred_cost
        
        stats['average_confidence'] = confidence_sum / len(self.prediction_history)
        return stats

# Initialize system
system = CS2PrizePicks()

# Set up logging
setup_logging(config)

@app.route('/')
def dashboard():
    """Render main dashboard."""
    return render_template('dashboard.html')

@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    return jsonify(system.get_model_stats())

@app.route('/api/predictions')
def get_predictions():
    """Get prediction history."""
    return jsonify(system.get_prediction_history())

@app.route('/api/predict', methods=['POST'])
async def make_prediction():
    """Make a new prediction."""
    try:
        data = request.json
        prediction = await system.predict(
            data['player_stats'],
            data['match_context'],
            PredictionContext(**data.get('prediction_context', {}))
        )
        return jsonify(prediction)
    except Exception as e:
        logging.error(f"Prediction API error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logging.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logging.info('Client disconnected')

if __name__ == '__main__':
    # Run the application
    socketio.run(
        app,
        host=config['host'],
        port=config['port'],
        debug=config['debug']
    ) 