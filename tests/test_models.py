"""Tests for prediction models."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from models.regression_model import CS2RegressionModel
from models.openai_model import CS2OpenAIModel
from models.ensemble import ModelEnsemble

@pytest.fixture
def example_player_stats():
    """Create example player statistics."""
    return {
        'player_id': '12345',
        'recent_matches': [
            {
                'map': 'dust2',
                'kills': 20,
                'headshots': 10,
                'awp_kills': 5,
                'first_bloods': 2,
                'team_kills': 0,
                'rounds_played': 30,
                'team_rounds_won': 16,
                'side_first_half': 'CT'
            } for _ in range(20)
        ],
        'map_stats': {
            'dust2': {
                'matches_played': 100,
                'average_kills': 18.5,
                'win_rate': 0.55,
                't_side_rounds_won_rate': 0.52,
                'ct_side_rounds_won_rate': 0.58
            }
        },
        'team_stats': {
            'team_id': 'team1',
            'win_rate': 0.60,
            'average_round_difference': 3.2,
            'playstyle_aggression': 0.7
        }
    }

@pytest.fixture
def example_match_context():
    """Create example match context."""
    return {
        'match_id': 'match1',
        'tournament_tier': 1,
        'is_lan': True,
        'stage': 'playoff',
        'maps': ['dust2'],
        'opponent_team': {
            'team_id': 'team2',
            'win_rate': 0.58,
            'average_round_difference': 2.8,
            'playstyle_aggression': 0.6
        },
        'scheduled_time': datetime.now() + timedelta(days=1),
        'format': 'bo3'
    }

@pytest.mark.asyncio
async def test_regression_model_prediction(example_player_stats, example_match_context):
    """Test regression model predictions."""
    model = CS2RegressionModel()
    
    # Test basic prediction
    prediction = model.predict(example_player_stats, example_match_context)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 100
    
    # Test confidence score
    confidence = model.get_prediction_confidence()
    assert 0 <= confidence <= 1
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    assert sum(importance.values()) == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_openai_model_prediction(example_player_stats, example_match_context):
    """Test OpenAI model predictions."""
    mock_response = {
        'choices': [{
            'message': {
                'content': json.dumps({
                    'prediction': 25.5,
                    'confidence': 0.85,
                    'explanation': 'Based on recent performance...'
                })
            }
        }]
    }
    
    with patch('openai.ChatCompletion.create', return_value=mock_response):
        model = CS2OpenAIModel()
        
        # Test prediction
        prediction = await model.predict(example_player_stats, example_match_context)
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 100
        
        # Test explanation
        explanation = model.get_prediction_explanation()
        assert isinstance(explanation, str)
        assert len(explanation) > 0

@pytest.mark.asyncio
async def test_model_ensemble(example_player_stats, example_match_context):
    """Test model ensemble predictions."""
    ensemble = ModelEnsemble()
    
    # Test ensemble prediction
    prediction = await ensemble.predict(example_player_stats, example_match_context)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 100
    
    # Test weight adjustment
    ensemble.update_weights(regression_accuracy=0.8, openai_accuracy=0.9)
    weights = ensemble.get_current_weights()
    assert sum(weights.values()) == pytest.approx(1.0)

def test_regression_model_feature_engineering(example_player_stats):
    """Test feature engineering in regression model."""
    model = CS2RegressionModel()
    
    # Test feature extraction
    features = model._extract_features(example_player_stats)
    assert 'average_kills' in features
    assert 'headshot_rate' in features
    assert 'awp_kill_rate' in features
    assert 'first_blood_rate' in features
    
    # Test map adjustments
    map_features = model._calculate_map_features(example_player_stats['map_stats']['dust2'])
    assert 'map_win_rate' in map_features
    assert 'map_rounds_avg' in map_features

def test_regression_model_validation(example_player_stats, example_match_context):
    """Test regression model validation rules."""
    model = CS2RegressionModel()
    
    # Test prediction bounds
    prediction = model.predict(example_player_stats, example_match_context)
    assert model.MIN_PREDICTION <= prediction <= model.MAX_PREDICTION
    
    # Test confidence thresholds
    confidence = model.get_prediction_confidence()
    assert model.MIN_CONFIDENCE <= confidence <= 1.0

@pytest.mark.asyncio
async def test_openai_model_error_handling():
    """Test OpenAI model error handling."""
    model = CS2OpenAIModel()
    
    # Test API error handling
    with patch('openai.ChatCompletion.create', side_effect=Exception('API Error')):
        with pytest.raises(Exception):
            await model.predict({}, {})
    
    # Test invalid response handling
    mock_response = {'choices': [{'message': {'content': 'invalid json'}}]}
    with patch('openai.ChatCompletion.create', return_value=mock_response):
        with pytest.raises(ValueError):
            await model.predict({}, {})

def test_model_performance_monitoring():
    """Test model performance monitoring."""
    monitor = ModelPerformanceMonitor()
    
    # Test accuracy tracking
    monitor.update_metrics(
        predicted=25.5,
        actual=24.0,
        model_type='regression',
        prediction_time=0.1,
        confidence=0.8
    )
    
    metrics = monitor.get_metrics()
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'prediction_count' in metrics

@pytest.mark.asyncio
async def test_ensemble_validation_rules(example_player_stats, example_match_context):
    """Test ensemble model validation rules."""
    ensemble = ModelEnsemble()
    
    # Test conflicting predictions handling
    with patch('models.regression_model.CS2RegressionModel.predict', return_value=20.0), \
         patch('models.openai_model.CS2OpenAIModel.predict', return_value=40.0):
        
        prediction = await ensemble.predict(example_player_stats, example_match_context)
        validation = ensemble.get_validation_details()
        
        assert 'conflict_detected' in validation
        assert 'resolution_method' in validation

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 