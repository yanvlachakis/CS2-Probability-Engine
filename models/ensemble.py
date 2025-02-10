"""
Hybrid model ensemble combining regression and OpenAI models.
"""

from typing import Dict, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import json
from datetime import datetime, date
import os
import numpy as np

from .regression_model import CS2RegressionModel
from .openai_model import CS2OpenAIModel

@dataclass
class PredictionContext:
    """Context information for making predictions."""
    match_id: str
    tournament_tier: int  # 1 (Major) to 3 (Minor)
    is_lan: bool
    stage: str  # 'group', 'playoff', 'final', etc.
    prize_pool: Optional[float] = None
    team_ranking_difference: Optional[int] = None

class ModelEnsemble:
    """Hybrid model ensemble combining regression and OpenAI approaches."""
    
    def __init__(self):
        """Initialize the model ensemble with both regression and OpenAI models."""
        # Load models
        self.regression_model = CS2RegressionModel()
        self.openai_model = CS2OpenAIModel()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize usage tracking
        self._init_usage_tracking()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        config_path = os.path.join('config', 'ensemble_config.json')
        default_config = {
            'openai_threshold': float(os.getenv('OPENAI_THRESHOLD', 0.8)),
            'hybrid_mode': os.getenv('HYBRID_MODE', 'true').lower() == 'true',
            'weights': {
                'regression': 0.6,
                'openai': 0.4
            },
            'high_stakes_threshold': {
                'prize_pool': 100000,
                'tournament_tier': 2,
                'stage_weights': {
                    'final': 1.0,
                    'playoff': 0.8,
                    'group': 0.6
                }
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            return default_config
        except Exception as e:
            print(f"Warning: Failed to load config: {str(e)}")
            return default_config
    
    def _init_usage_tracking(self):
        """Initialize usage tracking statistics."""
        self.usage_stats = {
            'daily_requests': {
                'date': str(date.today()),
                'regression_count': 0,
                'openai_count': 0,
                'total_cost': 0.0
            },
            'model_accuracy': {
                'regression': [],
                'openai': []
            }
        }
    
    def _is_high_stakes(self, context: PredictionContext) -> bool:
        """
        Determine if a prediction is high stakes based on context.
        
        Args:
            context: Prediction context
            
        Returns:
            Boolean indicating if prediction is high stakes
        """
        thresholds = self.config['high_stakes_threshold']
        
        # Check prize pool
        if context.prize_pool and context.prize_pool >= thresholds['prize_pool']:
            return True
        
        # Check tournament tier
        if context.tournament_tier <= thresholds['tournament_tier']:
            return True
        
        # Check stage importance
        stage_weight = thresholds['stage_weights'].get(
            context.stage.lower(),
            thresholds['stage_weights']['group']
        )
        if stage_weight >= 0.8:  # High importance stages
            return True
        
        return False
    
    def _calculate_complexity(self, stats: Dict[str, float],
                            context: Dict[str, any]) -> float:
        """
        Calculate prediction complexity score.
        
        Args:
            stats: Player statistics
            context: Match context
            
        Returns:
            Complexity score between 0 and 1
        """
        complexity_factors = []
        
        # Check for unusual stat patterns
        if stats.get('headshot_percentage', 0) > 80:
            complexity_factors.append(0.3)
        if stats.get('awp_kills', 0) / max(stats.get('kills', 1), 1) > 0.7:
            complexity_factors.append(0.2)
            
        # Consider team ranking difference
        if context.get('team_ranking_difference', 0) > 10:
            complexity_factors.append(0.2)
            
        # Consider map specific factors
        if context.get('map_name') in ['ancient', 'anubis']:  # Newer maps
            complexity_factors.append(0.1)
            
        return min(1.0, sum(complexity_factors))
    
    async def predict(self,
                     player_stats: Dict[str, float],
                     match_context: Dict[str, any],
                     prediction_context: PredictionContext) -> Dict:
        """
        Make a prediction using the appropriate model(s).
        
        Args:
            player_stats: Player statistics
            match_context: Match context information
            prediction_context: Additional context for model selection
            
        Returns:
            Dictionary containing prediction and metadata
        """
        result = {
            'prediction': None,
            'confidence': 0.0,
            'model_used': None,
            'explanation': None
        }
        
        # Update usage tracking
        today = str(date.today())
        if self.usage_stats['daily_requests']['date'] != today:
            self._init_usage_tracking()
        
        # Determine if this is a high stakes prediction
        is_high_stakes = self._is_high_stakes(prediction_context)
        
        # Calculate prediction complexity
        complexity = self._calculate_complexity(player_stats, match_context)
        
        # Decide which model(s) to use
        use_openai = (
            self.config['hybrid_mode'] and
            (is_high_stakes or complexity >= self.config['openai_threshold'])
        )
        
        # Get regression model prediction
        regression_prediction = self.regression_model.predict_score(player_stats)
        self.usage_stats['daily_requests']['regression_count'] += 1
        
        if use_openai:
            # Get OpenAI model prediction
            try:
                openai_result = await self.openai_model.predict_score(
                    player_stats,
                    match_context
                )
                openai_prediction = openai_result['score']
                explanation = openai_result['explanation']
                self.usage_stats['daily_requests']['openai_count'] += 1
                self.usage_stats['daily_requests']['total_cost'] += (
                    openai_result.get('cost', 0.0)
                )
                
                # Use weighted average
                weights = self.config['weights']
                final_prediction = (
                    regression_prediction * weights['regression'] +
                    openai_prediction * weights['openai']
                )
                
                # Calculate confidence based on prediction agreement
                confidence = 1.0 - abs(regression_prediction - openai_prediction) / max(
                    regression_prediction,
                    openai_prediction,
                    1.0
                )
                
                result.update({
                    'prediction': final_prediction,
                    'confidence': confidence,
                    'model_used': 'hybrid',
                    'explanation': explanation,
                    'regression_prediction': regression_prediction,
                    'openai_prediction': openai_prediction
                })
                
            except Exception as e:
                print(f"Warning: OpenAI prediction failed: {str(e)}")
                result.update({
                    'prediction': regression_prediction,
                    'confidence': 0.7,  # Default confidence for regression
                    'model_used': 'regression',
                    'explanation': "Used regression model due to OpenAI error"
                })
        else:
            # Use regression model only
            result.update({
                'prediction': regression_prediction,
                'confidence': 0.7,  # Default confidence for regression
                'model_used': 'regression',
                'explanation': (
                    "Used regression model based on prediction complexity "
                    f"and stakes (complexity: {complexity:.2f})"
                )
            })
        
        return result
    
    def get_usage_stats(self) -> Dict:
        """Get current usage statistics."""
        return self.usage_stats
    
    def update_weights(self, accuracy_data: Dict[str, float]):
        """
        Update model weights based on recent accuracy.
        
        Args:
            accuracy_data: Dictionary with recent accuracy scores
        """
        if not accuracy_data:
            return
            
        # Update accuracy history
        self.usage_stats['model_accuracy']['regression'].append(
            accuracy_data.get('regression', 0.0)
        )
        self.usage_stats['model_accuracy']['openai'].append(
            accuracy_data.get('openai', 0.0)
        )
        
        # Keep only last 100 accuracy points
        max_history = 100
        self.usage_stats['model_accuracy']['regression'] = (
            self.usage_stats['model_accuracy']['regression'][-max_history:]
        )
        self.usage_stats['model_accuracy']['openai'] = (
            self.usage_stats['model_accuracy']['openai'][-max_history:]
        )
        
        # Calculate new weights based on moving average
        reg_acc = np.mean(self.usage_stats['model_accuracy']['regression'])
        openai_acc = np.mean(self.usage_stats['model_accuracy']['openai'])
        
        total = reg_acc + openai_acc
        if total > 0:
            self.config['weights']['regression'] = reg_acc / total
            self.config['weights']['openai'] = openai_acc / total
            
            # Save updated config
            config_path = os.path.join('config', 'ensemble_config.json')
            try:
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            except Exception as e:
                print(f"Warning: Failed to save updated weights: {str(e)}") 