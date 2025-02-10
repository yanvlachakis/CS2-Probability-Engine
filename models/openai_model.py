import os
import openai
from typing import Dict, Optional, Union
from dotenv import load_dotenv
import json

class CS2OpenAIModel:
    def __init__(self):
        """Initialize the CS2 OpenAI model."""
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI API key
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.system_prompt = """You are an expert CS2 analyst specializing in 
        predicting player performance for PrizePicks. Your task is to analyze 
        player statistics and predict their PrizePicks score. The score is 
        calculated as follows: 1 point each for kills, headshots, AWP kills, 
        and first bloods."""
    
    def build_prompt(self, player_stats: Dict[str, float]) -> str:
        """
        Build a prompt for the OpenAI API based on player statistics.
        
        Args:
            player_stats: Dictionary containing player statistics
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the following CS2 player statistics, predict their 
        PrizePicks score. Consider both the raw statistics and performance biases.

        Player Statistics:
        - Kills: {player_stats.get('kills', 0):.1f}
        - Headshots: {player_stats.get('headshots', 0):.1f}
        - AWP Kills: {player_stats.get('awp_kills', 0):.1f}
        - First Bloods: {player_stats.get('first_bloods', 0):.1f}
        - Headshot Percentage: {player_stats.get('headshot_percentage', 0):.1f}%

        Map-specific Performance Bias:
        - Kills Bias: {player_stats.get('kills_bias', 0):.2f}
        - Headshots Bias: {player_stats.get('headshots_bias', 0):.2f}
        - AWP Kills Bias: {player_stats.get('awp_kills_bias', 0):.2f}
        - First Bloods Bias: {player_stats.get('first_bloods_bias', 0):.2f}

        Please analyze these statistics and provide:
        1. A predicted PrizePicks score (a single number)
        2. A brief explanation of the prediction

        Format your response as a JSON object with keys 'score' and 'explanation'.
        """
        return prompt
    
    def predict_score(self, player_stats: Dict[str, float], 
                     temperature: float = 0.3) -> Dict[str, Union[float, str]]:
        """
        Predict PrizePicks score using OpenAI API.
        
        Args:
            player_stats: Dictionary containing player statistics
            temperature: OpenAI API temperature parameter (0.0 to 1.0)
            
        Returns:
            Dictionary containing predicted score and explanation
        """
        try:
            # Build the prompt
            prompt = self.build_prompt(player_stats)
            
            # Make API call
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or another appropriate model
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=150
            )
            
            # Extract and parse response
            response_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(response_text)
                # Ensure the response has the required keys
                if not all(key in result for key in ['score', 'explanation']):
                    raise ValueError("Invalid response format from OpenAI API")
                
                # Ensure score is non-negative
                result['score'] = max(0.0, float(result['score']))
                
                return result
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                # Try to extract just the numeric score
                import re
                score_match = re.search(r'\b\d+\.?\d*\b', response_text)
                if score_match:
                    return {
                        'score': float(score_match.group()),
                        'explanation': "Score extracted from non-JSON response."
                    }
                else:
                    raise ValueError("Could not parse score from API response")
                
        except Exception as e:
            raise Exception(f"Error predicting score with OpenAI API: {str(e)}")

if __name__ == '__main__':
    # Example usage
    try:
        # Initialize model
        model = CS2OpenAIModel()
        
        # Example player stats
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
        
        # Get prediction
        result = model.predict_score(example_stats)
        print(f"Predicted PrizePicks Score: {result['score']:.2f}")
        print(f"Explanation: {result['explanation']}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 