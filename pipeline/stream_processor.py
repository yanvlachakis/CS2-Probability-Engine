"""
Real-time data streaming processor for CS2 match data.

This module handles live data ingestion and processing using Kafka
and WebSocket connections for real-time updates.
"""

import asyncio
import json
from typing import Dict, Optional, Union, List, Callable
from kafka import KafkaConsumer, KafkaProducer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
import aiofiles
from .scraper_interface import ScraperConfig
from models.scoring_rules import CS2ScoreCalculator

class StreamConfig(BaseModel):
    """Configuration for data streaming."""
    kafka_bootstrap_servers: List[str]
    kafka_input_topic: str
    kafka_output_topic: str
    websocket_port: int = 8000
    batch_size: int = 100
    batch_interval: float = 1.0  # seconds

class StreamProcessor:
    """Real-time data stream processor for CS2 match data."""
    
    def __init__(self, config: StreamConfig):
        """
        Initialize the stream processor.
        
        Args:
            config: StreamConfig object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.score_calculator = CS2ScoreCalculator()
        
        # Initialize Kafka
        self.producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        self.consumer = KafkaConsumer(
            config.kafka_input_topic,
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        # Initialize FastAPI for WebSocket
        self.app = FastAPI()
        self.active_connections: List[WebSocket] = []
        
        # Setup WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming WebSocket messages if needed
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connections."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected WebSocket clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {str(e)}")
                self.active_connections.remove(connection)
    
    async def process_stream(self):
        """Process incoming data stream."""
        batch = []
        last_process_time = datetime.now()
        
        async def process_batch(data_batch: List[Dict]):
            """Process a batch of data records."""
            try:
                # Convert batch to DataFrame
                df = pd.DataFrame(data_batch)
                
                # Calculate scores
                df['prizepicks_score'] = df.apply(
                    lambda row: self._calculate_score(row), axis=1
                )
                
                # Broadcast processed data
                for _, row in df.iterrows():
                    await self.broadcast(row.to_dict())
                
                # Send to Kafka output topic
                self.producer.send(
                    self.config.kafka_output_topic,
                    value=df.to_dict(orient='records')
                )
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")
        
        try:
            for message in self.consumer:
                batch.append(message.value)
                
                current_time = datetime.now()
                time_diff = (current_time - last_process_time).total_seconds()
                
                if len(batch) >= self.config.batch_size or time_diff >= self.config.batch_interval:
                    await process_batch(batch)
                    batch = []
                    last_process_time = current_time
                    
        except Exception as e:
            self.logger.error(f"Error in stream processing: {str(e)}")
            raise
    
    def _calculate_score(self, row: pd.Series) -> float:
        """Calculate PrizePicks score for a data record."""
        match_info = {
            'player_maps_played': [row['map_name']],
            'required_maps': [row['map_name']],
            'scheduled_time': row['scheduled_time'],
            'actual_start_time': row['actual_start_time']
        }
        
        player_stats = {
            'kills': row['kills'],
            'headshots': row['headshots'],
            'awp_kills': row['awp_kills'],
            'first_bloods': row['first_bloods'],
            'team_kills': row.get('team_kills', 0)
        }
        
        score = self.score_calculator.calculate_score(player_stats, match_info)
        return score.total_score if score is not None else 0
    
    async def run(self):
        """Run the stream processor."""
        import uvicorn
        
        # Start WebSocket server
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.websocket_port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run WebSocket server and stream processor concurrently
        await asyncio.gather(
            server.serve(),
            self.process_stream()
        )

class StreamMonitor:
    """Monitor for stream processing statistics."""
    
    def __init__(self):
        """Initialize the stream monitor."""
        self.stats = {
            'processed_records': 0,
            'errors': 0,
            'start_time': datetime.now(),
            'last_error': None,
            'processing_rate': 0.0
        }
    
    def update_stats(self, processed: int, error: Optional[Exception] = None):
        """Update monitoring statistics."""
        self.stats['processed_records'] += processed
        if error:
            self.stats['errors'] += 1
            self.stats['last_error'] = str(error)
        
        elapsed_time = (datetime.now() - self.stats['start_time']).total_seconds()
        self.stats['processing_rate'] = self.stats['processed_records'] / elapsed_time
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats

if __name__ == '__main__':
    # Example usage
    try:
        # Configure stream processor
        config = StreamConfig(
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_input_topic='cs2_raw_data',
            kafka_output_topic='cs2_processed_data',
            websocket_port=8000,
            batch_size=100,
            batch_interval=1.0
        )
        
        # Initialize and run processor
        processor = StreamProcessor(config)
        asyncio.run(processor.run())
        
    except Exception as e:
        logging.error(f"Error: {str(e)}") 