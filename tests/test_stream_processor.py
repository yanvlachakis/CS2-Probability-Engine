"""Tests for real-time stream processing functionality."""

import pytest
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import websockets
import aiohttp
from kafka import KafkaProducer, KafkaConsumer
from unittest.mock import Mock, patch

from pipeline.stream_processor import StreamProcessor, StreamConfig, StreamMonitor

@pytest.fixture
def stream_config():
    """Create test stream configuration."""
    return StreamConfig(
        kafka_bootstrap_servers=['localhost:9092'],
        kafka_input_topic='test_input',
        kafka_output_topic='test_output',
        websocket_port=8001,
        batch_size=10,
        batch_interval=0.5
    )

@pytest.fixture
def example_match_data():
    """Create example CS2 match data."""
    return {
        'player_id': '12345',
        'map_name': 'dust2',
        'kills': 20,
        'headshots': 10,
        'awp_kills': 5,
        'first_bloods': 2,
        'team_kills': 0,
        'match_id': 'test_match',
        'scheduled_time': datetime.now().isoformat(),
        'actual_start_time': datetime.now().isoformat(),
        'technical_reset': False
    }

@pytest.fixture
def mock_kafka_producer():
    """Create mock Kafka producer."""
    producer = Mock()
    producer.send = Mock(return_value=None)
    return producer

@pytest.fixture
def mock_kafka_consumer():
    """Create mock Kafka consumer."""
    consumer = Mock()
    consumer.__iter__ = Mock(return_value=iter([
        Mock(value=json.dumps({'test': 'data'}))
    ]))
    return consumer

@pytest.mark.asyncio
async def test_websocket_connection(stream_config):
    """Test WebSocket connection and message broadcasting."""
    processor = StreamProcessor(stream_config)
    
    # Start server
    server_task = asyncio.create_task(processor.run())
    await asyncio.sleep(1)  # Wait for server to start
    
    # Connect test client
    async with websockets.connect(f'ws://localhost:{stream_config.websocket_port}/ws') as websocket:
        # Test connection
        assert websocket.open
        
        # Test broadcasting
        test_message = {'test': 'data'}
        await processor.broadcast(test_message)
        
        # Receive broadcast
        received = await websocket.recv()
        assert json.loads(received) == test_message
    
    # Clean up
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_batch_processing(stream_config, example_match_data, mock_kafka_producer, mock_kafka_consumer):
    """Test batch processing of stream data."""
    with patch('pipeline.stream_processor.KafkaProducer', return_value=mock_kafka_producer), \
         patch('pipeline.stream_processor.KafkaConsumer', return_value=mock_kafka_consumer):
        
        processor = StreamProcessor(stream_config)
        
        # Process test batch
        test_batch = [example_match_data.copy() for _ in range(5)]
        await processor.process_batch(test_batch)
        
        # Verify producer was called
        assert mock_kafka_producer.send.called
        
        # Verify data was processed correctly
        call_args = mock_kafka_producer.send.call_args[1]['value']
        processed_data = json.loads(call_args)
        assert len(processed_data) == len(test_batch)
        assert all('prizepicks_score' in record for record in processed_data)

@pytest.mark.asyncio
async def test_score_calculation(stream_config, example_match_data):
    """Test score calculation in stream processing."""
    processor = StreamProcessor(stream_config)
    
    # Calculate score
    df = pd.DataFrame([example_match_data])
    score = processor._calculate_score(df.iloc[0])
    
    # Verify score calculation
    assert isinstance(score, (int, float))
    assert score >= 0
    expected_score = (
        example_match_data['kills'] +
        example_match_data['headshots'] +
        example_match_data['awp_kills'] +
        example_match_data['first_bloods']
    )
    assert score == expected_score

@pytest.mark.asyncio
async def test_stream_monitoring(stream_config):
    """Test stream monitoring functionality."""
    monitor = StreamMonitor()
    
    # Update stats
    monitor.update_stats(processed=100)
    monitor.update_stats(processed=50, error=Exception("Test error"))
    
    # Verify stats
    stats = monitor.get_stats()
    assert stats['processed_records'] == 150
    assert stats['errors'] == 1
    assert stats['last_error'] == "Test error"
    assert stats['processing_rate'] > 0

@pytest.mark.asyncio
async def test_error_handling(stream_config, example_match_data):
    """Test error handling in stream processing."""
    processor = StreamProcessor(stream_config)
    
    # Test invalid data
    invalid_data = example_match_data.copy()
    del invalid_data['kills']  # Remove required field
    
    # Process invalid data
    with pytest.raises(Exception):
        await processor.process_batch([invalid_data])

@pytest.mark.asyncio
async def test_batch_timing(stream_config, example_match_data, mock_kafka_producer, mock_kafka_consumer):
    """Test batch timing and interval processing."""
    with patch('pipeline.stream_processor.KafkaProducer', return_value=mock_kafka_producer), \
         patch('pipeline.stream_processor.KafkaConsumer', return_value=mock_kafka_consumer):
        
        processor = StreamProcessor(stream_config)
        
        # Start processing
        process_task = asyncio.create_task(processor.process_stream())
        
        # Wait for one batch interval
        await asyncio.sleep(stream_config.batch_interval + 0.1)
        
        # Verify processing occurred
        assert mock_kafka_producer.send.called
        
        # Clean up
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 