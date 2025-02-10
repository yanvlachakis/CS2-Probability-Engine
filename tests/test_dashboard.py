import pytest
from flask import url_for
from flask_socketio import SocketIOTestClient
from app import create_app, socketio
import json

@pytest.fixture
def app():
    app = create_app('testing')
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def socket_client(app):
    return SocketIOTestClient(app, socketio)

def test_dashboard_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'CS2 PrizePicks Probability Engine' in response.data

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_stats_endpoint(client):
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'total_predictions' in data
    assert 'average_confidence' in data
    assert 'daily_cost' in data
    assert 'total_cost' in data

def test_prediction_endpoint(client):
    test_data = {
        'player_name': 'test_player',
        'map': 'dust2',
        'tournament': 'Major',
        'kills_line': 20.5,
        'historical_stats': {
            'avg_kills': 19.8,
            'maps_played': 100
        }
    }
    response = client.post('/api/predict', 
                          data=json.dumps(test_data),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'model_used' in data

def test_websocket_connection(socket_client):
    assert socket_client.is_connected()
    received = socket_client.get_received()
    assert len(received) == 1
    assert received[0]['name'] == 'connect'

def test_websocket_prediction_update(socket_client):
    test_prediction = {
        'player_name': 'test_player',
        'prediction': 'over',
        'confidence': 0.85,
        'model_used': 'ensemble'
    }
    socket_client.emit('new_prediction', test_prediction)
    received = socket_client.get_received()
    assert len(received) > 0
    assert received[-1]['name'] == 'prediction_update'
    data = received[-1]['args'][0]
    assert data['player_name'] == 'test_player'
    assert data['prediction'] == 'over'

def test_error_handling(client):
    # Test invalid prediction request
    invalid_data = {'invalid': 'data'}
    response = client.post('/api/predict',
                          data=json.dumps(invalid_data),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

    # Test rate limiting
    for _ in range(101):  # Assuming rate limit is 100/hour
        response = client.get('/api/stats')
    assert response.status_code == 429

def test_cache_functionality(client):
    # First request should take longer and hit the database
    start_time = pytest.helpers.time.time()
    response1 = client.get('/api/stats')
    first_request_time = pytest.helpers.time.time() - start_time
    
    # Second request should be faster due to caching
    start_time = pytest.helpers.time.time()
    response2 = client.get('/api/stats')
    second_request_time = pytest.helpers.time.time() - start_time
    
    assert second_request_time < first_request_time
    assert response1.data == response2.data 