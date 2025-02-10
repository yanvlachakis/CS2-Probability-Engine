# CS2 PrizePicks Dashboard Documentation

## Overview
The CS2 PrizePicks Dashboard provides a real-time interface for monitoring and making predictions using the hybrid prediction system. It supports both development and production environments with appropriate configurations.

## Features
- Real-time prediction monitoring
- Model usage statistics
- Cost tracking and analysis
- Interactive prediction form
- WebSocket-based updates
- Configurable caching strategies

## Setup

### Development Environment
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export FLASK_ENV=development
export FLASK_APP=app.py
```

3. Run the development server:
```bash
flask run
```

### Production Environment
1. Install production dependencies:
```bash
pip install -r requirements.prod.txt
```

2. Configure environment:
```bash
export FLASK_ENV=production
export FLASK_APP=app.py
export SECRET_KEY=your-secure-key
```

3. Set up Redis (optional, for caching):
```bash
docker-compose up -d redis
```

4. Run with Gunicorn:
```bash
gunicorn -k eventlet -w 4 app:app
```

## Configuration

### Dashboard Config (`config/dashboard_config.json`)
- `development`: Local development settings
  - `host`: Server host (default: localhost)
  - `port`: Server port (default: 5000)
  - `debug`: Enable debug mode
  - `websocket`: WebSocket configuration
  - `cache`: Simple in-memory cache settings

- `production`: Production environment settings
  - `host`: Production host (default: 0.0.0.0)
  - `port`: Production port (default: 8000)
  - `websocket`: Production WebSocket settings with eventlet
  - `cache`: Redis-based caching configuration

## API Endpoints

### GET `/`
- Renders the main dashboard interface

### GET `/api/stats`
- Returns current system statistics
- Response includes:
  - Total predictions
  - Model usage distribution
  - Average confidence
  - Cost statistics

### POST `/api/predict`
- Makes a new prediction
- Request body:
```json
{
    "player_stats": {
        "kills": 20,
        "headshots": 10,
        "awp_kills": 5,
        "first_bloods": 2
    },
    "match_context": {
        "map_name": "dust2",
        "tournament_tier": 1,
        "stage": "playoff",
        "is_lan": true,
        "prize_pool": 1000000,
        "team_ranking_difference": 5
    }
}
```

### GET `/api/predictions`
- Returns prediction history
- Includes detailed prediction information and model selection rationale

## WebSocket Events

### `prediction_update`
- Emitted when a new prediction is made
- Payload includes:
  - Prediction value
  - Model used
  - Confidence score
  - Usage statistics

## Monitoring and Maintenance

### Health Checks
- `/health`: Basic health check endpoint
- `/metrics`: Prometheus metrics endpoint (if enabled)

### Logging
- Development: Console logging
- Production: File-based logging with rotation
- Log location: `/var/log/cs2prizepicks/dashboard.log`

### Performance Optimization
1. Redis Caching:
   - Enable in production for better performance
   - Configure cache lifetime in dashboard_config.json

2. WebSocket Optimization:
   - Adjust ping intervals based on load
   - Configure appropriate worker count

3. Database Connections:
   - Use connection pooling
   - Implement query optimization

## Security Considerations

1. Production Setup:
   - Use HTTPS
   - Set secure cookies
   - Configure CORS appropriately
   - Use strong SECRET_KEY

2. Rate Limiting:
   - API endpoints are rate-limited
   - WebSocket connections are monitored

3. Authentication:
   - Implement as needed for private deployments
   - Support for API keys

## Troubleshooting

### Common Issues
1. WebSocket Connection Failures:
   - Check CORS settings
   - Verify eventlet worker configuration
   - Ensure proper proxy settings

2. High Memory Usage:
   - Adjust cache settings
   - Monitor prediction history size
   - Check for memory leaks

3. Slow Response Times:
   - Enable Redis caching
   - Optimize database queries
   - Check network latency

## Development Guidelines

### Adding New Features
1. Create feature branch
2. Update configuration if needed
3. Add new API endpoints
4. Update documentation
5. Add tests
6. Submit pull request

### Code Style
- Follow PEP 8
- Use type hints
- Document all functions
- Add appropriate error handling 