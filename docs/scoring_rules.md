# CS2 PrizePicks Scoring Rules

## Overview
This document details the official PrizePicks scoring rules implementation for CS2 matches, including validation and special case handling.

## Basic Scoring Chart
| Action       | Points | Description |
|-------------|--------|-------------|
| AWP Kill    | 1 pt   | Kill achieved using the AWP sniper rifle |
| First Blood | 1 pt   | First kill of each round |
| Headshot    | 1 pt   | Kill achieved by hitting opponent's head hitbox |
| Kill        | 1 pt   | Any elimination of an opponent |

## Detailed Rules

### Kill Types
1. **Standard Kills**
   - Each elimination counts as 1 point
   - Multiple points can be earned for a single kill (e.g., AWP headshot = 2 points)

2. **Headshots**
   - Must be the final/killing blow
   - Must hit the uppermost portion of hitbox
   - Does not include neck/upper torso shots

3. **AWP Kills**
   - Must be achieved with the AWP sniper rifle
   - One-shot kills count the same as multiple-hit kills
   - No additional points for no-scope or quick-scope kills

4. **First Bloods**
   - Only the first kill of each round counts
   - Both teams can achieve first blood in the same round
   - Counts even if achieved through utility damage

### Special Cases

1. **Team Kills**
   - Do not result in point deductions
   - Do not count towards any scoring categories
   - Tracked separately for statistical purposes

2. **Technical Resets**
   - If a round is reset due to technical issues:
     - Only kills after the reset count
     - Previous stats in the round are discarded
     - Must be officially declared by match admin

3. **Disconnections/Substitutions**
   - Player must complete all specified maps
   - Substitutions result in DNP (Did Not Play)
   - Disconnections followed by reconnection are valid

## Match Requirements

### Map Completion
- Players must complete all specified maps
- Partial map completions are not counted
- Maps specified in projection must be played

### Match Timing
1. **Start Time**
   - Must begin within 12 hours of scheduled time
   - Time zone based on ET (Eastern Time)
   - Delays must be officially announced

2. **Postponements**
   - Within 24 hours: Projections remain active
   - Beyond 24 hours: Results in DNP
   - Rescheduled matches require new projections

### Technical Validation
1. **Stream Requirements**
   - All scoring based on official stream
   - Technical issues must be visible on stream
   - Admin decisions are final

2. **Data Validation**
   - Multiple data sources for verification
   - Automated validation of kill types
   - Manual review for disputed cases

## Implementation Details

### Validation Process
```python
def validate_match_completion(player_maps_played, required_maps):
    """Validate that player completed all required maps."""
    return all(map_name in player_maps_played for map_name in required_maps)

def validate_match_time(scheduled_time, actual_start_time):
    """Validate match timing according to PrizePicks rules."""
    time_diff = actual_start_time - scheduled_time
    return time_diff <= timedelta(hours=12)
```

### Score Calculation
```python
def calculate_score(player_stats, match_info):
    """Calculate player's score following PrizePicks rules."""
    score = CS2Score(
        awp_kills=player_stats.get('awp_kills', 0),
        first_bloods=player_stats.get('first_bloods', 0),
        headshots=player_stats.get('headshots', 0),
        kills=player_stats.get('kills', 0)
    )
    return score
```

## Error Handling

### Common Issues
1. **Data Discrepancies**
   - Multiple data source verification
   - Automated anomaly detection
   - Manual review process

2. **Technical Difficulties**
   - Clear documentation of reset events
   - Tracking of affected rounds
   - Validation of post-reset statistics

### Resolution Process
1. Monitor live data feeds
2. Validate against official stream
3. Apply technical reset rules if needed
4. Calculate final scores
5. Review edge cases manually

## Best Practices

1. **Data Collection**
   - Use multiple data sources
   - Implement real-time validation
   - Track all score components separately

2. **Score Calculation**
   - Validate all requirements first
   - Apply rules in consistent order
   - Document special case handling

3. **Quality Control**
   - Regular audits of scoring accuracy
   - Automated testing of edge cases
   - Continuous monitoring of live matches 