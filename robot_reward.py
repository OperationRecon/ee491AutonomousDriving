def calculate_reward(ax, ay, az, gx, gy, gz, sensorReading):
    # Example: reward for movement and staying on black road
    movement = abs(ax) + abs(ay) + abs(az) + abs(gx) + abs(gy) + abs(gz)
    # Assume sensorReading is high on black road, low otherwise
    road_reward = 1.0 if sensorReading > 500 else -1.0
    # Encourage movement, penalize stopping
    move_reward = movement * 0.01
    return road_reward + move_reward
