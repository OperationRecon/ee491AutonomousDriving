import gym
import numpy as np
import socket
import json
import cv2
import requests
import time
from robot_reward import calculate_reward

ROBOT_IP = '192.168.4.1'
ROBOT_PORT = 100

motor_map = {
    0: [0, 0],        # Stop
    1: [100, 100],    # Forward
    2: [120, 10],     # Turn Left
    3: [10, 120],     # Turn Right
}

def center_reward(ir_val):
    ir_clipped = np.clip(ir_val, 700, 900)
    normalized = (ir_clipped - 700) / (900 - 700)  # Range [0,1]
    return 1.0 - (1.0 - normalized) ** 2  # Smooth decay near edges

def turning_penalty(gz):
    turn_intensity = abs(gz) / 4094  # Normalize if known
    return -0.02 * turn_intensity

def acceleration_reward(ax):
    forward_bias = max(ax, 0) / 1080 - 0.6  # Normalize to [0,1]
    return 0.8 * forward_bias


class RobotEnv(gym.Env):
    """
    Gym environment for robot control and RL training.
    Actions: 0=stop, 1=forward, 2=stop, 3=left, 4=right
    Observation: stacked grayscale images (shape: [H, W, 4])
    Reward: calculated from sensor readings
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=6000):
        super(RobotEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(150, 200, 4), dtype=np.uint8
        )
        self.frame_image = None
        self.running = True
        import threading
        self.stream_thread = threading.Thread(target=self._stream_camera, daemon=True)
        self.stream_thread.start()
        time.sleep(1.0)  # Give the stream thread time to start and buffer frames
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ROBOT_IP, ROBOT_PORT))
        self.last_obs = None
        self.last_info = None
        self.episode_steps = 0
        self.max_steps = max_steps

    def _stream_camera(self):
        STREAM_URL = 'http://192.168.4.1:81/stream'
        r = requests.get(STREAM_URL, stream=True)
        if r.status_code != 200:
            print("Failed to connect to stream")
            self.running = False
            return
        bytes_buffer = b""
        while self.running:
            for chunk in r.iter_content(chunk_size=1024):
                if not self.running:
                    break
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                if a != -1 and b != -1 and b > a:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.frame_image = gray

    def _get_frame(self):
        # Return the latest frame from the background thread
        if self.frame_image is not None:
            return self.frame_image.copy()
        return None

    def _collect_images(self, num_images=4):
        collected = []
        while len(collected) < num_images:
            img = self._get_frame()
            if img is not None:
                # Resize to (150, 200)
                img = cv2.resize(img, (200, 150), interpolation=cv2.INTER_AREA)
                collected.append(img)
                time.sleep(0.05)
        stacked = np.stack(collected, axis=-1)
        return stacked

    def _send_command(self, action):
        command = {
            "N": 4,
            "D1": motor_map[action][0],
            "D2": motor_map[action][1],
        }
        json_command = json.dumps(command)
        self.s.send(json_command.encode('utf-8'))
        response = self.s.recv(800)
        return response

    def _send_e_command(self):
        command = {"N": 6}
        json_command = json.dumps(command)
        self.s.send(json_command.encode('utf-8'))
        response = self.s.recv(800)
        response_str = response.decode()
        return self._parse_sensor_response(response_str)

    def _parse_sensor_response(self, response_str):
        import re
        pattern = r"_ok_(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)"
        match = re.search(pattern, response_str)
        if match:
            ax, ay, az, gx, gy, gz, sensorReading = map(int, match.groups())
            return ax, ay, az, gx, gy, gz, sensorReading
        else:
            return None, None, None, None, None, None, None

    def reset(self):
        self.episode_steps = 0
        obs = self._collect_images(4)
        self.last_obs = obs
        self.last_info = self._send_e_command()
        return obs

    def step(self, action):
        self._send_command(action)
        obs = self._collect_images(4)
        info = self._send_e_command()
        self.episode_steps += 1
        done = self.episode_steps >= self.max_steps
        reward = self._calculate_reward(action, info)
        self.last_obs = obs
        self.last_info = info
        return obs, reward, done, {"info": info}

    def _calculate_reward(self, action, sensor_vals):
        ax, ay, az, gx, gy, gz, ir_val = sensor_vals

        # Movement encouragement
        move_r = 0.2 if action != 0 else -0.2

        # IR-based track alignment
        ir_r = center_reward(ir_val)

        # Turn stability
        turn_r = turning_penalty(gz)

        turn_r = 0.01* turn_r if turn_r > 0 and ir_val < 500 else turn_r
        # Optional: acceleration-based bonus
        acc_r = acceleration_reward(ax) 

        acc_r =  0.01 * acc_r if acc_r > 0  and ir_val < 500 else acc_r

        # Smoothness penalty (discourage repeated action)
        repeat_pen = -0.05 if getattr(self, 'last_action', -1) == action else 0
        self.last_action = action
        
        total_r = move_r + ir_r + turn_r + acc_r + repeat_pen
        return min(total_r, 0.98) if total_r > 0 else max(total_r, -0.98)


    def render(self, mode='human'):
        # Optionally show images using cv2.imshow
        if self.last_obs is not None:
            img = self.last_obs[..., 0]
            cv2.imshow("RobotEnv", img)

    def close(self):
        self.s.close()
        if hasattr(self, 'r'):
            self.r.close()
        cv2.destroyAllWindows()
