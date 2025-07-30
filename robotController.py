import socket
import json
import threading
import cv2
import requests
import numpy as np
import io
import time

ROBOT_IP = '192.168.4.1'
ROBOT_PORT = 100

direction_map = {
    pyglet.window.key.W: 1,  # Forward
    pyglet.window.key.S: 2,  # Stop
    pyglet.window.key.A: 3,  # Turn Left
    pyglet.window.key.D: 4,  # Turn right
}

motor_map = {
    0 : [0, 0],
    1: [100, 100], 
    2: [0, 0],
    3: [120, 10],
    4: [10, 120],
}

class RobotController(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_caption("Robot Controller - Use W/A/S/D to move, Q to quit")
        self.frame_image = None
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_camera, daemon=True)
        self.stream_thread.start()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ROBOT_IP, ROBOT_PORT))
        self.pressed_keys = set()
        self.current_action = 2  # Default to stop
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.label = pyglet.text.Label(
            "Use W/A/S/D to control the robot. Press Q to quit.",
            font_size=14, x=10, y=self.height - 30
        )
    def on_key_press(self, symbol, modifiers):
        # Map key presses to actions
        if symbol == pyglet.window.key.W:
            self.current_action = 1  # Forward
        elif symbol == pyglet.window.key.S:
            self.current_action = 2  # Stop
        elif symbol == pyglet.window.key.A:
            self.current_action = 3  # Turn Left
        elif symbol == pyglet.window.key.D:
            self.current_action = 4  # Turn Right
        elif symbol == pyglet.window.key.Q:
            self.running = False
            pyglet.app.exit()
        self.pressed_keys.add(symbol)
        if symbol == pyglet.window.key.E:
            self.step(0)



    def on_key_release(self, symbol, modifiers):
        # On key release, default to stop
        if symbol in self.pressed_keys:
            self.pressed_keys.remove(symbol)
        self.current_action = 2  # Stop

    def _stream_camera(self):
        STREAM_URL = 'http://192.168.4.1:81/stream'
        r = requests.get(STREAM_URL, stream=True)
        if r.status_code != 200:
            print("Failed to connect to stream")
            return
        bytes_buffer = b""
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
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed >= 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.start_time = time.time()
                    cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channels = frame.shape
                    image_data = frame.tobytes()
                    self.frame_image = pyglet.image.ImageData(width, height, 'RGB', image_data, pitch=width * -3)

    def on_draw(self):
        self.clear()
        if self.frame_image:
            # Center the image
            x = (self.width - self.frame_image.width) // 2
            y = (self.height - self.frame_image.height) // 2
            self.frame_image.blit(x, y)
        self.label.draw()

    def update(self, dt):
        # Send command based on current_action
        self.send_command(self.current_action)

    def send_command(self, d1):
        command = {
            "N": 4,
            "D1": motor_map[d1][0],
            "D2": motor_map[d1][1],
        }
        json_command = json.dumps(command)
        self.s.send(json_command.encode('utf-8'))
        response = self.s.recv(800)

    def step(self, movement_command):
        """Send movement command, collect 4 images (monochrome), stack channel-wise, send E command, return stacked image and E response."""
        self.send_command(movement_command)
        stacked_img = self._collect_images(4)
        e_response = self._send_e_command()
        print(e_response)
        return stacked_img, e_response

    def _collect_images(self, num_images):
        """Collect num_images frames, convert to grayscale, stack along channel axis."""
        collected = []
        count = 0
        while count < num_images:
            if self.frame_image:
                img_data = self.frame_image.get_data('RGB', self.frame_image.width * 3)
                arr = np.frombuffer(img_data, dtype=np.uint8)
                arr = arr.reshape((self.frame_image.height, self.frame_image.width, 3))
                # Convert to grayscale
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                collected.append(gray)
                count += 1
                time.sleep(0.05)
        # Stack along channel axis: shape (height, width, num_images)
        stacked = np.stack(collected, axis=-1)  # shape: (600, 800, 4)
        return stacked

    def _send_e_command(self):
        """Send special E command and return robot response as parsed variables."""
        command = {"N": 6}
        json_command = json.dumps(command)
        self.s.send(json_command.encode('utf-8'))
        response = self.s.recv(800)
        response_str = response.decode()
        ax, ay, az, gx, gy, gz, sensorReading = self._parse_sensor_response(response_str)
        print(f"ax: {ax}, ay: {ay}, az: {az}, gx: {gx}, gy: {gy}, gz: {gz}, sensorReading: {sensorReading}")
        return ax, ay, az, gx, gy, gz, sensorReading

    def _parse_sensor_response(self, response_str):
        """
        Parse response string like:
        "_ok_123,456,789,10,20,30,42}"
        Returns: ax, ay, az, gx, gy, gz, sensorReading
        """
        import re
        pattern = r"_ok_(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)"
        match = re.search(pattern, response_str)
        if match:
            ax, ay, az, gx, gy, gz, sensorReading = map(int, match.groups())
            return ax, ay, az, gx, gy, gz, sensorReading
        else:
            # If parsing fails, return None for all
            return None, None, None, None, None, None, None

    def _concat_images(self, images):
        """Concatenate images side-by-side."""
        if len(images) == 0:
            return None
        return np.concatenate(images, axis=1)

    def stepTest(self, d):
        """Call step, display concatenated images with E command response overlayed."""
        # Use forward command (1) for demonstration, or modify as needed
        concat_img, e_response = self.step(d)
        if concat_img is not None:
            # Overlay E command response
            display_img = concat_img.copy()
            cv2.putText(
                display_img,
                f"E response: {e_response}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("StepTest Images", display_img)

if __name__ == "__main__":
    window = RobotController(width=800, height=600)
    pyglet.clock.schedule_interval(window.update, 0.1)  # Call update every 0.1 seconds
    pyglet.app.run()