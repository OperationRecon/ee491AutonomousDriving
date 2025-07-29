import cv2
import requests
import numpy as np
import time

STREAM_URL = 'http://192.168.4.1:81/stream'

# Start the HTTP GET request with stream=True
r = requests.get(STREAM_URL, stream=True)
if r.status_code != 200:
    print("Failed to connect to stream")
    exit()

bytes_buffer = b""
frame_count = 0
start_time = time.time()
fps = 0

for chunk in r.iter_content(chunk_size=1024):
    bytes_buffer += chunk
    a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
    b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
    if a != -1 and b != -1 and b > a:
        jpg = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Robot Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
r.close()