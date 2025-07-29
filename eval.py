import time
import cv2
import os
import torch
from robot_env import RobotEnv
from robot_layers import DQN
from robot_buffer import Replay_Buffer
from robot_policy import Epsilon_Greedy_Policy
from torch.utils.tensorboard import SummaryWriter

episode_running = False
stopped = False
best_reward = 0
patience_counter = 0
epoch = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

EPOCHS = 10
TARGET_UPDATE_FREQUENCY = 20
SAMPLE_SIZE = 2000
BATCH_SIZE = 100
BUFFER_SIZE = 5000
OBSERVATION_CHANNELS = 4
LEARNING_RATE = 0.00006
DECAY = 0.5
STEP_COUNT = 400

model = DQN(learning_rate=LEARNING_RATE)

# Load checkpoint if exists

# Load model and replay buffer if they exist
checkpoint_path = 'robot_final_epoch_data.pth'
buffer_path = 'robot_replay_buffer.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint.get('epoch', 0)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Resumed from epoch {epoch}.")
else:
    print("no state to evaluate")


env = RobotEnv()

env.episode_steps = 0
# Main RL loop using Gym environment

env.episode_steps = 0
done = False
state = env.reset()

while not done:
    env.render()
    # DQN decision
    # state shape: (150, 200, 4) -> (4, 150, 200)
    state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    decision = model.forward(state_tensor)

    action = torch.argmax(decision).item()
    if action == 0:
        action = torch.argmax(decision[:, 1:]).item() + 1
    next_state, reward, done, info = env.step(action)
    state = next_state
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        print("paused")
        env._send_command(0)
        time.sleep(5)
    elif key == ord('q'):
        done = True
        stopped = True
        continue
    else:
        pass

    if env.episode_steps >= STEP_COUNT:
        print("Step limit reached.")

    print(f"step: {env.episode_steps}, reward: {reward:.2f}, action: {action} \nQ: {decision}",)
    env.episode_steps += 1

env.step(0)    
