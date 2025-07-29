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

EPOCHS = 40
TARGET_UPDATE_FREQUENCY = 18
SAMPLE_SIZE = 2000
BATCH_SIZE = 400
BUFFER_SIZE = 10000
OBSERVATION_CHANNELS = 4
LEARNING_RATE = 0.0001
DECAY = 0.85
STEP_COUNT = 4000
STARTING_EPSILON = 0.8
early_stopping_patience = 35
learning_iterations = 100

def save_checkpoint(model, target_model, replay_buffer, total_reward, high_episode_reward, epoch, model_path='robot_final_epoch_data.pth', buffer_prefix='robot_replay_buffer'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'total_reward': total_reward,
        'high_total_reward': high_episode_reward,
        'epoch': epoch
    }, model_path)

    # Save only the filled portion of the buffer
    if replay_buffer.full:
        N = replay_buffer.capacity
    else:
        N = replay_buffer.position
    # Save each buffer component to a separate file
    torch.save(replay_buffer.states[:N].cpu(), f'{buffer_prefix}_states.pth')
    torch.save(replay_buffer.actions[:N].cpu(), f'{buffer_prefix}_actions.pth')
    torch.save(replay_buffer.rewards[:N].cpu(), f'{buffer_prefix}_rewards.pth')
    torch.save(replay_buffer.next_states[:N].cpu(), f'{buffer_prefix}_next_states.pth')
    torch.save(replay_buffer.dones[:N].cpu(), f'{buffer_prefix}_dones.pth')
    # Save meta info
    torch.save({
        'position': N,
        'full': replay_buffer.full,
        'capacity': replay_buffer.capacity
    }, f'{buffer_prefix}_meta.pth')
    print(f"Checkpoint saved at epoch {epoch}.")


model = DQN(learning_rate=LEARNING_RATE)
target_model = DQN()
replay_buffer = Replay_Buffer(capacity=BUFFER_SIZE)

writer = SummaryWriter(log_dir='runs/robot_training')

# Load checkpoint if exists

# Load model and replay buffer if they exist
checkpoint_path = 'robot_final_epoch_data.pth'
buffer_path = 'robot_replay_buffer.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    total_reward = checkpoint.get('total_reward', 0)
    high_episode_reward = checkpoint.get('high_total_reward', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"Resumed from epoch {epoch}.")
else:
    print("No checkpoint found. Starting fresh training.")

# Load replay buffer if available
buffer_prefix = 'robot_replay_buffer'
meta_path = f'{buffer_prefix}_meta.pth'
if all(os.path.exists(f'{buffer_prefix}_{name}.pth') for name in ['states','actions','rewards','next_states','dones','meta']):
    print(f"Loading replay buffer from {buffer_prefix}_*.pth ...")
    meta = torch.load(meta_path, map_location=device)
    N = meta.get('position', BUFFER_SIZE//2)
    replay_buffer.full = meta.get('full', False)
    print(replay_buffer.full)
    if replay_buffer.full:
        N = meta.get('capacity', N)
    replay_buffer.states[:N] = torch.load(f'{buffer_prefix}_states.pth', map_location=device)[:N]
    replay_buffer.actions[:N] = torch.load(f'{buffer_prefix}_actions.pth', map_location=device)[:N]
    replay_buffer.rewards[:N] = torch.load(f'{buffer_prefix}_rewards.pth', map_location=device)[:N]
    replay_buffer.next_states[:N] = torch.load(f'{buffer_prefix}_next_states.pth', map_location=device)[:N]
    replay_buffer.dones[:N] = torch.load(f'{buffer_prefix}_dones.pth', map_location=device)[:N]
    replay_buffer.position = N
    print(f"Replay buffer loaded with {N} samples.")
else:
    print("No replay buffer found. Starting with empty buffer.")


POLICY = Epsilon_Greedy_Policy(epsilon=(STARTING_EPSILON*DECAY)**epoch, decay=DECAY)

env = RobotEnv(max_steps=STEP_COUNT)

total_reward = 0
high_episode_reward = 0
env.episode_steps = 0
prev_state = None
last_reward = 0
done = False

# Main RL loop using Gym environment
expert_seeded = False
while epoch < EPOCHS and not stopped:
    total_reward = 0
    high_episode_reward = 0
    env.episode_steps = 0
    prev_state = None
    last_reward = 0
    done = False
    state = env.reset()
    episode_running = False
    # Expert seeding: first episode controlled by user (optional, can be removed or replaced)
    if not expert_seeded:
        print("Expert seeding: Please control the robot for this episode.")
        print("Waiting for episode. Press e to start...")
        while not episode_running:
            env.render()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                print("starting episode")
                episode_running = True
                continue
            pass

        while not done:
            # Expert decision
            env.render()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'):
                action = 1
            elif key == ord('a'):
                action = 2
            elif key == ord('d'):
                action = 3
            elif key == ord('s'):
                action = 0
            elif key == ord('r'):
                print("Paused")
                env._send_command(0)
                time.sleep(5)
            elif key == ord('q'):
                print('ending..')
                done = True
                env.episode_steps = STEP_COUNT
            else:
                action = 0

            next_state, reward, _, info = env.step(action)
            # state shape: (150, 200, 4) -> (4, 150, 200)
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            total_reward += reward
            high_episode_reward = max(high_episode_reward, total_reward)
            if prev_state is not None:
                replay_buffer.add(prev_state.cpu().numpy(), action, reward, state_tensor.cpu().numpy(), done) if prev_state is not None else None
            prev_state = state_tensor.clone()
            state = next_state
            env.episode_steps += 1
            last_reward = reward
            print(f"step: {env.episode_steps}, reward: {reward:.2f}, action: {action}, idx: {replay_buffer.position % replay_buffer.capacity}",)
        expert_seeded = True
        print(f"expert seeding done:\navgr reward:{total_reward/env.episode_steps}")
        continue
    done = False
    env.episode_steps = 0
    print(f"episode: {epoch} starting...")
    while not done:
        use_next = True
        env.render()
        # DQN decision
        # state shape: (150, 200, 4) -> (4, 150, 200)
        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        if POLICY.select_action():
            decision = model.forward(state_tensor)
        else:
            decision = torch.rand(model.output_size)
        action = torch.argmax(decision).item()
        next_state, reward, done, info = env.step(action)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            print("Episode ended manually\n")
            reward = -0.98
            done = True
            env.episode_steps = STEP_COUNT
        elif key == ord('r'):
            print("pause")
            use_next = False
            env._send_command(0)
            time.sleep(5)
        elif key == ord('q'):
            done = True
            stopped = True
            continue
        else:
            pass
        total_reward += reward
        high_episode_reward = max(high_episode_reward, total_reward)

        if total_reward - high_episode_reward < -50:
            print("Early stopping episode due to reward drop.")
            done = True
        if env.episode_steps >= STEP_COUNT:
            print("Step limit reached.")

        if prev_state is not None:
            replay_buffer.add(prev_state.cpu().numpy(), action, reward, state_tensor.cpu().numpy(), done or not use_next) if prev_state is not None else None
        prev_state = state_tensor.clone()

        print(f"step: {env.episode_steps}, reward: {reward:.2f}, action: {action} \nQ: {decision}, idx: {replay_buffer.position % replay_buffer.capacity}",)
        state = next_state
        env.episode_steps += 1
        last_reward = reward
    POLICY.update_epsilon()
    env.step(0)    
    # Learning after episode
    save_checkpoint(model, target_model, replay_buffer, total_reward, high_episode_reward, epoch)
    for i in range(learning_iterations):
        loss = model.learn(replay_buffer, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, target_model=target_model)
        writer.add_scalar('Loss/model', loss, epoch * learning_iterations + i)
        print(f"Learning... {i}/{learning_iterations}", end='\r',flush=True)

        if i % TARGET_UPDATE_FREQUENCY == 0 and i > 0:
            model.update_target(target_model)

        if i % (learning_iterations//4) == 0 and i > 0:
            save_checkpoint(model, target_model, replay_buffer, total_reward, high_episode_reward, epoch)

    writer.add_scalar('Reward/total_reward', total_reward/env.episode_steps if env.episode_steps > 0 else 0, epoch)
    print(f"\nEpoch {epoch} - Total Reward: {total_reward/env.episode_steps if env.episode_steps > 0 else 0}")
    epoch += 1
    if total_reward > best_reward:
        best_reward = total_reward
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter > early_stopping_patience:
        print("Early stopping triggered.")
        break
    if epoch % 1 == 0:
        expert_seeded = False
        save_checkpoint(model, target_model, replay_buffer, total_reward, high_episode_reward, epoch)

# Final save
save_checkpoint(model, target_model, replay_buffer, total_reward, high_episode_reward, epoch)
writer.close()
