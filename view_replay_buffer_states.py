import torch
import matplotlib.pyplot as plt
import numpy as np

# Path to the saved replay buffer states file
states_path = 'robot_replay_buffer_states.pth'

# Load the states tensor
states = torch.load(states_path, map_location='cpu')

print(f"Loaded states shape: {states.shape}")

# Number of states to visualize
num_samples = min(10, len(states))

for i in range(1,num_samples):
    state = states[-i]
    # If state is numpy array, convert to tensor
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    # If state shape is (4, H, W), stack channels width-wise: (H, W*4)
    if state.shape[0] == 4:
        # Normalize or clip for visualization if needed
        imgs = [state[c].numpy() for c in range(4)]
        stacked_img = np.hstack(imgs)
        plt.figure(figsize=(12, 4))
        plt.title(f'State {i} (channels stacked width-wise)')
        plt.imshow(stacked_img.astype(np.uint8), cmap='gray')
    else:
        img = state.numpy()
        plt.figure(figsize=(6, 4))
        plt.title(f'State {i}')
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
