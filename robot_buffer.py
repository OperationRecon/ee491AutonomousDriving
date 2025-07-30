import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Replay_Buffer():
    def __init__(self, state_shape=[4, 150, 200], capacity=3000):
        self.capacity = capacity
        self.position = 0
        self.full = False

        # Pre-allocate buffers directly on device to avoid .to(device) calls
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        idx = (self.position % self.capacity) -1
        try:
            self.states[idx -1] = torch.from_numpy(state).to(device)
            self.actions[idx -1] = torch.tensor(action, dtype=torch.int64, device=device)
            self.rewards[idx -1] = torch.tensor(reward, dtype=torch.float32, device=device)
            self.next_states[idx -1] = torch.from_numpy(next_state).to(device)
            self.dones[idx - 1] = torch.tensor(done, dtype=torch.float32, device=device)

            self.position += 1
            self.full = self.full or self.position >= self.capacity
        except IndexError: 
            print(idx)
            idx = 0
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.position
        batch_size = min(batch_size, max_index)

        # Use torch.randint for GPU-side sampling
        idx = torch.randint(0, max_index, (batch_size,), device=device)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )
