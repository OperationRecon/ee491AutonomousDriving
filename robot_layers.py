import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class DQN(nn.Module):
    def __init__(self, output_size=4, learning_rate=1e-5):
        super(DQN, self).__init__()
        self.output_size = output_size
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.init_weights()
        self.optimizer = torch.optim.Adam(list(self.cnn.parameters()) + list(self.fc.parameters()), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()

    def init_weights(self):
        for m in list(self.cnn.modules()) + list(self.fc.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

    def learn(self, buffer, sample_size, batch_size, target_model, discount=0.98):
        states, actions, rewards, next_states, dones = buffer.sample(sample_size)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        total_size = states.shape[0]
        total_loss = 0.0

        for i in range(0, total_size, batch_size):
            s = states[i:min(i+batch_size, total_size)]
            a = actions[i:min(i+batch_size, total_size)]
            ns = next_states[i:min(i+batch_size, total_size)]
            r = rewards[i:min(i+batch_size, total_size)]
            d = dones[i:min(i+batch_size, total_size)]

            # Current Q-values
            q_values = self.forward(s)
            predicted_q_values = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            # Double DQN target Q-values
            with torch.no_grad():
                next_actions = self.forward(ns).argmax(dim=1).unsqueeze(1)
                next_q_values = target_model.forward(ns).gather(1, next_actions).squeeze(1)
                target_q_values = r + (1 - d) * discount * next_q_values

            loss = self.loss_function(predicted_q_values, target_q_values)

            # Regularization term to constrain Q-values
            reg_loss = 1e-4 * torch.mean(torch.square(q_values))
            total = loss + reg_loss

            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += total.item()

        return total_loss / (total_size // batch_size)

    def update_target(self, target_model):
        target_model.load_state_dict(self.state_dict())
