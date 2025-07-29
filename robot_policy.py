import random

class Epsilon_Greedy_Policy:
    def __init__(self, epsilon=1.0, decay=0.995, min_epsilon=0.05):
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    def select_action(self):
        return random.random() > self.epsilon

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
