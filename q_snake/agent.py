import random
from collections import deque
import numpy as np
from neural_network import DeepNeuralNetwork


class DeepQLearningAgent:
    def __init__(self, action_space, image_shape=None, network_path=None, gamma=0.9):
        self.action_space = action_space
        self.gamma = gamma  # discount factor

        self.experiences = deque(maxlen=2000)
        self.Q = DeepNeuralNetwork(action_space, input_shape=image_shape, load_path=network_path)

    def learn(self, state, action, reward, next_state, terminal=False, alpha=0.5):
        self.experiences.append((state, action, reward, next_state))
        self.Q.train([(state, action, reward, next_state)] + random.choices(self.experiences, k=1), terminal)
        # self.Q.train([(state, action, reward, next_state)], terminal)

    def choose_action(self, state, action_space, epsilon=0.0):
        print(f'Epsilon: {epsilon}')
        if random.random() < epsilon:
            # explore
            return random.choice(action_space)
        else:
            # pick best action
            print(f'Picking action {self.action_space[np.argmax(self.Q.predict(state))]}')
            return self.action_space[np.argmax(self.Q.predict(state))]
