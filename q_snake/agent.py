import random
from collections import deque
from q_snake.neural_network import DeepNeuralNetwork


class DeepQLearningAgent:
    def __init__(self, actions, gamma=0.9):
        self.gamma = gamma  # discount factor

        self.experiences = deque(maxlen=2000)
        self.Q = DeepNeuralNetwork(actions)

    def learn(self, state, action, reward, next_state, alpha=0.5):
        # use QLearning update rule to learn
        # self.Q[state][action] = \
        #     (1 - alpha) * self.Q[state][action] \
        #     + alpha * (reward + self.gamma * max(self.Q[next_state]))
        pass

    def choose_action(self, state, action_space, epsilon=0.0):
        if random.random() < epsilon:
            # explore
            return random.choice(action_space)
        else:
            # pick best action
            print(f'Picking action {self.Q.predict(state)}')
            return self.Q.predict(state)
