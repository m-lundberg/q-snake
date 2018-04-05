import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten


class DeepNeuralNetwork:
    def __init__(self, action_space, input_shape=None, load_path=None):
        self.action_space = action_space
        self.input_shape = input_shape

        if load_path:
            self.model = self.load(load_path)
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (8, 4), activation='relu', input_shape=input_shape))
            self.model.add(Conv2D(64, (4, 2), activation='relu'))
            self.model.add(Conv2D(64, (3, 1), activation='relu'))
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(len(action_space)))

            self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'acc'])

    def predict(self, state):
        state = state.reshape(1, *self.input_shape)
        return self.model.predict(state, batch_size=1)

    def train(self, batch, next_state_terminal=False, gamma=0.9):
        """
        :param batch: [(s, a, r, s'), ...]
        :param next_state_terminal: whether s' is a terminal state or not
        :param gamma: discount factor for future (estimated) rewards
        :return: 
        """
        # TODO: Make batch elements namedtuple

        x = [s[0].astype(np.float64) for s in batch]
        y = []

        for sample in batch:
            # pick max Q-value for next state
            next_q = np.max(self.predict(sample[3].astype(np.float64)).ravel())

            t = self.predict(sample[0])

            if next_state_terminal:
                t[0][self.action_space.index(sample[1])] = sample[2]
            else:
                t[0][self.action_space.index(sample[1])] = sample[2] + gamma * next_q

            y.append(t)

        x = np.asarray(x)
        x = x.reshape(x.shape[0], *self.input_shape)

        y = np.asarray(y).squeeze()

        self.model.fit(x, y, batch_size=32)

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load(filepath):
        return load_model(filepath)
