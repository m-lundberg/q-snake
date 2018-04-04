import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


class DeepNeuralNetwork:
    def __init__(self, actions):
        self.model = Sequential()

        self.model.add(Conv2D(32, 8, 4, activation='relu', input_shape=(30, 30, 1)))
        self.model.add(Conv2D(64, 4, 2, activation='relu'))
        self.model.add(Conv2D(64, 3, 1, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(actions))

        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'acc'])

    def predict(self, state):
        return self.model.predict(state, batch_size=1)

    def train(self, batch):
        """
        :param batch: [(s, a, r, s'), ...]
        :return: 
        """
        # TODO: Make batch namedtuple
        x, y = [], []

        for sample in batch:
            x += [sample[0].astype(np.float64)]

            next_q = np.max(self.predict(sample[3].astype(np.float64)).ravel())


