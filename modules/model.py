from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2 import optimizer_v2

import math
import numpy as np
from collections import deque

#  Same impors as before here
from environment import StocksNewsEnv


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class DeepLearningModel:
    def __init__(self, env: StocksNewsEnv):
        #  Same initializations as in our previous agent.
        self.env = env

        self.action_size = 2  # Sell, Buy
        self.state_size = 2  # Short, Long
        self.memory = deque(maxlen=1000)
        #  We'll have to finetune the ones below and see which ones give the best results.
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._model()

    def _model(self):
        model = Sequential()  # This allows us to specify the network layers in a sequential manner
        model.add(Dense(units=64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    #  To implement:
    #       -> learn(total_timesteps)

    def learn(self, total_timesteps):
        for episode in range(total_timesteps):
            state = self.env.reset()
            self.epsilon *= self.epsilon_decay
            done = False

            while not done:
                if self.epsilon > np.random.random():  # Let the agent explore by selecting a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    action = np.argmax(np.identity(self.env.observation_space.n)[state: state + 1])

                new_state, reward, done, _ = self.env.step(action)
                target = reward + self.gamma * np.max(
                    self.model.predict(np.identity(self.env.observation_space.n)[new_state: new_state + 1]))
                target_vector = self.model.predict(np.identity(self.env.observation_space.n)[state: state + 1])[0]
                target_vector[action] = target

                self.model.fit(
                    np.identity(self.env.observation_space.n)[state: state + 1],
                    target_vector.reshape(-1, self.env.action_space.n),
                    epochs=1,
                    verbose=0
                )

                state = new_state

    def predict(self, observations):
        return self.model.predict(observations)
