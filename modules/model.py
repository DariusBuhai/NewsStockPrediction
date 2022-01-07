from keras.models import Sequential
from keras.layers import Dense, InputLayer

import math
import numpy as np
from collections import deque

#  Same imports as before here
from environment import StocksNewsEnv


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class DeepLearningModel:
    def __init__(self, env: StocksNewsEnv):
        #  Same initializations as in our previous agent.
        self.env = env

        self.action_size = 2  # Sell, Buy
        self.state_size = 2  # Short, Long
        self.observation_size = self.env.window_size
        self.memory = deque(maxlen=1000)
        #  We'll have to finetune the ones below and see which ones give the best results.
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._model()

    def _model(self):
        model = Sequential()  # This allows us to specify the network layers in a sequential manner
        model.add(InputLayer(batch_input_shape=(1, self.observation_size)))
        model.add(Dense(units=64, input_dim=(1, self.observation_size), activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

    #  Receives a (1, observation_size) sized array.
    def normalize(self, tensor):
        for i in range(self.observation_size):
            tensor[0][i] = sigmoid(tensor[0][i])
        return tensor

    #  Currently the observations are (2, 5) shaped arrays, but we only use the second column.
    #  This function returns the desired (1, 5) tensor with normalized values..
    def get_input_tensor(self, observation):
        input_tensor = np.reshape(observation[:, 1], (1, self.observation_size))
        input_tensor = self.normalize(input_tensor)
        return input_tensor

    def learn(self, total_timesteps):
        for episode in range(total_timesteps):
            print('Episode: ' + str(episode))

            observation = self.env.reset()
            observation = self.get_input_tensor(observation)
            self.epsilon *= self.epsilon_decay
            done = False

            while not done:
                if self.epsilon > np.random.random():
                    #  Let the agent explore by selecting a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    #  Let the agent choose the best action according to what they learned so far.
                    action = self.predict(observation)

                new_observation, reward, done, _ = self.env.step(action)
                new_observation = self.get_input_tensor(new_observation)

                target = reward
                if not done:
                    #  Sum the reward and gamma * best_expected_value(next_state)
                    #  For us, next_state == next_observation, because the model learns based on
                    #  stock value fluctuations in the past 5 days -> that is stored in the observation array
                    target = reward + self.gamma * np.amax(self.model.predict(new_observation)[0])

                target_f = self.model.predict(observation)
                target_f[0][action] = target

                self.model.fit(
                    observation,
                    target_f,
                    epochs=1,
                    verbose=0
                )

                observation = new_observation

    #  Returns the predicted action based on given observation.
    def predict(self, observations):
        return np.argmax(self.model.predict(observations)[0])
