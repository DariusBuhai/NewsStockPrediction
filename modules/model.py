from collections import deque
import math
from os import path
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

import tensorflow as tf

#  Same imports as before here
from modules.environment import StocksNewsEnv


def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

#  Functia de loss este luata de aici: https://github.com/pskrunner14/trading-bot/blob/master/trading_bot/agent.py
#  Functia de loss este luata de aici: https://github.com/pskrunner14/trading-bot/blob/master/trading_bot/agent.py
def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class DeepLearningModel:
    BEST_MODEL = 'data/models/best_model.h5'

    def __init__(self, env: StocksNewsEnv):
        #  Same initializations as in our previous agent.
        self.env = env

        self.action_size = 2  # Sell, Buy
        self.state_size = 2  # Short, Long
        self.observation_size = self.env.observation_space
        self.memory = deque(maxlen=1000)
        #  We'll have to finetune the ones below and see which ones give the best results.
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.model = self._model()
        self.memory = deque(maxlen=50000)
        self.memory_y = deque(maxlen=50000)
        self.batch_size = 24

        self.model_checkpoint = ModelCheckpoint(self.BEST_MODEL, save_best_only=False, verbose=0)

    def load_best(self):
        if path.isfile(self.BEST_MODEL):
            self.model = load_model(self.BEST_MODEL)
            print('Am incarcat modelul salvat anterior!')

    def update_env(self, env_new: StocksNewsEnv):
        self.env = env_new

    def _model(self):
        model = Sequential()  # This allows us to specify the network layers in a sequential manner
        model.add(Dense(units=128, activation="relu", input_dim=self.observation_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=huber_loss, optimizer=opt, metrics=['mae', 'accuracy'])
        return model

    #  Receives a (1, observation_size) sized array.
    def normalize(self, tensor):
        #  TODO: de vazut daca sunt transformari care merita aplicate
        #  sigmoid() nu se comporta bine in cazul nostru si reteaua nu invata nimic daca o aplicam
        #  for i in range(self.observation_size):
        #    tensor[i] = sigmoid(tensor[i])
        #   tensor = keras.utils.normalize(tensor)
        return tensor

    # Converting shape (observation_size) into (1, observation_size)
    def get_input_tensor(self, observation):
        observation = self.normalize(observation)
        return observation
        # return np.array([observation])

    def learn(self, total_timesteps):
        for episode in range(total_timesteps):
            print('Episode: ' + str(episode))

            observation = self.env.reset()
            observation = self.get_input_tensor(observation)

            done = False
            num_model_decisions = 0

            while not done:
                action = 0
                if self.epsilon > np.random.random():
                    #  Let the agent explore by selecting a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    #  Let the agent choose the best action according to what they learned so far.
                    action = self.predict(np.array([observation]))
                    print(action, end="")
                    num_model_decisions += 1



                new_observation, reward, done, info = self.env.step(action)
                new_observation = self.get_input_tensor(new_observation)

                target = reward
                if not done:
                    #  Sum the reward and gamma * best_expected_value(next_state)
                    #  For us, next_state == next_observation, because the model learns based on
                    #  stock value fluctuations in the past 5 days -> that is stored in the observation array
                    target = reward + self.gamma * np.amax(self.model.predict(np.array([new_observation]))[0])

                target_f = self.model.predict(np.array([observation]))
                target_f[0][action] = target
                self.memory.append([observation, target_f])

                if len(self.memory) >= self.batch_size:
                    sample = random.sample(self.memory, self.batch_size)
                    x_train = [x[0] for x in sample]
                    y_train = [x[1] for x in sample]
                    x_train = np.array(x_train)
                    y_train = np.array(y_train)

                    self.model.fit(
                        x_train,
                        y_train,
                        epochs=1,
                        verbose=0,
                        callbacks=[self.model_checkpoint],
                        workers=8
                    )

                    self.epsilon *= self.epsilon_decay
                    if self.epsilon < self.epsilon_min:
                        self.epsilon = self.epsilon_min

                observation = new_observation
            print()
            print('Decizii luate de model: ' + str(num_model_decisions))
            self.evaluate(episode=episode)
            print()

    #  Returns the predicted action based on given observation.
    def predict(self, observations):
        return np.argmax(self.model.predict(observations)[0])

    def evaluate(self, episode=None):
        print('Evaluare:')

        observation = self.env.reset()
        observation = self.get_input_tensor(observation)
        done = False

        while not done:
            action = self.predict(np.array([observation]))
            new_observation, reward, done, info = self.env.step(action)
            print(action, end="")
            observation = self.get_input_tensor(new_observation)

            if done:
                print()
                print("info", info)

                if episode is not None:
                    self.model.save('data/models/episode_%d' % episode)
