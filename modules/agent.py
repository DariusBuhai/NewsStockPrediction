from matplotlib import pyplot as plt

from environment import StocksNewsEnv
from news import News
from model import DeepLearningModel

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv

# Processing libraries
import numpy as np
import pandas as pd


class Agent(News):
    def __init__(self, stock):
        super().__init__(stock)
        self.df = pd.read_csv(f'../data/stocks/{stock}.csv')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date')
        self.env = StocksNewsEnv(self.df, frame_bound=(5, 500), window_size=5)
        self.vec_env = DummyVecEnv([lambda: self.env])
        self.model = DeepLearningModel(self.env)

    def test(self):
        self.env.action_spacestate = self.env.reset()
        while True:
            action = self.env.action_space.sample()
            n_state, reward, done, info = self.env.step(action)
            if done:
                print("info", info)
                break
        self.show()

    def show(self):
        plt.figure(figsize=(15, 6))
        plt.cla()
        self.env.render_all()
        plt.show()

    def train(self, verbose=1, steps=100000):
        self.model.verbose = verbose
        self.model.learn(total_timesteps=steps)

    def evaluate(self):
        obs = self.env.reset()
        while True:
            obs = obs[np.newaxis, ...]
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            if done:
                print("info", info)
                break
