import sys

from modules.news import News

sys.path.append('/home/dariusbuhai/python/lib/python3.9/site-packages')

# Gym stuff
import gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Agent(News):
    def __init__(self, stock):
        super().__init__(stock)
        self.df = pd.read_csv(f'../data/stocks/{stock}.csv')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        env_maker = lambda: gym.make('stocks-v0', df=self.df, frame_bound=(5, 500), window_size=5)
        self.env = DummyVecEnv([env_maker])

    def train(self):
        # TODO: implement custom stable_baselines3 like agent but with news rewards
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", self.env, verbose=1)
        model.learn(total_timesteps=100000)
