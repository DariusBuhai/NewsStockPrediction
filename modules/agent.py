from matplotlib import pyplot as plt

from environment import StocksNewsEnv
from news import News
from model import DeepLearningModel
import pandas as pd


class Agent(News):
    def __init__(self, stock):
        super().__init__(stock)
        self.df = pd.read_csv(f'../data/stocks/{stock}.csv')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date')
        self.env = StocksNewsEnv(self.df, frame_bound=(5, 200), window_size=5)
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
        self.env = StocksNewsEnv(self.df, frame_bound=(200, 250), window_size=5)
        self.model.update_env(self.env)
        obs, balance, shares = self.env.reset()
        while True:
            # obs = obs[np.newaxis, ...]
            obs_fixed = self.model.get_input_tensor(obs)
            obs_fixed[0][0] = balance
            obs_fixed[0][1] = shares
            action = self.model.predict(obs_fixed)
            obs, rewards, done, info = self.env.step(action)
            balance = info['balance']
            shares = info['shares']
            if done:
                print("info", info)
                break
