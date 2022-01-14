import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Sell = 0
    Buy = 1


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stocks_df, news_df, bao, window_size, initial_balance=100):
        assert stocks_df.ndim == 2

        self.seed()
        self.stocks_df = stocks_df
        self.news_df = news_df
        self.bao = bao
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._profit_history = None
        self._total_reward = 0
        self._total_profit = 0
        self._first_rendering = None
        self.history = None

        self._balance = initial_balance
        self._last_balance = initial_balance
        self._initial_balance = initial_balance
        self._shares = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._profit_history = (self.window_size * [0]) + [0]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick or (self._balance == 0 and self._shares == 0):
            self._done = True

        self._update_profit(action)
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._profit_history.append(self._total_profit)
        observation, balance, shares = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            shares=shares,
            balance=balance
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        observation = self.signal_features[(self._current_tick - self.window_size):self._current_tick]
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick], self._balance, self._shares

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position > 0:
                color = 'green'
            elif position <= 0:
                color = 'red'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._profit_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._profit_history))
        plt.plot(self.prices)

        good_ticks = []
        bad_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._profit_history[i] > 0:
                good_ticks.append(tick)
            elif self._profit_history[i] <= 0:
                bad_ticks.append(tick)

        plt.plot(good_ticks, self.prices[good_ticks], 'go')
        plt.plot(bad_ticks, self.prices[bad_ticks], 'ro')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
