import numpy as np
from trading_environment import TradingEnv, Actions


# TODO: Edit this environment and add news rewards
class StocksNewsEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, initial_balance=100):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, initial_balance)

        self.trade_fee_bid_percent = 0.01
        self.trade_fee_ask_percent = 0.005

    # def update_frame_bounds(self, frame_bound):
    #     self.frame_bound = frame_bound
    #     super().__init__()

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        if action == Actions.Sell.value:
            return self._balance - self._last_balance
        else:
            return 0

    def _update_profit(self, action):

        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        current_share = min(1, self._balance * (1 - self.trade_fee_ask_percent) / current_price)

        # Update balance and shares
        if action == Actions.Buy.value:
            self._last_balance = self._balance
            self._balance -= current_price * current_share
            self._shares += current_share
        if action == Actions.Sell.value:
            self._balance += current_price * self._shares
            self._shares = 0

        self._total_profit = (self._balance + self._shares * current_price) - self._initial_balance

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                # position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                # position = Positions.Long

            # if position == Positions.Long:
            #     current_price = self.prices[current_tick - 1]
            #     last_trade_price = self.prices[last_trade_tick]
            #     shares = profit / last_trade_price
            #     profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
