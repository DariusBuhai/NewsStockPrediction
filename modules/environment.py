import numpy as np
from modules.trading_environment import TradingEnv, Actions
from modules.wordprocessing import WordProcessing


class StocksNewsEnv(TradingEnv):

    def __init__(self, stocks_df, news_df, bao, window_size, frame_bound, initial_balance=100):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(stocks_df=stocks_df, news_df=news_df, bao=bao, window_size=window_size,
                         initial_balance=initial_balance)

        self.trade_fee_bid_percent = 0.01
        self.trade_fee_ask_percent = 0.005

    # TODO: Adapt and modify features
    # Current features: (36) -> [price - size(5), bag of words - size(30), sentimental analysis = size(2)]
    def _process_data(self):
        # Prices
        prices = self.stocks_df.loc[:, 'Close'].to_numpy()
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        # Calculate price features for each trade day
        price_features = []
        for idx in range(self.frame_bound[1] - self.window_size):
            price_features.append(prices[idx:idx + self.window_size])
        price_features = np.array(price_features)

        # Articles for each day
        trade_days = [x.strftime("%Y-%m-%d") for x in
                      list(self.stocks_df.loc[:, 'Date'][self.frame_bound[0]:self.frame_bound[1]])]
        articles = []
        for trade_day in trade_days:
            if trade_day in self.news_df.keys():
                articles.append(self.news_df[trade_day]['articles'])
            else:
                articles.append([])
        article_features = []

        for day in articles:
            words_usages, sentimental_analysis = np.zeros(30), np.zeros(2)
            for article in day:
                words_usages = np.add(words_usages, WordProcessing.getWordsUsages(self.bao, article))
                sentimental_analysis = np.add(sentimental_analysis, WordProcessing.getSentimentalAnalysis(article))
            article_features.append(np.concatenate((words_usages, sentimental_analysis)))
        article_features = np.array(article_features)
        # Concatenate features
        features = np.concatenate((price_features, article_features), axis=1)

        return prices, features

    def _calculate_reward(self, action):
        if action == Actions.Sell.value:
            return self._balance - self._last_balance
        else:
            return 0

    def _update_profit(self, action):

        current_price = self.prices[self._current_tick]
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
        profit = 0

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1

        return profit
