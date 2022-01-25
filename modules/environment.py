import numpy as np
from modules.trading_environment import TradingEnv, Actions
from modules.wordprocessing import WordProcessing
from collections import deque


class StocksNewsEnv(TradingEnv):

    def __init__(self, stocks_df, news_df, bao, window_size, frame_bound, initial_balance=1000):
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
        # ATENTIE: price_features stocheaza diferentele de preturi dintre zilele consecutive
        price_features = []
        for idx in range(self.frame_bound[1] - self.window_size):
            price_features.append(prices[idx + 1:idx + self.window_size] - prices[idx:idx + self.window_size - 1])
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

        #  Returnam price_features in loc de features - momentan modelul nu reuseste sa invete cu
        #  toate feature-urile.
        return prices, price_features

    def _calculate_reward(self, action):
        if self._current_tick == self._end_tick:
            return 0

        current_price = self.prices[self._current_tick]
        next_price = self.prices[self._current_tick + 1]

        #  Recompensam pozitiv/negativ agentul in functie de actiunea pe care o alege
        #  si de pretul stockului la momentul de timp urmator.
        #  ex: daca agentul decide sa vanda share-uri, iar in ziua urmatoare pretul share-urilor
        #      creste, atunci recompensa va fi -1. Daca in ziua urmatoare pretul scade, atunci
        #      recompensa va fi 1.
        #  Am observat ca aceste valori (-1, +1) se comporta bine in invatarea modelului.
        #  Am incercat si valori mai mici, si valori mai mari, iar modelul ajungea sa nu mai invete.
        if action == Actions.Buy.value:
            if next_price > current_price:
                return 1
            else:
                return -1
        else:
            if next_price > current_price:
                return -1
            else:
                return 1

    def _update_profit(self, action):
        current_price = self.prices[self._current_tick]
        final_action = -1

        # Update balance and shares
        if action == Actions.Buy.value:
            shares_to_buy = min(2, self._balance * (1 - self.trade_fee_ask_percent) / current_price)
            if shares_to_buy == 2:
                self._last_balance = self._balance
                self._balance -= current_price * shares_to_buy
                self._shares += shares_to_buy
                final_action = 1
        if action == Actions.Sell.value:
            shares_to_sell = min(2, self._shares)
            if shares_to_sell != 0:
                self._last_balance = self._balance
                self._balance += current_price * shares_to_sell
                self._shares -= shares_to_sell
                final_action = 0

        self._total_profit = (self._balance + self._shares * current_price) - self._initial_balance
        return final_action

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
