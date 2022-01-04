import json

import requests
from datetime import date, timedelta

from modules.wordprocessing import WordProcessing

NEWS_URI = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "2fdb18e9297a437ab91a7ec7669bed35"
NEWS_SOURCES = ""


class News(WordProcessing):

    def __init__(self, stock):
        self.stock = stock
        self.loadNews()

    @staticmethod
    def getNewsFromApi(keywords="", interval=None):
        if not interval:
            interval = (date.today(), date.today())
        # defining a params dict for the parameters to be sent to the API
        PARAMS = {'apiKey': NEWS_API_KEY,
                  'qInTitle': keywords,
                  'language': "en",
                  'sortBy': 'relevancy',
                  'from': str(interval[0]),
                  'to': str(interval[1])}

        # sending get request and saving the response as response object
        r = requests.get(url=NEWS_URI, params=PARAMS)

        # extracting data
        return r.json()

    @staticmethod
    def categorizeNewsPerDays(keywords="", interval=None):
        if not interval:
            interval = (date.today(), date.today())
        categorized_news = dict()
        current_day = interval[1]
        print("Loading news: ")
        while current_day >= interval[0]:
            categorized_news[str(current_day)] = News.getNewsFromApi(keywords, (current_day, current_day))
            current_day -= timedelta(days=1)
            print("#", end="")
        print("\nDone.")
        return categorized_news

    @staticmethod
    def saveNews(keyword, stock):
        news_per_day = News.categorizeNewsPerDays(keyword, (date.today() - timedelta(days=30), date.today()))
        filepath = f"../data/news/{stock}.json"
        with open(filepath, "w") as f:
            f.write(json.dumps(news_per_day))
        print(f"News saved to {filepath}")

    def loadNews(self):
        # TODO: Process each news using wordprocessing
        filepath = f"../data/news/{self.stock}.json"
        with open(filepath, "r") as r:
            news_per_days = json.loads(r.read())
        # for day in news_per_days.keys():
        #     for article in news_per_days[day]['articles']:
        #         article['score'] = self.getTextScore(article['content'])
        return news_per_days


if __name__ == '__main__':
    News.saveNews("UiPath", "PATH")
