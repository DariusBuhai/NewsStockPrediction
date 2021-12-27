import json

import requests
from datetime import date, timedelta
import csv

NEWS_URI = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "2fdb18e9297a437ab91a7ec7669bed35"
NEWS_SOURCES = ""


class News:
    @staticmethod
    def getNews(keywords="", interval=None):
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

        # extracting data in json format
        return r.json()

    @staticmethod
    def categorizeNewsPerDays(keywords="", interval=None):
        if not interval:
            interval = (date.today(), date.today())
        categorized_news = dict()
        current_day = interval[1]
        while current_day >= interval[0]:
            categorized_news[str(current_day)] = News.getNews(keywords, (current_day, current_day))
            current_day -= timedelta(days=1)
        return categorized_news

    @staticmethod
    def saveNews(stockname, filename):
        news_per_day = News.categorizeNewsPerDays(stockname, (date.today() - timedelta(days=365), date.today()))
        filepath = f"data/news/{filename}"
        with open(filepath, "w") as f:
            f.write(json.dumps(news_per_day))


if __name__ == '__main__':
    News.saveNews("Pfizer", "PFE.json")
