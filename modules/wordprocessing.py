from cube.api import Cube

from modules.news import News


class WordProcessing(News):
    @staticmethod
    def getTextScore(text) -> int:
        cube = Cube(verbose=False)
        cube.load("en", device='cpu')
        document = cube(text)
        return 1

    def getBagOfWords(self):
        useless_words = ["Pfizer", "to", "for", "-", "and", ""]
        bao = dict()
        for day in self.news_per_days.keys():
            for article in self.news_per_days[day]['articles']:
                if 'title' in article:
                    for word in article['title'].split():
                        if word in useless_words or len(word) < 3:
                            continue
                        if word not in bao.keys():
                            bao[word] = 1
                        else:
                            bao[word] += 1
        words_used = list()
        for word in bao:
            words_used.append((word, bao[word]))
        words_used.sort(key=lambda x: x[1], reverse=True)
        return words_used


if __name__ == '__main__':
    # print(WordProcessing.getTextScore("Moderna has lost 10% of it's revenew"))
    word_processing = WordProcessing("PFE")
    bao = word_processing.getBagOfWords()
    print(bao[:10])
