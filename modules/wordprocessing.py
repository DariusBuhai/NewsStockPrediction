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
        useless_words = [self.KEYWORDS[self.stock].lower(), self.stock.lower(), "to", "for", "-", "and", "", "the", "are", "has", "chars]", "that", "from", "with", "have", "its", "was", "(Reuters)"]
        bao = dict()
        for day in self.news_per_days.keys():
            for article in self.news_per_days[day]['articles']:
                if 'content' in article:
                    for word in article['content'].split():
                        if word.lower() in useless_words or len(word) < 3:
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

    def getMostUsedWords(self, limit=30):
        words_used = self.getBagOfWords()
        return [x[0] for x in words_used[:limit]]


if __name__ == '__main__':

    word_processing = WordProcessing("PFE")
    bao = word_processing.getBagOfWords()
    print(bao[:30])
