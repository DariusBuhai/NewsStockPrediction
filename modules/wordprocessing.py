from cube.api import Cube


class WordProcessing:
    @staticmethod
    def getTextScore(text) -> int:
        cube = Cube(verbose=False)
        cube.load("en", device='cpu')
        document = cube(text)
        return 1


if __name__ == '__main__':
    print(WordProcessing.getTextScore("Moderna has lost 10% of it's revenew"))
