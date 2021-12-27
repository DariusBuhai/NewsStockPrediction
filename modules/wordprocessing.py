from cube.api import Cube

if __name__ == '__main__':
    cube = Cube(verbose=True)  # initialize it
    cube.load("en", device='cpu')  # select the desired language (it will auto-download the model on first run)
    text = "This is the text I want segmented, tokenized, lemmatized and annotated with POS and dependencies."
    text2 = "Moderna has lost 13% of it's revenew"
    document = cube(text)
    print(document.sentences[0][2].upos)
