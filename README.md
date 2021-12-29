# News Stock Prediction

## Team
[Buhai Darius](https://github.com/DariusBuhai) and [Andrei Arnautu](https://github.com/andreiarnautu)

## Project description
Predict and learn how to trade based on stock prices and news analytics.

## Implementation

 - News Analysis:
   - News are brought from [Google News API](https://newsapi.org/s/google-news-api) and stored locally (due to api limitations);
   - We use [NLP Cube](https://github.com/adobe/NLP-Cube) to extract words features;
   - We use [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) in order to classify news and determine their impact on stocks;
 - Stocks Analysis:
   - Stocks are brought from Yahoo Finance and stored locally as CSV files; 
   - We use Gym in order to generate our environment (Default environment: **stocks-v0**);
   - We use Stable Baseline (V3) in order to train our model (MlpPolicy);

## Algorithm
**TO BE DESCRIBED**

## Results and Plots
**TO BE DESCRIBED**

## Project Requirements (RO):

Cerintele postate sunt orientative. Va puteti alege din lista data sau puteti veni cu alte idei,
dar care trebuie discutate in prealabil la curs/laborator. Tema aleasa trebuie sa aiba o
aplicabilitate in industrie.

### In prezentarea proiectului, este necesar:
- Sa aveti algoritmul, pe baza caruia vom discuta;
- Rezultatele algoritmului, metrici, grafice;
- Un document care sa explice sumar ideea proiectului;
