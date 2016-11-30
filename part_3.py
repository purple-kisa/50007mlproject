from part_2 import symbols, get_symbol_word_counts, estimate_emission_params, emission_probability
from collections import defaultdict

def get_symbol_symbol_counts(training_data):
    """
    Takes a training data file formatted with lines like
    (word) (symbol)
    and returns a tuple of two dictionaries
    The first is a nested dictionary of symbol-to-symbol transition counts
    where the value of d[symbol1][symbol2] is the counts of
    (word) (symbol1)
    (word) (symbol2)
    The second is a dictionary of symbol counts
    where the value of d[symbol] is the total count of the symbol
    """

    symbol_symbol_counts = { 'START': {} }
    for symbol1 in symbols:
        symbol_symbol_counts[symbol1] = { 'STOP': {} }
        symbol_symbol_counts[symbol1]['STOP'] = 0
        for symbol2 in symbols:
            symbol_symbol_counts['START'][symbol2] = 0
            symbol_symbol_counts[symbol1][symbol2] = 0

    prev_symbol = 'START'

    with open(training_data) as f:
        for line in f:
            if line.isspace():
                continue
            symbol = line.split(' ')[-1].strip()
            symbol_symbol_counts[prev_symbol][symbol] += 1
            prev_symbol = symbol

    symbol_symbol_counts[prev_symbol]['STOP'] += 1

    symbol_counts = {}
    for symbol in symbol_symbol_counts:
        symbol_counts[symbol] = sum(symbol_symbol_counts[symbol].values())

    return symbol_symbol_counts, symbol_counts

def estimate_transition_params(symbol_symbol_counts, symbol_counts):
    """
    Returns a nested dictionary of transition probabilities
    where the value of d[symbol1][symbol2] is the probability
    of transitioning from symbol1 to symbol2
    """

    transition_probabilities = {}
    for symbol1 in symbol_symbol_counts:
        transition_probabilities[symbol1] = {}
        for symbol2 in symbol_symbol_counts[symbol1]:
            if symbol_counts[symbol1] == 0:
                transition_probabilities[symbol1][symbol2] = 0
            else:
                transition_probabilities[symbol1][symbol2] = float(symbol_symbol_counts[symbol1][symbol2])/symbol_counts[symbol1]

    return transition_probabilities

def viterbi(transition_probabiltiies, emission_probabilities, dev_file):
    tweets = []
    with open(dev_file) as f:
        tweet = []
        for line in f:
            if line!="":
                word = line.strip()
                tweet.append(word)
            else:
                tweets.append(tweet)
                tweet=[]

    for tweet in tweets:
        n = len(tweet)
        scores = [[0 for i in range(len(symbols) + 2)] for j in range(n)]
        optimal_symbols = [[0 for i in range(len(symbols) + 2)] for j in range(n)]
        for symbol in symbols:
            scores[0][symbol]=0
            optimal_symbols[0][symbol]="START"
        scores[0]["STOP"]=0
        scores[0]["START"]=1


    for k in range(1,n+1):
        for v in range(len(symbols)):
            u = scores[k-1].index(max(scores[k-1]))
            scores[k][v] = max(scores[k-1]) * transition_probabiltiies[u][v] * emission_probability[v][]



symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts('data/test')
print(symbol_symbol_counts)
print(symbol_counts)
print("TRANSITION PARAMS")
print(estimate_transition_params(symbol_symbol_counts, symbol_counts))
