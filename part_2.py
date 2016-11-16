from collections import defaultdict

def estimate_emission_params(training_data):
    """
    Takes a training data file formatted with lines like
    (word) (symbol)
    and returns a nested dictionary of emission probabilies
    where the [symbol][word] element gives the emission probability
    for that word from that symbol
    """

    symbols = ['START', 'STOP', 'O', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative', 'I-negative']

    # initialize dictionary of counts
    symbol_word_counts = {}
    for symbol in symbols:
        symbol_word_counts[symbol] = defaultdict(int)

    with open(training_data) as f:
        symbol = 'START'
        for line in f:
            if line.isspace():
                continue
            word = line.split(' ')[0].strip()
            symbol_word_counts[symbol][word] += 1
            symbol = line.split(' ')[-1].strip()

    # find total symbol counts
    symbol_counts = {}
    for symbol in symbol_word_counts:
        symbol_counts[symbol] = sum(symbol_word_counts[symbol].values())

    # return nested dictionary of emission probabilities
    emission_probabilities = {}
    for symbol in symbol_word_counts:
        emission_probabilities[symbol] = defaultdict(lambda: 1/(symbol_counts[symbol] +1))
        for word in symbol_word_counts[symbol]:
            emission_probabilities[symbol][word] = float(symbol_word_counts[symbol][word])/symbol_counts[symbol] + 1

    return emission_probabilities
