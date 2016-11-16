from collections import defaultdict

symbols = ['START', 'STOP', 'O', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative', 'I-negative']

def estimate_emission_params(training_data):
    """
    Takes a training data file formatted with lines like
    (word) (symbol)
    and returns a nested dictionary of emission probabilies
    where the [symbol][word] element gives the emission probability
    for that word from that symbol
    """

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

def find_symbol_estimate(dev_file, emission_probabilities):
    #using dev.in
    predicted_symbols = []
    with open(dev_file) as f:
        for line in f:
            word = line.strip()
            current_arg_max = symbols[0]
            current_max = 0
            for symbol in symbols:
                if emission_probabilities[symbol][word] > current_max:
                    current_arg_max = symbol
                    current_max = emission_probabilities[symbol][word]
            predicted_symbols.append(current_arg_max)

    return predicted_symbols

print("hi")
e = estimate_emission_params("data/EN/train")
predicted_symbols = find_symbol_estimate("data/EN/dev.in", e)
print predicted_symbols