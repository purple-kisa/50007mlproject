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

def get_observation_sequences(dev_file):
    sequences = []
    with open(dev_file) as f:
        sequence = []
        for line in f:
            if line.isspace():
                sequences.append(sequence)
                sequence = []
            else:
                word = line.strip()
                sequence.append(word)
        sequences.append(sequence)
    return sequences

def viterbi(transition_probabilities, emission_probabilities, symbol_counts, observation_sequences):
   for sequence in observation_sequences:

        # Initialize probability score and optimal symbol matrices
        n = len(sequence)
        scores = {}
        optimal_symbols = {}
        for k in range(n + 1):
            scores[k] = {}
            optimal_symbols[k] = {}
            for symbol in symbols:
                scores[k][symbol] = 0
                optimal_symbols[k][symbol] = 0

        # Set base case
        for symbol in symbols:
            scores[0][symbol]= 0
            scores[1][symbol] = transition_probabilities["START"][symbol] * emission_probability(symbol, sequence[0], emission_probabilities, symbol_counts)
            optimal_symbols[0][symbol]="START"
            optimal_symbols[1][symbol]="START"
        scores[0]["STOP"]= 0
        scores[0]["START"]= 1

        # Move forward recursively
        for k in range(2, n + 2):
            for v in symbols:

                # Get the max probability score
                if k != n + 1: # k = n + 1 is the final case
                    kth_word = sequence[k-1]
                    probabilities = [scores[k-1][u] * transition_probabilities[u][v] * emission_probability(v, kth_word, emission_probabilities, symbol_counts) for u in symbols]
                    scores[k][v] = max(probabilities)

                # Get the optimal symbol
                max_probability = 0
                for u in symbols:
                    probability = scores[k-1][u] * transition_probabilities[u][v]
                    if probability >= max_probability:
                        max_probability = probability
                        argmax = u
                if k == n + 1: # Final case
                    optimal_symbols[k] = {}
                    optimal_symbols[k]["STOP"] = argmax
                    break
                else:
                    optimal_symbols[k][v] = argmax

        # Predict symbol sequence
        predicted_symbols = ["STOP"]
        for k in range(n + 1, 0, -1):
            predicted_symbols.insert(0, optimal_symbols[k][predicted_symbols[0]])

        return predicted_symbols

def decode_file(training_data, dev_in):
    symbol_word_counts, symbol_counts = get_symbol_word_counts(training_data)
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)

    symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts(training_data)
    transition_probabilities = estimate_transition_params(symbol_symbol_counts, symbol_counts)

    observation_sequences = get_observation_sequences(dev_in)
    predicted_symbols = viterbi(transition_probabilities, emission_probabilities, symbol_counts, observation_sequences)

    print(predicted_symbols)


symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts('data/test')
#  print(symbol_symbol_counts)
#  print(symbol_counts)
#  print("TRANSITION PARAMS")
#  print(estimate_transition_params(symbol_symbol_counts, symbol_counts))

decode_file('data/test', 'data/test_dev')
