import math
import sys
from part_2 import symbols, get_symbol_word_counts, estimate_emission_params, emission_probability
from collections import defaultdict

def log(x):
    if x == 0:
        return math.log(sys.float_info.min)
    else:
        return math.log(x)

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

    with open(training_data, encoding="utf8") as f:
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

def get_transition_probabilities(training_file):
    symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts(training_file)
    return estimate_transition_params(symbol_symbol_counts, symbol_counts)

def get_observation_sequences(dev_file):
    sequences = []
    with open(dev_file, encoding="utf8") as f:
        sequence = []
        for line in f:
            if line.isspace():
                if sequence!=[]:
                    sequences.append(sequence)
                sequence = []
            else:
                word = line.strip()
                sequence.append(word)
        if sequence!=[]:
            sequences.append(sequence)
    return sequences

def viterbi(transition_probabilities, emission_probabilities, symbol_counts, observation_sequences):
    all_predicted_symbols = []
    for sequence in observation_sequences:

        # Initialize probability score and optimal symbol matrices
        n = len(sequence)
        scores_and_previous_symbols = {}
        scores_and_previous_symbols[n+1] = {}
        for k in range(n + 1):
            scores_and_previous_symbols[k] = {}
            for symbol in symbols:
                scores_and_previous_symbols[k][symbol] = []


        # Set base case
        for symbol in symbols:
            scores_and_previous_symbols[0][symbol]= (0, "NA")
            scores_and_previous_symbols[1][symbol] = (log(transition_probabilities["START"][symbol]) + log(emission_probability(symbol, sequence[0], emission_probabilities, symbol_counts)), "START")
            #  optimal_symbols[0][symbol]="START"
            #  optimal_symbols[1][symbol]="START"
        scores_and_previous_symbols[0]["STOP"]= (0, "NA")
        scores_and_previous_symbols[0]["START"]= (1, "NA")

        # Move forward recursively
        for k in range(2, n + 1):
            for v in symbols:
                # Get the max probability score
                kth_word = sequence[k-1]
                probabilities_and_previous_symbols = [(scores_and_previous_symbols[k-1][u][0] + log(transition_probabilities[u][v]) + log(emission_probability(v, kth_word, emission_probabilities, symbol_counts)), u) for u in symbols]
                scores_and_previous_symbols[k][v] = max(probabilities_and_previous_symbols, key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0])

        probabilities_and_previous_symbols = [(scores_and_previous_symbols[n][u][0] + log(transition_probabilities[u]["STOP"]), u) for u in symbols]
        scores_and_previous_symbols[n+1]["STOP"] = max(probabilities_and_previous_symbols, key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0])

        # Predict symbol sequence
        predicted_symbols = ["STOP"]
        for k in range(n + 1, 0, -1):
            predicted_symbols.insert(0, scores_and_previous_symbols[k][predicted_symbols[0]][1])

        all_predicted_symbols.append(predicted_symbols)

    return all_predicted_symbols

def decode_file(training_data, dev_in):
    symbol_word_counts, symbol_counts = get_symbol_word_counts(training_data)
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)

    symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts(training_data)
    transition_probabilities = estimate_transition_params(symbol_symbol_counts, symbol_counts)

    observation_sequences = get_observation_sequences(dev_in)
    predicted_symbols = viterbi(transition_probabilities, emission_probabilities, symbol_counts, observation_sequences)

    print(predicted_symbols)
    return predicted_symbols

def add_predicted_symbols_to_file(predicted_symbols, dev_in, prediction_file):
    result_file = open(prediction_file, "w", encoding="utf8")
    symbols_list = []
    for sequence_symbol in predicted_symbols:
        for symbol in sequence_symbol:
            if symbol!="START":
                if symbol=="STOP":
                    symbols_list.append("")
                else:
                    symbols_list.append(symbol)
    print (symbols_list)

    with open(dev_in, encoding="utf8") as f:
        for i,line in enumerate(f):
            word_label = line.strip() + " " + symbols_list[i] + "\n"
            result_file.write(word_label)



symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts('data/test')
#  print(symbol_symbol_counts)
#  print(symbol_counts)
#  print("TRANSITION PARAMS")
#  print(estimate_transition_params(symbol_symbol_counts, symbol_counts))

decode_file('data/test', 'data/test_dev')
predicted_symbols = decode_file('data/CN/train', 'data/CN/dev.in')
add_predicted_symbols_to_file(predicted_symbols, 'data/CN/dev.in', 'data/CN/dev.p3.out')
