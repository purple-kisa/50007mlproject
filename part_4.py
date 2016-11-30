import math
import sys
from part_3 import symbols, get_symbol_word_counts, get_symbol_symbol_counts, estimate_emission_params, emission_probability, estimate_transition_params, get_observation_sequences

def log(x):
    if x == 0:
        return math.log(sys.float_info.min)
    else:
        return math.log(x)

def top_m_viterbi(m, transition_probabilities, emission_probabilities, symbol_counts, observation_sequences):
    for sequence in observation_sequences:

        # Initialize probability score and optimal symbol matrices
        n = len(sequence)
        scores = {}
        scores[n+1] = {}
        for k in range(n + 1):
            scores[k] = {}
            for symbol in symbols:
                scores[k][symbol] = []

        # Set base case
        for symbol in symbols:
            scores[0][symbol]= [(0, "NA")]
            scores[1][symbol] = [(log(transition_probabilities["START"][symbol]) + log(emission_probability(symbol, sequence[0], emission_probabilities, symbol_counts)), "START")]
        scores[0]["STOP"]= [(0, "NA")]
        scores[0]["START"]= [(1, "NA")]

        # Move forward recursively
        for k in range(2, n + 1):
            for v in symbols:
                # Get the k highest probability scores and associated previous symbols
                kth_word = sequence[k-1]
                probabilities_and_previous_symbols = [(score_and_symbol[0] + log(transition_probabilities[u][v]) + log(emission_probability(v, kth_word, emission_probabilities, symbol_counts)), u) for u in symbols for score_and_symbol in scores[k-1][u]]
                probabilities_and_previous_symbols.sort(key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0], reverse=True)
                scores[k][v] = probabilities_and_previous_symbols[0:m]

        # Final entry
        probabilities_and_previous_symbols = [(score_and_symbol[0] + log(transition_probabilities[u]["STOP"]), u) for u in symbols for score_and_symbol in scores[k-1][u]]
        probabilities_and_previous_symbols.sort(key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0], reverse=True)
        scores[n+1]["STOP"] = probabilities_and_previous_symbols[0:m]

        # Get matrix of top m paths at each step in the sequence
        # The [k][symbol] element gives a list of tuples of the top-k paths to
        # the kth observation if it is tagged with symbol, and their score
        top_m_scores_and_paths= {}
        top_m_scores_and_paths[n+1] = {}
        for k in range(1, n + 1):
            top_m_scores_and_paths[k] = {}
            for symbol in symbols:
                top_m_scores_and_paths[k][symbol] = []

        # Set base case
        for symbol in symbols:
            top_m_scores_and_paths[1][symbol] = [(log(transition_probabilities["START"][symbol]), ["START"])]

        for k in range(2, n + 1):
            for v in symbols:
                kth_word = sequence[k-1]
                probabilities_and_paths = [(score_and_symbol[0] + score_and_path[0], score_and_path[1] + [score_and_symbol[1]]) for score_and_symbol in scores[k][v] for score_and_path in top_m_scores_and_paths[k-1][score_and_symbol[1]]]
                probabilities_and_paths.sort(key=lambda probability_and_path: probability_and_path[0], reverse=True)
                top_m_scores_and_paths[k][v] = probabilities_and_paths[0:m]

        # Final entry
        probabilities_and_paths = [(score_and_symbol[0] + score_and_path[0], score_and_path[1] + [score_and_symbol[1]]) for score_and_symbol in scores[n+1]["STOP"] for score_and_path in top_m_scores_and_paths[n][score_and_symbol[1]]]
        probabilities_and_paths.sort(key=lambda probability_and_path: probability_and_path[0], reverse=True)
        top_m_scores_and_paths[n+1]["STOP"] = probabilities_and_paths[0:m]

        return top_m_scores_and_paths[n+1]["STOP"]

def top_m_decode_file(m, training_data, dev_in):
    symbol_word_counts, symbol_counts = get_symbol_word_counts(training_data)
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)

    symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts(training_data)
    transition_probabilities = estimate_transition_params(symbol_symbol_counts, symbol_counts)

    observation_sequences = get_observation_sequences(dev_in)
    top_m_paths = top_m_viterbi(m, transition_probabilities, emission_probabilities, symbol_counts, observation_sequences)

    print(top_m_paths)

top_m_decode_file(3, 'data/test', 'data/test_dev')
