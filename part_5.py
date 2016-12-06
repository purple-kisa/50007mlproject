from part_4 import symbols, get_symbol_word_counts, get_symbol_symbol_counts, estimate_emission_params, emission_probability, estimate_transition_params, get_observation_sequences, top_m_viterbi, log

# We try to learn a second order Markov model,
# where the transition probabilities are now conidtioned on the previous two states # instead of just the previous state

def get_symbol_symbol_symbol_counts(training_data):
    """
    Takes a training data file formatted with lines like
    (word) (symbol)
    and returns a tuple of two dictionaries
    The first is a doubly nested dictionary of
    prev-symbol-symbol-to-symbol transition counts
    where the value of d[symbol1][symbol2][symbol3] is the counts of
    (word) (symbol1)
    (word) (symbol2)
    (word) (symbol3)
    The second is a dictionary of symbol-symbol counts
    where the value of d[symbol1][symbol2] is the total count of
    (word) (symbol1)
    (word) (symbol2)
    """

    symbol_symbol_symbol_counts = { 'START': {} }
    for symbol1 in symbols:
        symbol_symbol_symbol_counts[symbol1] = {}
        for symbol2 in symbols:
            symbol_symbol_symbol_counts[symbol1][symbol2] = { 'STOP': {} }
            symbol_symbol_symbol_counts[symbol1][symbol2]['STOP'] = 0
            symbol_symbol_symbol_counts['START'][symbol2] = {}
            for symbol3 in symbols:
                symbol_symbol_symbol_counts['START'][symbol2][symbol3] = 0
                symbol_symbol_symbol_counts[symbol1][symbol2][symbol3] = 0

    prev_prev_symbol = 'START'
    prev_symbol = 'START'

    with open(training_data, encoding='utf8') as f:
        for line in f:
            if line.isspace():
                continue
            symbol = line.split(' ')[-1].strip()
            if prev_symbol == 'START':
                prev_symbol = symbol
                continue
            symbol_symbol_symbol_counts[prev_prev_symbol][prev_symbol][symbol] += 1
            prev_prev_symbol = prev_symbol
            prev_symbol = symbol

    symbol_symbol_symbol_counts[prev_prev_symbol][prev_symbol]['STOP'] += 1

    symbol_symbol_counts = {}
    for symbol1 in symbol_symbol_symbol_counts:
        symbol_symbol_counts[symbol1] = {}
        for symbol2 in symbol_symbol_symbol_counts[symbol1]:
            symbol_symbol_counts[symbol1][symbol2] = sum(symbol_symbol_symbol_counts[symbol1][symbol2].values())

    return symbol_symbol_symbol_counts, symbol_symbol_counts

def estimate_second_order_transition_params(symbol_symbol_symbol_counts, symbol_symbol_counts):
    """
    Returns a doubly nested dictionary of transition probabilities
    where the value of d[symbol1][symbol2][symbol3] is the probability
    of transitioning from symbol1, symbol2 to symbol3
    """

    transition_probabilities = {}
    for symbol1 in symbol_symbol_symbol_counts:
        transition_probabilities[symbol1] = {}
        for symbol2 in symbol_symbol_symbol_counts[symbol1]:
            transition_probabilities[symbol1][symbol2] = {}
            for symbol3 in symbol_symbol_symbol_counts[symbol1][symbol2]:
                if symbol_symbol_counts[symbol1][symbol2] == 0:
                    transition_probabilities[symbol1][symbol2][symbol3] = 0
                else:
                    transition_probabilities[symbol1][symbol2][symbol3] = float(symbol_symbol_symbol_counts[symbol1][symbol2][symbol3])/symbol_symbol_counts[symbol1][symbol2]

    return transition_probabilities

def get_second_order_transition_probabilities(training_data):
    symbol_symbol_symbol_counts, symbol_symbol_counts = get_symbol_symbol_symbol_counts(training_data)
    return estimate_second_order_transition_params(symbol_symbol_symbol_counts, symbol_symbol_counts)

def second_order_viterbi(second_order_transition_probabilities, emission_probabilities, symbol_symbol_counts, symbol_counts, observation_sequences):
    all_predicted_symbols = []
    for sequence in observation_sequences:

        # Initialize probability score and optimal symbol matrix
        n = len(sequence)
        scores_and_previous_symbols = {}
        scores_and_previous_symbols[n+1] = {}
        for k in range(n + 1):
            if k == 1:
                scores_and_previous_symbols[k] = { 'START': {} }
            else:
                scores_and_previous_symbols[k] = {}
            for symbol1 in symbols:
                scores_and_previous_symbols[k][symbol1] = {}
                for symbol2 in symbols:
                    scores_and_previous_symbols[k][symbol1][symbol2] = []

        if len(sequence) == 1:
            for symbol in symbols:
                first_observation_transition_probability = symbol_symbol_counts['START'][symbol1]
                scores_and_previous_symbols[1]['START'][symbol] = (log(first_observation_transition_probability) + log(emission_probability(symbol, sequence[0], emission_probabilities, symbol_counts)), 'START', 'NA')
            predicted_symbols = [max([(scores_and_previous_symbols[1]['START'][symbol], symbol) for symbol in symbols], key=lambda score_and_previous_symbol : score_and_previous_symbol[0])[1]]
            all_predicted_symbols.append(predicted_symbols)
            continue

        # Set base case
        for symbol in symbols:
            first_observation_transition_probability = symbol_symbol_counts['START'][symbol1]
            scores_and_previous_symbols[1]['START'][symbol] = (log(first_observation_transition_probability) + log(emission_probability(symbol, sequence[0], emission_probabilities, symbol_counts)), 'START', 'NA')
        for symbol1 in symbols:
            for symbol2 in symbols:
                scores_and_previous_symbols[2][symbol1][symbol2] = (scores_and_previous_symbols[1]['START'][symbol1][0] + log(second_order_transition_probabilities['START'][symbol1][symbol2]) + log(emission_probability(symbol2, sequence[1], emission_probabilities, symbol_counts)), symbol1, 'START')

        # Move forward recursively
        for k in range(3, n + 1):
            for v in symbols:
                for u in symbols:
                    # Get the max probability score
                    kth_word = sequence[k-1]
                    #  for w in symbols:
                        #  for u in symbols:
                            #  print(scores_and_previous_symbols[k-1][w][u])
                    probabilities_and_previous_symbols = [(scores_and_previous_symbols[k-1][w][u][0] + log(second_order_transition_probabilities[w][u][v]) + log(emission_probability(v, kth_word, emission_probabilities, symbol_counts)), u, w) for w in symbols]
                    scores_and_previous_symbols[k][u][v] = max(probabilities_and_previous_symbols, key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0])

        for u in symbols:
            probabilities_and_previous_symbols = [(scores_and_previous_symbols[n][w][u][0] + log(second_order_transition_probabilities[w][u]["STOP"]), u, w) for w in symbols]
            scores_and_previous_symbols[n+1][u] = {}
            scores_and_previous_symbols[n+1][u]["STOP"] = max(probabilities_and_previous_symbols, key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0])

        # Predict symbol sequence

        # Get last two optimal tags
        # The last (n+1)th is STOP
        # From this we can get the tag of the nth observation
        # by finding the u that gives us the max score
        predicted_symbols = ["STOP"]
        max_value = max([scores_and_previous_symbols[n+1][u][predicted_symbols[0]] for u in symbols], key=lambda score_and_previous_symbol: score_and_previous_symbol[0])
        for u in scores_and_previous_symbols[n+1]:
            if scores_and_previous_symbols[n+1][u][predicted_symbols[0]]== max_value:
                predicted_symbols.insert(0, u)
                break

        # Given the two subsequent symbols, we can get the current symbol from our matrix
        for k in range(n + 1, 1, -1):
            predicted_symbols.insert(0, scores_and_previous_symbols[k][predicted_symbols[0]][predicted_symbols[1]][2])

        all_predicted_symbols.append(predicted_symbols)

    return all_predicted_symbols

def decode_file(training_data, dev_in):
    symbol_word_counts, symbol_counts = get_symbol_word_counts(training_data)
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)

    symbol_symbol_symbol_counts, symbol_symbol_counts = get_symbol_symbol_symbol_counts(training_data)
    second_order_transition_probabilities = estimate_second_order_transition_params(symbol_symbol_symbol_counts, symbol_symbol_counts)

    observation_sequences = get_observation_sequences(dev_in)
    predicted_symbols = second_order_viterbi(second_order_transition_probabilities, emission_probabilities, symbol_symbol_counts, symbol_counts, observation_sequences)

    print(predicted_symbols)

#  decode_file('data/test', 'data/test_dev')
#  decode_file('data/EN/train', 'data/EN/dev.in')
#  decode_file('data/ES/train', 'data/ES/dev.in')
