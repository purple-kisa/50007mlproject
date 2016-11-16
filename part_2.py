from collections import defaultdict

symbols = ['O', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative', 'I-negative']

def get_symbol_word_counts(training_data):
    """
    Takes a training data file formatted with lines like
    (word) (symbol)
    and returns a tuple of two dictionaries
    The first is a nested dictionary of word counts
    where the value of d[symbol][word] is the counts of (word) (symbol)
    The second is a dictionary of symbol counts
    where the value of d[symbol] is the total count of the symbol
    """

    # initialize dictionary of counts
    symbol_word_counts = {}
    for symbol in symbols:
        symbol_word_counts[symbol] = defaultdict(int)

    with open(training_data) as f:
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

    return symbol_word_counts, symbol_counts

def estimate_emission_params(symbol_word_counts, symbol_counts):
    """
    Returns a nested dictionary of emission probabilities
    where the value of d[symbol][word] is the emission probability
    for that word and symbol.
    If the word is unseen in the training data used to produce the
    symbol_word_counts and symbol_counts, trying to access it from
    the dictionary will raise a KeyError
    """

    emission_probabilities = {}
    for symbol in symbol_word_counts:
        emission_probabilities[symbol] = {}
        for word in symbol_word_counts[symbol]:
            emission_probabilities[symbol][word] = float(symbol_word_counts[symbol][word])/(symbol_counts[symbol] + 1)

    return emission_probabilities

def emission_probability(symbol, word, emission_probabilities, symbol_counts):
    """
    Takes a symbol, a word, a nested dictionary of emission probabilities,
    and a dictionary of symbol counts
    and returns the emission probability for that symbol and word
    If the word has not been encountered in the training data
    we assign it a fixed probability based on the symbol count
    """

    emission_probabilities = estimate_emission_params(training_data)
    unseen_word = True

    for sym in emission_probabilities:
        if word in emission_probabilities[sym]:
            unseen_word = False

    if unseen_word:
        return 1/(1 + symbol_counts[symbol])
    else:
        emission_probabilities[symbol][word]

def find_symbol_estimate(dev_file, emission_probabilities):
    # using dev.in
    predicted_symbols = []
    with open(dev_file) as f:
        for line in f:
            word = line.strip()
            current_arg_max = symbols[0]
            current_max = 0
            for symbol in symbols:
                print(word, symbol,emission_probabilities[symbol][word])
                if emission_probabilities[symbol][word] > current_max:
                    current_arg_max = symbol
                    current_max = emission_probabilities[symbol][word]
            predicted_symbols.append(current_arg_max)

    return predicted_symbols

def get_symbol_sequence(dev_out_file):
    with open(dev_out_file) as f:
        return [line.split(' ')[-1].strip() for line in f if not line.isspace()]

def get_entity_count(symbol_sequence):
    entity_count = 0
    inside_entity = False

    for symbol in symbol_sequence:
        if inside_entity:
            if symbol[0] != 'I':
                entity_count += 1
                if symbol[0] == 'O':
                    inside_entity = False
        else:
            if symbol[0] != 'O':
                inside_entity = True

    # In case our sequence ends with an entity
    # we need to count this last entity
    if symbol[0] != 'O':
        entity_count += 1

    return entity_count

def compute_precision(predicted_symbols, gold_standard):
    correct_count = 0
    inside_entity = False
    correctly_predicted = False

    for i, gold_symbol in enumerate(gold_standard):
        if inside_entity:
            if gold_symbol != predicted_symbols[i]:
                correctly_predicted = False

            # if we move out of an entity
            if gold_symbol[0] != 'I':
                if correctly_predicted:
                    correct_count += 1
                if gold_symbol[0] == 'O':
                    inside_entity = False
        else:
            if gold_symbol[0] != 'O':
                inside_entity = True
                if gold_symbol != predicted_symbols[i]:
                    correctly_predicted = False
                else:
                    correctly_predicted = True

    return correct_count



e = estimate_emission_params("data/EN/train")
predicted_symbols = find_symbol_estimate("data/EN/dev.in", e)
gold_standard = get_symbol_sequence('data/EN/dev.out')
test_sequence = ['B-p', 'I', 'O', 'O', 'I', 'I', 'B', 'O', 'I', 'B', 'B', 'B', 'B', 'O', 'I', 'O']
print(get_entity_count(test_sequence))
print(get_entity_count(predicted_symbols))
print(compute_precision(predicted_symbols, gold_standard))

test_e = estimate_emission_params('data/test')
print(find_symbol_estimate('data/test_dev', test_e))
print(test_e)
