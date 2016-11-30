from copy import deepcopy
from part_2 import symbols, get_symbol_word_counts, estimate_emission_params, emission_probability, find_symbol_estimate, write_part_2_dev_out

def test_get_symbol_word_counts():
    symbol_word_counts, symbol_counts = get_symbol_word_counts('data/test')

    # Symbol word counts
    assert symbol_word_counts['B-neutral']['A'] == 1
    assert symbol_word_counts['O']['C'] == 5
    assert symbol_word_counts['O']['A'] == 1
    assert symbol_word_counts['O']['B'] == 1
    assert symbol_word_counts['I-neutral']['A'] == 1
    assert symbol_word_counts['I-negative']['A'] == 0
    assert symbol_word_counts['B-negative']['A'] == 0

    # Symbol counts
    assert symbol_counts['O'] == 7
    assert symbol_counts['B-neutral'] == 2
    assert symbol_counts['I-neutral'] == 2
    assert symbol_counts['B-negative'] == 0
    assert symbol_counts['I-negative'] == 0
    assert symbol_counts['B-positive'] == 1
    assert symbol_counts['I-positive'] == 0

def test_emission_probability():
    symbol_word_counts, symbol_counts = get_symbol_word_counts('data/test')
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)

    words = ['A', 'B', 'C', 'D']
    # Unseen symbols
    for word in words:
        assert emission_probability('B-negative', word, emission_probabilities, symbol_counts) == 0
        assert emission_probability('I-negative', word, emission_probabilities, symbol_counts) == 0
        assert emission_probability('I-positive', word, emission_probabilities, symbol_counts) == 0

    # Unseen words
    assert emission_probability('O', 'U', emission_probabilities, symbol_counts) == 1.0/(1 + 7)
    assert emission_probability('B-neutral', 'U', emission_probabilities, symbol_counts) == 1.0/(1 + 2)
    assert emission_probability('I-neutral', 'U', emission_probabilities, symbol_counts) == 1.0/(1 + 2)
    assert emission_probability('B-positive', 'U', emission_probabilities, symbol_counts) == 1.0/(1 + 1)

    # Others
    assert emission_probability('O', 'C', emission_probabilities, symbol_counts) == 5.0/(1 + 7)
    assert emission_probability('O', 'A', emission_probabilities, symbol_counts) == 1.0/(1 + 7)
    assert emission_probability('O', 'B', emission_probabilities, symbol_counts) == 1.0/(1 + 7)
    assert emission_probability('B-neutral', 'A', emission_probabilities, symbol_counts) == 1.0/(1 + 2)
    assert emission_probability('I-neutral', 'A', emission_probabilities, symbol_counts) == 1.0/(1 + 2)

def test_find_symbol_estimate():
    symbol_word_counts, symbol_counts = get_symbol_word_counts('data/test')
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)
    predicted_word_symbol_sequence = find_symbol_estimate('data/test_dev', emission_probabilities, symbol_counts)

    assert predicted_word_symbol_sequence[0] == ('A', 'B-neutral')
    assert predicted_word_symbol_sequence[1] == ('B', 'B-neutral')
    assert predicted_word_symbol_sequence[2] == ('C', 'O')
    assert predicted_word_symbol_sequence[3] == ('C', 'O')
    assert predicted_word_symbol_sequence[4] == ('C', 'O')
    assert predicted_word_symbol_sequence[5] == ('A', 'B-neutral')
    assert predicted_word_symbol_sequence[6] == ('B', 'B-neutral')
    assert predicted_word_symbol_sequence[7] == ('A', 'B-neutral')
    assert predicted_word_symbol_sequence[8] == ('B', 'B-neutral')
    assert predicted_word_symbol_sequence[9] == ('C', 'O')
    assert predicted_word_symbol_sequence[10] == ('D', 'B-positive')
    assert predicted_word_symbol_sequence[11] == ('C', 'O')

    write_part_2_dev_out('data/test_dev.p2.out', predicted_word_symbol_sequence)
