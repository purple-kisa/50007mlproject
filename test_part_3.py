from part_2 import get_symbol_word_counts, get_emission_probabilities
from part_3 import symbols, get_symbol_symbol_counts, get_transition_probabilities, get_observation_sequences, viterbi

def test_get_symbol_symbol_counts():
    symbol_symbol_counts, symbol_counts = get_symbol_symbol_counts('data/test')

    # Symbol symbol counts
    assert symbol_symbol_counts['B-neutral']['I-neutral'] == 2
    assert symbol_symbol_counts['I-neutral']['O'] == 2
    assert symbol_symbol_counts['O']['O'] == 4
    assert symbol_symbol_counts['O']['B-neutral'] == 1
    assert symbol_symbol_counts['O']['B-positive'] == 1
    assert symbol_symbol_counts['B-positive']['O'] == 1
    assert symbol_symbol_counts['O']['B-negative'] == 0
    assert symbol_symbol_counts['O']['I-negative'] == 0
    assert symbol_symbol_counts['O']['I-neutral'] == 0
    assert symbol_symbol_counts['B-negative']['I-neutral'] == 0

    # Symbol counts
    assert symbol_counts['O'] == 7
    assert symbol_counts['B-neutral'] == 2
    assert symbol_counts['I-neutral'] == 2
    assert symbol_counts['B-negative'] == 0
    assert symbol_counts['I-negative'] == 0
    assert symbol_counts['B-positive'] == 1
    assert symbol_counts['I-positive'] == 0

def test_get_transition_probabilities():
    transition_probabilities = get_transition_probabilities('data/test')

    assert transition_probabilities['B-neutral']['I-neutral'] == 1
    assert transition_probabilities['I-neutral']['O'] == 1
    assert transition_probabilities['O']['O'] == 4.0/7
    assert transition_probabilities['B-positive']['O'] == 1

    for symbol in symbols:
        assert transition_probabilities['B-negative'][symbol] == 0
        assert transition_probabilities['I-positive'][symbol] == 0
        assert transition_probabilities['I-negative'][symbol] == 0

def test_get_observation_sequences():
    observation_sequences = get_observation_sequences('data/test_dev')

    assert observation_sequences == [['A', 'B', 'C', 'C', 'C', 'A', 'B', 'A', 'B', 'C', 'D', 'C']]

def test_viterbi():
    symbol_counts = get_symbol_word_counts('data/test')[1]
    emission_probabilities = get_emission_probabilities('data/test')
    transition_probabilities = get_transition_probabilities('data/test')
    observation_sequences = get_observation_sequences('data/test_dev')

    assert viterbi(transition_probabilities, emission_probabilities, symbol_counts, observation_sequences) == [['START', 'B-neutral', 'I-neutral', 'O', 'O', 'O', 'O', 'B-neutral', 'I-neutral', 'O', 'O', 'B-positive', 'O', 'STOP']]
