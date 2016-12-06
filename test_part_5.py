from part_5 import symbols, get_symbol_word_counts, estimate_emission_params, get_symbol_symbol_symbol_counts, estimate_second_order_transition_params, get_second_order_transition_probabilities, get_observation_sequences, second_order_viterbi

def test_get_symbol_symbol_symbol_counts():
    symbol_symbol_symbol_counts, symbol_symbol_counts = get_symbol_symbol_symbol_counts('data/test')

    # SYmbol symbol symbol counts
    assert symbol_symbol_symbol_counts['START']['B-neutral']['I-neutral'] == 1
    assert symbol_symbol_symbol_counts['B-neutral']['I-neutral']['O'] == 2
    assert symbol_symbol_symbol_counts['I-neutral']['O']['O'] == 2
    assert symbol_symbol_symbol_counts['O']['O']['O'] == 2
    assert symbol_symbol_symbol_counts['O']['O']['B-neutral'] == 1
    assert symbol_symbol_symbol_counts['O']['O']['B-positive'] == 1
    assert symbol_symbol_symbol_counts['B-positive']['O']['STOP'] == 1
    assert symbol_symbol_symbol_counts['START']['I-neutral']['I-neutral'] == 0
    assert symbol_symbol_symbol_counts['START']['I-neutral']['B-neutral'] == 0
    assert symbol_symbol_symbol_counts['O']['O']['I-positive'] == 0
    assert symbol_symbol_symbol_counts['O']['O']['B-negative'] == 0

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

def test_get_second_order_transition_probabilities():
    second_order_transition_probabilities = get_second_order_transition_probabilities('data/test')

    assert second_order_transition_probabilities['START']['B-neutral']['I-neutral'] == 1
    assert second_order_transition_probabilities['B-neutral']['I-neutral']['O'] == 1
    assert second_order_transition_probabilities['I-neutral']['O']['O'] == 1
    assert second_order_transition_probabilities['O']['O']['O'] == 2.0/4
    assert second_order_transition_probabilities['O']['O']['B-neutral'] == 1.0/4
    assert second_order_transition_probabilities['O']['O']['B-positive'] == 1.0/4
    assert second_order_transition_probabilities['B-positive']['O']['STOP'] == 1

    for symbol in symbols:
        assert second_order_transition_probabilities['START']['I-neutral'][symbol] == 0
        assert second_order_transition_probabilities['START']['I-negative'][symbol] == 0
        assert second_order_transition_probabilities['START']['I-positive'][symbol] == 0
        assert second_order_transition_probabilities['I-neutral']['B-positive'][symbol] == 0
        assert second_order_transition_probabilities['I-neutral']['B-negative'][symbol] == 0
        assert second_order_transition_probabilities['B-neutral']['B-negative'][symbol] == 0

def test_second_order_viterbi():
    symbol_word_counts, symbol_counts = get_symbol_word_counts('data/test')
    emission_probabilities = estimate_emission_params(symbol_word_counts, symbol_counts)

    symbol_symbol_symbol_counts, symbol_symbol_counts = get_symbol_symbol_symbol_counts('data/test')
    second_order_transition_probabilities = estimate_second_order_transition_params(symbol_symbol_symbol_counts, symbol_symbol_counts)

    observation_sequences = get_observation_sequences('data/test_dev')

    assert second_order_viterbi(second_order_transition_probabilities, emission_probabilities, symbol_symbol_counts, symbol_counts, observation_sequences) == [['START', 'B-neutral', 'I-neutral', 'O', 'O', 'O', 'O', 'B-neutral', 'I-neutral', 'O', 'O', 'B-positive', 'O', 'STOP']]
