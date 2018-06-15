import random

import state
import param

# Translate actions in MCTS from (choice, value) in states.
# @ret action in [1, 100] for SMALL/BIG choices, and [101, 200] for TRIPLE choices.
def action_from_choice_value(choice, value):
	assert choice == state.SMALL or choice == state.BIG or choice >= state.TRIPLE_ONE
	if choice == state.SMALL or choice == state.BIG:
		return value
	else:
		return 100 + value


# @ret (choice, value), where choice is default to SMALL without distinguishing it from BIG,
# or is to TRIPLE_ONE without distinguishing it from other TRIPLE_***
def action_to_state_choice(action):
	if not 1 <= action <= 200:
		print(action)
	assert 1 <= action <= 200
	if action <= 100:
		return state.SMALL, action
	else:
		return state.TRIPLE_ONE, action - 100


# Action a for MCTS is in the domain of [1, 200],
# mapping to p[0..199]
def action_to_p_index(a):
	assert 1 <= a <= 200
	return a - 1


# Probability p[0..199] maps to actions a in [1, 200]
# for MCTS
def p_index_to_action(p):
	assert 0 <= p <= 199
	return p + 1


# Select the indexes of the max in list/array p.
# Return a list of the indexes of the max.
def max_indexes(p):
	m = max(p)
	return [i for i, pi in enumerate(p) if pi == m]


# Randomly select one index of the max in list/array p.
def random_max_index(p):
	indexes = max_indexes(p)
	return random.choice(indexes)


# Check whether the game should be ended.
def is_game_ended(stat):
	return stat.balance() == 0 or stat.balance() - 100 >= stat.goal() or stat.step() >= param.QUIT_THRESHOLD


# Transform a state to its reward.
def state_to_reward(stat):
	return (stat.balance() - 100) / 100.0