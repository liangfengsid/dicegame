import math
import os.path
import numpy as np

import model
import param
import state
import util


class MCTS:
	def __init__(self, path=""):
		self._state_init()
		if path == "":
			self.model = model.SoftmaxModel()
			self.model.init_variables()
		else:
			self.load(path)

	def _state_init(self):
		self.N = dict()
		self.W = dict()
		self.Q = dict()
		self.P = dict()

	def _select(self, stat):
		max_reward = float("-inf")
		max_action = 0
		n_sum_sqrt = 0

		sv = stat.tuple_value()
		available_actions = stat.available_actions()
		actions = []
		for a in available_actions:
			if self.N.get((sv, a)) is not None:
				actions.append(a)
				n_sum_sqrt += self.N[(sv, a)]
		n_sum_sqrt = math.sqrt(n_sum_sqrt)

		# 0 stands for the choice of SMALL/BIG, 1 for TRIPLE***
		rewards = []
		for a in actions:
			reward = self.Q[(sv, a)] + \
					param.C_PUCT * self.P[(sv, a)] * \
					n_sum_sqrt / (1 + self.N[(sv, a)])
			rewards.append(reward)
			if reward > max_reward:
				max_reward = reward
				max_action = a

		return actions, max_action, max_reward

	def _expand(self, stat):
		actions = stat.available_actions()
		sv = stat.tuple_value()
		p, v = self.model.inference([stat.value()])
		for a in actions:
			self.N[(sv, a)] = 0
			self.W[(sv, a)] = 0
			self.Q[(sv, a)] = 0
			self.P[(sv, a)] = p[0][util.action_to_p_index(a)]
		return v[0]

	# TODO
	# This function can be parallelized for different states.
	def _backup(self, stats, v):
		for s, a in reversed(stats):
			sv = s.tuple_value()
			if self.N.get((sv, a)) is None:
				print("ERROR")
			assert self.N.get((sv, a)) is not None
			self.N[(sv, a)] += 1

			assert self.W.get((sv, a)) is not None
			self.W[(sv, a)] += v[0]
			self.Q[(sv, a)] = self.W[(sv, a)] / self.N[(sv, a)]

	def _play(self, stat):
		actions = stat.available_actions()
		assert len(actions) != 0
		sv = stat.tuple_value()
		p_sum = 0
		max_a = 0
		max_p = 0
		for a in actions:
			assert self.N.get((sv, a)) is not None
			pi = math.pow(self.N[(sv, a)], 1.0 / param.TEMPRETURE)
			if pi > max_p:
				max_p = pi
				max_a = a
			p_sum += pi
		p = [0.0] * state.NUM_ACTIONS
		for a in actions:
			p[util.action_to_p_index(a)] = math.pow(self.N[(sv, a)], 1.0 / param.TEMPRETURE) / p_sum

		choice = util.action_to_state_choice(max_a)
		new_stat = stat.copy_transform(state.DEFAULT_TRANSFORM, choice)

		return choice, p, new_stat

	# Discard the statistics of the root of MC tree
	# Assuming keys (sv, a) for all available actions a exist in N, W, Q, P
	def _discard(self, root):
		sv = root.tuple_value()
		for a in root.available_actions():
			del self.N[(sv, a)]
			del self.W[(sv, a)]
			del self.Q[(sv, a)]
			del self.P[(sv, a)]


	# Play a move from a root state,
	# @return the choice and value of the move, distributions of model inference choice, and a new state.
	def play_one_move(self, root):
		# For each simulation, select, expand/evaluate, and backup
		for num_sim in range(0, param.NUM_SIMULATION):
			is_leaf = False
			stat = root
			stat_actions = []

			# Select
			for _ in range(0, param.QUIT_THRESHOLD):
				if util.is_game_ended(stat):
					# lose all jettons or wins to goal,
					# TODO
					break

				# Select
				acts, act, reward = self._select(stat)

				# Reach to the leave node.
				if len(acts) == 0:
					is_leaf = True
					break

				# Store the state and action list for backup
				stat_actions.append((stat, act))

				# Transform the state
				choice = util.action_to_state_choice(act)
				stat = stat.copy_transform(state.DEFAULT_TRANSFORM, choice)

			# expand and backup
			if is_leaf:
				v = self._expand(stat)
				# TODO
				# It is not clear whether the expanded state should be backed up.
				for a in stat.available_actions():
					stat_actions.append((stat, a))
				self._backup(stat_actions, v)
		# End the simulation for loop

		# Play
		state_choice, p, new_root = self._play(root)
		self._discard(root)
		return state_choice, p, new_root

	# Play a game by MCTS. The play statistics is used to train the model.
	def play_one_game(self, root):
		self._state_init()
		root_list = list()
		p_list = list()
		# For each play, play to the end and back propagate.
		while not util.is_game_ended(root):
			# Play one move
			root_list.append(root.value())
			_, p, root = self.play_one_move(root)
			p_list.append(np.asarray(p))
		# End the play while loop

		# Train the model
		root_array = np.asarray(root_list)
		p_array = np.asarray(p_list)
		# v is normalized to the precision of 0.01.
		v_array = np.asarray([[util.state_to_reward(root)]] * len(root_list))
		for i in range(0, (len(root_array) - 1) / param.BATCH_SIZE + 1):
			self.model.train(root_array[i * param.BATCH_SIZE:(i + 1) * param.BATCH_SIZE],
					p_array[i * param.BATCH_SIZE:(i + 1) * param.BATCH_SIZE],
					v_array[i * param.BATCH_SIZE:(i + 1) * param.BATCH_SIZE])

	def save(self, path):
		model_path = path + '.model'
		self.model.save(model_path)

	def load(self, path):
		model_path = path + '.model'
		self.model = model.SoftmaxModel()
		if os.path.exists(model_path):
			self.model.load(model_path)
		else:
			self.model.init_variables()



if __name__ == "__main__":
	tree = MCTS(path="model/dice")
	p, v = tree.model.inference([[0, 100, 1]])
	print(p, v)

	s = [[0, i, 1] for i in range(1, 101)]
	# s = [[i, 100, 1] for i in range(0, 100)]
	ps, vs = tree.model.inference(s)
	print(ps)
	for p in ps:
		print(util.random_max_index(p))

	tree.save("model/dice")

	new_model = model.SoftmaxModel()
	new_model.load("model/dice.model")
	new_p, new_v = new_model.inference([[0, 100, 1]])
	print(new_p, new_v)
