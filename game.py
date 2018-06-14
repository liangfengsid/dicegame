import sys
import threading
from multiprocessing import Value

import state
import util
from mcts import MCTS
from state import DEFAULT_TRANSFORM
from state import State

MCTS_GAME = 0
OPT_GAME = 1

class mcts_game:
	def __init__(self, path=''):
		self.mcts = MCTS(path)

	def learn(self):
		self.mcts.learn()

	def play(self, stat):
		choice, value, _, _ = self.mcts.play_one_move(stat)
		return choice, value

# Opt_game always bet a value just cover the lost plus the goal.
class opt_game:
	def play(self, stat):
		value = min(stat.balance(), 100 + stat.goal() - stat.balance())
		return state.SMALL, value


if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == 'opt':
		game_type = OPT_GAME
	else:
		game_type = MCTS_GAME
	num_thread = 4
	num_game = 1000

	lock = threading.Lock()
	sum_balance = Value('i', 0)
	sum_step = Value('i', 0)

	class game_thread(threading.Thread):
		# type is either MCTS_TYPE or OPT_TYPE
		def __init__(self, id, num_game, type, balance_ref, step_ref, lock):
			threading.Thread.__init__(self)
			self.__id = id
			self.__num_game = num_game
			self.__type = type
			self.__sum_balance = balance_ref
			self.__sum_step = step_ref
			self.__lock = lock

		def run(self):
			balances = 0
			steps = 0
			for i in range(0, self.__num_game):
				# A game
				if self.__type == MCTS_GAME:
					g = mcts_game(path='model/dice')
				else:
					g = opt_game()
				stat = State(0, 100, 1)
				while not util.is_game_ended(stat):
					choice, value = g.play(stat)
					stat.transform(DEFAULT_TRANSFORM, choice, value)
				#print("==Game ends with state: ", stat.value())
				balances += stat.balance()
				steps += stat.step()
				if (i + 1) % 50 == 0:
					print('====Thread ', self.__id, ': game ends with state: ', stat.value(),
							", average balance: ", 1.0 * balances / (1 + i))
			with self.__lock:
				self.__sum_balance.value += balances
				self.__sum_step.value += steps
			print('Thread ', self.__id, ': average balance ', 1.0 * balances / self.__num_game,
					', average step: ', 1.0 * steps / self.__num_game)

	threads = [game_thread(i, num_game, game_type, sum_balance, sum_step, lock) for i in range(0, num_thread)]
	for t in threads:
		t.start()
	for t in threads:
		t.join()

	print('Average balance ', 1.0 * sum_balance.value / num_game / num_thread,
			', average step: ', 1.0 * sum_step.value / num_game / num_thread)



