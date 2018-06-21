import random
import sys
import time
import threading
from multiprocessing import Value

import state
import util
import param
from mcts import MCTS
from state import DEFAULT_TRANSFORM
from state import State

MCTS_GAME = 0
OPT_GAME = 1
RAND_GAME = 2
NORMAL_GAME = 3

class MCTSGame:
	def __init__(self, path=''):
		self.mcts = MCTS(path)

	def learn(self, checkpoint_path=""):
		start = time.time()
		for n in range(0, param.NUM_PLAY):
			root = state.State(0, 100, (n % 100) + 1)
			self.mcts.play_one_game(root)
			end = time.time()
			print(n, "======:", end - start)
			start = end
			if (n + 1) % param.CHECKPOINT_INTERVAL == 0 and checkpoint_path != "":
				self.mcts.save(checkpoint_path)

	def play(self, stat):
		(choice, value), _, _ = self.mcts.play_one_move(stat)
		return choice, value


# OptGame always bet a value just cover the lost plus the goal.
class OptGame:
	def play(self, stat):
		value = min(stat.balance(), 100 + stat.goal() - stat.balance())
		return state.SMALL, value


# RandGame always bet a random value within balance.
class RandGame:
	def play(self, stat):
		random.seed(time.time())
		value = random.randint(1, stat.balance())
		return state.SMALL, value


class NormalGame:
	def play(self, stat):
		return state.SMALL, 1


class GameThread(threading.Thread):
		# type is either MCTS_TYPE or OPT_TYPE
		def __init__(self, id, model_path, num_game, type, balance_ref, step_ref, lock):
			threading.Thread.__init__(self)
			self.__id = id
			self.__model_path = model_path
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
					g = MCTSGame(path=self.__model_path)
				elif self.__type == NORMAL_GAME:
					g = NormalGame()
				elif self.__type == RAND_GAME:
					g = RandGame()
				else:
					g = OptGame()
				stat = State(0, 100, 10)
				while not util.is_game_ended(stat):
					choice, value = g.play(stat)
					stat.transform(DEFAULT_TRANSFORM, choice, value)
				#print("==Game ends with state: ", stat.value())
				balances += stat.balance()
				steps += stat.step()
				if (i + 1) % 1 == 0:
					print(i, '====Thread ', self.__id, ': game ends with state: ', stat.value(),
							", average balance: ", 1.0 * balances / (1 + i))
			with self.__lock:
				self.__sum_balance.value += balances
				self.__sum_step.value += steps
			print('Thread ', self.__id, ': average balance ', 1.0 * balances / self.__num_game,
					', average step: ', 1.0 * steps / self.__num_game)


if __name__ == '__main__':
	if len(sys.argv) == 3 and sys.argv[1] == 'learn':
		path = sys.argv[2]
		g = MCTSGame(path)
		g.learn(checkpoint_path=path)
	elif (len(sys.argv) == 3 or len(sys.argv) == 4) and sys.argv[1] == 'play':
		if len(sys.argv) == 3 and sys.argv[2] == 'opt':
			path = ''
			game_type = OPT_GAME
		elif len(sys.argv) == 3 and sys.argv[2] == 'random':
			path = ''
			game_type = RAND_GAME
		elif len(sys.argv) == 3 and sys.argv[2] == 'normal':
			path = ''
			game_type = NORMAL_GAME
		else:
			path = sys.argv[3]
			game_type = MCTS_GAME
		num_thread = param.NUM_THREAD
		num_game = param.GAME_PER_THREAD

		lock = threading.Lock()
		sum_balance = Value('i', 0)
		sum_step = Value('i', 0)

		threads = [GameThread(i, path, num_game, game_type, sum_balance, sum_step, lock) for i in range(0, num_thread)]
		for t in threads:
			t.start()
		for t in threads:
			t.join()

		print('Average balance ', 1.0 * sum_balance.value / num_game / num_thread,
				', average step: ', 1.0 * sum_step.value / num_game / num_thread)
	else:
		print('Wrong argument. The input command should be in the format of "game.py [learn|play [opt|mcts]]"')