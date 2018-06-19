import tensorflow as tf

from model import SoftmaxModel
import param

def draw_sess_graph(path):
	m = SoftmaxModel()
	writer = tf.summary.FileWriter(path, m.sess.graph)
	writer.flush()


if __name__ == '__main__':
	draw_sess_graph(param.SUMMARY_PATH)