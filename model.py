import tensorflow as tf

import param

class SoftmaxModel:
	def __init__(self, rate=0.5, summary_path=param.SUMMARY_PATH):
		graph = tf.Graph()
		with graph.as_default():
			self.x = tf.placeholder(tf.float32, shape=[None, 3])
			# [101] represents 100 probabilities from 1% to 100%, plus a reward value.
			self.w = tf.Variable(tf.random_normal([3, 201], name="w", mean=0, stddev=1.0, dtype=tf.float32))
			self.b = tf.Variable(tf.random_normal([201], name="b", mean=0, stddev=1.0, dtype=tf.float32))
			z = tf.matmul(self.x, self.w) + self.b
			z_p = tf.slice(z, [0, 0], [tf.shape(z)[0], 200])
			z_v = tf.slice(z, [0, 200], [tf.shape(z)[0], 1])
			self.p = tf.nn.softmax(z_p)
			self.v = tf.nn.sigmoid(z_v)

			self.p_ = tf.placeholder(tf.float32, shape=[None, 200])
			self.v_ = tf.placeholder(tf.float32, shape=[None, 1])
			# TODO weights of differences between p and v
			min_bound = tf.ones([tf.shape(z)[0], 200], dtype=tf.float32) / 10000
			self.cross_entropy = -tf.reduce_sum(tf.reduce_sum(tf.maximum(self.p_, min_bound) * tf.log(tf.maximum(self.p, min_bound)), reduction_indices=[1])) + \
					tf.reduce_sum(tf.square(self.v - self.v_)) + param.C * \
					(tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.b))
			self.optimizer = tf.train.GradientDescentOptimizer(rate).minimize(self.cross_entropy)
			tf.summary.scalar('cross_entropy', self.cross_entropy)
			self.merged = tf.summary.merge_all()

		self.train_index = 0
		self.sess = tf.Session(graph=graph)
		self.writer = tf.summary.FileWriter(summary_path)


	def init_variables(self):
		with self.sess.as_default():
			with self.sess.graph.as_default():
				tf.global_variables_initializer().run()

	def save(self, path):
		with self.sess.graph.as_default():
			saver = tf.train.Saver()
			saver.save(self.sess, path)

	def load(self, path):
		with self.sess.graph.as_default():
			saver = tf.train.Saver()
			saver.restore(self.sess, path)

	# def __del__(self):
	# 	if self.sess is not None:
	# 		self.sess.close()

	def train(self, x, p, v):
		# type: ([None, 3], [None, 200], [None, 1]) -> object
		summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.x: x, self.p_: p, self.v_: v})
		self.writer.add_summary(summary, self.train_index)
		self.train_index += 1

	def inference(self, x):
		return self.sess.run([self.p, self.v], feed_dict={self.x: x})


if __name__ == "__main__":
	model = SoftmaxModel()
	model.init_variables()
	print(model.inference([[0, 1, 0.1]]))
	model.save("model/test.model")

	new_model = SoftmaxModel()
	new_model.load("model/test.model")
	print(new_model.inference([[0, 1, 0.1]]))
