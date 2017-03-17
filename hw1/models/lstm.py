import csv
import preprocess.utils
import tensorflow as tf
import numpy as np
import os
import random
import sys

class lstm:

	# Config
	num_layers = 1
	hidden_size = [256]
	dropout_prob_c = 0.0
	batch_size = 48
	save_dir = 'data/model_lstm'
	save_path = 'data/model_lstm/model.ckpt'
	submit_file = 'data/submit_lstm.csv'

	# Constants
	training_data = None
	val_data = None
	testing_data = None
	word_vec_dict = None

	word_vec_dim = None
	max_seq_len = None

	# Placeholder tensors
	x = None
	y_label = None
	seq_len = None
	dropout_prob = None

	# Output tensors
	train_step = None
	loss = None
	losses = None

	def __init__(self):
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.fillConstants()
		self.defineModel()

	def fillConstants(self):
		# Read data
		print 'Preparing data...'
		self.training_data = preprocess.utils.getTrainingData()
		self.val_data = preprocess.utils.getValData()
		self.testing_data = preprocess.utils.getTestingData()
		self.testing_choices = preprocess.utils.getTestingChoiceList()
		self.word_vec_dict = preprocess.utils.getWordVecDict()

		self.word_vec_dim = self.word_vec_dict.values()[0].shape[0]
		self.max_seq_len = max(map(
			lambda sentences : max(map(
			lambda sentence : len(sentence), sentences)), 
			[self.training_data, self.val_data, self.testing_data]))

	def defineModel(self):
		## Define model topology
		print 'Defining model...'

		# Input
		self.x = x = tf.placeholder(tf.float32, shape=[None, self.max_seq_len, self.word_vec_dim], name='x')
		self.seq_len = seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
		self.dropout_prob = dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
		self.y_label = y_label = tf.placeholder(tf.float32, shape=[None, self.max_seq_len, self.word_vec_dim], name='y_label')
		
		# RNN
		basic_cells = [None] * self.num_layers
		for i in range(self.num_layers):
			basic_cells_lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size[i])
			basic_cells[i] = tf.contrib.rnn.DropoutWrapper(basic_cells_lstm, output_keep_prob=(1.0 - dropout_prob))
		stack_cell = tf.contrib.rnn.MultiRNNCell(basic_cells)
		stack_outputs, stack_states = tf.nn.dynamic_rnn(
			stack_cell, 
			inputs=x, 
			sequence_length=seq_len,
			dtype=tf.float32) # Shape: [batch_size, max_seq_len, hidden_size[-1]]
		
		
		# Output
		w_output = tf.get_variable('w_output', shape=[self.hidden_size[-1], self.word_vec_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_output = tf.get_variable('b_output', shape=[self.word_vec_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))

		stack_outputs_flatten = tf.reshape(
			stack_outputs, [-1, self.hidden_size[-1]]) # Shape: [batch_size*max_seq_len, hidden_size[-1]]
		y_flatten = tf.matmul(stack_outputs_flatten, w_output) + b_output # Shape: [batch_size*max_seq_len, word_vec_dim]
		y = tf.reshape(y_flatten, [-1, self.max_seq_len, self.word_vec_dim], name='y')

		## Define Loss (cosine distance)
		print 'Defining loss...'
		
		mask = tf.sequence_mask(seq_len, maxlen=self.max_seq_len) # Shape: [batch_size, max_seq_len]

		y_masked = tf.boolean_mask(y, mask, name='y_masked')
		y_masked_flatten = tf.reshape(y_masked, [-1, self.word_vec_dim], name='y_masked_flatten')
		y_masked_flatten_l2 = tf.nn.l2_normalize(y_masked_flatten, dim=1, name='y_masked_flatten_l2')

		y_label_masked = tf.boolean_mask(y_label, mask, name='y_label_masked')
		y_label_masked_flatten = tf.reshape(y_label_masked, [-1, self.word_vec_dim], name='y_label_masked_flatten')
		y_label_masked_flatten_l2 = tf.nn.l2_normalize(y_label_masked_flatten, dim=1, name='y_label_masked_flatten_l2')

		self.losses = losses = 1.0 - tf.reduce_sum(tf.multiply(y_masked_flatten_l2, y_label_masked_flatten_l2), axis=1, name='losses')
		self.loss = loss = tf.reduce_mean(losses, name='loss')

		adam = tf.train.AdamOptimizer()
        #global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))
		#self.train_step = train_step = adam.minimize(loss, global_step=global_step)
		self.train_step = train_step = adam.minimize(loss)

	def genFeedDict(self, batch_start, data, batch_size=None, testing=False):
		if batch_size == None:
			batch_size = self.batch_size

		x_data = np.zeros((batch_size, self.max_seq_len, self.word_vec_dim), dtype=np.float32)
		for j, sentence in enumerate(data[batch_start: batch_start + batch_size]):
			for k, word in enumerate(sentence[:-1]):
				try:
					x_data[j, k + 1] = self.word_vec_dict[word]
				except Exception as e:
					if not word == '_____':
						raise e

		y_label_data = np.zeros((batch_size, self.max_seq_len, self.word_vec_dim), dtype=np.float32)
		for j, sentence in enumerate(data[batch_start: batch_start + batch_size]):
			for k, word in enumerate(sentence):
				try:
					y_label_data[j, k] = self.word_vec_dict[word]
				except Exception as e:
					if not word == '_____':
						raise e

		seq_len_data = np.zeros((batch_size), dtype=np.int32)
		for j, sentence in enumerate(data[batch_start: batch_start + batch_size]):
			seq_len_data[j] = len(sentence)

		feed_dict = {
			self.x: x_data, 
			self.y_label: y_label_data, 
			self.seq_len: seq_len_data, 
			self.dropout_prob: self.dropout_prob_c if not testing else 0.0}

		return feed_dict

	def getPseudoAccuracy(self, data, sess):
		word_vec_list = self.word_vec_dict.values()
		wrong_count = 0
		for i in range(0, len(data) - self.batch_size, self.batch_size):
			correct_losses = sess.run(self.losses, feed_dict=self.genFeedDict(i, data))
			wrong_vec = np.ones(correct_losses.shape, dtype=bool)
			
			for j in range(4):
				# Randomly generate 4 wrong answers
				wrong_feed_dict = self.genFeedDict(i, data)

				wrong_feed_dict[self.y_label] = np.zeros((self.batch_size, self.max_seq_len, self.word_vec_dim), dtype=np.float32)
				for l, sentence in enumerate(data[i: i + self.batch_size]):
					for k, word in enumerate(sentence):
						wrong_feed_dict[self.y_label][l, k] = random.choice(word_vec_list)

				wrong_losses = sess.run(self.losses, feed_dict=wrong_feed_dict)
				
				wrong_vec &= wrong_losses < correct_losses
			wrong_count += np.sum(wrong_vec)

		return 1.0 - float(wrong_count) / len(data)

	def train(self, load_savepoint=None):
		print 'Start training...'

		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if load_savepoint != None:
				tf.train.Saver().restore(sess, load_savepoint)
			last_val_loss = 999.9
			for epoch in range(100):
				print '[Epoch #' + str(epoch) + ']'

				percent = 0
				for i in range(0, len(self.training_data) - self.batch_size, self.batch_size):
					feed_dict = self.genFeedDict(i, self.training_data)
					sess.run(self.train_step, feed_dict=feed_dict)

					if int(i / (len(self.training_data) / 100)) > percent:
						percent = int(i / (len(self.training_data) / 100))
						print str(percent) + '%, Training loss:' + str(sess.run(self.loss, feed_dict=feed_dict))

				val_loss = 0.0
				count = 0
				for i in range(0, len(self.val_data) - self.batch_size, self.batch_size):
					feed_dict = self.genFeedDict(i, self.val_data)
					val_loss += sess.run(self.loss, feed_dict=feed_dict)
					count += 1
				print 'Validation loss:' + str(val_loss / count)
				print 'Psuedo accuracy:' + str(self.getPseudoAccuracy(random.sample(self.val_data, len(self.val_data) / 4), sess))

				if (val_loss / count) >= last_val_loss:
					break

				last_val_loss = (val_loss / count)
				tf.train.Saver().save(sess, self.save_path)
				random.shuffle(self.training_data)

	def test(self, load_savepoint=None):
		print 'Start testing...'

		with tf.Session() as sess:
			if load_savepoint != None:
				tf.train.Saver().restore(sess, load_savepoint)
			else:
				tf.train.Saver().restore(sess, self.save_path)

			answers = [None] * len(self.testing_data)
			percent = 0
			for i in range(len(self.testing_data)):
				blank_index = self.testing_data[i].index('_____')
				test_feed_dict = self.genFeedDict(i, self.testing_data, batch_size=1, testing=True)
				best_loss = 99999.9
				best_choice = None
				for j in range(5):
					test_feed_dict[self.y_label][0][blank_index] = self.word_vec_dict[self.testing_choices[i][j]]
					losses = sess.run(self.losses, feed_dict=test_feed_dict)
					loss = losses[blank_index]
					if loss < best_loss:
						best_choice = j
						best_loss = loss
				answers[i] = best_choice

				if int(i * 100 / len(self.testing_data)) > percent:
					percent = int(i * 100 / len(self.testing_data))
					print str(percent) + '% on testing'
			
			with open(self.submit_file, 'wb') as file:
				writer = csv.DictWriter(file, ['id', 'answer'])
				writer.writeheader()
				for i, ans in enumerate(answers):
					lut = ['a', 'b', 'c', 'd', 'e']
					writer.writerow({'id': i + 1, 'answer': lut[ans]})

