from collections import defaultdict
import csv
import preprocess.utils as utils
import tensorflow as tf
import math
import numpy as np
import os
import random
import sys

class lstm_softmax:

	# Config
	num_layers = 2
	hidden_size = [256, 256]
	dropout_prob_c = 0.0
	batch_size = 32
	save_dir = 'data/model_lstm_softmax'
	save_path = 'data/model_lstm_softmax/model.ckpt'
	submit_file = 'data/submit_lstm_softmax.csv'
	nce_sample = 6000
	lr_init = 0.001
	lr_decay = 0.97
	subsample_rate = 0.0001

	# Constants
	training_data = None
	val_data = None
	testing_data = None
	testing_choices = None
	word_vec_dict = None
	word_index_dict = None
	word_frequency = None

	word_vec_dim = None
	max_seq_len = None
	total_word_count = None

	# Placeholder tensors
	x = None
	y_label = None
	seq_len = None
	dropout_prob = None
	lr = None
	weight = None

	# Output tensors
	train_step = None
	loss = None
	losses = None
	softmax = None

	def __init__(self):
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.fillConstants()
		self.defineModel()

	def fillConstants(self):
		# Read data
		print 'Preparing data...'
		self.training_data = utils.getTrainingData()
		self.val_data = utils.getValData()
		self.testing_data = utils.getTestingData()
		self.testing_choices = utils.getTestingChoiceList()
		self.word_vec_dict = utils.getWordVecDict()
		self.word_index_dict = utils.getWordIndexDict(min_occurence=3)
		word_occurence = utils.getWordOccurence()
		total_occurence = sum(word_occurence.values())
		self.word_frequency = defaultdict(int)
		for word in word_occurence:
			self.word_frequency[word] = float(word_occurence[word]) / total_occurence

		self.word_vec_dim = self.word_vec_dict.values()[0].shape[0]
		self.val_data
		self.max_seq_len = max(map(
			lambda sentence : len(sentence), self.training_data + self.val_data + self.testing_data))
		self.total_word_count = len(self.word_index_dict)

	def defineModel(self):
		## Define model topology
		print 'Defining model...'

		# Input
		self.x = x = tf.placeholder(tf.float32, shape=[None, self.max_seq_len, self.word_vec_dim], name='x')
		self.seq_len = seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
		self.dropout_prob = dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
		self.y_label = y_label = tf.placeholder(tf.int64, shape=[None, self.max_seq_len], name='y_label')
		self.lr = lr = tf.placeholder_with_default(self.lr_init, [], name='lr')
		self.weight = weight = tf.placeholder(tf.float32, shape=[None, self.max_seq_len], name='weight')
		
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
		y = stack_outputs

		## Define Loss (cross entropy)
		print 'Defining loss...'
		
		mask = tf.sequence_mask(seq_len, maxlen=self.max_seq_len) # Shape: [batch_size, max_seq_len]

		y_masked = tf.boolean_mask(y, mask, name='y_masked')
		y_masked_flatten = tf.reshape(y_masked, [-1, self.hidden_size[-1]], name='y_masked_flatten')

		y_label_masked = tf.boolean_mask(y_label, mask, name='y_label_masked')
		y_label_masked_flatten = tf.reshape(y_label_masked, [-1], name='y_label_masked_flatten')
		y_label_masked_flatten_nce = tf.reshape(y_label_masked, [-1, 1], name='y_label_masked_flatten_nce')

		weight_masked = tf.boolean_mask(weight, mask, name='weight_masked')
		weight_masked_flatten = tf.reshape(weight_masked, [-1], name='weight_masked_flatten')

		w_nce = tf.get_variable('w_nce', shape=[self.total_word_count, self.hidden_size[-1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_nce = tf.get_variable('b_nce', shape=[self.total_word_count], dtype=tf.float32, initializer=tf.constant_initializer(0))

		nce_losses = tf.nn.nce_loss(w_nce, b_nce, y_label_masked_flatten_nce, y_masked_flatten, self.nce_sample, self.total_word_count)
		nce_loss = tf.reduce_mean(nce_losses * weight_masked_flatten, name='nce_loss')

		logits = tf.matmul(y_masked_flatten, w_nce, transpose_b=True) + b_nce
		self.softmax = softmax = tf.nn.softmax(logits, name='softmax')
		self.losses = losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y_label_masked_flatten, self.total_word_count), name='losses')
		self.loss = loss = tf.reduce_mean(losses * weight_masked_flatten, name='loss')

		adam = tf.train.AdamOptimizer(lr)
		global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))
		self.train_step = train_step = adam.minimize(nce_loss, global_step=global_step)

	def genFeedDict(self, batch_start, data, lr=None, batch_size=None, testing=False):
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

		y_label_data = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
		for j, sentence in enumerate(data[batch_start: batch_start + batch_size]):
			for k, word in enumerate(sentence):
				try:
					y_label_data[j, k] = self.word_index_dict[word]
				except Exception as e:
					if not word == '_____':
						raise e

		seq_len_data = np.zeros((batch_size), dtype=np.int32)
		for j, sentence in enumerate(data[batch_start: batch_start + batch_size]):
			seq_len_data[j] = len(sentence)

		weight_data = np.zeros((batch_size, self.max_seq_len), dtype=np.float32)
		for j, sentence in enumerate(data[batch_start: batch_start + batch_size]):
			for k, word in enumerate(sentence):
				try:
					freq = self.word_frequency[word]
					weight_data[j, k] = (math.sqrt(freq / self.subsample_rate) + 1) * self.subsample_rate / freq
				except Exception as e:
					if not word == '_____':
						raise e

		feed_dict = {
			self.x: x_data, 
			self.y_label: y_label_data, 
			self.seq_len: seq_len_data, 
			self.dropout_prob: self.dropout_prob_c if not testing else 0.0,
			self.lr: lr if lr != None else self.lr_init,
			self.weight: weight_data}

		return feed_dict

	def getPseudoAccuracy(self, data, sess):
		wrong_count = 0
		total_count = 0
		for i in range(0, len(data) - self.batch_size, self.batch_size):
			correct_losses = sess.run(self.losses, feed_dict=self.genFeedDict(i, data))
			wrong_vec = np.ones(correct_losses.shape, dtype=bool)
			
			for j in range(4):
				# Randomly generate 4 wrong answers
				wrong_feed_dict = self.genFeedDict(i, data)

				for l, sentence in enumerate(data[i: i + self.batch_size]):
					for k, word in enumerate(sentence):
						wrong_feed_dict[self.y_label][l, k] = random.randrange(self.total_word_count)

				wrong_losses = sess.run(self.losses, feed_dict=wrong_feed_dict)

				wrong_vec &= wrong_losses < correct_losses
			wrong_count += np.sum(wrong_vec)
			total_count += correct_losses.shape[0]

		return 1.0 - float(wrong_count) / total_count

	def train(self, load_savepoint=None):
		print 'Start training...'

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if load_savepoint != None:
				tf.train.Saver().restore(sess, load_savepoint)
			last_val_loss = 999.9
			lr = self.lr_init
			for epoch in range(100):
				print '[Epoch #' + str(epoch) + ']'

				percent = 0
				for i in range(0, len(self.training_data) - self.batch_size, self.batch_size):
					feed_dict = self.genFeedDict(i, self.training_data, lr)
					sess.run(self.train_step, feed_dict=feed_dict)

					if int(i * 100 / len(self.training_data)) > percent:
						percent = int(i * 100 / len(self.training_data))
						lr *= self.lr_decay
						print str(percent) + '%, Training loss:' + str(sess.run(self.loss, feed_dict=feed_dict))
						print 'Psuedo accuracy:' + str(self.getPseudoAccuracy(random.sample(self.val_data, self.batch_size + 1), sess))
						if percent % 10 == 0:
							tf.train.Saver().save(sess, self.save_path)
							print 'Saved!'
							

				val_loss = 0.0
				count = 0
				for i in range(0, len(self.val_data) - self.batch_size, self.batch_size):
					feed_dict = self.genFeedDict(i, self.val_data)
					val_loss += sess.run(self.loss, feed_dict=feed_dict)
					count += 1
				val_loss /= count
				print 'Validation loss:' + str(val_loss)
				print 'Psuedo accuracy:' + str(self.getPseudoAccuracy(random.sample(self.val_data, len(self.val_data) / 8), sess))

				if val_loss >= last_val_loss:
					break

				last_val_loss = val_loss
				tf.train.Saver().save(sess, self.save_path)
				print 'Saved!'
				random.shuffle(self.training_data)

	def test(self, load_savepoint=None):
		print 'Start testing...'

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			if load_savepoint != None:
				tf.train.Saver().restore(sess, load_savepoint)
			else:
				tf.train.Saver().restore(sess, self.save_path)

			answers = [None] * len(self.testing_data)
			percent = 0
			for i in range(len(self.testing_data)):
				blank_index = self.testing_data[i].index('_____')
				test_feed_dict = self.genFeedDict(i, self.testing_data, batch_size=1, testing=True)
				softmax = sess.run(self.softmax, feed_dict=test_feed_dict)[blank_index]
				best_score = float('-inf')
				best_choice = None
				for j in range(5):
					choice = self.testing_choices[i][j]
					score = softmax[self.word_index_dict[choice]]
					print choice, score
					if (score > best_score):
						best_choice = j
						best_score = score
				answers[i] = best_choice
				raw_input()

				if int(i * 100 / len(self.testing_data)) > percent:
					percent = int(i * 100 / len(self.testing_data))
					print str(percent) + '% on testing'
			
			with open(self.submit_file, 'wb') as file:
				writer = csv.DictWriter(file, ['id', 'answer'])
				writer.writeheader()
				for i, ans in enumerate(answers):
					lut = ['a', 'b', 'c', 'd', 'e']
					writer.writerow({'id': i + 1, 'answer': lut[ans]})

