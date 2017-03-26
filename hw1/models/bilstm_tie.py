from collections import defaultdict
import csv
import preprocess.utils as utils
import tensorflow as tf
import math
import numpy as np
import os
import random
import sys

class bilstm_tie:

	# Config
	num_layers = 2
	hidden_size = [250, 250]
	dropout_prob_c = 0.0
	batch_size = 20
	save_dir = 'data/model_bilstm_tie5'
	save_path = 'data/model_bilstm_tie5/model.ckpt'
	submit_file = 'data/submit_bilstm_tie5.csv'
	nce_sample = 20 * batch_size * 20
	lr_init = 0.001
	lr_decay = 0.7
	lr_decay_patient = 7
	subsample_rate = 0.0001

	# Constants
	training_data = None
	val_data = None
	testing_data = None
	testing_choices = None
	word_vec_dict = None
	word_index_dict = None
	word_frequency = None
	index_word_dict = None

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
	trainable = None

	# Output tensors
	train_step = None
	loss = None
	losses = None
	softmax = None
	mask = None

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
		self.index_word_dict = {v: k for k, v in self.word_index_dict.iteritems()}
		word_occurence = utils.getWordOccurence(min_occurence=3)
		total_occurence = sum(word_occurence)
		self.word_frequency = defaultdict(int)
		for i, occ in enumerate(word_occurence):
			self.word_frequency[i] = float(occ) / total_occurence

		self.word_vec_dim = self.word_vec_dict.values()[0].shape[0]
		self.val_data
		self.max_seq_len = max(map(
			lambda sentence : len(sentence), self.training_data + self.val_data + self.testing_data)) + 2
		self.total_word_count = len(self.word_index_dict)

	def defineModel(self):
		## Define model topology
		print 'Defining model...'

		# Input
		self.x = x = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len], name='x')
		self.seq_len = seq_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='seq_len')
		self.dropout_prob = dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
		self.y_label = y_label = tf.placeholder(tf.int64, shape=[self.batch_size, self.max_seq_len], name='y_label')
		self.lr = lr = tf.placeholder_with_default(self.lr_init, [], name='lr')
		self.weight = weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_seq_len], name='weight')
		self.trainable = trainable = tf.placeholder(bool, shape=[], name='trainable')
		
		# Input embedding
		
		w_emb_init = tf.constant(np.array([self.word_vec_dict[self.index_word_dict[i]] for i in range(self.total_word_count)], dtype=np.float32), dtype=tf.float32)
		w_emb = tf.get_variable('w_emb', dtype=tf.float32, initializer=w_emb_init)
		w_emb_fixed = tf.stop_gradient(w_emb, name='w_emb_fixed')

		x_emb = tf.cond(trainable, 
			lambda: tf.nn.embedding_lookup(w_emb, x, name='x_emb_flatten_trainable'),
			lambda: tf.nn.embedding_lookup(w_emb_fixed, x, name='x_emb_flatten_untrainable'))
		x_emb_reshape = tf.reshape(x_emb, [-1, self.word_vec_dim], name='x_emb_reshape')

		# Highway
		w_highway_t= tf.get_variable('w_highway_t', [self.word_vec_dim, self.word_vec_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_highway_t = tf.get_variable('b_highway_t', [self.word_vec_dim], dtype=tf.float32, initializer=tf.constant_initializer(-1.0))

		w_highway = tf.get_variable('w_highway', [self.word_vec_dim, self.word_vec_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_highway = tf.get_variable('b_highway', [self.word_vec_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		t_highway = tf.sigmoid(tf.matmul(x_emb_reshape, w_highway_t) + b_highway_t, name='transform_gate_highway')
		h_highway = tf.nn.relu(tf.matmul(x_emb_reshape, w_highway) + b_highway, name='activation_highway')
		c_highway = tf.subtract(1.0, t_highway, name='carry_gate_highway')

		highway_outputs = tf.add(h_highway * t_highway, x_emb_reshape * c_highway, name='highway_outputs')
		highway_outputs_reshape = tf.reshape(highway_outputs, [-1, self.max_seq_len, self.word_vec_dim], name='highway_outputs_reshape')

		# RNN
		basic_cells = [None] * self.num_layers
		for i in range(self.num_layers):
			basic_cells_lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size[i])
			basic_cells[i] = tf.contrib.rnn.DropoutWrapper(basic_cells_lstm, output_keep_prob=(1.0 - dropout_prob))
		stack_cell = tf.contrib.rnn.MultiRNNCell(basic_cells)
		(stack_outputs_fw, stack_outputs_bw), stack_states = tf.nn.bidirectional_dynamic_rnn(
			stack_cell, 
			stack_cell,
			inputs=highway_outputs_reshape, 
			sequence_length=seq_len + 2, # Add 1 for backward RNN
			dtype=tf.float32) # Shape: [batch_size, max_seq_len, hidden_size[-1]]
		
		# Output
		stack_outputs_bw_shift = tf.slice(stack_outputs_bw, [0, 2, 0], [-1, -1, -1])
		stack_outputs_bw_shift = tf.pad(stack_outputs_bw_shift, [[0, 0], [0, 2], [0, 0]])
			
		mask = tf.sequence_mask(seq_len, maxlen=self.max_seq_len) # sequence length mask
		mask = tf.logical_and(mask, tf.not_equal(y_label, 0)) # unknown word mask
		self.mask = mask
		mask_flatten = tf.reshape(mask, [-1])

		stack_outputs = tf.concat([stack_outputs_fw, stack_outputs_bw_shift], axis=2, name='stack_outputs')
		stack_outputs_flatten = tf.reshape(stack_outputs, [-1, 2 * self.hidden_size[-1]])
		stack_outputs_masked_flatten = tf.boolean_mask(stack_outputs_flatten, mask_flatten)

		w_fc = tf.get_variable('w_fc', shape=[2 * self.hidden_size[-1], self.word_vec_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_fc = tf.get_variable('b_fc', shape=[self.word_vec_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))

		y_masked_flatten = tf.matmul(stack_outputs_masked_flatten, w_fc) + b_fc

		## Define Loss (cross entropy)
		print 'Defining loss...'

		y_label_flatten = tf.reshape(y_label, [-1], name='y_label_flatten')
		y_label_masked_flatten = tf.boolean_mask(y_label_flatten, mask_flatten, name='y_label_masked_flatten')
		y_label_masked_flatten_nce = tf.reshape(y_label_masked_flatten, [-1, 1], name='y_label_masked_flatten_nce')

		weight_flatten = tf.reshape(weight, [-1], name='weight_flatten')
		weight_masked_flatten = tf.boolean_mask(weight_flatten, mask_flatten, name='weight_masked_flatten')

		b_nce = tf.zeros([self.total_word_count])

		nce_losses = tf.cond(trainable, 
			lambda: tf.nn.sampled_softmax_loss(w_emb, b_nce, y_label_masked_flatten_nce, y_masked_flatten, self.nce_sample, self.total_word_count),
			lambda: tf.nn.sampled_softmax_loss(w_emb_fixed, b_nce, y_label_masked_flatten_nce, y_masked_flatten, self.nce_sample, self.total_word_count))
		self.loss = nce_loss = tf.reduce_mean(nce_losses * weight_masked_flatten, name='nce_loss')

		logits = tf.matmul(y_masked_flatten, w_emb, transpose_b=True) + b_nce
		self.softmax = softmax = tf.nn.softmax(logits, name='softmax')
		self.losses = losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y_label_masked_flatten, self.total_word_count), name='losses')
		#self.loss = loss = tf.reduce_mean(losses * weight_masked_flatten, name='loss')

		adam = tf.train.AdamOptimizer(lr)
		global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))
		self.train_step = train_step = adam.minimize(nce_loss, global_step=global_step)

	def genFeedDict(self, batch_start, data, lr=None, actual_batch_size=None, testing=False, trainable=True):
		if actual_batch_size == None:
			actual_batch_size = self.batch_size

		x_data = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int32)
		for j, sentence in enumerate(data[batch_start: batch_start + actual_batch_size]):
			x_data[j, 0] = self.word_index_dict['<START>']
			for k, word in enumerate(sentence):
				x_data[j, k + 1] = self.word_index_dict[word]
			x_data[j, len(sentence) + 1] = self.word_index_dict['<END>']

		y_label_data = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int64)
		for j, sentence in enumerate(data[batch_start: batch_start + actual_batch_size]):
			for k, word in enumerate(sentence):
				y_label_data[j, k] = self.word_index_dict[word]

		seq_len_data = np.zeros((self.batch_size), dtype=np.int32)
		for j, sentence in enumerate(data[batch_start: batch_start + actual_batch_size]):
			seq_len_data[j] = len(sentence)

		weight_data = np.zeros((self.batch_size, self.max_seq_len), dtype=np.float32)
		for j, sentence in enumerate(data[batch_start: batch_start + actual_batch_size]):
			for k, word in enumerate(sentence):
				freq = self.word_frequency[self.word_index_dict[word]]
				weight_data[j, k] = (math.sqrt(freq / self.subsample_rate) + 1) * self.subsample_rate / freq

		feed_dict = {
			self.x: x_data, 
			self.y_label: y_label_data, 
			self.seq_len: seq_len_data, 
			self.dropout_prob: self.dropout_prob_c if not testing else 0.0,
			self.lr: lr if lr != None else self.lr_init,
			self.weight: weight_data,
			self.trainable: trainable}

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
						if wrong_feed_dict[self.y_label][l, k] != 0:
							wrong_feed_dict[self.y_label][l, k] = random.randrange(1, self.total_word_count)

				wrong_losses = sess.run(self.losses, feed_dict=wrong_feed_dict)

				wrong_vec &= wrong_losses < correct_losses
			wrong_count += np.sum(wrong_vec)
			total_count += correct_losses.shape[0]

		return 1.0 - float(wrong_count) / total_count

	def train(self, load_savepoint=None):
		print 'Start training...'

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=False
		with tf.Session(config=config) as sess:
			writer = tf.summary.FileWriter('logs/', sess.graph)
			init = tf.global_variables_initializer()
			sess.run(init)
			if load_savepoint != None:
				tf.train.Saver().restore(sess, load_savepoint)
			last_val_loss = 999.9
			lr = self.lr_init
			for epoch in range(100):
				print '[Epoch #' + str(epoch) + ']'

				percent = 0
				loss_sum = 0.0
				counter = 0
				patient_counter = 0
				last_loss = 999.9
				trainable = False
				STATE_LEARNING = 0
				STATE_ADJUSTMENT = 1
				state = STATE_LEARNING
				for i in range(0, len(self.training_data) - self.batch_size, self.batch_size):

					feed_dict = self.genFeedDict(i, self.training_data, lr, trainable=trainable)
					_, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
					loss_sum += loss
					counter += 1

					if int(i * 100 / len(self.training_data)) > percent:
						percent = int(i * 100 / len(self.training_data))
						#lr *= self.lr_decay
						#print str(percent) + '%, Training loss:' + str(sess.run(self.loss, feed_dict=feed_dict))
						if state == STATE_LEARNING:
							if loss_sum / counter > last_loss:
								patient_counter += 1
								if patient_counter > self.lr_decay_patient:
									trainable = True
									patient_counter = 0
									lr *= self.lr_decay
									print 'LR changes to ' + str(lr)
									state = STATE_ADJUSTMENT
							else:
								patient_counter = 0
								last_loss = loss_sum / counter
						else:
							if loss_sum / counter < last_loss:
								state = STATE_LEARNING
								last_loss = loss_sum / counter
								patient_counter = 0
							
						print str(percent) + '%, Training loss:' + str(loss_sum / counter)
						print 'Psuedo accuracy:' + str(self.getPseudoAccuracy(random.sample(self.val_data, 5 * self.batch_size + 1), sess))
						if percent % 10 == 0:
							tf.train.Saver().save(sess, self.save_path)
							print 'Saved!'
						loss_sum = 0.0
						counter = 0
							

				val_loss = 0.0
				count = 0
				for i in range(0, len(self.val_data) - self.batch_size, self.batch_size):
					feed_dict = self.genFeedDict(i, self.val_data)
					val_loss += sess.run(self.loss, feed_dict=feed_dict)
					count += 1
				val_loss /= count
				print 'Validation loss:' + str(val_loss)
				#print 'Psuedo accuracy:' + str(self.getPseudoAccuracy(random.sample(self.val_data, len(self.val_data) / 8), sess))

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
				test_feed_dict = self.genFeedDict(i, self.testing_data, actual_batch_size=1, testing=True)
				softmax = sess.run(self.softmax, feed_dict=test_feed_dict)[blank_index]
				'''
				inv = dict((v, k) for k, v in self.word_index_dict.iteritems())
				for j in range(len(softmax)):
					if softmax[j] > 1e-2:
						print j, inv[j], softmax[j]
				'''
				best_score = float('-inf')
				best_choice = None
				for j in range(5):
					choice = self.testing_choices[i][j]
					score = softmax[self.word_index_dict[choice]]
					#print choice, score
					if (score > best_score):
						best_choice = j
						best_score = score
				answers[i] = best_choice
				#raw_input()
				if int(i * 100 / len(self.testing_data)) > percent:
					percent = int(i * 100 / len(self.testing_data))
					print str(percent) + '% on testing'
			
			with open(self.submit_file, 'wb') as file:
				writer = csv.DictWriter(file, ['id', 'answer'])
				writer.writeheader()
				for i, ans in enumerate(answers):
					lut = ['a', 'b', 'c', 'd', 'e']
					writer.writerow({'id': i + 1, 'answer': lut[ans]})

