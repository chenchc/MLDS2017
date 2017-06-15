import json
import math
import numpy as np
import os
import random
import tensorflow as tf
import utils

class ModelS2S:

	## Config
	lstm1_size = 500
	lstm2_size = 500
	batch_size = 30

	model_name = 's2s'
	save_dir = 'data/model_' + model_name
	save_path = save_dir + '/model.ckpt'

	lr_init = 0.001
	lr_decay = 0.5

	epoch_init = 0

	## Constants
	data = None
	
	word_vec_dim = None
	max_question_len = None
	max_answer_len = None
	total_word_count = None

	## Placeholder tensors
	question = None
	question_len = None
	ref_answer = None
	ref_answer_len = None
	training = None

	## Output tensors
	loss = None
	answer = None
	train_step = None
	lr = None

	def __init__(self):
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.__fillConstants()
		self.__defineModel()
	
	def __fillConstants(self):
		self.data = utils.Utils()

		self.word_vec_dim = self.data.word_vec_dict.values()[0].shape[0]
		self.max_question_len = max([len(pair[0]) for pair in self.data.conv_pair])
		print 'Max question length: {}'.format(self.max_question_len)
		self.max_answer_len = max([len(pair[1]) for pair in self.data.conv_pair])
		print 'Max answer length: {}'.format(self.max_answer_len)
		self.total_word_count = len(self.data.word_list)
		print 'Total word count: {}'.format(self.total_word_count)
		print 'Total training samples: {}'.format(len(self.data.conv_pair))
	
	def __defineModel(self):
		print 'Defining model...'

		## Global step
		global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))

		## Input placeholders
		self.question = question = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_question_len], name='question')
		self.question_len = question_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='question_len')
		self.ref_answer = ref_answer = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_answer_len], name='ref_answer')
		self.ref_answer_len = ref_answer_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='ref_answer_len')
		self.training = training = tf.placeholder(tf.bool, shape=[], name='training')

		## Word embedding
		w_word_emb = tf.constant(np.array([self.data.word_vec_dict[self.data.word_list[i]] for i in range(self.total_word_count)], dtype=np.float32), dtype=tf.float32)

		## Question 
		question_word_emb = tf.nn.embedding_lookup(w_word_emb, question)
		lstm1_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm1_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
		lstm1_output, lstm1_state = tf.nn.dynamic_rnn(
			lstm1_basic_cell, 
			inputs=question_word_emb, 
			sequence_length=question_len,
			dtype=tf.float32,
			scope='lstm1')
		#question_emb = tf.gather(tf.transpose(lstm1_output, [1, 0, 2]), -1)
		
		## Answer
		lstm2_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm2_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))

		w_logits = tf.get_variable('w_logits', shape=[self.lstm2_size, self.total_word_count], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_logits = tf.get_variable('b_logits', shape=[self.total_word_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		ref_answer_time_major = tf.transpose(ref_answer, [1, 0])

		def lstm2_loop_fn(time, cell_output, cell_state, loop_state):
			# elements_finished
			elements_finished = (time >= ref_answer_len)
			all_finished = tf.reduce_all(elements_finished)

			## Softmax Logits (dummy)
			answer = tf.zeros([self.batch_size], dtype=tf.int32)
			if cell_output != None:
				cell_output_masked = tf.where(
					elements_finished,
					tf.zeros([self.batch_size, self.lstm2_size], dtype=tf.float32),
					cell_output)

				logits = tf.add(tf.matmul(cell_output_masked, w_logits), b_logits)
				answer_candidate = tf.transpose(tf.nn.top_k(logits, k=2)[1], [1, 0])
				answer = tf.where(tf.equal(answer_candidate[0], loop_state), answer_candidate[1], answer_candidate[0])

			# emit_output
			emit_output = cell_output

			# next_input
			if cell_output is None:
				input_word = tf.fill([self.batch_size], self.data.word_index_dict[utils.Utils.MARKER_BOS])
				input_word_emb = tf.nn.embedding_lookup(w_word_emb, input_word)
				bos_bit = tf.fill([self.batch_size, 1], 1.0)
				oov_bit = tf.zeros([self.batch_size, 1])
				#next_input = tf.concat([bos_bit, oov_bit, input_word_emb, question_emb], axis=1)
				next_input = tf.concat([bos_bit, oov_bit, input_word_emb], axis=1)
			else:
				input_word = tf.cond(
					training,
					lambda: ref_answer_time_major[time-1],
					lambda: answer)
				input_word_emb = tf.nn.embedding_lookup(w_word_emb, input_word)
				bos_bit = tf.fill([self.batch_size, 1], 0.0)
				oov_bit = tf.expand_dims(tf.cast(tf.equal(input_word, self.data.word_index_dict[utils.Utils.MARKER_OOV]), tf.float32), axis=-1)
				next_input = tf.cond(
					all_finished,
					lambda: tf.zeros([self.batch_size, self.word_vec_dim + 2]),
					lambda: tf.concat([bos_bit, oov_bit, input_word_emb], axis=1))
					#lambda: tf.zeros([self.batch_size, self.word_vec_dim + self.lstm1_size + 2]),
					#lambda: tf.concat([oov_bit, bos_bit, input_word_emb, question_emb], axis=1))

			# next_cell_state
			if cell_output is None:
				#next_cell_state = lstm2_basic_cell.zero_state(self.batch_size, tf.float32)
				next_cell_state = lstm1_state
			else:
				next_cell_state = cell_state

			# next_loop_state
			next_loop_state = answer

			return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
			
		with tf.variable_scope('lstm2'):
			lstm2_output_ta, _, _ = tf.nn.raw_rnn(lstm2_basic_cell, lstm2_loop_fn)
	
		lstm2_output_trunc = tf.transpose(lstm2_output_ta.stack(), [1, 0, 2])
		lstm2_output = tf.pad(lstm2_output_trunc, [[0, 0], [0, self.max_answer_len - lstm2_output_ta.size()], [0, 0]], name='lstm2_output')
		lstm2_output_flat = tf.reshape(lstm2_output, [-1, self.lstm2_size])

		## Softmax Logits
		logits_flat = tf.add(tf.matmul(lstm2_output_flat, w_logits), b_logits)
		logits = tf.reshape(logits_flat, [self.batch_size, self.max_answer_len, self.total_word_count], name='logits') 
		answer_candidate = tf.transpose(tf.nn.top_k(logits, k=2)[1], [2, 1, 0])
		answer_list = []
		for time in range(self.max_answer_len):
			if time == 0:
				answer_list.append(answer_candidate[0][0])
			else:
				answer_list.append(tf.where(tf.equal(answer_candidate[0][time], answer_list[-1]),
					answer_candidate[1][time],
					answer_candidate[0][time]))
		self.answer = answer = tf.transpose(tf.stack(answer_list), [1, 0])
	
		## Loss
		#correct_mask = tf.cast(tf.cumprod(tf.cast(tf.equal(answer, ref_answer), tf.int32), axis=-1, exclusive=True), tf.bool)
		len_mask = tf.sequence_mask(ref_answer_len, maxlen=self.max_answer_len, name='len_mask')
		oov_mask = tf.not_equal(ref_answer, self.data.word_index_dict[utils.Utils.MARKER_OOV])
		#mask = tf.logical_and(correct_mask, tf.logical_and(len_mask, oov_mask))
		mask = tf.logical_and(len_mask, oov_mask)
		#mask = len_mask

		logits_mask = tf.boolean_mask(logits, mask, name='logits_mask')
		ref_answer_mask = tf.boolean_mask(ref_answer, mask, name='ref_answer_mask')
		losses = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(ref_answer_mask, self.total_word_count, dtype=tf.float32),
			logits=logits_mask, name='losses')
		self.loss = loss = tf.reduce_mean(losses)

		## Optimize
		self.lr = lr = tf.get_variable('lr', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.lr_init))
		optimizer = tf.train.AdamOptimizer(lr)
		self.train_step = train_step = optimizer.minimize(loss, global_step=global_step)

		## Print trainable variables
		print 'Variables:'
		for i in tf.global_variables():
			print i.name, i.get_shape()

	def _prepareTrainPairList(self):
		random.shuffle(self.data.conv_pair)
		
		return self.data.conv_pair

	def _prepareTrainFeedDictList(self, pair_list, index):

		question_data = np.zeros(shape=(self.batch_size, self.max_question_len), dtype=int)
		question_len_data = np.zeros(shape=(self.batch_size), dtype=int)
		ref_answer_data = np.zeros(shape=(self.batch_size, self.max_answer_len), dtype=int)
		ref_answer_len_data = np.zeros(shape=(self.batch_size), dtype=int)

		for j in range(self.batch_size):
			question_len_data[j] = len(pair_list[index + j][0])
			for k in range(question_len_data[j]):
				question_data[j, k] = self.data.word_index_dict[pair_list[index + j][0][k]]
			ref_answer_len_data[j] = len(pair_list[index + j][1])
			for k in range(ref_answer_len_data[j]):
				ref_answer_data[j, k] = self.data.word_index_dict[pair_list[index + j][1][k]]
			
		feed_dict = {
			self.question: question_data,
			self.question_len: question_len_data,
			self.ref_answer: ref_answer_data,
			self.ref_answer_len: ref_answer_len_data,
			self.training: True,
			}

		return feed_dict
	
	def _prepareTestFeedDictList(self, q):

		question_data = np.zeros(shape=(self.batch_size, self.max_question_len), dtype=int)
		question_len_data = np.zeros(shape=(self.batch_size), dtype=int)
		ref_answer_data = np.zeros(shape=(self.batch_size, self.max_answer_len), dtype=int)
		ref_answer_len_data = np.full(shape=(self.batch_size), fill_value=self.max_answer_len, dtype=int)

		question_len_data[0] = len(q)
		for k in range(question_len_data[0]):
			question_data[0, k] = self.data.word_index_dict[q[k]]

		feed_dict = {
			self.question: question_data,
			self.question_len: question_len_data,
			self.ref_answer: ref_answer_data,
			self.ref_answer_len: ref_answer_len_data,
			self.training: False,
			}

		return feed_dict

	def train(self, savepoint=None):
		print 'Start training...'

		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if savepoint != None:
				tf.train.Saver().restore(sess, savepoint)

			last_global_loss = float('inf')
			global_loss_history = []
			for epoch in range(self.epoch_init, 1000, 1):
				print '[Epoch #' + str(epoch) + ']'

				train_pair_list = self._prepareTrainPairList()
				
				percent = 0
				loss_history = []
				for i in range(0, len(train_pair_list) - self.batch_size, self.batch_size):
					feed_dict = self._prepareTrainFeedDictList(train_pair_list, i)
					_, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
					loss_history.append(loss)
					global_loss_history.append(loss)

					if i * 100 / len(train_pair_list) > percent:
						percent += 1
						print 'Process: {}%, Training loss: {}'.format(percent, np.mean(loss_history))
						print 'Samples:'
						for i in range(len(self.data.sample_test_questions)):
							question = self.data.sample_test_questions[i]
							feed_dict = self._prepareTestFeedDictList(question)
							answer = sess.run(self.answer, feed_dict=feed_dict)
							answer = [self.data.word_list[idx] for idx in answer.tolist()[0]]
							answer = self.data.tokenListToCaption(answer)
							question = self.data.tokenListToCaption(question)
							print 'Q: {} A: {}'.format(question, answer)
						
				global_loss = np.mean(global_loss_history)
				print 'Epoch Loss: {}'.format(global_loss)
				global_loss_history = []
				if last_global_loss < global_loss:
					lr = sess.run(self.lr)
					lr *= self.lr_decay
					sess.run(tf.assign(self.lr, lr))
					print 'LR changes to {}'.format(lr)
				last_global_loss = global_loss

				tf.train.Saver().save(sess, self.save_path, global_step=epoch)

	def test(self, savepoint):
		print 'Start training...'

		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			tf.train.Saver().restore(sess, savepoint)

			answer_list = []
			for i in range(len(self.data.test_questions)):
				question = self.data.test_questions[i]
				feed_dict = self._prepareTestFeedDictList(question)
				answer = sess.run(self.answer, feed_dict=feed_dict)
				answer = [self.data.word_list[idx] for idx in answer.tolist()[0]]
				answer = self.data.tokenListToCaption(answer)
				question = self.data.tokenListToCaption(question)
				print 'Q: {} A: {}'.format(question, answer)
				answer_list.append(answer)

			file = open('output.txt', 'w')
			for answer in answer_list:
				file.write('{}\n'.format(answer))
		

model = ModelS2S()
#model.train('data/model_s2s/model.ckpt-9')
model.test('data/model_s2s/model.ckpt-10')
