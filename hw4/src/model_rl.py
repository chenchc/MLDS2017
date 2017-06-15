import bleu_eval
import json
import math
import numpy as np
import os
import random
import tensorflow as tf
import utils

class ModelRL:

	## Config
	lstm1_size = 500
	lstm2_size = 500
	batch_size = 30
	baseline = 0.0

	model_name = 'rl'
	save_dir = 'data/model_' + model_name
	save_path = save_dir + '/model.ckpt'

	lr_init = 0.0001
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
	#ref_answer_len = None
	training = None

	## Output tensors
	loss = None
	actor_loss = None
	critic_loss = None
	mean_reward = None
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
		#self.ref_answer_len = ref_answer_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='ref_answer_len')
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

		def lstm2_loop_fn(time, cell_output, cell_state, loop_state):
			## Softmax Logits (dummy)
			answer = tf.zeros([self.batch_size], dtype=tf.int32)
			if cell_output != None:
				logits = tf.add(tf.matmul(cell_output, w_logits), b_logits)
				answer = tf.cond(training,
					lambda: tf.squeeze(tf.cast(tf.multinomial(logits, 1), tf.int32)),
					#lambda: tf.cast(tf.argmax(logits, -1), tf.int32),
					lambda: tf.cast(tf.argmax(logits, -1), tf.int32))

			# elements_finished
			elements_finished = tf.logical_or(tf.equal(answer, tf.constant(self.data.word_index_dict[utils.Utils.MARKER_EOS])), time >= self.max_answer_len)
			all_finished = tf.reduce_all(elements_finished)

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
				input_word = answer
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
			if cell_output is None:
				next_loop_state = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
			else:
				next_loop_state = loop_state.write(time - 1, answer)

			return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
			
		with tf.variable_scope('lstm2'):
			lstm2_output_ta, _, answer_ta = tf.nn.raw_rnn(lstm2_basic_cell, lstm2_loop_fn)
	
		lstm2_output_trunc = tf.transpose(lstm2_output_ta.stack(), [1, 0, 2])
		lstm2_output = tf.pad(lstm2_output_trunc, [[0, 0], [0, self.max_answer_len - lstm2_output_ta.size()], [0, 0]], name='lstm2_output')
		lstm2_output = tf.reshape(lstm2_output, [self.batch_size, self.max_answer_len, self.lstm2_size])
		lstm2_output_flat = tf.reshape(lstm2_output, [-1, self.lstm2_size])

		## Actor
		logits_flat = tf.add(tf.matmul(lstm2_output_flat, w_logits), b_logits)
		logits = tf.reshape(logits_flat, [self.batch_size, self.max_answer_len, self.total_word_count], name='logits') 
		#self.answer = answer = tf.cast(tf.argmax(logits, axis=-1, name='answer'), tf.int32)
		self.answer = answer = tf.pad(tf.transpose(answer_ta.stack(), [1, 0]), [[0, 0], [0, self.max_answer_len - answer_ta.size()]])
		actor_prob = tf.nn.softmax(logits)
		#actor_prob_ans = tf.reduce_max(actor_prob, axis=-1)
		answer_indice = tf.concat([
			tf.expand_dims(tf.constant(np.tile(np.expand_dims(np.arange(self.batch_size, dtype=np.int32), -1), (1, self.max_answer_len))), -1),
			tf.expand_dims(tf.constant(np.tile(np.expand_dims(np.arange(self.max_answer_len, dtype=np.int32), 0), (self.batch_size, 1))), -1),
			tf.expand_dims(answer, -1)], -1)
		actor_prob_ans = tf.gather_nd(actor_prob, answer_indice)

		## Critic
		answer_word_emb = tf.nn.embedding_lookup(w_word_emb, answer)
		with tf.variable_scope('rl_only'):
			lstm_critic_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm2_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
			seq_len = tf.fill([self.batch_size], answer_ta.size())
			lstm_critic_output, _ = tf.nn.dynamic_rnn(
				lstm_critic_basic_cell, 
				inputs=answer_word_emb,
				sequence_length=seq_len,
				initial_state=lstm1_state,
				dtype=tf.float32,
				scope='lstm_critic')
			fc_value = tf.contrib.layers.fully_connected(lstm_critic_output, self.lstm2_size, activation_fn=tf.sigmoid)
			value_ans = tf.squeeze(tf.contrib.layers.fully_connected(fc_value, 1, weights_initializer=tf.random_uniform_initializer(minval=-0.0001, maxval=0.0001), activation_fn=None))
			baseline = tf.get_variable('baseline', dtype=tf.float32, shape=[], initializer=tf.constant_initializer(self.baseline))
	
		'''
		lstm_critic_output_flat = tf.reshape(lstm_critic_output, [-1, self.lstm2_size])
		value_flat = tf.add(tf.matmul(lstm_critic_output_flat, w_value), b_value)
		value = tf.reshape(value_flat, [self.batch_size, self.max_answer_len, self.total_word_count], name='value') 
		value_ans = tf.gather_nd(value, answer_indice)
		'''

		reward_by_critic = value_ans - tf.concat([tf.fill([self.batch_size, 1], self.baseline), tf.slice(value_ans, [0, 0], [-1, self.max_answer_len - 1])], axis=-1)
		#reward_by_critic = value_ans 

		## Reward
		def tfBleu(ref, ans):
			ans = ans.tolist()
			ans = [[self.data.word_list[x] for x in y] for y in ans]
			ans = [self.data.tokenListToCaption(y) for y in ans]
			ref = ref.tolist()
			ref = [[self.data.word_list[x] for x in y] for y in ref]
			ref = [self.data.tokenListToCaption(y) for y in ref]
			bleu = [bleu_eval.BLEU_fn(ans[i], ref[i]) for i in range(self.batch_size)]
			bleu = [np.exp(i) - 1.0 for i in bleu]
			for i in range(len(ans)):
				if "don't know" in ans[i] or "dont know" in ans[i]:
					bleu[i] = -1.0
				else:
					bleu[i] = 0.0
			return np.array(bleu).astype(np.float32)

		reward = tf.py_func(tfBleu, [ref_answer, answer], tf.float32, stateful=False, name='reward')


		## Loss
		answer_eos_mask = tf.equal(answer, tf.constant(self.data.word_index_dict[utils.Utils.MARKER_EOS]))
		answer_valid_mask = tf.logical_not(tf.cast(tf.cumsum(tf.cast(answer_eos_mask, tf.int32), axis=-1, exclusive=True), tf.bool))

		#actor_objs = tf.stop_gradient(reward_by_critic) * tf.log(actor_prob_ans)
		actor_objs = tf.tile(tf.expand_dims(reward, -1), [1, self.max_answer_len]) * tf.log(actor_prob_ans)
		self.actor_loss = actor_loss = -tf.reduce_mean(tf.reduce_mean(actor_objs * tf.cast(answer_valid_mask, tf.float32), axis=-1))

		'''
		critic_losses = tf.squared_difference(tf.where(
			answer_eos_mask,
			tf.tile(tf.expand_dims(reward, -1), [1, self.max_answer_len]),
			tf.concat([tf.slice(value_ans, [0, 1], [-1, -1]), tf.expand_dims(reward, -1)], axis=-1)), value_ans)
		'''
		critic_losses = tf.squared_difference(
			tf.tile(tf.expand_dims(reward, -1), [1, self.max_answer_len]),
			value_ans)
		#critic_losses = tf.Print(critic_losses, [value_ans, reward_by_critic], summarize=20)
		self.critic_loss = critic_loss = tf.reduce_mean(tf.reduce_mean(critic_losses * tf.cast(answer_valid_mask, tf.float32), axis=-1))
		baseline_loss = tf.squared_difference(baseline, tf.reduce_mean(reward))
		#baseline_loss = tf.Print(baseline_loss, [baseline], summarize=20)

		self.loss = loss = actor_loss #+ critic_loss + 10.0 * baseline_loss
		self.mean_reward = tf.reduce_mean(reward)

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
			#self.ref_answer_len: ref_answer_len_data,
			self.training: True,
			}

		return feed_dict
	
	def _prepareTestFeedDictList(self, q):

		question_data = np.zeros(shape=(self.batch_size, self.max_question_len), dtype=int)
		question_len_data = np.zeros(shape=(self.batch_size), dtype=int)
		ref_answer_data = np.zeros(shape=(self.batch_size, self.max_answer_len), dtype=int)
		#ref_answer_len_data = np.full(shape=(self.batch_size), fill_value=self.max_answer_len, dtype=int)

		question_len_data[0] = len(q)
		for k in range(question_len_data[0]):
			question_data[0, k] = self.data.word_index_dict[q[k]]

		feed_dict = {
			self.question: question_data,
			self.question_len: question_len_data,
			self.ref_answer: ref_answer_data,
			#self.ref_answer_len: ref_answer_len_data,
			self.training: False,
			}

		return feed_dict

	def train(self, savepoint=None):
		print 'Start training...'

		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if savepoint != None:
				tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='^(?!rl_only/).*$')).restore(sess, savepoint)

			last_global_loss = float('inf')
			global_loss_history = []
			for epoch in range(self.epoch_init, 1000, 1):
				print '[Epoch #' + str(epoch) + ']'

				train_pair_list = self._prepareTrainPairList()
				
				percent = 0
				loss_history = []
				actor_loss_history = []
				critic_loss_history = []
				metric_history = []
				for i in range(0, len(train_pair_list) - self.batch_size, self.batch_size):
					feed_dict = self._prepareTrainFeedDictList(train_pair_list, i)
					_, loss, actor_loss, critic_loss, metric = sess.run([self.train_step, self.loss, self.actor_loss, self.critic_loss, self.mean_reward], feed_dict=feed_dict)
					loss_history.append(loss)
					actor_loss_history.append(actor_loss)
					critic_loss_history.append(critic_loss)
					metric_history.append(metric)
					global_loss_history.append(loss)

					if i * 100 / len(train_pair_list) > percent:
						percent += 1
						print 'Process: {}%, Training loss: {}, Actor loss: {}, Critic loss: {}, Avg score: {}'.format(percent, np.mean(loss_history), np.mean(actor_loss_history), np.mean(critic_loss_history),np.mean(metric_history))
						loss_history = []
						actor_loss_history = []
						critic_loss_history = []
						metric_history = []
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
		

model = ModelRL()
#model.train()
#model.train('data/model_s2s/model.ckpt-10')
model.test('data/model_rl/model.ckpt-0')
