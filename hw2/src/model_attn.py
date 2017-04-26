import bleu_eval
import math
import numpy as np
import os
import random
import read_util
import tensorflow as tf
import pyter

class ModelAttn:

	## Config
	frame_emb_size = 500
	lstm1_size = 1000
	lstm2_size = 1000
	batch_size = 80
	frame_drop_prob = 0.0
	word_emb_drop_prob = 0.0
	lstm1_drop_prob = 0.0
	lstm2_drop_prob = 0.5
	subsample_rate = 0.0
	word_list_emb_size = 500

	model_name = 'attn'
	save_dir = 'data/model_' + model_name
	save_path = save_dir + '/model.ckpt'
	submit_path = 'data/submit_' + model_name
	lr_init = 0.0001
	lr_decay_steps = 500
	lr_decay_rate = 1.0
	lr_momentum = 0.9
	schd_sampling_decay = 0.0
	patience = 1000

	## Constants
	data = None
	
	word_vec_dim = None
	frame_vec_dim = None
	max_caption_len = None
	max_video_len = None
	total_word_count = None

	## Placeholder tensors
	frames = None
	ref_caption_len = None
	ref_caption = None
	training = None
	weight = None

	## Output tensors
	loss = None
	caption = None
	train_step = None
	word_predict_loss = None
	seq_loss = None

	def __init__(self):
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.__fillConstants()
		self.__defineModel()

	def __fillConstants(self):
		self.data = read_util.ReadUtil()

		self.word_vec_dim = self.data.word_vec_dict.values()[0].shape[0]
		self.frame_vec_dim = self.data.train_feat_list[0].shape[1]
		self.max_caption_len = max([len(c) for sublist in self.data.train_caption_list for c in sublist]) 
		self.max_video_len = max([v.shape[0] for v in self.data.train_feat_list])
		self.total_word_count = len(self.data.word_list)

	def __defineModel(self):
		print 'Defining model...'

		## Global step
		global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))

		## Input placeholders
		self.frames = frames = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_video_len, self.frame_vec_dim], name='frames')
		self.ref_caption_len = ref_caption_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='ref_caption_len')
		self.ref_caption = ref_caption = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_caption_len], name='ref_caption')
		self.training = training = tf.placeholder(tf.bool, shape=[], name='training')
		self.weight = weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_caption_len], name='weight')
		self.ref_word_list = ref_word_list = tf.placeholder(tf.float32, shape=[self.batch_size, self.total_word_count], name='ref_word_list')

		## Frame embedding
		frame_drop_prob = tf.cond(training, lambda: tf.constant(self.frame_drop_prob), lambda: tf.constant(0.0)) ###
		frames = tf.nn.dropout(frames, 1.0 - frame_drop_prob) ###
		frames_2 = tf.reshape(frames, [-1, self.frame_vec_dim], name='frames_2')
		w_frames_emb = tf.get_variable('w_frames_emb', shape=[self.frame_vec_dim, self.frame_emb_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_frames_emb = tf.get_variable('b_frames_emb', shape=[self.frame_emb_size], dtype=tf.float32, initializer=tf.constant_initializer(0.2))
		frames_emb = tf.add(tf.matmul(frames_2, w_frames_emb), b_frames_emb, name='frames_emb')
		frames_emb_2 = tf.reshape(frames_emb, [self.batch_size, self.max_video_len, self.frame_emb_size], name='frames_emb_2')

		## LSTM 1
		frames_emb_padded = tf.pad(frames_emb_2, [[0, 0], [0, self.max_caption_len], [0, 0]], name='frames_emb_padded')
		lstm1_drop_prob = tf.cond(training, lambda: tf.constant(self.lstm1_drop_prob), lambda: tf.constant(0.0))
		#lstm1_basic_cell_layer = []
		#for i in range(2):
		#	lstm1_basic_cell_layer.append(tf.contrib.rnn.LSTMCell(self.lstm1_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08)))
		#lstm1_basic_cell = tf.contrib.rnn.MultiRNNCell(lstm1_basic_cell_layer)
		lstm1_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm1_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
		lstm1_basic_cell = tf.contrib.rnn.DropoutWrapper(lstm1_basic_cell, output_keep_prob=(1.0 - lstm1_drop_prob))

		with tf.variable_scope('lstm1'):
			lstm1_output, lstm1_state = tf.nn.dynamic_rnn(
				lstm1_basic_cell, inputs=frames_emb_2, 
				dtype=tf.float32)
		'''
		## Word prediction
		w_hidden_word_predict = tf.get_variable('w_hidden_word_predict', shape=[self.lstm1_size, 2000], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_hidden_word_predict = tf.get_variable('b_hidden_word_predict', shape=[2000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		hidden_word_predict = tf.nn.relu(tf.matmul(video_emb, w_hidden_word_predict) + b_hidden_word_predict)
		hidden_word_predict = tf.nn.dropout(hidden_word_predict, tf.cond(training, lambda: tf.constant(0.5), lambda: tf.constant(1.0)))

		w_word_predict = tf.get_variable('w_word_predict', shape=[2000, self.total_word_count], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_word_predict = tf.get_variable('b_word_predict', shape=[self.total_word_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		
		word_predict_logits = tf.matmul(hidden_word_predict, w_word_predict) + b_word_predict
		word_predict = tf.sigmoid(word_predict_logits)
		word_predict_stop_gd = tf.stop_gradient(word_predict)
		self.word_predict_loss = word_predict_loss = tf.losses.mean_squared_error(ref_word_list, word_predict)
		'''
		#w_word_list_emb = tf.get_variable('w_word_list_emb', shape=[self.total_word_count, self.word_list_emb_size], dtype=tf.float32, initializer=tf.diag(tf.ones(self.word_list_emb_size)))
		#word_list_emb = tf.matmul(word_predict_stop_gd, w_word_list_emb)
		#word_list_emb = tf.slice(word_predict_stop_gd, [0, 0], [-1, self.word_list_emb_size])

		## LSTM 2
		lstm2_drop_prob = tf.cond(training, lambda: tf.constant(self.lstm2_drop_prob), lambda: tf.constant(0.0))
		#lstm2_basic_cell_layer = []
		#for i in range(2):
		#	lstm2_basic_cell_layer.append(tf.contrib.rnn.LSTMCell(self.lstm2_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08)))
		#lstm2_basic_cell = tf.contrib.rnn.MultiRNNCell(lstm2_basic_cell_layer)
		lstm2_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm2_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
		lstm2_basic_cell = tf.contrib.rnn.DropoutWrapper(lstm2_basic_cell, output_keep_prob=(1.0 - lstm2_drop_prob))

		ref_caption_2 = tf.transpose(ref_caption, perm=[1, 0])

		w_logits = tf.get_variable('w_logits', shape=[self.lstm2_size, self.total_word_count], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
		b_logits = tf.get_variable('b_logits', shape=[self.total_word_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		w_word_emb_init = tf.constant(np.array([self.data.word_vec_dict[self.data.word_list[i]] for i in range(self.total_word_count)], dtype=np.float32), dtype=tf.float32)
		#w_word_emb = tf.get_variable('w_word_emb', dtype=tf.float32, initializer=w_word_emb_init)
		w_word_emb = w_word_emb_init

		#self.prior_factor = prior_factor = tf.get_variable('prior_factor', initializer=0.1, trainable=True)
		#word_predict_smooth = tf.pow(word_predict_stop_gd, prior_factor)
		#word_predict_smooth_inv = tf.log(word_predict_smooth)

		def lstm2_loop_fn(time, cell_output, cell_state, loop_state):
			# elements_finished
			elements_finished = (time >= ref_caption_len)
			all_finished = tf.reduce_all(elements_finished)


			## Softmax Logits (dummy)
			caption = tf.zeros([self.batch_size], dtype=tf.int32)
			if cell_output != None:
				cell_output_masked = tf.where(
					elements_finished,
					tf.zeros([self.batch_size, self.lstm2_size], dtype=tf.float32),
					cell_output)

				logits = tf.add(tf.matmul(cell_output_masked, w_logits), b_logits)
				aggregated_logits = logits #+ word_predict_smooth_inv
				caption_candidate = tf.transpose(tf.nn.top_k(aggregated_logits, k=2)[1], [1, 0])
				caption = tf.where(tf.equal(caption_candidate[0], loop_state), caption_candidate[1], caption_candidate[0])
				#caption = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

			# emit_output
			emit_output = cell_output

			# next_input
			if cell_output is None:
				## Attention
				attn_alpha = tf.reduce_sum(tf.nn.l2_normalize(tf.zeros([self.lstm1_size]), -1) * tf.nn.l2_normalize(lstm1_output, -1), axis=-1)
				attn_alpha_softmax = tf.nn.softmax(attn_alpha, dim=-1)
				attn_c = tf.reduce_sum(tf.tile(tf.expand_dims(attn_alpha_softmax, 2), [1, 1, self.lstm1_size]) * lstm1_output, axis=1)

				input_word = tf.fill([self.batch_size], self.data.word_index_dict[read_util.ReadUtil.MARKER_BOS])
				input_word_emb = tf.nn.embedding_lookup(w_word_emb, input_word)
				word_emb_drop_prob = tf.cond(training, lambda: tf.constant(self.word_emb_drop_prob), lambda: tf.constant(0.0)) ###
				input_word_emb = tf.nn.dropout(input_word_emb, 1.0 - word_emb_drop_prob) ###
				bos_bit = tf.fill([self.batch_size, 1], 1.0)
				next_input = tf.concat([bos_bit, input_word_emb, attn_c], axis=1)
			else:
				## Attention
				attn_alpha = tf.reduce_sum(tf.nn.l2_normalize(cell_output, -1) * tf.nn.l2_normalize(lstm1_output, -1), axis=-1)
				attn_alpha_softmax = tf.nn.softmax(attn_alpha, dim=-1)
				attn_c = tf.reduce_sum(tf.tile(tf.expand_dims(attn_alpha_softmax, 2), [1, 1, self.lstm1_size]) * lstm1_output, axis=1)

				select_ref_prob = tf.maximum(1.0 - self.schd_sampling_decay * tf.cast(global_step, tf.float32), 0.0)
				input_word = tf.cond(
					training,
					lambda: tf.where(tf.random_uniform([self.batch_size]) < select_ref_prob, ref_caption_2[time-1], caption),
					lambda: caption)
				input_word_emb = tf.nn.embedding_lookup(w_word_emb, input_word)
				word_emb_drop_prob = tf.cond(training, lambda: tf.constant(self.word_emb_drop_prob), lambda: tf.constant(0.0)) ###
				input_word_emb = tf.nn.dropout(input_word_emb, 1.0 - word_emb_drop_prob) ###
				bos_bit = tf.fill([self.batch_size, 1], 0.0)
				next_input = tf.cond(
					all_finished,
					lambda: tf.zeros([self.batch_size, self.lstm1_size + self.word_vec_dim + 1]),
					lambda: tf.concat([bos_bit, input_word_emb, attn_c], axis=1))

			# next_cell_state
			if cell_output is None:
				next_cell_state = lstm2_basic_cell.zero_state(self.batch_size, tf.float32)
			else:
				next_cell_state = cell_state

			# next_loop_state
			next_loop_state = caption

			return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

		with tf.variable_scope('lstm2'):
			lstm2_output_ta, _, _ = tf.nn.raw_rnn(lstm2_basic_cell, lstm2_loop_fn)
	
		lstm2_output_trunc = tf.transpose(lstm2_output_ta.stack(), [1, 0, 2])
		lstm2_output = tf.pad(lstm2_output_trunc, [[0, 0], [0, self.max_caption_len - lstm2_output_ta.size()], [0, 0]], name='lstm2_output')
		lstm2_output_flat = tf.reshape(lstm2_output, [-1, self.lstm2_size])

		## Softmax Logits
		logits_flat = tf.add(tf.matmul(lstm2_output_flat, w_logits), b_logits)
		logits = tf.reshape(logits_flat, [self.batch_size, self.max_caption_len, self.total_word_count], name='logits') 
		aggregated_logits = logits #+ tf.tile(tf.expand_dims(word_predict_smooth_inv, axis=1), [1, self.max_caption_len, 1])
		#aggregated_logits = logits * tf.pow(tf.tile(tf.expand_dims(word_predict_stop_gd, axis=1), [1, self.max_caption_len, 1]), prior_factor)
		#aggregated_logits = tf.log(aggregated_sigmoid / (1 - aggregated_sigmoid))
		caption_candidate = tf.transpose(tf.nn.top_k(aggregated_logits, k=2)[1], [2, 1, 0])
		caption_list = []
		for time in range(self.max_caption_len):
			if time == 0:
				caption_list.append(caption_candidate[0][0])
			else:
				caption_list.append(tf.where(tf.equal(caption_candidate[0][time], caption_list[-1]),
					caption_candidate[1][time],
					caption_candidate[0][time]))
		self.caption = caption = tf.transpose(tf.stack(caption_list), [1, 0])
		#self.caption = caption = tf.argmax(logits, axis=2, name='caption')

		## Correct mask
		#correct_mask = tf.concat([tf.fill([self.batch_size, 1], True), tf.slice(tf.equal(caption, ref_caption), [0, 0], [-1, self.max_caption_len - 1])], axis=-1)
		correct_mask = tf.cast(tf.cumprod(tf.cast(tf.equal(caption, ref_caption), tf.int32), axis=-1, exclusive=True), tf.bool)
		#critical_mask = tf.logical_xor(correct_mask, tf.cast(tf.cumprod(tf.cast(tf.equal(caption, ref_caption), tf.int32), axis=-1, exclusive=False), tf.bool))

		#position_weight = 1.0 / tf.cumsum(tf.fill([self.batch_size, self.max_caption_len], 1.0), axis=-1, exclusive=False, reverse=True)
		#position_weight = tf.cumsum(tf.tile(tf.expand_dims(1.0 / tf.cast(ref_caption_len, tf.float32), axis=-1), [1, self.max_caption_len]), exclusive=False, reverse=True)
		#position_weight = tf.concat([1.0 / tf.cumsum(tf.fill([self.batch_size, self.max_caption_len / 2], 1.0), axis=-1, exclusive=False, reverse=True), tf.fill([self.batch_size, self.max_caption_len - self.max_caption_len / 2], 1.0)], axis=-1)

		## Loss
		mask = tf.sequence_mask(ref_caption_len, maxlen=self.max_caption_len, name='mask')
		mask = tf.logical_and(mask, correct_mask)

		#critical_mask_masked = tf.boolean_mask(critical_mask, mask)

		logits_mask = tf.boolean_mask(aggregated_logits, mask, name='logits_mask')
		ref_caption_mask = tf.boolean_mask(ref_caption, mask, name='ref_caption_mask')
		weight_mask = tf.boolean_mask(weight, mask, name='weight_mask')
		seq_losses = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(ref_caption_mask, self.total_word_count, dtype=tf.float32),
			logits=logits_mask, name='seq_losses') * weight_mask

		self.seq_loss = seq_loss = tf.reduce_mean(seq_losses)
		loss = seq_loss #+ 1000.0 * word_predict_loss

		## Optimize
		lr = tf.train.exponential_decay(self.lr_init, global_step, self.lr_decay_steps, self.lr_decay_rate)
		#optimizer = tf.train.MomentumOptimizer(lr, self.lr_momentum)
		optimizer = tf.train.AdamOptimizer(lr)
		self.train_step = train_step = optimizer.minimize(loss, global_step=global_step)

	def _prepareTrainPairList(self):
		train_pair_list = []
		for video_id in range(len(self.data.train_caption_list)):
			for caption in self.data.train_caption_list[video_id]:
				train_pair_list.append((self.data.train_feat_list[video_id], caption))
		random.shuffle(train_pair_list)
		
		return train_pair_list

	def _prepareTrainFeedDictList(self, pair_list, index):

		frames_data = np.zeros(shape=(self.batch_size, self.max_video_len, self.frame_vec_dim), dtype=float)
		ref_caption_len_data = np.zeros(shape=(self.batch_size), dtype=int)
		ref_caption_data = np.zeros(shape=(self.batch_size, self.max_caption_len), dtype=int)
		#ref_caption_len_data = np.full((self.batch_size), self.max_caption_len, dtype=int)
		#ref_caption_data = np.full((self.batch_size, self.max_caption_len), self.data.word_index_dict[self.data.MARKER_EOS], dtype=int)
		weight_data = np.ones(shape=(self.batch_size, self.max_caption_len), dtype=float)
		ref_word_list_data = np.zeros(shape=(self.batch_size, self.total_word_count), dtype=float)

		for j in range(self.batch_size):
			frames_data[j] = pair_list[index + j][0]
			ref_caption_len_data[j] = len(pair_list[index + j][1])
			
			for k, word in enumerate(pair_list[index + j][1]):
				ref_caption_data[j, k] = self.data.word_index_dict[word]
				#weight_data[j, k] = (math.sqrt(self.data.word_freq_dict[word] / self.subsample_rate) + 1) * self.subsample_rate / self.data.word_freq_dict[word]
				#weight_data[j, k] = math.pow(self.data.word_freq_dict[word], 0.0)
				ref_word_list_data[j, self.data.word_index_dict[word]] = 1.0

		feed_dict = {
			self.frames: frames_data,
			self.ref_caption_len: ref_caption_len_data,
			self.ref_caption: ref_caption_data,
			self.training: True,
			self.weight: weight_data,
			self.ref_word_list: ref_word_list_data
			}

		return feed_dict

	def _prepareTestFeedDictList(self, feat_list, index):

		frames_data = np.zeros(shape=(self.batch_size, self.max_video_len, self.frame_vec_dim), dtype=float)
		ref_caption_len_data = np.zeros((self.batch_size), dtype=int)
		ref_caption_data = np.zeros(shape=(self.batch_size, self.max_caption_len), dtype=int)

		frames_data[0] = feat_list[index]
		ref_caption_len_data[0] = self.max_caption_len

		feed_dict = {
			self.frames: frames_data,
			self.ref_caption_len: ref_caption_len_data,
			self.ref_caption: ref_caption_data,
			self.training: False
			}

		return feed_dict

	def train(self, savepoint=None):
		print 'Start training...'

		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if savepoint != None:
				tf.train.Saver().restore(sess, savepoint)
			last_val_ter = 100.0
			patience_counter = 0
			for epoch in range(1000):
				print '[Epoch #' + str(epoch) + ']'

				train_pair_list = self._prepareTrainPairList()
				
				percent = 0
				for i in range(0, len(train_pair_list) - self.batch_size, self.batch_size):
					feed_dict = self._prepareTrainFeedDictList(train_pair_list, i)
					#_, caption, wp_loss, seq_loss, prior_factor = sess.run([self.train_step, self.caption, self.word_predict_loss, self.seq_loss, self.prior_factor], feed_dict=feed_dict)
					_, caption, seq_loss = sess.run([self.train_step, self.caption, self.seq_loss], feed_dict=feed_dict)
					caption_str = self.data.tokenListToCaption([self.data.word_list[word] for word in caption[0].tolist()])
					#print 'Caption: "{}", WP loss: {}, Seq loss: {}'.format(caption_str, wp_loss, seq_loss)
					print 'Caption: "{}", Seq loss: {}'.format(caption_str, seq_loss)
					if i * 100 / len(train_pair_list) > percent:
						percent += 1
						print '{}%'.format(percent)
				
				if epoch > 1:
					'''
					mean_bleu_list = []
					max_bleu_list = []
					'''
					ter_score_list = []
					for i in range(len(self.data.val_feat_list)):
						feed_dict = self._prepareTestFeedDictList(self.data.val_feat_list, i)
						caption = sess.run(self.caption, feed_dict=feed_dict)
						caption_str = self.data.tokenListToCaption([self.data.word_list[word] for word in caption[0].tolist()])
						bleu_list = []
						'''
						for ref_caption in self.data.val_caption_str_list[i]:
							if caption_str != '':
								#bleu = bleu_eval.BLEU_fn(caption_str, ref_caption)
								bleu = pyter.ter(caption_str, ref_caption)
							else:
								bleu = 0.0
							bleu_list.append(bleu)
						mean_bleu = np.mean(bleu_list)
						max_bleu = max(bleu_list)
						print 'Caption: "{}", Correct: {}, Average BLEU: {}, Best BLEU: {}'.format(caption_str, random.choice(self.data.val_caption_str_list[i]), mean_bleu, max_bleu)
						mean_bleu_list.append(mean_bleu)
						max_bleu_list.append(max_bleu)
						'''
						ter_score = pyter.ter(caption_str, random.choice(self.data.val_caption_str_list[i]))
						ter_score_list.append(ter_score)
						print 'Caption: "{}", Correct: {}, TER: {}'.format(caption_str, random.choice(self.data.val_caption_str_list[i]), ter_score)
					'''
					val_bleu = np.mean(max_bleu_list)
					print 'Validation BLEU: {}'.format(val_bleu)
					'''
					val_ter = np.mean(ter_score_list)
					print 'Validation TER: {}'.format(val_ter)

					if val_ter > last_val_ter:
						patience_counter += 1
						print 'Patience Counter: {}'.format(patience_counter)
						if patience_counter > self.patience:
							break
					else:
						patience_counter = 0
						last_val_ter = val_ter
					tf.train.Saver().save(sess, self.save_path, global_step=epoch)

	def predict(self, feat, savepoint=None):
		if savepoint == None:
			savepoint = self.save_path

		dummy_feat_list = np.array([feat])

		feed_dict = self._prepareTestFeedDictList(dummy_feat_list, 0)
		caption = sess.run(self.caption, feed_dict=feed_dict)
		caption_str = self.data.tokenListToCaption([self.data.word_list[word] for word in caption[0].tolist()])
		print 'Caption: "{}"'.format(caption_str)

		return caption_str

	def testLimited(self, savepoint=None):
		if savepoint == None:
			savepoint = self.save_path
		
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if savepoint != None:
				tf.train.Saver().restore(sess, savepoint)
			for i in range(len(self.data.test_limited_feat_list)):
				feed_dict = self._prepareTestFeedDictList(self.data.test_limited_feat_list, i)
 				caption = sess.run(self.caption, feed_dict=feed_dict)
				caption_str = self.data.tokenListToCaption([self.data.word_list[word] for word in caption[0].tolist()])
				print 'Caption: "{}"'.format(caption_str)
		
model = ModelAttn()
#model.train()
model.testLimited('/tmp3/chenchc/MLDS2017/hw2/data/model_attn/model.ckpt-125')
