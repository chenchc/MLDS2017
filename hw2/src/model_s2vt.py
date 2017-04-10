import bleu_eval
import numpy as np
import os
import random
import read_util
import tensorflow as tf

class ModelS2VT:

	## Config
	frame_emb_size = 500
	lstm1_size = 1000
	lstm2_size = 1000
	batch_size = 80
	lstm1_drop_prob = 0.5
	lstm2_drop_prob = 0.5

	model_name = 's2vt'
	save_dir = 'data/model_' + model_name
	save_path = save_dir + '/model.ckpt'
	submit_path = 'data/submit_' + model_name
	lr_init = 0.0001
	lr_decay_steps = 20000
	lr_decay_rate = 0.5
	lr_momentum = 0.9
	schd_sampling_decay = 0.0005
	patience = 3

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

	## Output tensors
	loss = None
	caption = None
	train_step = None

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

		## Frame embedding
		frames_2 = tf.reshape(frames, [-1, self.frame_vec_dim], name='frames_2')
		w_frames_emb = tf.get_variable('w_frames_emb', shape=[self.frame_vec_dim, self.frame_emb_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_frames_emb = tf.get_variable('b_frames_emb', shape=[self.frame_emb_size], dtype=tf.float32, initializer=tf.constant_initializer(0.2))
		frames_emb = tf.add(tf.matmul(frames_2, w_frames_emb), b_frames_emb, name='frames_emb')
		frames_emb_2 = tf.reshape(frames_emb, [self.batch_size, self.max_video_len, self.frame_emb_size], name='frames_emb_2')

		## LSTM 1
		frames_emb_padded = tf.pad(frames_emb_2, [[0, 0], [0, self.max_caption_len], [0, 0]], name='frames_emb_padded')
		lstm1_drop_prob = tf.cond(training, lambda: tf.constant(self.lstm1_drop_prob), lambda: tf.constant(0.0))
		lstm1_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm1_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))

		with tf.variable_scope('lstm1'):
			lstm1_output, lstm1_state = tf.nn.dynamic_rnn(
				lstm1_basic_cell, inputs=frames_emb_2, 
				dtype=tf.float32)
		video_emb = tf.gather(tf.transpose(lstm1_output, [1, 0, 2]), self.max_video_len - 1, name='video_emb')

		## LSTM 2
		lstm2_basic_cell = tf.contrib.rnn.LSTMCell(self.lstm2_size, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))

		ref_caption_2 = tf.transpose(ref_caption, perm=[1, 0])

		w_logits = tf.get_variable('w_logits', shape=[self.lstm2_size, self.total_word_count], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
		b_logits = tf.get_variable('b_logits', shape=[self.total_word_count], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		w_word_emb_init = tf.constant(np.array([self.data.word_vec_dict[self.data.word_list[i]] for i in range(self.total_word_count)], dtype=np.float32), dtype=tf.float32)
		#w_word_emb = tf.get_variable('w_word_emb', dtype=tf.float32, initializer=w_word_emb_init)
		w_word_emb = w_word_emb_init

		def lstm2_loop_fn(time, cell_output, cell_state, loop_state):
			# elements_finished
			elements_finished = (time >= ref_caption_len)
			all_finished = tf.reduce_all(elements_finished)

			## Softmax Logits (dummy)
			if cell_output != None:
				cell_output_masked = tf.where(
					elements_finished,
					tf.zeros([self.batch_size, self.lstm2_size], dtype=tf.float32),
					cell_output)

				logits = tf.add(tf.matmul(cell_output_masked, w_logits), b_logits)
				caption = tf.cast(tf.argmax(logits, axis=1), tf.int32)
			# emit_output
			emit_output = cell_output

			# next_input
			if cell_output is None:
				input_word = tf.fill([self.batch_size], self.data.word_index_dict[read_util.ReadUtil.MARKER_BOS])
				input_word_emb = tf.nn.embedding_lookup(w_word_emb, input_word)
				bos_bit = tf.fill([self.batch_size, 1], 1.0)
				next_input = tf.concat([bos_bit, input_word_emb, video_emb], axis=1)
			else:
				select_ref_prob = tf.maximum(1.0 - self.schd_sampling_decay * tf.cast(global_step, tf.float32), 0.0)
				input_word = tf.cond(
					training,
					lambda: tf.where(tf.random_uniform([self.batch_size]) < select_ref_prob, ref_caption_2[time-1], caption),
					lambda: caption)
				input_word_emb = tf.nn.embedding_lookup(w_word_emb, input_word)
				bos_bit = tf.fill([self.batch_size, 1], 0.0)
				next_input = tf.cond(
					all_finished,
					lambda: tf.zeros([self.batch_size, self.lstm1_size + self.word_vec_dim + 1]),
					lambda: tf.concat([bos_bit, input_word_emb, video_emb], axis=1))

			# next_cell_state
			if cell_output is None:
				next_cell_state = lstm2_basic_cell.zero_state(self.batch_size, tf.float32)
			else:
				next_cell_state = cell_state

			# next_loop_state
			next_loop_state = None

			return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

		with tf.variable_scope('lstm2'):
			lstm2_output_ta, _, _ = tf.nn.raw_rnn(lstm2_basic_cell, lstm2_loop_fn)
	
		lstm2_output_trunc = tf.transpose(lstm2_output_ta.stack(), [1, 0, 2])
		lstm2_output = tf.pad(lstm2_output_trunc, [[0, 0], [0, self.max_caption_len - lstm2_output_ta.size()], [0, 0]], name='lstm2_output')
		lstm2_drop_prob = tf.cond(training, lambda: tf.constant(self.lstm2_drop_prob), lambda: tf.constant(0.0))
		lstm2_output = tf.nn.dropout(lstm2_output, 1.0 - lstm2_drop_prob, name='lstm2_output_drop')
		lstm2_output_flat = tf.reshape(lstm2_output, [-1, self.lstm2_size])

		## Softmax Logits
		logits_flat = tf.add(tf.matmul(lstm2_output_flat, w_logits), b_logits)
		logits = tf.reshape(logits_flat, [self.batch_size, self.max_caption_len, self.total_word_count], name='logits')
		self.caption = caption = tf.argmax(logits, axis=2, name='caption')

		## Loss
		mask = tf.sequence_mask(ref_caption_len, maxlen=self.max_caption_len, name='mask')
		logits_mask = tf.boolean_mask(logits, mask, name='logits_mask')
		ref_caption_mask = tf.boolean_mask(ref_caption, mask, name='ref_caption_mask')
		losses = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(ref_caption_mask, self.total_word_count, dtype=tf.float32),
			logits=logits_mask, name='losses')

		loss = tf.reduce_sum(losses, name='loss')
		self.mean_loss = mean_loss = tf.reduce_mean(losses, name='mean_loss')

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

		for j in range(self.batch_size):
			frames_data[j] = pair_list[index + j][0]
			ref_caption_len_data[j] = len(pair_list[index + j][1])
			
			for k, word in enumerate(pair_list[index + j][1]):
				ref_caption_data[j, k] = self.data.word_index_dict[word]

		feed_dict = {
			self.frames: frames_data,
			self.ref_caption_len: ref_caption_len_data,
			self.ref_caption: ref_caption_data,
			self.training: True
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
			last_val_bleu = 0.0
			patience_counter = 0
			for epoch in range(1000):
				print '[Epoch #' + str(epoch) + ']'

				train_pair_list = self._prepareTrainPairList()
				
				percent = 0
				for i in range(0, len(train_pair_list) - self.batch_size, self.batch_size):
					feed_dict = self._prepareTrainFeedDictList(train_pair_list, i)
					_ = sess.run(self.train_step, feed_dict=feed_dict)
					if i * 100 / len(train_pair_list) > percent:
						percent += 1
						print '{}%'.format(percent)
				
				mean_bleu_list = []
				for i in range(len(self.data.val_feat_list)):
					feed_dict = self._prepareTestFeedDictList(self.data.val_feat_list, i)
					caption = sess.run(self.caption, feed_dict=feed_dict)
					caption_str = self.data.tokenListToCaption([self.data.word_list[word] for word in caption[0].tolist()])
					bleu_list = []
					for ref_caption in self.data.val_caption_str_list[i]:
						bleu = bleu_eval.BLEU_fn(caption_str, ref_caption)
						bleu_list.append(bleu)
					mean_bleu = np.mean(bleu_list)
					print 'Caption: "{}", Correct: {}, BLEU: {}'.format(caption_str, random.choice(self.data.val_caption_str_list[i]), mean_bleu)
					mean_bleu_list.append(mean_bleu)
				val_bleu = np.mean(mean_bleu_list)
				print 'Validation BLEU: {}'.format(val_bleu)

				if val_bleu < last_val_bleu:
					patience_counter += 1
					print 'Patience Counter: {}'.format(patience_counter)
					if patience_counter > self.patience:
						break
				else:
					patience_counter = 0
					tf.train.Saver().save(sess, self.save_path)
					last_val_bleu = val_bleu


model = ModelS2VT()
model.train()
