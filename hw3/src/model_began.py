import math
import numpy as np
import os
import sys
import random
import tensorflow as tf
import time
import utils
import gc

class ModelBEGAN:
	
	## Config
	batch_size = 16
	filter_dim = 128
	latent_dim = 64
	color_emb_dim = 16
	gamma_init = 0.5
	gamma_decay = 0.005
	gamma_min = 0.5
	lambda_k = 0.001
	gamma_for_wrong = 1.1
	lr_init = 0.000025
	lr_decay_rate = 0.5
	num_sample = 5
	real_image_threshold = 2.0

	model_name = 'began'
	save_dir = 'data/model_' + model_name
	save_path = save_dir + '/model.ckpt'

	## Constants
	data = None
	num_color = None
	color_unspecified_index = None
	color_begin = None
	color_end = None

	## Placeholder tensors
	real_image = None
	hair_color = None
	eyes_color = None
	training = None

	## Variable
	loss_real_image_threshold = None
	lr = None

	## Output tensors
	update_gamma = None
	gamma = None
	update_k = None
	convergence_measure = None
	train_step = None
	gen_image_uint8 = None
	loss_gen_image_batch = None

	def __init__(self):
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.__fillConstants()
		self.__defineModel()

	def __fillConstants(self):
		self.data = utils.Utils()
		self.num_color = len(self.data.color_list)
		self.color_unspecified_index = self.data.color_index_dict[self.data.COLOR_UNSPECIFIED]
		self.color_begin = 1
		self.color_end = self.num_color

	def __defineModel(self):
		print 'Defining model...'

		## Global step
		global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

		## Real image loss threshold
		self.loss_real_image_threshold = loss_real_image_threshold = tf.get_variable('loss_real_image_threshold', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

		## Annealing Gamma
		self.gamma = gamma = tf.get_variable('gamma', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.gamma_init), trainable=False)
		self.update_gamma = tf.assign(gamma, tf.maximum(self.gamma_min, gamma - self.gamma_decay), name='update_gamma')

		## Input placeholders
		self.training = training = tf.placeholder(tf.bool, shape=[], name='training')
		self.real_image = real_image = tf.placeholder(tf.uint8, shape=[self.batch_size]+self.data.TRAIN_IMAGE_SHAPE, name='real_image')
		real_image = tf.image.convert_image_dtype(real_image, tf.float32, saturate=True)
		self.hair_color = hair_color = tf.placeholder(tf.int32, shape=[self.batch_size], name='hair_color')
		self.eyes_color = eyes_color = tf.placeholder(tf.int32, shape=[self.batch_size], name='eyes_color')

		## Real image colors and their embeddings
		def colorEmb(color, dim, variable_scope, reuse=False):
			with tf.variable_scope(variable_scope) as scope:
				if reuse:
					scope.reuse_variables()
				w = tf.get_variable('w', shape=[self.num_color, dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
				color_emb = tf.where(
					tf.equal(color, self.color_unspecified_index),
					tf.fill([self.batch_size, dim], 0.0),
					tf.nn.embedding_lookup(w, color, name='color_emb'))
				#color_emb = tf.nn.embedding_lookup(w, color)
				#color_emb = tf.tanh(color_emb)
			return color_emb

		with tf.variable_scope('generator'):
			hair_color_emb_gen = colorEmb(hair_color, self.color_emb_dim, 'hair_color_emb_gen')
			eyes_color_emb_gen = colorEmb(eyes_color, self.color_emb_dim, 'eyes_color_emb_gen')
			color_emb_gen = tf.concat([hair_color_emb_gen, eyes_color_emb_gen], -1, name='color_emb_gen')
		with tf.variable_scope('discriminator'):
			hair_color_emb_disc = colorEmb(hair_color, self.color_emb_dim, 'hair_color_emb_disc')
			eyes_color_emb_disc = colorEmb(eyes_color, self.color_emb_dim, 'eyes_color_emb_disc')
			color_emb_disc = tf.concat([hair_color_emb_disc, eyes_color_emb_disc], -1, name='color_emb_disc')

		## Wrong colors and their embeddings
		wrong_hair_color = tf.where(
			tf.equal(hair_color, self.color_unspecified_index), 
			tf.fill([self.batch_size], self.color_unspecified_index),
			tf.random_uniform([self.batch_size], minval=self.color_begin, maxval=self.color_end, dtype=tf.int32))
		wrong_eyes_color = tf.where(
			tf.equal(eyes_color, self.color_unspecified_index), 
			tf.fill([self.batch_size], self.color_unspecified_index),
			tf.random_uniform([self.batch_size], minval=self.color_begin, maxval=self.color_end, dtype=tf.int32))

		with tf.variable_scope('discriminator'):
			wrong_hair_color_emb_disc = colorEmb(wrong_hair_color, self.color_emb_dim, 'hair_color_emb_disc', reuse=True)
			wrong_eyes_color_emb_disc = colorEmb(wrong_eyes_color, self.color_emb_dim, 'eyes_color_emb_disc', reuse=True)
			wrong_color_emb_disc = tf.concat([wrong_hair_color_emb_disc, wrong_eyes_color_emb_disc], -1, name='wrong_color_emb_disc')
		
		## Generator
		z = tf.random_uniform([self.batch_size, self.latent_dim], minval=-1.0, maxval=1.0, dtype=tf.float32, name='z')
		z_for_d = tf.random_uniform([self.batch_size, self.latent_dim], minval=-1.0, maxval=1.0, dtype=tf.float32, name='z_for_d')
		gen_image = self.__defineGenerator(z, color_emb_gen)
		gen_image_for_d = self.__defineGenerator(z_for_d, color_emb_gen, reuse_scope=True)
		self.gen_image_uint8 = gen_image_uint8 = tf.image.convert_image_dtype(gen_image, tf.uint8, saturate=True, name='gen_image_uint8')
		
		## Discriminator
		restored_real_image = self.__defineDiscriminator(real_image, color_emb_disc, False)
		restored_real_image_wrong_cond = self.__defineDiscriminator(real_image, wrong_color_emb_disc, True, reuse_scope=True)
		restored_gen_image = self.__defineDiscriminator(gen_image, color_emb_disc, False, reuse_scope=True)
		restored_gen_image_for_d = self.__defineDiscriminator(gen_image_for_d, color_emb_disc, False, reuse_scope=True)
		
		## Define reconstruction loss
		def reconLoss(img, restored_img, scope, reserve_batch=False):
			with tf.variable_scope(scope):
				recon_loss = tf.abs(img - restored_img)
				#recon_loss = tf.reduce_sum(recon_loss * tf.constant([0.299, 0.587, 0.114]), axis=-1)
				recon_loss = tf.reduce_mean(recon_loss, axis=-1)
				if reserve_batch:
					recon_loss = tf.reshape(recon_loss, [self.batch_size, -1])
					recon_loss = tf.reduce_mean(recon_loss, axis=1, name='recon_loss')
				else:
					recon_loss = tf.reduce_mean(recon_loss, name='recon_loss')
			return recon_loss
	
		def outOfBoundLoss(img, scope, reserve_batch=False):
			with tf.variable_scope(scope):
				out_of_bound_loss = (tf.square(img - 0.5) - 0.25) * tf.cast(tf.logical_or(tf.less(img, 0.0), tf.greater(img, 1.0)), tf.float32)
				out_of_bound_loss = tf.reduce_mean(out_of_bound_loss, axis=-1)
				if reserve_batch:
					out_of_bound_loss = tf.reshape(out_of_bound_loss, [self.batch_size, -1])
					out_of_bound_loss = tf.reduce_mean(out_of_bound_loss, axis=1, name='out_of_bound_loss')
				else:
					out_of_bound_loss = tf.reduce_mean(out_of_bound_loss, name='out_of_bound_loss')
			return out_of_bound_loss

		## Loss
		self.k = k = tf.get_variable('k', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)

		loss_real_image_batch = reconLoss(real_image, restored_real_image, 'real_image_batch', reserve_batch=True)
		qualified_mask = tf.less(loss_real_image_batch, loss_real_image_threshold)
		#qualified_mask = tf.fill([self.batch_size], True)
		self.loss_real_image_batch = loss_real_image_batch = tf.boolean_mask(loss_real_image_batch, qualified_mask)
		self.loss_real_image = loss_real_image = tf.reduce_mean(loss_real_image_batch, name='loss_real_image')

		loss_real_image_wrong_cond_batch = reconLoss(real_image, restored_real_image_wrong_cond, 'real_image_wrong_cond_batch', reserve_batch=True)
		loss_real_image_wrong_cond_batch = tf.boolean_mask(loss_real_image_wrong_cond_batch, qualified_mask)
		self.loss_real_image_wrong_cond = loss_real_image_wrong_cond = tf.reduce_mean(loss_real_image_wrong_cond_batch, name='loss_real_image_wrong_cond')

		#loss_cond = tf.reduce_mean(tf.maximum(self.gamma_for_wrong * loss_real_image_batch - loss_real_image_wrong_cond_batch, 0.0))
		loss_cond = tf.reduce_mean(self.gamma_for_wrong * loss_real_image_batch - loss_real_image_wrong_cond_batch)

		self.oob_loss_gen_image = oob_loss_gen_image = outOfBoundLoss(gen_image, 'oob_loss_gen_image')
		self.loss_gen_image = loss_gen_image = reconLoss(gen_image, restored_gen_image, 'gen_image')
		self.loss_gen_image_batch = reconLoss(gen_image, restored_gen_image, 'gen_image_batch', reserve_batch=True) + outOfBoundLoss(gen_image, 'gen_image_batch', reserve_batch=True)
		loss_gen_image_for_d = reconLoss(gen_image_for_d, restored_gen_image_for_d, 'gen_image_for_d')
		loss_generator = loss_gen_image + oob_loss_gen_image
		loss_discriminator = loss_real_image + loss_cond - k * loss_gen_image_for_d
		
		self.update_k = update_k = tf.assign(k, tf.maximum(0.0, k + self.lambda_k * (self.gamma * loss_real_image - loss_gen_image)), name='update_k')
		self.convergence_measure = convergence_measure = tf.add(loss_real_image, tf.abs(self.gamma * loss_real_image - loss_gen_image), name='convergence_measure')
		
		## Optimization
		self.lr = lr = tf.get_variable('lr', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.lr_init))
		optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
		grad_loss_generator = optimizer.compute_gradients(loss_generator, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator/'))
		grad_loss_discriminator = optimizer.compute_gradients(loss_discriminator, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator/'))
		self.train_step = train_step = optimizer.apply_gradients(grad_loss_generator + grad_loss_discriminator, global_step=global_step, name='train_step')
		
		## Print trainable variables
		print 'Variables:'
		for i in tf.global_variables():
			print i.name, i.get_shape()

	def __defineGenerator(self, z, color_emb, reuse_scope=False):
		with tf.variable_scope('generator') as scope:
			if reuse_scope:
				scope.reuse_variables()

			def doConv(input, filters):
				output = tf.layers.conv2d(input, filters, [3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				#output = tf.layers.batch_normalization(output, training=self.training)
				output = tf.nn.elu(output, name='output')
				return output

			def doUpscaling(input, size):
				output = tf.image.resize_bilinear(input, size, name='output')
				return output

			with tf.variable_scope('layer1'):
				extended_z = tf.concat([z, color_emb], -1)
				extended_z = tf.layers.dense(extended_z, self.latent_dim, kernel_initializer=tf.contrib.layers.xavier_initializer())
				extended_z = tf.nn.elu(extended_z, name='extended_z')
				w = tf.get_variable('w', shape=[self.latent_dim, 8 * 8 * self.filter_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
				b = tf.get_variable('b', shape=[8 * 8 * self.filter_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
				layer1 = tf.reshape(tf.add(tf.matmul(extended_z, w), b), [self.batch_size, 8, 8, self.filter_dim], name='output')

			with tf.variable_scope('conv1_1'):
				conv1_1 = doConv(layer1, self.filter_dim)

			with tf.variable_scope('conv1_2'):
				conv1_2 = doConv(conv1_1, self.filter_dim)

			with tf.variable_scope('layer2'):
				layer2 = doUpscaling(conv1_2, [16, 16])

			with tf.variable_scope('conv2_1'):
				conv2_1 = doConv(layer2, self.filter_dim)

			with tf.variable_scope('conv2_2'):
				conv2_2 = doConv(conv2_1, self.filter_dim)

			with tf.variable_scope('layer3'):
				layer3 = doUpscaling(conv2_2, [32, 32])

			with tf.variable_scope('conv3_1'):
				conv3_1 = doConv(layer3, self.filter_dim)

			with tf.variable_scope('conv3_2'):
				conv3_2 = doConv(conv3_1, self.filter_dim)

			with tf.variable_scope('layer4'):
				layer4 = doUpscaling(conv3_2, [64, 64])

			with tf.variable_scope('conv4_1'):
				conv4_1 = doConv(layer4, self.filter_dim)

			with tf.variable_scope('conv4_2'):
				conv4_2 = doConv(conv4_1, self.filter_dim)

			with tf.variable_scope('output'):
				output = tf.layers.conv2d(conv4_2, 3, [3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.5), padding='SAME')

		return output
				
	def __defineDiscriminator(self, image, color_emb, isWrong, reuse_scope=False):
		
		def doConv(input, filters):
			output = tf.layers.conv2d(input, filters, [3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
			#output = tf.layers.batch_normalization(output, training=self.training)
			output = tf.nn.elu(output, name='output')
			return output

		def doUpscaling(input, size):
			output = tf.image.resize_bilinear(input, size, name='output')
			return output

		def doDownscaling(input, size):
			output = tf.image.resize_bilinear(input, size, name='output')
			return output
			
		with tf.variable_scope('discriminator') as scope:
			if reuse_scope:
				scope.reuse_variables()
			with tf.variable_scope('encoder'):
				with tf.variable_scope('conv1_1'):
					conv1_1 = doConv(image, self.filter_dim)

				with tf.variable_scope('conv1_2'):
					conv1_2 = doConv(conv1_1, self.filter_dim)

				with tf.variable_scope('layer2'):
					layer2 = doDownscaling(conv1_2, [32, 32])

				with tf.variable_scope('conv2_1'):
					conv2_1 = doConv(layer2, self.filter_dim * 2)

				with tf.variable_scope('conv2_2'):
					conv2_2 = doConv(conv2_1, self.filter_dim * 2)

				with tf.variable_scope('layer3'):
					layer3 = doDownscaling(conv2_2, [16, 16])

				with tf.variable_scope('conv3_1'):
					conv3_1 = doConv(layer3, self.filter_dim * 3)

				with tf.variable_scope('conv3_2'):
					conv3_2 = doConv(conv3_1, self.filter_dim * 3)

				with tf.variable_scope('layer4'):
					layer4 = doDownscaling(conv3_2, [8, 8])

				with tf.variable_scope('conv4_1'):
					conv4_1 = doConv(layer4, self.filter_dim * 4)

				with tf.variable_scope('conv4_2'):
					conv4_2 = doConv(conv4_1, self.filter_dim * 4)

				with tf.variable_scope('z'):
					conv4_2 = tf.reshape(conv4_2, [self.batch_size, 8 * 8 * self.filter_dim * 4])
					w = tf.get_variable('w', shape=[8 * 8 * self.filter_dim * 4, self.latent_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
					b = tf.get_variable('b', shape=[self.latent_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
					z = tf.add(tf.matmul(conv4_2, w), b, name='z')

		with tf.variable_scope('discriminator') as scope:
			if reuse_scope:
				scope.reuse_variables()
			with tf.variable_scope('extended_z'):
				extended_z = tf.concat([z, color_emb], -1)
				extended_z = tf.layers.dense(extended_z, self.latent_dim, kernel_initializer=tf.contrib.layers.xavier_initializer())
				extended_z = tf.nn.elu(extended_z, name='extended_z')

		with tf.variable_scope('discriminator') as scope:
			if reuse_scope:
				scope.reuse_variables()
			with tf.variable_scope('decoder'):
				with tf.variable_scope('layer1'):
					w = tf.get_variable('w', shape=[self.latent_dim, 8 * 8 * self.filter_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
					b = tf.get_variable('b', shape=[8 * 8 * self.filter_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
					dec_layer1 = tf.reshape(tf.add(tf.matmul(extended_z, w), b), [self.batch_size, 8, 8, self.filter_dim], name='output')

				with tf.variable_scope('conv1_1'):
					dec_conv1_1 = doConv(dec_layer1, self.filter_dim)

				with tf.variable_scope('conv1_2'):
					dec_conv1_2 = doConv(dec_conv1_1, self.filter_dim)

				with tf.variable_scope('layer2'):
					dec_layer2 = doUpscaling(dec_conv1_2, [16, 16])

				with tf.variable_scope('conv2_1'):
					dec_conv2_1 = doConv(dec_layer2, self.filter_dim)

				with tf.variable_scope('conv2_2'):
					dec_conv2_2 = doConv(dec_conv2_1, self.filter_dim)

				with tf.variable_scope('layer3'):
					dec_layer3 = doUpscaling(dec_conv2_2, [32, 32])

				with tf.variable_scope('conv3_1'):
					dec_conv3_1 = doConv(dec_layer3, self.filter_dim)

				with tf.variable_scope('conv3_2'):
					dec_conv3_2 = doConv(dec_conv3_1, self.filter_dim)

				with tf.variable_scope('layer4'):
					dec_layer4 = doUpscaling(dec_conv3_2, [64, 64])

				with tf.variable_scope('conv4_1'):
					dec_conv4_1 = doConv(dec_layer4, self.filter_dim)

				with tf.variable_scope('conv4_2'):
					dec_conv4_2 = doConv(dec_conv4_1, self.filter_dim)

				with tf.variable_scope('output'):
					output = tf.layers.conv2d(dec_conv4_2, 3, [3, 3], kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.5), padding='SAME')
			
		return output

	def __prepareTrainPairList(self):
		train_pair_list = []
		for idx in range(len(self.data.train_image_list)):
			train_pair_list.append((self.data.train_image_list[idx], self.data.train_hair_color_list[idx], self.data.train_eyes_color_list[idx]))
		random.shuffle(train_pair_list)

		return train_pair_list

	def __prepareTrainFeedDictList(self, pair_list, index):
		real_image_data = np.zeros(shape=[self.batch_size]+self.data.TRAIN_IMAGE_SHAPE, dtype=np.uint8)
		hair_color_data = np.zeros(shape=(self.batch_size), dtype=np.int32)
		eyes_color_data = np.zeros(shape=(self.batch_size), dtype=np.int32)

		for i in range(self.batch_size):
			# If the index is out of range, shift the cursor back to the beginning
			while index + i >= len(pair_list):
				index -= len(pair_list)
			real_image_data[i] = pair_list[index + i][0]
			hair_color_data[i] = self.data.color_index_dict[pair_list[index + i][1]]
			eyes_color_data[i] = self.data.color_index_dict[pair_list[index + i][2]]
				

		feed_dict = {
			self.training: True,
			self.real_image: real_image_data,
			self.hair_color: hair_color_data,
			self.eyes_color: eyes_color_data
			}

		return feed_dict

	def __prepareTestFeedDictList(self, testing_id):
		hair_color_data = np.zeros(shape=(self.batch_size), dtype=np.int32)
		eyes_color_data = np.zeros(shape=(self.batch_size), dtype=np.int32)

		for i in range(self.batch_size):
			hair_color_data[i] = self.data.color_index_dict[self.data.test_list[testing_id]['hair']]
			eyes_color_data[i] = self.data.color_index_dict[self.data.test_list[testing_id]['eyes']]

		feed_dict = {
			self.training: False,
			self.hair_color: hair_color_data,
			self.eyes_color: eyes_color_data
			}

		return feed_dict

	def __testFast(self, sess):
		feed_dict = {
			self.training: False,
			self.hair_color: np.zeros(shape=(self.batch_size), dtype=np.int32),
			self.eyes_color: np.zeros(shape=(self.batch_size), dtype=np.int32)
			}
		gen_image, loss = sess.run([self.gen_image_uint8, self.loss_gen_image_batch], feed_dict=feed_dict)
		self.data.saveImageTile('training', 'tile', gen_image)
		feed_dict = {
			self.training: False,
			self.hair_color: np.full([self.batch_size], self.data.color_index_dict['blue'], dtype=np.int32),
			self.eyes_color: np.full([self.batch_size], self.data.color_index_dict['blue'], dtype=np.int32)
			}
		gen_image, loss = sess.run([self.gen_image_uint8, self.loss_gen_image_batch], feed_dict=feed_dict)
		self.data.saveImageTile('training', 'tile_bb', gen_image)
		feed_dict = {
			self.training: False,
			self.hair_color: np.full([self.batch_size], self.data.color_index_dict['blue'], dtype=np.int32),
			self.eyes_color: np.full([self.batch_size], self.data.color_index_dict['red'], dtype=np.int32)
			}
		gen_image, loss = sess.run([self.gen_image_uint8, self.loss_gen_image_batch], feed_dict=feed_dict)
		self.data.saveImageTile('training', 'tile_br', gen_image)
		feed_dict = {
			self.training: False,
			self.hair_color: np.full([self.batch_size], self.data.color_index_dict['red'], dtype=np.int32),
			self.eyes_color: np.full([self.batch_size], self.data.color_index_dict['red'], dtype=np.int32)
			}
		gen_image, loss = sess.run([self.gen_image_uint8, self.loss_gen_image_batch], feed_dict=feed_dict)
		self.data.saveImageTile('training', 'tile_rr', gen_image)
		feed_dict = {
			self.training: False,
			self.hair_color: np.full([self.batch_size], self.data.color_index_dict['yellow'], dtype=np.int32),
			self.eyes_color: np.full([self.batch_size], self.data.color_index_dict['yellow'], dtype=np.int32)
			}
		gen_image, loss = sess.run([self.gen_image_uint8, self.loss_gen_image_batch], feed_dict=feed_dict)
		self.data.saveImageTile('training', 'tile_yy', gen_image)
		feed_dict = {
			self.training: False,
			self.hair_color: np.full([self.batch_size], self.data.color_index_dict['white'], dtype=np.int32),
			self.eyes_color: np.full([self.batch_size], self.data.color_index_dict['yellow'], dtype=np.int32)
			}
		gen_image, loss = sess.run([self.gen_image_uint8, self.loss_gen_image_batch], feed_dict=feed_dict)
		self.data.saveImageTile('training', 'tile_wy', gen_image)

	def train(self, savepoint=None):
		print 'Start training...'

		gc.disable()
		with tf.device('/cpu:0'):
			saver = tf.train.Saver(allow_empty=True)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			if savepoint != None:
				saver.restore(sess, savepoint)

			gamma = sess.run([self.update_gamma])
			print 'Gamma change to: {}'.format(gamma)

			global_convergence_measure_history = []
			last_convergence_measure = 1.0
			lr = self.lr_init
			last_checkpoint = None
			for epoch in range(16, 1000, 1):
				print '[Epoch #{}]'.format(str(epoch))

				train_pair_list = self.__prepareTrainPairList()
				
				convergence_measure_history = []
				loss_real_image_history = np.array([], dtype=np.float32)
				percent = 1
				for step in range(0, len(train_pair_list), self.batch_size):
					feed_dict = self.__prepareTrainFeedDictList(train_pair_list, step)
					_, _, convergence_measure, loss_real_image, k, loss_real, loss_gen, loss_wrong, loss_oob = sess.run([self.train_step, self.update_k, self.convergence_measure, self.loss_real_image_batch, self.k, self.loss_real_image, self.loss_gen_image, self.loss_real_image_wrong_cond, self.oob_loss_gen_image], feed_dict=feed_dict)
					print 'step: {}, k: {}, real loss: {}, gen loss: {}, wrong loss: {}, OOB loss: {}'.format(step, k, loss_real, loss_gen, loss_wrong, loss_oob)
					## Handling NaN
					if math.isnan(convergence_measure):
						print 'NaN is coming!!!'
						tvars = tf.trainable_variables()
						tvars_vals = sess.run(tvars)

						for var, val in zip(tvars, tvars_vals):
							print(var.name, val)  # Prints the name of the variable alongside its value.
							if np.isnan(val).any():
								print 'BOOM!!!'

						saver.restore(sess, last_checkpoint)
						lr = lr * self.lr_decay_rate
						update_lr = tf.assign(self.lr, tf.constant(lr, dtype=tf.float32))
						sess.run([update_lr])
						print 'LR change to: {}'.format(lr)
						global_convergence_measure_history = []
						continue
						
					convergence_measure_history.append(convergence_measure)
					loss_real_image_history = np.concatenate((loss_real_image_history, loss_real_image))
					
					if step > percent * len(train_pair_list) / 100:
						print 'Progress: {}%, Convergence measure: {}'.format(percent, np.mean(convergence_measure_history))
						global_convergence_measure_history += convergence_measure_history
						convergence_measure_history = []
						self.__testFast(sess)
						
						percent += 1

				## Update threshold
				loss_real_image_threshold = np.mean(loss_real_image_history) + self.real_image_threshold * np.std(loss_real_image_history)
				update_threshold = tf.assign(self.loss_real_image_threshold, tf.constant(loss_real_image_threshold, dtype=tf.float32))
				sess.run([update_threshold])
				print 'Real Image Threshold: {}'.format(loss_real_image_threshold)

				## Update LR
				new_convergence_measure = np.mean(global_convergence_measure_history)
				print 'Avg Convergence Measure: {}'.format(new_convergence_measure)
				if new_convergence_measure > last_convergence_measure:
					lr = lr * self.lr_decay_rate
					update_lr = tf.assign(self.lr, tf.constant(lr, dtype=tf.float32))
					sess.run([update_lr])
					print 'LR change to: {}'.format(lr)
					#self.gamma_for_wrong = 0.0
					#print 'Cond Throttling!'
				last_convergence_measure = new_convergence_measure
				global_convergence_measure_history = []

				## Update gamma
				gamma = sess.run([self.update_gamma])
				print 'Gamma change to: {}'.format(gamma)

				last_checkpoint = saver.save(sess, self.save_path, global_step=epoch)
			
		
if __name__ == '__main__':
	model = ModelBEGAN()
	model.train('data/model_began/model.ckpt-15')
	#model.train()
