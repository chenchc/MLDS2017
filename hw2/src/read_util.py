import config
import json
import numpy as np
import re
from collections import defaultdict

class ReadUtil:

	train_caption_list = None
	train_feat_list = None
	word_freq_dict = None
	word_list = None
	
	def __init__(self):
		self.__readTrain()
		self.__genWordList()

	def __readTrain(self):
		## Read train label
		with open(config.train_label_path, 'r') as file:
			content = file.read()
		train_label = json.loads(content)

		## Generate caption list
		self.train_caption_list = []
		for video in train_label:
			caption_list = video['caption']
			self.train_caption_list.append([])
			for caption in caption_list:
				if not all(ord(c) < 128 for c in caption):
					continue
				token_list = self.captionToTokenList(caption)
				self.train_caption_list[-1].append(token_list)

		## Generate feature list
		self.train_feat_list = []
		for video in train_label:
			video_id = video['id']
			feat = np.load(config.train_feat_folder_path + video_id + '.npy')
			self.train_feat_list.append(feat)
	
	@staticmethod
	def captionToTokenList(caption):
		# Remove trailing punctuations, e.g. '.'
		caption = re.sub('\W+$', '', caption)
		
		# Convert to lowercase
		caption = caption.lower()

		# "blabla" -> blabla
		caption = re.sub('"(?P<bla>([a-zA-Z]+))"', lambda m: m.group('bla'), caption)

		# Isolate comma
		caption = re.sub('(?P<letter>\w),', lambda m: m.group('letter') + ' , ', caption)

		# Isolate 's
		caption = re.sub("(?P<letter>\w)'s", lambda m: m.group('letter') + " 's ", caption)

		# Tokenize
		token_list = re.split('\s+', caption)

		return token_list

	@staticmethod
	def tokenListToCaption(token_list):
		# Convert back to sentence
		caption = reduce(lambda s, term: s + ' ' + term, token_list[1:], token_list[0])
		
		# Captalize the first letter
		caption = caption[0:1].upper() + caption[1:]

		# Remove space before comma
		caption = re.sub(' , ', ', ', caption)

		# Remove space before 's
		caption = re.sub(" 's ", "'s ", caption)

		return caption

	def __genWordList(self):
		self.word_freq_dict = defaultdict(int)
		total_word_count = 0.0
		for video in self.train_caption_list:
			for caption in video:
				for token in caption:
					self.word_freq_dict[token] += 1
					total_word_count += 1.0
		for word in self.word_freq_dict:
			self.word_freq_dict[word] /= total_word_count

		word_freq_list = sorted(self.word_freq_dict.iteritems(), key=lambda (k, v): v, reverse=True)
		self.word_list = [k for (k, v) in word_freq_list]

	def getWordList(self):
		return self.word_list

