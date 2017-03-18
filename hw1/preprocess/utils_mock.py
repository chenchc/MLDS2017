from collections import defaultdict
import config
import csv
import numpy as np
import operator
import os.path
import subprocess

def getTrainingLikeData(filename):
	# Check if the file exist
	if not os.path.isfile(filename):
		subprocess.call(['make', filename])

	# Load
	data = []
	with open(filename, 'r') as file:
		for line in file:
			line = line[:-1] # remove \n
			line = line.lower()	# to lowercase
			data.append(line.split(' '))

	return data

def getTrainingData():
	return [s for s in getTrainingLikeData(config.train_file) if 'sherlock' in s]

def getValData():
	return [s for s in getTrainingLikeData(config.train_file) if 'sherlock' in s]

def getTestingData():
	return [['sherlock', '_____', ',', 'i', 'want', 'to', 'play', 'piano']]
	
def getTestingChoiceList():
	return [['want', 'i', 'holmes', 'piano', 'to']]

word_vec_dict = None # Lazy initilized

def getWordVecDict():
	global word_vec_dict

	if word_vec_dict != None:
		return word_vec_dict

	# Check if the file exist
	if not os.path.isfile(config.word_vec_file):
		subprocess.call(['make', config.word_vec_file])

	# Load
	word_vec_dict = dict()
	with open(config.word_vec_file, 'r') as file:
		for line in file:
			content = line.split(' ')
			content.remove('\n')
			word_vec_dict[content[0]] = np.array(map(lambda string: float(string), content[1:]), dtype=np.float32)

	return word_vec_dict

word_occurence = None
word_essential = None

def getWordIndexDict(min_occurence=1):

	if word_occurence == None:
		getWordOccurence()

	word_list = [word for word in word_occurence if word in word_essential or word_occurence[word] >= min_occurence]
	word_index_dict = defaultdict(int) # Unknown word to be 0
	word_index_dict['<UNK>'] = 0
	for i, word in enumerate(word_list):
		word_index_dict[word] = i + 1

	return word_index_dict

def getWordOccurence():
	global word_occurence
	global word_essential

	if word_occurence != None:
		return word_occurence

	word_occurence = defaultdict(int)
	word_essential = set()

	train = getTrainingData()
	for s in train:
		for word in s:
			word_occurence[word] += 1

	val = getValData()
	for s in val:
		for word in s:
			word_occurence[word] += 1

	test = getTestingData()
	for s in test:
		for word in s:
			if word != '_____':
				word_occurence[word] += 1

	choice = getTestingChoiceList()
	for s in choice:
		for word in s:
			word_occurence[word] += 1
			word_essential.add(word)
	
	return word_occurence	
