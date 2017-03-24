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
	return getTrainingLikeData(config.train_file)

def getValData():
	return getTrainingLikeData(config.val_file)

def getTestingData():
	data = []
	with open(config.test_file, 'rb') as file:
		reader = csv.DictReader(file)
		for row in reader:
			question = row['question']
			question = question[:-1] # Remove the trailing .
			question = question.lower()
			data.append(question.split(' '))

	return data
	
def getTestingChoiceList():
	data = []
	with open(config.test_file, 'rb') as file:
		reader = csv.DictReader(file)
		for row in reader:
			data.append(
				map(
				lambda string : string.lower(), 
				[row['a)'], row['b)'], row['c)'], row['d)'], row['e)']])
				)

	return data

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
			word_vec_dict[content[0]] = np.concatenate((np.array(map(lambda string: float(string), content[1:]), dtype=np.float32), np.zeros([50])))

	# Special words
	word_vec_dict['<UNK>'] = np.zeros([350], dtype=np.float32)
	word_vec_dict['<START>'] = np.zeros([350], dtype=np.float32)
	word_vec_dict['<END>'] = np.zeros([350], dtype=np.float32)

	return word_vec_dict

word_occurence = None
word_essential = None

def getWordIndexDict(min_occurence=1):

	if word_occurence == None:
		getWordOccurencePrologue()

	word_list = map(lambda (key, value) : key, sorted(word_occurence.items(), key=operator.itemgetter(1), reverse=True))
	word_list = [word for word in word_list if word in word_essential or word_occurence[word] >= min_occurence]
	word_index_dict = defaultdict(int) # Unknown word to be 0
	word_index_dict['<UNK>'] = 0
	for i, word in enumerate(word_list):
		word_index_dict[word] = i + 1
	word_index_dict['<START>'] = len(word_index_dict)
	word_index_dict['<END>'] = len(word_index_dict)
	
	return word_index_dict

def getWordOccurencePrologue():
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
				word_essential.add(word)

	choice = getTestingChoiceList()
	for s in choice:
		for word in s:
			word_occurence[word] += 1
			word_essential.add(word)
	
	return word_occurence	

def getWordOccurence(min_occurence=1):
	global word_occurence
	global word_essential
	
	if word_occurence == None:
		getWordOccurencePrologue()

	word_index_dict = getWordIndexDict(min_occurence)

	unk_count = 0
	for word in word_occurence:
		if word_occurence[word] < min_occurence and word not in word_essential:
			unk_count += word_occurence[word]
	
	fixed_word_occurence = [0] * len(word_index_dict)
	for word in word_occurence:
		fixed_word_occurence[word_index_dict[word]] += word_occurence[word]

	return fixed_word_occurence

