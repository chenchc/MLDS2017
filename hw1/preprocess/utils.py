import config
import csv
import numpy as np
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
			data.append(question.split(' '))

	return data
	
def getTestingChoiceList():
	data = []
	with open(config.test_file, 'rb') as file:
		reader = csv.DictReader(file)
		for row in reader:
			data.append([row['a)'], row['b)'], row['c)'], row['d)'], row['e)']]) 

	return data

def getWordVecDict():
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
