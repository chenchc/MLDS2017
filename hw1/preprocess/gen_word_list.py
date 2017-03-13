import config
import os.path
import subprocess
import utils

def genWordListOnTrainLikeFile(filename):
	# Check if the file exist
	if not os.path.isfile(filename):
		subprocess.call(['make', filename])

	# Load data
	data = utils.getTrainingLikeData(filename)
	unique_words = set()
	for instance in data:
		for word in instance:
			unique_words.add(word)

	return unique_words

def genWordListOnTestFile():
	# Load data
	data = utils.getTestingData()
	unique_words = set()
	for instance in data:
		for word in instance:
			if word != '_____':
				unique_words.add(word)

	# Load choices
	choiceList = utils.getTestingChoiceList()
	for instance in choiceList:
		for word in instance:
			unique_words.add(word)

	return unique_words

def genWordList():
	word_list = genWordListOnTrainLikeFile(config.train_file)
	word_list |= genWordListOnTrainLikeFile(config.val_file)
	word_list |= genWordListOnTestFile()

	with open(config.word_list_file, 'w') as file:
		for word in word_list:
			file.write(word + ' ')
			

if __name__ == '__main__':
	genWordList()
