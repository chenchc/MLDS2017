import config
import os
import re

#excluded_files = ['AESOP11.TXT']
excluded_files = []

def transformTrainFile(content):
	# Eliminate cariage return
	content = content.replace('\r', '')

	# Split into paragraphs
	paragraphs = re.split('\n[ \n]*\n', content)

	# Eliminate newline in each paragraph
	paragraphs = [paragraph.replace('\n', '') for paragraph in paragraphs]

	# Strip whitespace
	paragraphs = [paragraph.strip(' ') for paragraph in paragraphs]
	
	# Trim the header
	start_line = None
	for idx, paragraph in enumerate(paragraphs):
		if re.match('(^.*\*?end\*?the small print.*$)', paragraph.lower()) != None:
			start_line = idx
	if start_line == None:
		raise Exception('Start Line Error')
	del paragraphs[: start_line + 1]

	# Trim the trailer
	end_line = None
	for idx, paragraph in reversed(list(enumerate(paragraphs))):
		if re.match('(^(\*{3})?((the )?end of )?(the )?project gutenberg.*$)|(^\ *the end.*$)', paragraph.lower()) != None:
			end_line = idx
	if end_line != None:
		del paragraphs[end_line:]
	
	# Remove those paragraphs don't end with a period
	paragraphs = [p for p in paragraphs if re.search('\.$', p) != None]

	# Trim chapter titles (no lowercase letter)
	paragraphs = [p for p in paragraphs if re.search('[a-z]', p) != None]

	# Make sentences
	sentences = []
	for paragraph in paragraphs:
		new_sentences = paragraph.split('.')
		new_sentences = [s + '.' for s in new_sentences if s != '']
		sentences.extend(new_sentences)
	
	# Strip whitespace
	sentences = [s.strip(' ') for s in sentences]

	return sentences

def transformTrainData():
	trainData = []
	valData = []
	for dirname, subdirnames, filenames in os.walk(config.train_folder):
		for filename in filenames:
			if filename in excluded_files:
				continue
			print 'Reading ' + filename + '...'
			file = open(os.path.join(dirname, filename))
			content = file.read()
			try:
				data = transformTrainFile(content)
			except Exception as e:
				print e
				raw_input()
			if hash(filename) % config.val_split == 0:
				valData.extend(data)
			else:
				trainData.extend(data)
	print trainData

transformTrainData()
