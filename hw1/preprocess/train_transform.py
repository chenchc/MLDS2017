import config
import os
import random
import re

excluded_files = []

def transformTrainFile(content):
	# Eliminate cariage return
	content = content.replace('\r', '')

	# Split into paragraphs
	paragraphs = re.split('\n[ \n]*\n', content)

	# Eliminate newline in each paragraph
	paragraphs = [re.sub('(?=\W)- *\n', '', paragraph) for paragraph in paragraphs]
	paragraphs = [paragraph.replace('\n', ' ') for paragraph in paragraphs]

	# Strip whitespace
	paragraphs = [paragraph.strip(' ') for paragraph in paragraphs]
	
	# Trim the header
	start_line = None
	for idx, paragraph in list(enumerate(paragraphs)):
		if re.match('(^.*\*?end\*?the small print.*$)', paragraph.lower()) != None:
			start_line = idx

	if start_line == None:
		raise Exception('Start Line Error')
	del paragraphs[: start_line + 1]

	# Trim the trailer
	end_line = None
	trailer_not_found = True
	for idx, paragraph in reversed(list(enumerate(paragraphs))[len(paragraphs)/2:]):
		if re.match('(^.*((the )?end of )?(the )?project gutenberg.*$)', paragraph.lower()) != None:
			end_line = idx
			trailer_not_found = False
	if not trailer_not_found:
		del paragraphs[end_line:]


	# Remove those paragraphs don't end with a period (or question mark, etc.)
	paragraphs = [p for p in paragraphs if re.search('[.;!?]$', p) != None]

	# Trim chapter titles (no lowercase letter)
	paragraphs = [p for p in paragraphs if re.search('[a-z]', p) != None]

	# "blabla" or 'blabla' -> blabla
	for idx, paragraph in enumerate(paragraphs):
		paragraphs[idx] = re.sub('"(?P<bla>([a-zA-Z]+))"', lambda m: m.group('bla'), paragraphs[idx])
		paragraphs[idx] = re.sub("'+(?P<bla>([a-zA-Z]+))'+", lambda m: m.group('bla'), paragraphs[idx])
	
	# Extract quotes (from " ")
	for idx, paragraph in enumerate(paragraphs):
		if re.search('"[^"]*"', paragraph) != None:
			for quote in re.findall('"(?P<quote>[^"]*)"', paragraph):
				paragraphs.append(quote)
			paragraphs[idx] = re.sub('[^.;!?]*"[^"]*".*?[.;!?]', '', paragraph)

	# Extract quotes (from `` '')
	for idx, paragraph in enumerate(paragraphs):
		if re.search("``.*?''", paragraph) != None:
			for quote in re.findall("``(?P<quote>.*?)''", paragraph):
				paragraphs.append(quote)
			paragraphs[idx] = re.sub("[^.;!?]*``.*?''.*?[.;!?]", '', paragraph)

	# Extract quotes (from '' '')
	for idx, paragraph in enumerate(paragraphs):
		if re.search("''.*?''", paragraph) != None:
			for quote in re.findall("''(?P<quote>.*?)''", paragraph):
				paragraphs.append(quote)
			paragraphs[idx] = re.sub("[^.;!?]*''.*?''.*?[.;!?]", '', paragraph)

	# Make sentences
	sentences = []
	for paragraph in paragraphs:
		new_sentences = re.split('(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)[.;!?]', paragraph)
		new_sentences = [s for s in new_sentences if s != '']
		sentences.extend(new_sentences)

	# Strip whitespace
	sentences = [s.strip(' ') for s in sentences]

	# Remove parenthesed content
	sentences = [re.sub('\(.*?\)', '', s) for s in sentences]

	# Delete weird sentences
	sentences = [s for s in sentences if re.search("([^a-zA-Z,.'\- ])|([,'\-]{2,})", s) == None]
	sentences = [s for s in sentences if re.match("^[A-Z].*[a-z]$", s) != None]

	# Delete empty sentences
	sentences = [s for s in sentences if s != '']

	# Tokenize
	tokens_of_sentences = []
	for s in sentences:
		s = re.sub(',', ' , ', s)
		tokens = re.split('\s+', s);
		tokens_of_sentences.append(tokens)

	# Delete sentences with less than 3 words
	tokens_of_sentences = [t for t in tokens_of_sentences if len(t) >= 3]

	# Delete sentences with punctuations which is not ','
	#tokens_of_sentences = [t for t in tokens_of_sentences if all(re.match('(^[A-Za-z].*$)|(^,$)', token) for token in t)]

	# Refill to sentences
	tokenized_sentences = [reduce(lambda x, y: x + ' ' + y, t[1:], t[0]) for t in tokens_of_sentences]

	return tokenized_sentences

def transformTrainData():
	trainData = []
	valData = []

	# Read files
	for dirname, subdirnames, filenames in os.walk(config.train_folder):
		for filename in filenames:
			if filename in excluded_files:
				continue
			print 'Reading ' + filename + '...'
			with open(os.path.join(dirname, filename), 'r') as file:
				content = file.read()
				try:
					data = transformTrainFile(content)
				except Exception as e:
					print e

				if hash(filename) % config.val_split == 0:
					valData.extend(data)
				else:
					trainData.extend(data)

	# Randomize
	print 'Shufflng...'
	random.seed(config.random_seed)
	random.shuffle(trainData)
	random.shuffle(valData)

	# Write files
	print 'Writing...'
	with open(config.train_file, 'w') as file:
		for sentence in trainData:
			file.write(sentence + '\n')
	with open(config.val_file, 'w') as file:
		for sentence in valData:
			file.write(sentence + '\n')
	
if __name__ == '__main__':
	transformTrainData()
