import config
import read_util

if __name__ == '__main__':
	util = read_util.ReadUtil()
	word_list = util.word_list
	with open(config.word_list_path, 'w') as file:
		for word in word_list:
			file.write(word + ' ')

