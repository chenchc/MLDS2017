PYTHON=python

all: preprocess

preprocess: data/training_data.txt data/val_data.txt data/word_vec.txt

data/Holmes_Training_Data/.dirstamp: kaggle_data/training_data.tgz
	tar xvf $< -C data/
	touch $@

data/training_data.txt data/val_data.txt: data/Holmes_Training_Data/.dirstamp preprocess/config.py preprocess/train_transform.py
	$(PYTHON) preprocess/train_transform.py

data/wiki.en.zip:
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip -O data/wiki.en.zip

data/wiki.en.bin: data/wiki.en.zip
	cd data; unzip wiki.en.zip; cd -

data/word_list.txt: data/training_data.txt data/val_data.txt kaggle_data/testing_data.csv preprocess/gen_word_list.py preprocess/utils.py
	$(PYTHON) preprocess/gen_word_list.py

data/word_vec.txt: data/word_list.txt data/wiki.en.bin
	git submodule init
	git submodule update
	cd fastText; make; cd -
	fastText/fasttext print-vectors data/wiki.en.bin < $< > $@

.PHONY=clean
clean:
	rm -rf data/*
