PYTHON=python

MLDS_hw2_data/:
	wget http://speech.ee.ntu.edu.tw/~yangchiyi/MLDS_hw2/MLDS_hw2_data.tar.gz
	tar xvf MLDS_hw2_data.tar.gz

data/wiki.en.zip:
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip -O data/wiki.en.zip

data/wiki.en.bin: data/wiki.en.zip
	cd data; unzip wiki.en.zip; cd -

data/word_list.txt: src/gen_word_list.py src/read_util.py src/config.py MLDS_hw2_data/
	$(PYTHON) $<

data/word_vec.txt: data/word_list.txt data/wiki.en.bin
	git submodule init
	git submodule update
	cd ../fastText; make; cd -
	../fastText/fasttext print-vectors data/wiki.en.bin < $< > $@

.PHONY=clean
clean:
	rm -rf data/*
	rm -rf MLDS_hw2_data

