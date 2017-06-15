#!/bin/bash

wget https://www.dropbox.com/s/5p614n46b092orv/hw4_data.tar?dl=0 -O hw4_data.tar
tar xvf hw4_data.tar
wget https://www.dropbox.com/s/1kl7kjx2og2q54h/word_vec.tar?dl=0 -O word_vec.tar
tar xvf word_vec.tar
cp $2 test_input.txt
if [[ $1 == 'S2S' ]]; then
	python src/model_s2s.py
else
	python src/model_rl.py
fi
cp output.txt $3

