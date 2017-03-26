#!/bin/bash

cd data; wget https://www.dropbox.com/s/v98ldfp9wzj8p64/model_final.tar?dl=0 -O model_final.tar; tar xf model_final.tar; cd -
cp $1 kaggle_data/testing_data.csv
python test.py
cp data/submit_bilstm_hybrid6.csv $2
