#!/bin/bash

wget http://speech.ee.ntu.edu.tw/~yangchiyi/MLDS_hw2/MLDS_hw2_data.tar.gz -O MLDS_hw2_data.tar; tar xf MLDS_hw2_data.tar
wget https://www.dropbox.com/s/kc28n3zpu228368/s2vt_final.tar?dl=0 -O s2vt_final.tar; tar xf s2vt_final.tar
rm -rf test
mkdir test
cp $1 test/testing_id.txt
cd test; mkdir feat; cd -
cp $2/* test/feat/
python src/model_s2vt.py

