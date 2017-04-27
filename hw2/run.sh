#!/bin/bash

wget https://www.dropbox.com/s/wp57qiorhl0xoy3/s2vt_final.tar?dl=0 -O s2vt_final.tar; tar xf s2vt_final.tar
rm -rf test
mkdir test
cp $1 test/testing_id.txt
cd test; mkdir feat; cd -
cp $2/* test/feat/
python src/model_s2vt.py

