wget https://www.dropbox.com/s/wot3adc7kk1mjhv/began_final.tar?dl=0 -O began_final.tar
tar xvf began_final.tar
cp $1 data/sample_testing_text.txt
python generate.py

