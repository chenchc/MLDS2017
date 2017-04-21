import os

cwd = os.getcwd()

train_feat_folder_path = cwd + '/MLDS_hw2_data/training_data/feat/'
train_label_path = cwd + '/MLDS_hw2_data/training_label.json'
test_feat_folder_path = cwd + '/MLDS_hw2_data/testing_data/feat/'
test_id_path = cwd + '/MLDS_hw2_data/testing_id.txt'
test_public_label_path = cwd + '/MLDS_hw2_data/testing_public_label.json'
test_limited_feat_folder_path = cwd + '/MLDS_hw2_time_limited/feat/'
test_limited_id_path = cwd + '/MLDS_hw2_time_limited/testing_id.txt'

word_list_path = cwd + '/data/word_list.txt'
word_vec_path = cwd + '/data/word_vec.txt'

word_vec_cached_path = cwd + '/data/word_vec.p'
