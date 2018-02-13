import os


'''
文件路径
'''
current_path=os.path.abspath(os.curdir)
positive_file_path="./data//pos.txt"
negative_file_path="./data//neg.txt"
embedding_model_path="./data//embedding_64.bin"
train_data_path="./data//training_params.pickle"


'''
模型超参
'''
test_sample_percentage=0.1
num_labels=2
embedding_dim=64
filter_sizes=[3,4,5]
num_filter=128
dropout_keep_prob=0.5
l2_reg_lambda=0.0
batch_size=64
num_epochs=200
evaluate_every=100
checkpoint_every=100
num_checkpoints=5
allow_soft_placement=True
log_device_placement=False