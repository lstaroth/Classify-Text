import os
import readdata
import word2vec
import Cnn_Model
import numpy as np
import tensorflow as tf



#文件路径
current_path=os.path.abspath(os.curdir)
positive_file_path="./data//pos.txt"
negative_file_path="./data//neg.txt"
embedding_model_path="./data//embedding_64.bin"
train_data_path="./data//training_params.pickle"



#模型超参
test_sample_percentage=0.1
num_labels=2
embedding_size=64
filter_sizes=[3,4,5]
num_filters=128
dropout_keep_prob=0.5
l2_reg_lambda=0.0
batch_size=64
num_epochs=200
evaluate_every=100
checkpoint_every=100
num_checkpoints=5
allow_soft_placement=True
log_device_placement=False


#加载数据
all_sample_lists,all_label_arrays,max_sentences_length=readdata.get_all_data_from_file(positive_file_path,negative_file_path)
all_sample_arrays=np.array(word2vec.get_embedding_vector(all_sample_lists))
print("sample.shape = {}".format(all_sample_arrays.shape))
print("label.shape = {}".format(all_label_arrays.shape))

#存储训练参数
params={"num_labels":num_labels,"max_sentences_length":max_sentences_length}
readdata.save(params,train_data_path)

#打乱样本顺序
np.random.seed(10)
random_index=np.random.permutation(np.arange(len(all_label_arrays)))
random_sample_lists=all_sample_lists[random_index]
random_label_lists=all_label_arrays[random_index]

#按比例抽取测试样本
num_tests=int(test_sample_percentage*len(all_label_arrays))
test_sample_arrays=all_sample_arrays[:num_tests]
test_label_arrays=all_label_arrays[:num_tests]
train_sample_arrays=all_sample_arrays[num_tests:]
train_label_arrays=all_label_arrays[num_tests:]
print("Train/Test split: {:d}/{:d}".format(len(train_label_arrays), len(test_label_arrays)))


#开始训练
with tf.Graph().as_default():
    sess=tf.Session()
    with sess.as_default():
        CnnModel=Cnn_Model.TextCNN(
            sequence_length=train_sample_arrays.shape[1],
            num_classes=train_label_arrays.shape[1],
            embedding_size=embedding_size,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda
        )
