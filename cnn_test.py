import tensorflow as tf
import numpy as np
import readdata
import word2vec
import os
import Cnn_Model


test_file_path="./data//test.txt"
train_data_path="./data//training_params.pickle"
embedding_model_path="./data//embedding_64.bin"
test_sample_percentage=0.1
num_labels=2
embedding_size=64
filter_sizes=[3,4,5]
num_filters=128
l2_reg_lambda=0.0


if not os.path.exists(embedding_model_path):
    print("word2vec model is not found")

if not os.path.exists(train_data_path):
    print("train params is not found")

params = readdata.loadDict(train_data_path)
num_labels = int(params['num_labels'])
train_length = int(params['max_sentences_length'])



test_sample_lists = readdata.get_cleaned_list(test_file_path)
test_sample_lists,max_sentences_length = readdata.padding_sentences(test_sample_lists,padding_token='<PADDING>',padding_sentence_length=train_length)
test_sample_arrays=np.array(word2vec.get_embedding_vector(test_sample_lists,embedding_model_path))
print("sample.shape = {}".format(test_sample_arrays.shape))

with tf.Graph().as_default():
    sess=tf.Session()
    with sess.as_default():
        cnn=Cnn_Model.TextCNN(
            sequence_length=max_sentences_length,
            num_classes=num_labels,
            embedding_size=embedding_size,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda
        )

        #加载参数
        saver = tf.train.Saver()
        saver.restore(sess, "data/text_model")

        #定义测试函数
        def test_step(x_batch):
            feed_dict={
                cnn.input_x:x_batch,
                cnn.dropout_keep_prob:1.0
            }
            predictions=sess.run(
                [cnn.predictions],
                feed_dict=feed_dict
            )
            return predictions


        #拿到结果
        predictions=test_step(test_sample_arrays)
        print("(0->neg & 1->pos)the result is:")
        print(predictions)
