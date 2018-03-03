import tensorflow as tf
import numpy as np
import readdata
import word2vec
import os
import cnn_model


test_file_path="./data//test.txt"
train_data_path="./data//cnn//training_params.pickle"
embedding_model_path="./data//embedding_64.bin"

class config():
    test_sample_percentage=0.2
    num_labels=2
    embedding_size=64
    filter_sizes=[2,3,4,5]
    num_filters=128
    dropout_keep_prob=1.0
    l2_reg_lambda=0.5
    batch_size=64
    num_epochs=200
    max_sentences_length=0
    lr_rate=1e-3


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
testconfig=config()
testconfig.max_sentences_length=max_sentences_length

sess=tf.InteractiveSession()
cnn=cnn_model.TextCNN(config=testconfig)

#加载参数
saver = tf.train.Saver()
saver.restore(sess, "data/cnn/text_model")

#定义测试函数
def test_step(x_batch):
    feed_dict={
        cnn.input_x:x_batch,
        cnn.dropout_keep_prob:1.0
        }
    predictions,scores=sess.run(
        [cnn.predictions,cnn.softmax_result],
        feed_dict=feed_dict
        )
    return (predictions,scores)


#拿到结果
predictions,scores=test_step(test_sample_arrays)
print("(0->neg & 1->pos)the result is:")
print(predictions)
print("********************************")
print("the scores is:")
print(scores)
print(scores.shape)
print()
