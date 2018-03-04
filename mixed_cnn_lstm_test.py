import os
import readdata
import word2vec
import lstm_model
import cnn_model
import numpy as np
import tensorflow as tf



#文件路径
current_path=os.path.abspath(os.curdir)
test_file_path="./data//test.txt"
embedding_model_path="./data//embedding_64.bin"
lstm_train_data_path="./data//lstm//training_params.pickle"
cnn_train_data_path="./data//cnn//training_params.pickle"


#模型超参
class lstmconfig():
    test_sample_percentage=0.2
    num_labels=2
    embedding_size=64
    dropout_keep_prob=1.0
    batch_size=64
    num_epochs=200
    max_sentences_length=40
    num_layers=2
    max_grad_norm=5
    l2_rate=0.001

class cnnconfig():
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

if not os.path.exists(lstm_train_data_path):
    print("lstm train params is not found")

lstm_params = readdata.loadDict(lstm_train_data_path)
lstm_num_labels = int(lstm_params['num_labels'])
lstm_train_length = int(lstm_params['max_sentences_length'])

if not os.path.exists(cnn_train_data_path):
    print("cnn train params is not found")

cnn_params = readdata.loadDict(cnn_train_data_path)
cnn_num_labels = int(cnn_params['num_labels'])
cnn_train_length = int(cnn_params['max_sentences_length'])


test_sample_lists = readdata.get_cleaned_list(test_file_path)
lstm_test_sample_lists,lstm_max_sentences_length = readdata.padding_sentences(test_sample_lists,padding_token='<PADDING>',padding_sentence_length=lstm_train_length)
cnn_test_sample_lists,cnn_max_sentences_length = readdata.padding_sentences(test_sample_lists,padding_token='<PADDING>',padding_sentence_length=cnn_train_length)
lstm_test_sample_arrays=np.array(word2vec.get_embedding_vector(lstm_test_sample_lists,embedding_model_path))
cnn_test_sample_arrays=np.array(word2vec.get_embedding_vector(cnn_test_sample_lists,embedding_model_path))
lstm_config=lstmconfig()
cnn_config=cnnconfig()
lstm_config.max_sentences_length=lstm_max_sentences_length
cnn_config.max_sentences_length=cnn_max_sentences_length



lstm_graph=tf.Graph()
cnn_graph=tf.Graph()
lstm_sess=tf.Session(graph=lstm_graph)
cnn_sess=tf.Session(graph=cnn_graph)


with lstm_sess.as_default():
    with lstm_graph.as_default():
        lstm = lstm_model.TextLSTM(config=lstm_config)
        lstm_saver = tf.train.Saver()
        lstm_saver.restore(lstm_sess, "data/lstm/text_model")
        def lstm_test_step(x_batch):
            feed_dict={
                lstm.input_x:x_batch,
                lstm.dropout_keep_prob:lstm_config.dropout_keep_prob
            }
            scores=lstm_sess.run(
                [lstm.softmax_result],
                feed_dict=feed_dict
            )
            return scores


        lstm_scores = lstm_test_step(lstm_test_sample_arrays)


with cnn_sess.as_default():
    with cnn_graph.as_default():
        cnn = cnn_model.TextCNN(config=cnn_config)
        cnn_saver = tf.train.Saver()
        cnn_saver.restore(cnn_sess, "data/cnn/text_model")
        def cnn_test_step(x_batch):
            feed_dict={
                cnn.input_x:x_batch,
                cnn.dropout_keep_prob:cnn_config.dropout_keep_prob
            }
            scores=cnn_sess.run(
                [cnn.softmax_result],
                feed_dict=feed_dict
            )
            return scores


        cnn_scores = cnn_test_step(cnn_test_sample_arrays)

lstm_sess.close()
cnn_sess.close()
mixed_scores=np.sum([lstm_scores,cnn_scores],axis=0)
print(mixed_scores)
predictions=np.argmax(mixed_scores,axis=2)
print(predictions)