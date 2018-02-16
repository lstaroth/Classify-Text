**学习研究项目：基于微博评论的数据挖掘与情感分析**  


**项目简介**  
学习cnn，rnn在实际环境下的应用，提升实践能力，了解深度学习在自然语言处理方面的进展。  
分别测试cnn与rnn模型在微博评论分类任务下的表现  

cnn_for_emotion_classify  
模型架构参考论文：[Implementing a CNN for Text Classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)  

lstm_for_emotion_classify  
待完善

word2vec:  
项目使用的词向量：embedding_64.bin(1.5G)  
训练语料：百度百科800w条 20G+搜狐新闻400w条 12G+小说：90G左右  
模型参数：window=5 min_count=5 size=64  
下载链接：[百度网盘链接](https://pan.baidu.com/s/1o7MWrnc)      密码:wzqv


**文件功能介绍**

./  
weibo.py：微博评论爬虫  
readdata.py：为情感分析模型提供多种数据加载相关API  
word2vec.py：为情感分析模型提供多种词向量的相关API  
Cnn_Model.py：CNN文本分类模型图结构  
cnn_train.py：CNN文本分类训练代码
cnn_test.py： CNN文本分类测试代码


./data  
pos.txt：正面评价数据集  
neg.txt：负面评价数据集  
test.txt：自己放样本测试  
embedding_64.bin：训练好的词向量模型
training_params.pickle：存着训练时的类别数量和句子允许的最长单词量  
模型保存参数文件  


**推荐运行环境**  

python 3.6  
tensorflow-gpu 1.4  
gensim 3.3  
Ubuntu 64bit / windows10 64bit  