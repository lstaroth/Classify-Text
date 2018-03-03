# 学习研究项目：基于微博评论的数据挖掘与情感分析


## 项目简介
学习卷积神经网络，循环神经网络在实际环境下的应用，提升实践能力，了解深度学习在自然语言处理方面的进展

## cnn_for_text_classify
具备较强的自动关键词提取能力，在酒店评论测试集上达到95%的准确率  
采用l2正则和dropout来控制过拟合现象  
4种卷积核使其能提取局部高效的短特征  

## lstm_for_text_classify
具有较强的对长难句，反问句，阴阳怪气句的判断能力，在在酒店评论测试集上达到97%的准确率  
采用双向LSTM网络  
对输入数据进行dropout，模拟增大样本空间  
LSTM层与层之间进行dropout  
对LSTM网络权重，偏置进行l2正则，抗过拟合  
网络采用正交初始化，加快收敛速度，提升训练集上的正确率，大幅提升测试集上的正确率  
采用Clipping Gradients，防止梯度爆炸，提升测试集上的正确率

## word2vec:
项目使用的词向量：embedding_64.bin(1.5G)  
训练语料：百度百科800w条 20G+搜狐新闻400w条 12G+小说：90G左右  
模型参数：window=5 min_count=5 size=64  
下载链接：[百度网盘链接](https://pan.baidu.com/s/1o7MWrnc)      密码:wzqv


## 文件功能介绍
./  
weibo.py：微博评论爬虫  
readdata.py：为情感分析模型提供多种数据加载相关API  
word2vec.py：为情感分析模型提供多种词向量的相关API  
cnn_model.py：CNN文本分类模型图结构  
cnn_train.py：CNN文本分类训练代码
cnn_test.py： CNN文本分类测试代码
lstm_model.py：lstm文本分类模型图结构  
lstm_train.py：lstm文本分类训练代码
lstm_test.py： lstm文本分类测试代码
mixed_cnn_lstm_test.py:采用模型融合方式将cnn与lstm的结果进行融合投票绝对最终结果

./data  
pos.txt：正面评价数据集  
neg.txt：负面评价数据集  
test.txt：自己放样本测试  
embedding_64.bin：训练好的词向量模型  
/cnn:cnn模型训练完成的相关数据参数  
/lstm：lstm模型训练完成的相关数据参数  


## 推荐运行环境
python 3.6  
tensorflow-gpu 1.4  
gensim 3.3  
Ubuntu 64 Bit / windows10 64 Bit  

## 使用模型注意事项
1.文本TXT文件必须采用UTF-8编码格式，非UTF-8格式的，去记事本中另存为的时候选择UTF-8  
2.pos.txt、neg.txt、test.txt 文件一行为一条评论，长度不限，可以有英文和标点（反正都会去除的），不要词性标注信息  
3.词向量模型一定要用我放的那个64维度的bin文件  
4.模型代码在windows上测试过基本没bug，linux平台没测试过，不过肯定需要自行修改文件路径  
