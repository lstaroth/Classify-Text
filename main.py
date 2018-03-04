import numpy as np
from weibo import start as spider
from cnn_test import get_cnn_result
from lstm_test import get_lstm_result
from mixed_cnn_lstm_test import get_mixed_result
from visual import show_emtion



if __name__=="__main__":
    prediction = np.array([])
    print("********************欢迎使用微博舆情分析工具***********************")
    url = input("请输入需要分析的微博url:\n")
    #调用weibo.py接口开始爬取相关微博评论
    spider(url)
    #选择模型
    model_index=int(input("请输入你想选择的AI模型：\n1.CNN\n2.LSTM\n3.CNN & LSTM融合模型\n"))
    #调用AI模型接口返回结果
    if model_index == 1:
        prediction=get_cnn_result()
    elif model_index == 2:
        prediction=get_lstm_result()
    elif model_index == 3:
        prediction=get_mixed_result()
    else:
        print("输入信息错误")
    #移交可视化模块完成数据视化
    show_emtion(prediction)