import matplotlib.pyplot as plt


def show_emtion(prediction):
    positive=0
    negative=0
    for i, num in enumerate(prediction):
        if num == 0:
            positive+=1
        else:
            negative+=1
    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']
    #用来正常显示负号
    plt.rcParams['axes.unicode_minus']=False
    #调节图形大小(宽,高)
    plt.figure(figsize=(12,8))
    #定义饼状图的标签，标签是列表
    labels = [u'喜悦',u'低落']
    #每个标签的占比,不一定要和为100%
    sizes = [positive,negative]
    colors = ['lightskyblue','FireBrick']
    explode = (0.05,0)
    patches,l_text,p_text = plt.pie(sizes,explode=explode,labels=labels,colors=colors,labeldistance = 1.1,autopct = '%3.1f%%',shadow = False,startangle = 90,pctdistance = 0.5)

    #改变文本的大小
    for t in l_text:
        t.set_size(20)
    for t in p_text:
        t.set_size(20)
    #圆
    plt.axis('equal')
    plt.legend()
    plt.show()