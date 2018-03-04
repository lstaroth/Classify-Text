#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/1 19:54
# @Author  : 孑曦曦孑
# @File    : visitor_weibo_login.py

import requests
import re
import json
from lxml import etree
import time
import os

# url="https://weibo.com/2447680824/G5nMd0MBJ?type=comment#_rnd1519906057635"
#模拟游客登录获取cookies
class visitor():
    def __init__(self,url):
        try:
            self.cookies,self.headers=self.get_cookies()
            self.id=self.weibo_spider(url)
            # self.word=0
        except:
            print("模拟失败-。-")


    def get_cookies(self):
        # 获取dict_data
        print("正在模拟游客登录")
        S = requests.session()
        url = "https://passport.weibo.com/visitor/genvisitor"
        S.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:41.0) Gecko/20100101 Firefox/41.0"
            }
        )
        data = {
            "cb": "gen_callback",
            "fp": '{"os":"1","browser":"Gecko58,0,0,0","fonts":"undefined","screenInfo":"1536*864*24","plugins":""}'
        }
        response = S.post(url, data=data)
        pattren=re.compile(r"\((.*)\)")
        data=pattren.findall(response.text)[0]
        dict_data=json.loads(data)["data"]
        tid=dict_data["tid"]
        confidence=dict_data["confidence"]
        where=dict_data["new_tid"]
        if where:
            where=3
        else:
            where=2
        # print(tid,confidence,where)
        url="https://passport.weibo.com/visitor/visitor?a=incarnate&t="\
            +str(tid)+"&w="\
            +str(where)+"&c="\
            +str(confidence)+"&cb=cross_domain&from=weibo"
        response=S.get(url)
        data=pattren.findall(response.text)[0]
        dict_data=json.loads(data)["data"]
        sub=dict_data["sub"]
        subp=dict_data["subp"]
        # print(sub,subp)
        url="https://login.sina.com.cn/visitor/visitor?a=crossdomain&cb=return_back"\
            +"&s="+str(sub)\
            +"&sp="+str(subp)\
            +"&from=weibo"
        response=S.get(url)
        print("成功获取游客Cookies")
        return S.cookies,S.headers

    #base62解码
    def base62(self,string):
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = len(alphabet)
        strlen = len(string)
        num = 0
        idx = 0
        for char in string:
            power = (strlen - (idx + 1))
            num += alphabet.index(char) * (base ** power)
            idx += 1
        return num

    def weibo_spider(self,url):
        S=requests.session()
        S.cookies=self.cookies
        S.headers=self.headers
        response=S.get(url)
        #获取微博大致内容
        selector = etree.HTML(response.text)
        weibo=selector.xpath('/title/text()')
        page = etree.HTML(response.text)
        content = page.xpath('//title/text()')[0]
        content=str(content).replace("\n"," ")
        print("爬取ing\n\t",content)
        # #获取异步加载url中的id
        pattern=re.compile(r"\d\/(.*)\?type")
        content=pattern.search(url).group(1)
        string1=str(self.base62(content[0]))
        string2=str(self.base62(content[1:5]))
        if(len(string2)==6):
            string2="0"+string2
        string3=str(self.base62(content[5:]))
        id=string1+string2+string3

        return id

    def catch_comments(self,page=1,past_html=None):
        #模拟异步加载
        #https://weibo.com/aj/v6/comment/big?ajwvr=6&id=4213083327566698&filter=hot&page=1
        if(page==1):
            print("开始爬取~")
        S=requests.session()
        S.cookies=self.cookies
        S.headers=self.headers
        url="https://weibo.com/aj/v6/comment/big?ajwvr=6"\
            +"&id="+str(self.id)\
            +"&filter=hot"\
            +"&page="+str(page)
        # print(url)
        response=S.get(url)
        html=json.loads(response.text)["data"]["html"]
        #如果两次相同表示结束了 -。-
        if past_html==html:
            print("爬取结束")
            # print(self.id)
            print("共",page,"页")
            return
        #搜索评论
        text=etree.HTML(html)
        #评论数-xpath
        # comments=text.xpath('//div[@class="list_li S_line1 clearfix"]//div[@class="WB_text"]//text()')
        comments = text.xpath('//div[@class="list_li S_line1 clearfix"]/*/div[@class="WB_text"]')
        # 评论数
        points = text.xpath('//div[@class="list_li S_line1 clearfix"]//*/span[@node-type="like_status"]/child::*[2]//text()')
        #点赞数-xpath
        # points
        pattern = re.compile(r'\：(.*)')
        # try:
        if page==1:
            wa="w"
        else:
            wa="a"
        f=open("weibo_comments.txt", wa,encoding='utf-8')
        # f2=open("weibo_points.txt",wa,encoding='utf-8')
        for i in range(len(comments)):
            comment = comments[i].xpath("text()")
            comment = ",".join(comment[1:])[1:].strip()
            point = points[i]
            if point == "赞":
                point = "0"
            #点赞数为权重0.2
            weights=int(0.2*int(point))
            #写入评论
            comment=comment+"\n"
            f.write(comment)
            for i in range(weights):
                f.write(comment)
            #写入点赞数
        print("已写入", page, "页")
        f.close()
    # except :
    #     print("写入文件失败")

        page+=1
        self.catch_comments(page,html)

    #获取图片
    def catch_pictures(self,page=1,past_html=None):
        # 模拟异步加载
        # https://weibo.com/aj/v6/comment/big?ajwvr=6&id=4213083327566698&filter=hot&page=1
        path = "./weibo-pic"
        if (page == 1):
            print("开始爬取~")
            #创建文件夹
            if not os.path.exists(path):
                os.makedirs(path)
        S = requests.session()
        S.cookies = self.cookies
        S.headers = self.headers
        url = "https://weibo.com/aj/v6/comment/big?ajwvr=6" \
              + "&id=" + str(self.id) \
              + "&filter=hot" \
              + "&page=" + str(page)
        response = S.get(url)
        html = json.loads(response.text)["data"]["html"]
        # 如果两次相同表示结束了 -。-
        if past_html == html:
            print("爬取结束")
            # print(self.id)
            print("共", page, "页")
            return
        # 搜索图片链接
        text = etree.HTML(html)
        ids = text.xpath('//li[@action-type="comment_media_img"]/attribute::action-data')
        # 写入图片
        try:
            for id in ids:
                id = id.split("&")
                id = id[0][4:]
                url = "https://wx3.sinaimg.cn/bmiddle/" + id + ".jpg"
                filename = path + "/" + str(id) + ".jpg"
                response = requests.get(url, stream=True)
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(128):
                        f.write(chunk)
        except:
            print("写入失败")
        page += 1
        print(page)
        self.catch_pictures(page, html)

#模块调用
def start(url=None):
    if url==None:
        print("请输入正确的url")
        return
    else:
        spider=visitor(url)
        spider.catch_comments()

if __name__=="__main__":
    url="https://weibo.com/2387903701/G5bn7s5CS?type=comment"
    url = input("输入需要爬取的微博url:\n")
    spider=visitor(url)
    spider.catch_comments()
    # spider.catch_pictures()
#https://weibo.com/1840483562/G48Ajgfhq?type=comment
#https://weibo.com/aj/v6/comment/big?ajwvr=6&id=4209863153871988&filter=hot&page=1

#https://weibo.com/2387903701/G5bn7s5CS?type=comment
#https://weibo.com/aj/v6/comment/big?ajwvr=6&id=4212353576694814&filter=hot&page=12


#------------------
#      ~。~   nice
#------------------