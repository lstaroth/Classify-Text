# -*- coding:gbk -*-
import re
import requests
import json
from lxml import html
#测试微博4054483400791767
comments=[]

def get_page(weibo_id):
	url='https://m.weibo.cn/status/{}'.format(weibo_id)
	html=requests.get(url).text
	regcount=r'"comments_count": (.*?),'
	comments_count=re.findall(regcount,html)[-1]
	comments_count_number=int(comments_count)
	page=int(comments_count_number/10)
	return page-1
    
def opt_comment(comment):
    tree=html.fromstring(comment)
    strcom=tree.xpath('string(.)')
    reg1=r'回复@.*?:'
    reg2=r'回覆@.*?:'
    reg3=r'//@.*'
    newstr=''
    comment1=re.subn(reg1,newstr,strcom)[0]
    comment2=re.subn(reg2,newstr,comment1)[0]
    comment3=re.subn(reg3,newstr,comment2)[0]
    return comment3
    
def get_responses(id,page):
    url="https://m.weibo.cn/api/comments/show?id={}&page={}".format(id,page)
    response=requests.get(url)
    return response

def get_weibo_comments(response):
    json_response=json.loads(response.text)
    for i in range(0,len(json_response['data'])):
        comment=opt_comment(json_response['data'][i]['text'])
        comments.append(comment)


weibo_id=input("输入微博id，自动返回前5页评论：")
weibo_id=int(weibo_id)
print('\n')
page=get_page(weibo_id)
for page in range(1,page+1):
    response=get_responses(weibo_id,page)
    get_weibo_comments(response)

for com in comments:
    print(com)
print(len(comments))
