# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:21:01 2022

@author: Bruno Tabet
"""


# import requests
# r = requests.get('https://github.com/typicode/lowdb/blob/main/.eslintrc.json')
# print (r.json())



import urllib.request, json 

# with urllib.request.urlopen('https://github.com/typicode/lowdb/blob/main/.eslintrc.json') as url:
#     data = json.load(url)
#     print(data)
    
    
# with urllib.request.urlopen("https://www.back4app.com/database/tag/download-json") as url:
#     data = json.load(url)
#     print(data)


# with urllib.request.urlopen('https://drive.google.com/file/d/18_EBCdDP_EWq1Mu6G3oWFMOBVj5LbRzg/view?usp=sharing') as url:
#     data = json.load(url)
#     print(data)
    
    
    
with urllib.request.urlopen('https://beesme-my.sharepoint.com/:u:/g/personal/bruno_tabet_enova-me_com/EcXcGKDkc8xCs2mOLoQKSXcBiW3m28QZJN1RtSGQ6_k6oA?e=buuZe5') as url:
    data= json.load(url)
    print(data)
    
    
