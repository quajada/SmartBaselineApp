# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:52:23 2022

@author: Bruno Tabet
"""

from sharepoint import SharePointSite

# set file name
file_name = 'database.json'

# set the folder name
folder_name = 'Documents'

#get file
# file = SharePointSite().download_folder(file_name, folder_name)


# #save file
# with open('https://beesme-my.sharepoint.com/:u:/g/personal/bruno_tabet_enova-me_com/EcXcGKDkc8xCs2mOLoQKSXcBiW3m28QZJN1RtSGQ6_k6oA?e=Fsc1A7', 'wb') as f:
#     print(f)
    
from urllib.request import urlopen  # the lib that handles the url stuff

# data = urlopen('https://beesme-my.sharepoint.com/:u:/g/personal/bruno_tabet_enova-me_com/EcXcGKDkc8xCs2mOLoQKSXcBiW3m28QZJN1RtSGQ6_k6oA?e=Fsc1A7')
# data = urlopen('https://stackoverflow.com/questions/2792650/import-error-no-module-name-urllib2')
# data = urlopen('https://github.com/BrunoTabet/SmartBaselineApp/edit/master/database.json')
# data = urlopen("https://drive.google.com/drive/folders/1nvc9DQwEk5IuEsetDeMstCsQ8iuAlRvd")
data = urlopen('https://drive.google.com/file/d/18_EBCdDP_EWq1Mu6G3oWFMOBVj5LbRzg/view?usp=sharing')
for line in data:
    print (line)