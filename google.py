# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:46:42 2022

@author: Bruno Tabet
"""

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials


# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# # https://drive.google.com/file/d/18_EBCdDP_EWq1Mu6G3oWFMOBVj5LbRzg/view?usp=sharing
# fileDownloaded = drive.CreateFile({'id': '18_EBCdDP_EWq1Mu6G3oWFMOBVj5LbRzg'})



import urllib.request, json 
with urllib.request.urlopen("https://drive.google.com/file/d/18_EBCdDP_EWq1Mu6G3oWFMOBVj5LbRzg/view?usp=sharing") as url:
    data = json.load(url)
    print(data)