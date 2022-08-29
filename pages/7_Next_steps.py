# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:24:23 2022

@author: Bruno Tabet
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly_express as px
import openpyxl
import xlsxwriter as xl
import xlwt
import xlrd
from xlutils.copy import copy
import os
import csv

# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")


from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from helpful_funcs.useful_funcs import nav_page
from helpful_funcs.useful_funcs import initialization



st.header('Next steps')

st.write('- For the synthetic features, include the average, max, min, sum and std dev of features')
st.write('- Include the deltatime as a feature')
st.write('- Finalise database')
st.write('- Discuss baseline normalization')