from pyspark import SparkConf,SparkContext   # from pyspark.conf import SparkConf 
from pyspark.streaming import StreamingContext 
from pyspark.sql import Row,SQLContext
import sys
#import kafka
import requests
from pyspark.streaming.kafka import KafkaUtils    # ERROR
import json
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2 import service_account
import re
import operator
import random
import time

new_csv = pd.read_csv("group_simulation.csv")
gc = gspread.service_account(filename='tweets-97ab5693ec9b.json') # Google API credentials
sh = gc.open_by_key('1dId7bMBrMCJsFGdIGuAL8Mzhpd7mCRgmnBg86eOYg54') # your_google_sheet_ID
sheet_index_no = 0
worksheet = sh.get_worksheet(sheet_index_no)

range_of_cells = worksheet.range('A2:E1000') #-> Select the range you want to clear
for cell in range_of_cells:
    cell.value = ''
worksheet.update_cells(range_of_cells)
for i in range(len(new_csv)):
    i2 = i*9+9
    print(new_csv.iloc[i*9:i2,:])
    set_with_dataframe(worksheet, new_csv.iloc[i*9:i2,:],row=2,include_column_header=False)
    time_value = random.uniform(1, 2.5)
    time.sleep(time_value)
