#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
#import spark-csv
#import pygsheets

#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /Users/franciscomansillanavarro/Documents/Streaming/spark-2.4.7-bin-hadoop2.7/jars/spark-streaming-kafka-0-8-assembly_2.11-2.4.7.jar pyspark-shell' 
# In[ ]:



# In[ ]:


# create spark configuration
conf = SparkConf() 
conf.setAppName("polutionApp")


# In[ ]:


sc = SparkContext(conf=conf) 
sc.setLogLevel("ERROR")


# In[ ]:Text
ssc = StreamingContext(sc, 2)

# setting a checkpoint to allow RDD recovery
ssc.checkpoint("checkpoint_group")


# In[ ]:
topic_name = 'testClusterTopic'

from pyspark.streaming.kafka import KafkaUtils
dataStream = KafkaUtils.createDirectStream(ssc, [topic_name], {"bootstrap.servers": 'data3:6667'})
dataStream.pprint(num=3)
# ./bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.12:2.4.7


# In[ ]:


def aggregate_tags_count(new_values, total_sum): 
    
    if (len(new_values) == 0): 
        return total_sum
    else:
        return sum(new_values) - (total_sum or 0)



def get_sql_context_instance(spark_context):
    if ('sqlContextSingletonInstance' not in globals()):
        globals()['sqlContextSingletonInstance'] = SQLContext(spark_context) 
    return globals()['sqlContextSingletonInstance']

def avg_map(row):
    return (row[0], (row[1],1))
def avg(v1,v2):
    return ((v1[0] + v2[0],v1[1]+v2[1]))

# In[ ]:

def process_rdd2(time, rdd):
    print("----------- %s -----------" % str(time))
    try:
        # Get spark sql singleton context from the current context 
        sql_context = get_sql_context_instance(rdd.context)
        # convert the RDD to Row RDD
        row_rdd = rdd.map(lambda w: Row(Station_Code=w[0],City = w[1],Latitude = w[2] ,Longitude= w[3],AQI = w[4],PM10=w[5],PM2=w[6]))

        if row_rdd.isEmpty():
            print('RDD is empty')
        else:
            # create a DF from the Row RDD
            hashtags_df = sql_context.createDataFrame(row_rdd)
            # Register the dataframe as table 
            hashtags_df.registerTempTable("hashtags")
            # get the top 10 stations from the table using SQL and print them
            hashtag_counts_df = sql_context.sql("select Station_Code,City,Latitude,Longitude,AQI,PM10,PM2 from hashtags order by Station_Code desc limit 10")

            hashtag_counts_df.show()
            hashtag_counts_df.toPandas().to_csv('chekit.csv') 
            print('Saved')
	    # ACCES GOOGLE SHEET
            gc = gspread.service_account(filename='tweets-97ab5693ec9b.json') # Google API credentials
            sh = gc.open_by_key('1dId7bMBrMCJsFGdIGuAL8Mzhpd7mCRgmnBg86eOYg54') # your_google_sheet_ID
            sheet_index_no = 0
            worksheet = sh.get_worksheet(sheet_index_no) #-> 0 - first sheet, 1 - second sheet etc. 
#              worksheet.insert_row(hashtag_counts_df, 1)
            print("worksheet opened")
            #hasdf = hashtag_counts_df.toPandas()
            #hasdf.pprint()
            print(hashtags_df.columns)
            print("CONVERTEDd")
            #worksheet.update_cells([hashtags_df.toPandas().tolist()])
            print("pastedd")
            # CLEAR SHEET CONTENT
            range_of_cells = worksheet.range('A12:G1000') #-> Select the range you want to clear
            for cell in range_of_cells:
                cell.value = ''
            worksheet.update_cells(range_of_cells) 

#             # APPEND DATA TO SHEET
            set_with_dataframe(worksheet, hashtag_counts_df,row=15) #-> THIS EXPORTS YOUR DATAFRAME TO THE GOOGLE SHEET
    except:
        e = sys.exc_info()[0] 
        print("Error: %s" % e)





def process_rdd(time, rdd):
    print("----------- %s -----------" % str(time))
    try:
        # Get spark sql singleton context from the current context 
        sql_context = get_sql_context_instance(rdd.context)
        # convert the RDD to Row RDD
        row_rdd = rdd.map(lambda w: Row(Station_Code=w[0],CO = w[1] ))

        if row_rdd.isEmpty():
            print('RDD is empty')
        else:
            # create a DF from the Row RDD
            hashtags_df = sql_context.createDataFrame(row_rdd)
            # Register the dataframe as table 
            hashtags_df.registerTempTable("hashtags")
            # get the top 10 stations from the table using SQL and print them

            hashtag_counts_df = sql_context.sql("select Station_Code, CO from hashtags order by Station_Code")

            hashtag_counts_df.show()
            hashtag_counts_df.toPandas().to_csv('checklist2.csv') 
            print('Saved')
	    
	    # ACCES GOOGLE SHEET
            gc = gspread.service_account(filename='tweets-97ab5693ec9b.json') # Google API credentials
            sh = gc.open_by_key('1dId7bMBrMCJsFGdIGuAL8Mzhpd7mCRgmnBg86eOYg54') # your_google_sheet_ID
            sheet_index_no = 0
            worksheet = sh.get_worksheet(sheet_index_no) #-> 0 - first sheet, 1 - second sheet etc. 
            print("worksheet opened")

            #worksheet.update_cells([hashtags_df.toPandas().tolist()])
            
            #worksheet.update([hashtag_counts_df.columns.values.tolist()] + hashtag_counts_df.values.tolist())
            # CLEAR SHEET CONTENT
            range_of_cells = worksheet.range('H12:I1000') #-> Select the range you want to clear
            for cell in range_of_cells:
                cell.value = ''
            worksheet.update_cells(range_of_cells) 

#             # APPEND DATA TO SHEET
            set_with_dataframe(worksheet, hashtag_counts_df,col=8,row=15) #-> THIS EXPORTS YOUR DATAFRAME TO THE GOOGLE SHEET
            print("Pasted")
    except:
        e = sys.exc_info()[0] 
        print("Error: %s" % e)

#def cleanupWords(word):
#	return re.search(r"\\’", re.UNICODE).split(word.lower())
# In[ ]:

#words = dataStream.map(lambda x: json.loads(x[1])).flatMap(lambda line: line.split("\{"))
wordsr= dataStream.map(lambda kaf: kaf[1]).map(lambda x: json.loads(x))
wordsr.pprint(2)

#trial = wordsr.map(lambda x:(x['Station code'],(x['SO2'], x['NO2'], x['O3'], x['CO'], x['PM10'], x['PM2.5'])))\
    #.updateStateByKey(aggregate_tags_count)

trial = wordsr.map(lambda x:(x['Station code'],x['CO']))

trial = trial.updateStateByKey(aggregate_tags_count)
trial.pprint(2)

trial2 = wordsr.map(lambda x:(x['Station code'],x['Address'], x['Latitude'], x['Longitude'], (x['SO2'] + x['NO2'] +  x['O3'] + x['CO']/4), x['PM10'], x['PM2.5'] ))

trial2.pprint(2)

def map_to_kvp(row):
    if len(row) < 3:
        return row
    return (row[0], tuple(row[1:]))

#ex = trial2.map(map_to_kvp)
#ex.pprint(2)

#trial3 = trial.map(map_to_kvp).join(trial2.map(map_to_kvp))
#trial3 = trial.join(trial2)


#print(trial3)
#trial3.pprint(2)

trial2.foreachRDD(process_rdd2)
trial.foreachRDD(process_rdd)
#trial3.foreachRDD(process_rdd)   The join transformation would not work with the keys


# In[ ]:


# start the streaming computation 
ssc.start()
ssc.awaitTermination()

