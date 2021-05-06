#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np



# In[2]:


#import socket
import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages/')
import requests
import requests_oauthlib 
import json
# import kafka
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka import KafkaClient
import time
import random

# In[3]:


data = pd.read_csv("Measurement_summary.csv", index_col=0, header=0)
data
data101 = data.loc[data["Station code"]==101,:]
data101.reset_index()

data = data.reset_index().sort_values("Measurement date")
# In[4]:


# data101.iloc[1,2:].to_json()



# In[6]:


# Send data to a Kafka Topic
#producer = KafkaProducer(bootstrap_servers='localhost:9092') #Same port as the Kafka server
topic_name = 'testClusterTopic'
#producer.send(topic_name, str.encode(resp, errors="ignore"))

def injest_data(dataset):

    #serialize dict to string via json and encode to bytes via utf-8
    p = KafkaProducer(bootstrap_servers='data3:6667',api_version=(0,10,1), acks='all', linger_ms=5)# ,value_serializer=lambda m: json.dumps(m).encode('utf-8'))#, batch_size=1024)
        
    for i,row in dataset.iterrows():
        try:
            rows = dataset.loc[i,:].to_json()
            print(rows)
   	    print( dataset.loc[i,:])
            print ("------------------------------------------") 
            p.send(topic_name, value=rows) #line
            time_value = random.uniform(0.5, 2.5)
            time.sleep(time_value)
        except:
            e = sys.exc_info()[0]
            print("Error: %s" % e)
    #p.flush(100) #The Python client provides a flush() method which can be used to make writes synchronous. This is typically a bad idea since it effectively limits throughput to the broker round trip time, but may be justified in some cases.
    #p.close()




# In[ ]:


injest_data(data)


# In[ ]:
