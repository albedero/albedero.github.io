#!/usr/bin/env python
# coding: utf-8


#Import libraries
#pip install wordcloud

#!pip install plotly

from __future__ import division
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
import altair as alt


import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
import altair as alt
from collections import defaultdict
from pathlib import Path
import glob
import os
import json
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

import time
from math import log
from nltk.stem import *


############################### CHANGE FILE PATH HERE ######################################
file_path = r'C:\Users\34603\IE Students\Las Bombas - Documents\NLP\NLP_Final_Submission_TeamB'


############################### Read File ##################################################
path = os.path.join(file_path, "8_code_output.csv")
dataset = pd.read_csv(path,index_col = 0)

# Set the wide layout for the dashboard
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

logo = Image.open(os.path.join(file_path, "3_logo.jpeg"))

# Home page:
def pageHome():
    st.title('JOBIFY')
    st.subheader('Welcome to JOBIFY, your one-stop shop to help you find out everything you need to know about a company.')

    about = Image.open(os.path.join(file_path, "4_home_about.jpg"))
    companies = Image.open(os.path.join(file_path, "5_companies.jpg"))
    st.markdown("<h1 style='text-align: center; class = 'big-font'; color: Black; '>Cutting Edge Possibilities</h1>", unsafe_allow_html=True)
    #st.header('Cutting Edge Possibilities')
    st.image(about, use_column_width= True)
    st.markdown("<h1 style='text-align: center; class = 'big-font'; color: Black; '>Unified Platform</h1>", unsafe_allow_html=True)

    #st.header('Unified Platform')
    st.image(companies, use_column_width= True)

#About page:
def pageAbout():
    st.title('ABOUT US')
    st.write('Job-seeking people have expressed their need for a unified platform that stores company profiles of employers. Our application efficiently condenses information from various sources, allowing us to translate textual reviews into a quantifiable score that can easily be summarized. We have applied state of the art NLP techniques to gain maximum insight. Using Sentiment Analysis and Information Retrieval in combination with real-time data streaming, we bring you the most accurate and up-to-date overview of any company of your choice. Our goal is to facilitate your personal job research, making it more time-optimal and accurate.')
    st.title('Meet the Team')
    st.write ('Our team of experts come from a wide range of backgrounds specialising in Micheladas and Money Making.')
    team1 = Image.open(os.path.join(file_path, "6_team1.jpeg"))
    st.image(team1,use_column_width=True)
    team2 = Image.open(os.path.join(file_path, "7_team2.jpeg"))
    st.image(team2,use_column_width=True)


# Dashboard page:
def pageDashboard():
    st.title('OVERALL DASHBOARD')
   
    industry = list(dataset['Industry'].unique())
    #industry.append('Select all')
    industry_choice = st.sidebar.selectbox('Select your industry:', industry)
    #country = list(dataset["Country"].loc[dataset["Industry"] == industry_choice].unique())
    country = list(dataset['Country'].unique())
    #country.append('Select all')
    country_choice = st.sidebar.selectbox('Select your country:', country)
    state = list(dataset['State_Name'].loc[dataset['Country'] == country_choice].unique())
    #state.append('Select all')
    state_choice = st.sidebar.selectbox('Select your state:', state)
    city = list(dataset['City'].loc[dataset['State_Name'] == state_choice].unique())
    #city.append('Select all')
    city_choice = st.sidebar.selectbox('Select your city:', city)
    dataset_withfilter = dataset.loc[(dataset['Industry']==industry_choice) & (dataset['Country']==country_choice) & (dataset['State_Name']==state_choice) & (dataset['City']==city_choice)]
    st.subheader('Top 10 highest rated companies')
    #st.write('\n')
    company_rank = dataset_withfilter.groupby('Company').agg(Reviews=('review_title', 'size'), Score=('overall_score', 'mean')).sort_values(by='Score',ascending=False)
    st.table(company_rank)

    # Get the date from the user
    df_companies = dataset_withfilter
    df_companies["date"] = df_companies["date"].astype(str).str[0:24]
    df_companies["date"] = pd.to_datetime(df_companies["date"]).apply(lambda x:x.date())


    st.subheader('Distribution of Scores:')
    col1, col2 = st.beta_columns([1,2])

    with col1:
        
        start_date = st.date_input('Start date', df_companies.date.min())
        end_date = st.date_input('End date', df_companies.date.max())

        if (start_date < end_date):
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
        else:
            st.error('Error: End date must fall after start date.')

        # Adjust the slider based on the dates selected
        values = st.slider("Score Range", float(df_companies[(start_date <= df_companies.date) & (end_date <= end_date)].overall_score.min()),float(0.5), (float(df_companies.overall_score.min()), float(df_companies.overall_score.max())))
         
        
    with col2:
        # Draw the histogram of the selected time frame and the selected company                        
        f = px.histogram(df_companies[(values[0] <= df_companies.overall_score) & (df_companies.overall_score <= values[1]) & (start_date <= df_companies.date) & (start_date <= end_date)], 
                    x='overall_score', nbins=15, title='Distribution of Overall Score',
                    template="simple_white", color_discrete_sequence = ["cornflowerblue"])

        f.update_xaxes(title="Overall Score")
        f.update_yaxes(title="Number of Reviews")
        st.plotly_chart(f)

# #################Drilldown Page:#######################################
def pageDrilldown():
    st.title('Company Drilldown')

    company = list(dataset['Company'].unique())
    company_choice = st.sidebar.selectbox('Select your company:', company)
    dataset_filtered = dataset.loc[dataset['Company']==company_choice]
    dataset_filtered["date"] = dataset_filtered["date"].astype(str).str[0:24]
    dataset_filtered["date"] = pd.to_datetime(dataset_filtered["date"]).apply(lambda x:x.date())

    ####################################### Distribution of scores ############################3
    st.header("Distribution of Scores")
    #Columns for layout:
    h1_s, h1_e  = st.beta_columns([1,2])

    with h1_s:
        start_date = st.date_input('Start date', dataset_filtered.date.min())
        end_date = st.date_input('End date', dataset_filtered.date.max())

        if (start_date < end_date):
            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
        else:
            st.error('Error: End date must fall after start date.')
    
        values = st.slider("Score Range", float(dataset_filtered[(start_date <= dataset_filtered.date) & (end_date <= end_date)].overall_score.min()),float(0.5), (float(dataset_filtered.overall_score.min()), float(dataset_filtered.overall_score.max())))

    with h1_e: 
        with st.beta_container():                   
            f = px.histogram(dataset_filtered[(values[0] <= dataset_filtered.overall_score) & (dataset_filtered.overall_score <= values[1]) & (start_date <= dataset_filtered.date) & (start_date <= end_date)], 
                        x='overall_score', nbins=15, title='Distribution of Overall Score',
                        template="simple_white", color_discrete_sequence = ["cornflowerblue"])

            f.update_xaxes(title="Overall Score")
            f.update_yaxes(title="Number of Reviews")
            st.plotly_chart(f)
        
    ############################ Distribution of Reviews #####################################
    # Distribution by country, date and employment types
    st.header("Distribution of Reviews")
    display_choice = st.selectbox('Select your analysis level:', list(['Country', 'Employment Type']))
    h2_s, h2_e = st.beta_columns([1,2])
         
    if display_choice == 'Country':
        # Display and get the country choice
        with h2_s:
            country_choice = st.multiselect("Choose countries that company operates in",
                                            list(dataset_filtered['Country'].unique()) + ['ALL'], 
                                            default = 'ALL')
            if "ALL" in country_choice:
                df_temp = dataset_filtered
            else:
                df_temp = dataset_filtered.reset_index(drop=True).set_index('Country')[
                    dataset_filtered.reset_index(drop=True).set_index('Country').index.isin(country_choice)].reset_index()     
            #Get the date from the user
            start_date_2 = st.date_input('Start date', df_temp.date.min(),key=3)
            end_date_2 = st.date_input('End date', df_temp.date.max(),key=4)

            if (start_date_2 < end_date_2):
                st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date_2, end_date_2))
            else:
                st.error('Error: End date must fall after start date.')
        
        with h2_e:
            # Draw the histogram of the selected time frame and the selected company                        
            figure = px.histogram(df_temp[(start_date_2 <= df_temp.date) & (df_temp.date <= end_date_2)], 
                        x='overall_score', nbins=15, title='Distribution of Reviews',
                        template="simple_white", color_discrete_sequence = ["cornflowerblue"])
            figure.update_xaxes(title="Overall Score")
            figure.update_yaxes(title="Number of Reviews")
            st.plotly_chart(figure)
        
    elif display_choice == 'Employment Type':
        with h2_s:
            status_choice = st.multiselect("Choose employment situation",
                                            list(dataset_filtered['status'].unique()), 
                                            default = list(dataset_filtered['status'].unique()))
            df_temp = dataset_filtered.reset_index(drop=True).set_index('status')[
                dataset_filtered.reset_index(drop=True).set_index('status').index.isin(status_choice)].reset_index()

            emp_type_choice = st.multiselect("Choose employment type",
                                            list(df_temp['emptype'].unique()), 
                                            default = list(df_temp['emptype'].unique()))
            df_temp = df_temp.set_index('status')[df_temp.set_index('emptype').index.isin(emp_type_choice)].reset_index()
    
            # Get the date from the user
            start_date_2 = st.date_input('Start date', df_temp.date.min(),key=5)
            end_date_2 = st.date_input('End date', df_temp.date.max(),key=6)

            if (start_date_2 < end_date_2):
                st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date_2, end_date_2))
            else:
                st.error('Error: End date must fall after start date.')
        
        with h2_e:
            # Draw the histogram of the selected time frame and the selected company                        
            figure = px.histogram(df_temp[(start_date_2 <= df_temp.date) & (df_temp.date <= end_date_2)], 
                                x='overall_score', nbins=15, title='Distribution of Reviews',
                                template="simple_white", color_discrete_sequence = ["cornflowerblue"])
            figure.update_xaxes(title="Overall Score")
            figure.update_yaxes(title="Number of Reviews")
            st.plotly_chart(figure)
        
    else:
        st.write("Please make a selection for further analysis")

    ####################### Word Cloud     ###################################
    st.header('Search for most common words used:')
    #st.write('Most common words')

    stars = list(dataset_filtered['Rank'].unique())
    sentiment = list(['Pros','Cons'])
    w1,w2 = st.beta_columns(2)

    with w1:
        star_choice = st.selectbox('Number of stars:', stars)
    
    with w2: 
        sentiment_choice = st.selectbox("Pro's or Con's", sentiment)

    
    def wc(data,bgcolor,title):
        plt.figure(figsize = (15,10))
        wc = WordCloud(background_color = bgcolor, max_words = 20, random_state=42, max_font_size = 50)
        wc.generate(' '.join(data))
        plt.imshow(wc)
        plt.axis('off')
        plt.show()
    if sentiment_choice == 'Pros':
        wc(dataset_filtered['pros'][dataset.Rank==star_choice],'black','Common Words')
        st.pyplot()
    elif sentiment_choice == 'Cons':
        wc(dataset_filtered['cons'][dataset.Rank==star_choice],'black','Common Words')
        st.pyplot()
    else:
        st.error("Please choose pros or cons")

    #####################################Information Retrieval ######################################
    st.header('Search for relevant reviews based on key words or sentences:')

    df = dataset_filtered.reset_index(drop=True)
    # User decides whether to choose pros or cons
    select_pro = st.sidebar.checkbox("Show pros", True, key=1)
    select_con = st.sidebar.checkbox("Show cons", False, key=2)
    # Based on user selection change the search space
    if (select_pro==True & select_con==True):
        #merge pros and cons and search on both
        df["both"] = "PROS: " + df["pros"] + " " + "CONS: " + df["cons"]
        columna = len(df.columns)-1
        selection = "Pros and Cons"
    elif select_con:
        columna = df.columns.get_loc('cons')
        selection = "Cons"
    else:
        #By default, if user does not specify, get pros
        columna = df.columns.get_loc('pros')
        selection = "Pros"         
    # Get the stopwords
    stopwords_english = set(stopwords.words('english'))
    # Define required functions going to be used
    def get_list_tokens_string(string):
        list_words = re.split(r'\W+',string)
        list_words = [w.lower() for w in list_words if w.isalpha() and len(w)>1 and w.lower() not in stopwords_english]  
        stemmer = PorterStemmer()
        list_words = [stemmer.stem(w) for w in list_words]
        return list_words
    def index_one_file(termlist):
        fileIndex = {}
        for index, word in enumerate(termlist):
            if word in fileIndex.keys():
                fileIndex[word].append(index)
            else:
                fileIndex[word] = [index]
        return fileIndex
    # Get the search type from the user
    select_status = st.sidebar.radio("Search Type", ('Exact Match','All Relevant'))
    # Get the search query from the user
    st.subheader('Input your search query')
    user_query_input = st.text_input("", "salary") 

    # We need to come up with something for the default search 
    # query else it is not working -- DUNNO HOW TO SOLVE THIS!

    df_companies = df

    # ################################## Perform Information Retrieval ###############################################
    if select_status == "Exact Match":
        # Algorithm for the exact match
        start_time = time.time()    
        list_document_ids =  df_companies.index #get_list_fileids(corpus)   
        num_docs = len(list_document_ids)    

        ##1) Create a inverted index
        inverted_index = {}

        ## 2)Loop through the dataset, to get the entire text from  each file
        for document_index in list_document_ids:
            list_words = get_list_tokens_string(df_companies.iloc[document_index,columna])
            list_words_pos = index_one_file(list_words)

            ## 3) Update the dictionary with the words in the document and the related document_id
            for w in set(list_words):
                if w in inverted_index.keys():
                    if document_index in inverted_index[w].keys():
                        inverted_index[w][document_index].extend(list_words_pos[w][:])
                    else:
                        inverted_index[w][document_index] = list_words_pos[w]
                else:
                    inverted_index[w] = {document_index: list_words_pos[w]}

        ##5) Getting the query from the user
        query = user_query_input
        result_file_indices= []
        result = []

        start_time = time.time()

        ##6) Tokenizing query string to get individual words
        query_list = get_list_tokens_string(query)

        ##7) Get the documents including the query terms
        for q in query_list:
            if q in inverted_index.keys():
                result_file_indices.append([filename for filename in inverted_index[q.lower()]])

        setted = set(result_file_indices[0]).intersection(*result_file_indices)

        for filename in setted:
            temp = []
            for word in query_list:
                temp.append(inverted_index[word][filename][:])
            for i in range(len(temp)):
                for ind in range(len(temp[i])):
                    temp[i][ind] -= i
            if set(temp[0]).intersection(*temp):
                result.append(filename)

        ##8) Sorting the dictionary based on its values
        if(len(result)==0):
            st.write("Sorry No matches found")
        else:
            st.write("Number of search results : " , len(result))
            st.subheader("Results:")
            st.write("\n")
            column_taken = df_companies.columns[columna]
            display = pd.DataFrame(df_companies[['Source','emptype','date','Company', 
                                'Country','City','review_title',
                                column_taken]].iloc[result])
            
            display = display.rename(columns={'date': 'Date',
                                   'review_title': 'Review Title',
                                   column_taken: 'Reviews',
                                   'emptype': 'Employment Type',
                                  })
           
            st.table(display)

    #        for index in result:
    #            st.write(df_companies.iloc[index,columna])
    #            st.write("\n")

    elif select_status == "All Relevant":
        # Algorithm for TF-IDF
        start_time = time.time()    
        list_document_ids =  df_companies.index #get_list_fileids(corpus)   
        num_docs = len(list_document_ids)      

        ##1) Create a inverted index
        inverted_index = {}

        ##2)Loop through the df_companies, to get the entire text from  each file
        for document_index in list_document_ids:

            ##3) Parse the string to get individual words    
            list_words = get_list_tokens_string(df_companies.iloc[document_index,columna])

            ##4) Update the dictionary to use tf-idf 
            for w in list_words:
                if inverted_index.get(w,0)==0:
                    inverted_index[w]={}
                    inverted_index[w][document_index]=1
                else:
                    inverted_index[w][document_index] = inverted_index[w].get(document_index,0)
                    inverted_index[w][document_index] += 1
            for w in set(list_words):
                inverted_index[w][document_index]=1+log(inverted_index[w][document_index])

        ##5) Getting the query from the user
        query = user_query_input
        result_file_dict={}

        start_time = time.time()

        ##6) Tokenizing query string to get individual words
        query_list = get_list_tokens_string(query)

        ##7) Calculating tf-idf scores 
        for q in query_list:
            d = inverted_index.get(q,0) 
            if d!=0:
                length=len(d)
                for file_index in d.keys():
                    result_file_dict[file_index] = result_file_dict.get(file_index,0)
                    result_file_dict[file_index]+=((1+log(d[file_index]))*(log(num_docs/length)/log(10)))
                                                #1st term is tf # 2nd term is idf

        ##8) Sorting the dictionary based on its values            
        result_file_indices = sorted(result_file_dict.items(), key=lambda x:x[1],reverse = True)

        if(len(result_file_indices)==0):
            st.write("Sorry No matches found")
        else:
            st.write("Number of search results : " , len(result_file_indices))
            st.write("\n")
            st.write("Results:")
            st.write("\n")
            if len(result_file_indices) > 10:
                 st.write("Returning 1st 10 results")
            indexing = []
            for (index,tup) in enumerate(result_file_indices):
                indexing.append(tup[0])
                if index==10:
                    break
               # st.write(df_companies.iloc[tup[0],columna])
            st.write("\n")
            column_taken = df_companies.columns[columna]
            display = pd.DataFrame(df_companies[['Source','emptype','date','Company', 
                                'Country','City','review_title',
                                column_taken]].iloc[indexing])

            display = display.rename(columns={'date': 'Date',
                                   'review_title': 'Review Title',
                                   column_taken: 'Reviews',
                                   'emptype': 'Employment Type',
                                  })
            st.table(display)

# ##############################################Overall website format ################################################
PAGES = {"Home": pageHome, "About Us": pageAbout, "Dashboard": pageDashboard, "Drilldown": pageDrilldown}

st.sidebar.image(logo, use_column_width= True )
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()

# Finally done