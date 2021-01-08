import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from collections import Counter
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


st.title('NLP Project deployment')


st.markdown('Inappropriate emails would demotivates and spoil the positive environment that would lead to more attrition rate and low productivity and Inappropriate emails could be on form of bullying, racism, sexual favourtism and hate in the gender or culture, in today’s world so dominated by email no organization is immune to these hate emails.The goal of the project is to identify such emails in the given day based on the above inappropriate content.')

st.sidebar.header('Group 1 ,Project grp P38')
st.sidebar.markdown('Group members')
st.sidebar.markdown('Vikas Bevoor')
st.sidebar.markdown('Sumit Yenugwar')
st.sidebar.markdown('Chandramohan')
st.sidebar.markdown('Harsh Joshi')


st.title('Model Deployment: Logistic Regression')

msg = st.text_input("Please paste the mail content here to check if its Abusive or Not", " ")

df = pd.read_csv("cleandata2.csv")
df.reset_index(inplace= True)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def split_into_words(i):
    return [word for word in i.split(" ")]

# Preparing email texts into word count matrix format 
emails_bow = CountVectorizer(analyzer=split_into_words).fit(df.CleanContent)

# For all messages
all_emails_matrix = emails_bow.transform(df.CleanContent)

X = all_emails_matrix
Y = df.Class

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=500,random_state=0)
classifier.fit(X,Y)

# Taking input

f = [msg]
fpd = pd.DataFrame(f,columns = ['CleanContent'])

# For input message
f_matrix = emails_bow.transform(fpd.CleanContent)

f_pred = classifier.predict(f_matrix)

answer = f_pred[0]

st.subheader('Predicted Result')
st.write(answer)
