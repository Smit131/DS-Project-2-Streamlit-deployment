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
model = pickle.load(open('model.pkl','rb'))

with open('matrix.pkl', 'rb') as f:
    emails_bow1 = pickle.load(f)

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



# Taking input

f = [msg]
fpd = pd.DataFrame(f,columns = ['CleanContent'])

# For input message
f_matrix = emails_bow1.transform(fpd.CleanContent)

f_pred = model.predict(f_matrix)

answer = f_pred[0]

st.subheader('Predicted Result')
st.write(answer)
