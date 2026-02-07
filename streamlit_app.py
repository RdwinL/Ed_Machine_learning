import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('This build a machine learning model')

with st.expander('Data')
 st.write(**Raw data**)
 df = pd.read_csv('https://raw.githubusercontent.com/RdwinL/Ed_Machine_learning/refs/heads/master/crop_yield_data.csv')
df
