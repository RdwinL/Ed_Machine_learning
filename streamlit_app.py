import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Machine Learning App')

st.info('This build a machine learning model')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/RdwinL/Ed_Machine_learning/refs/heads/master/crop_yield_data.csv')
  df
  
with st.expander('X_Data'):
  st.write('**X**')
  X =df.drop('crop_yield', axis=1)
  X
  
with st.expander('y_data'):
  st.write('**y**')
  y = df["crop_yield"]
  y
st.subheader("Initial Data Analysis")

st.write('Target distribution')
fig, ax = plt.subplots()
sns.histplot(df["crop_yield"], kde=True, ax=ax)
st.pyplot(fig)

st.write('Correlation heatmap')
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
