import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Machine Learning App')

st.info('Machine learning model For Crop Yield Prediction')

with st.expander('Raw Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/RdwinL/Ed_Machine_learning/refs/heads/master/crop_yield_data.csv')
  df
  
st.header("Data Overview")
# Expander
with st.expander("View Dataset Information", expanded=False):

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.markdown("---")

    st.subheader("First 5 Rows (Head)")
    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("Last 5 Rows (Tail)")
    st.dataframe(df.tail())

    st.markdown("---")

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.markdown("---")

    st.subheader("Missing Values Count")
    missing = df.isnull().sum()
    st.dataframe(missing)

    st.markdown("---")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
  
with st.expander('X_Data'):
  st.write('**X**')
  X =df.drop('crop_yield', axis=1)
  X


with st.expander('y_data'):
  st.write('**y**')
  y = df["crop_yield"]
  y
st.header("Initial Data Visualization")
st.subheader("Variable Distributions")

with st.expander("View Histograms for All Variables"):

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        st.subheader(f"Histogram: {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
      
with st.expander("Correlation Heatmap"):

    st.subheader("Correlation Matrix")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
