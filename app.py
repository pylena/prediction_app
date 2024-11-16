import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Football Player Clustering")

# Input fields
current_value = st.number_input("Enter Current Value:", min_value=0)
goals = st.number_input("Enter Number of Goals:", min_value=0)
appearance = st.number_input("Enter Number of appearance:", min_value=0)
position_numeric = st.number_input("Enter the position:", min_value=0)

if st.button("Get Prediction"):
    # API call
    response = requests.post("https://prediction-app-v1.onrender.com/predict", json={
        "current_value": current_value,
        "goals": goals,
        "appearance": appearance,
        "position_numeric": position_numeric,
        
    })
    cluster = response.json().get("cluster")
    st.write(f"The player belongs to Cluster {cluster}")

    # Load dataset for scatterplot
    data = pd.read_csv('./final_data.csv')  
    sns.scatterplot(data=data, x='current_value', y='goals', hue='cluster')
    plt.axhline(y=goals, color='gray', linestyle='--', label='Input Goals')
    plt.axvline(x=current_value, color='gray', linestyle='--', label='Input Current Value')
    st.pyplot(plt)
