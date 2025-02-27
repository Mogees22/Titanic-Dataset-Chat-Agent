import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Streamlit UI Setup
st.title("Titanic Chat Agent 🚢")
st.write("Ask a question about the Titanic dataset!")

# Load dataset
df = pd.read_csv("titanic.csv")

# User input
question = st.text_input("Enter your question:")

if st.button("Ask"): 
    if question:
        response = requests.post("https://titanic-dataset-chat-agent-2ad3.onrender.com", json={"question": question})
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.write("**Answer:**", answer)
            
            # Example: Show histogram if user asks for age distribution
            if "histogram of passenger ages" in question.lower():
                plt.figure(figsize=(8,5))
                plt.hist(df['Age'].dropna(), bins=20, edgecolor='black')
                plt.xlabel("Age")
                plt.ylabel("Number of Passengers")
                plt.title("Age Distribution of Titanic Passengers")
                st.pyplot(plt)
            
            # Show bar chart for embarked passengers
            elif "how many passengers embarked from each port" in question.lower():
                plt.figure(figsize=(8,5))
                sns.countplot(x=df['Embarked'], palette='viridis')
                plt.xlabel("Port of Embarkation")
                plt.ylabel("Passenger Count")
                plt.title("Number of Passengers by Embarkation Port")
                st.pyplot(plt)
            
            # Show pie chart for gender distribution
            elif "percentage of passengers were male" in question.lower():
                gender_counts = df['Sex'].value_counts()
                plt.figure(figsize=(6,6))
                plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
                plt.title("Male vs Female Passenger Distribution")
                st.pyplot(plt)
            
            # Show boxplot for fare distribution
            elif "average ticket fare" in question.lower():
                plt.figure(figsize=(8,5))
                sns.boxplot(x=df['Fare'], color='blue')
                plt.xlabel("Ticket Fare")
                plt.title("Distribution of Ticket Fares")
                st.pyplot(plt)
        else:
            st.write("Error fetching response from backend.")
