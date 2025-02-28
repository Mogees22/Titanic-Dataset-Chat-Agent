from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pydantic import BaseModel
import os
import requests
import streamlit as st
from dotenv import load_dotenv

app = FastAPI()

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Load environment variables
load_dotenv()

# Retrieve API key from Streamlit secrets or fallback to environment variable
api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY", os.getenv("GOOGLE_GEMINI_API_KEY"))

if not api_key:
    st.error("Google Gemini API key is missing. Please set it in Streamlit Secrets or environment variables.")
else:
    st.success("API Key Loaded Successfully!")

# Configure Gemini model with a correct model name
try:
    model_name = "gemini-1.5-pro"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model Initialization Error: {str(e)}")

# Define request model
class QueryRequest(BaseModel):
    question: str

def is_dataset_query(question: str):
    dataset_keywords = ["average fare", "percentage of passengers", "histogram", "survival rate", "distribution"]
    return any(keyword in question.lower() for keyword in dataset_keywords)

def process_dataset_query(question: str):
    if "average fare" in question.lower():
        return f"The average fare was ${df['Fare'].mean():.2f}."
    elif "percentage of passengers" in question.lower() and "male" in question.lower():
        male_percentage = (df[df['Sex'] == 'male'].shape[0] / df.shape[0]) * 100
        return f"The percentage of male passengers was {male_percentage:.2f}%."
    elif "histogram" in question.lower() and "age" in question.lower():
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Age'].dropna(), bins=20, kde=True)
        plt.xlabel("Age")
        plt.ylabel("Number of Passengers")
        plt.title("Age Distribution of Titanic Passengers")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"image": img_base64}
    
    return "Query not recognized."

def process_query(question):
    try:
        if is_dataset_query(question):
            return process_dataset_query(question)
        
        url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        data = {"contents": [{"parts": [{"text": question}]}]}

        response = requests.post(url, headers=headers, params=params, json=data)
        response_json = response.json()

        # Debugging log
        print(f"Raw API Response: {response_json}")

        # Handle errors in API response
        if "error" in response_json:
            error_message = response_json["error"].get("message", "Unknown API error.")
            raise HTTPException(status_code=500, detail=f"API Error: {error_message}")

        # Extract and return generated text safely
        if "candidates" in response_json and response_json["candidates"]:
            parts = response_json["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "No valid response found.")
        
        return "No valid response from API."
    
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error: {str(req_err)}")
        raise HTTPException(status_code=500, detail=f"API Request Error: {str(req_err)}")
    
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/query")
def query(request: QueryRequest):
    try:
        print(f"Processing query: {request.question}")  # Debugging log
        response = process_query(request.question)
        print(f"Response from LLM: {response}")  # Debugging log
        return {"answer": response}
    except HTTPException as http_err:
        print(f"HTTP Error: {str(http_err)}")
        raise http_err  # Raise HTTP exception with correct status code
    except Exception as e:
        print(f"Unexpected Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
