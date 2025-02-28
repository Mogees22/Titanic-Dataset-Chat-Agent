from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import pandas as pd
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Retrieve API key from environment variable
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

if not api_key:
    raise ValueError("Google Gemini API key is missing. Set it as an environment variable.")

# Configure Gemini model with a correct model name
try:
    model_name = "gemini-1.5-pro"  # Change if needed after listing available models
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model Initialization Error: {str(e)}")

# Define request model
class QueryRequest(BaseModel):
    question: str

def is_dataset_query(question):
    keywords = ["average ticket fare", "survival rate", "total passengers", "fare", "age", "class"]
    return any(keyword in question.lower() for keyword in keywords)

def process_query(question):
    try:
        # Handle dataset-related questions
        if is_dataset_query(question):
            question_lower = question.lower()
            
            if "average ticket fare" in question_lower:
                avg_fare = df["Fare"].mean()
                return f"The average ticket fare was ${avg_fare:.2f}."
            
            elif "survival rate" in question_lower:
                survival_rate = df["Survived"].mean() * 100
                return f"The survival rate was {survival_rate:.2f}%."
            
            elif "total passengers" in question_lower:
                total_passengers = len(df)
                return f"The total number of passengers was {total_passengers}."
            
            elif "percentage of passengers were male" in question_lower or "male percentage" in question_lower:
                male_count = df[df["Sex"] == "male"].shape[0]
                total_count = df.shape[0]
                male_percentage = (male_count / total_count) * 100
                return f"The percentage of male passengers on the Titanic was {male_percentage:.2f}%."
            
            else:
                return "I couldn't find relevant data in the dataset."

        # Otherwise, use Gemini for general questions
        url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        data = {"contents": [{"parts": [{"text": question}]}]}  

        response = requests.post(url, headers=headers, params=params, json=data)
        response_json = response.json()

        # Debugging log
        print(f"Raw API Response: {response_json}")

        # Handle API errors
        if "error" in response_json:
            error_message = response_json["error"].get("message", "Unknown API error.")
            raise HTTPException(status_code=500, detail=f"API Error: {error_message}")

        # Extract generated text
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



