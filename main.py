
import logging
from pathlib import Path

import joblib
import pandas as pd

from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().with_name("Mental_Health_Model.pkl")
model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(title="Mental Health Predictor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define input data validation
class StudentData(BaseModel):
    Age: int = Field(..., gt=0, description="Age of the student")

    Gender: Literal["Male", "Female", "Other"]
    Country: str

    Academic_Level: Literal[
        "Undergraduate", "Graduate", "High School"
    ]

    Most_Used_Platform: Literal[
        "Instagram", "TikTok", "Facebook", "LinkedIn",
        "YouTube", "Twitter", "Snapchat", "WhatsApp",
        "LINE", "VKontakte", "WeChat", "KakaoTalk"
    ]

    Purpose_Of_Use: Literal[
        "Networking", "Education", "Entertainment", "News"
    ]

    Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24)
    Daily_Unlocks: int = Field(..., ge=0)
    Study_Hours: float = Field(..., ge=0, le=24)
    Physical_Activity_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)

    Stress_Level: Literal["Low", "Medium", "High", "Very High"]


# Define response format
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# List of supported countries
top_countries = [
    "Other", "India", "USA", "Canada", "Australia",
    "UK", "Germany", "Mexico", "Turkey", "France"
]


# Home endpoint
@app.get("/")
def greet():
    return {"message": "Welcome to Mental Health Predictor"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    # Group less common countries
    country_group = (
        data.Country if data.Country in top_countries else "Other"
    )

    # Prepare input data for the model
    input_row = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender,
        "Academic_Level": data.Academic_Level,
        "Most_Used_Platform": data.Most_Used_Platform,
        "Purpose_Of_Use": data.Purpose_Of_Use,
        "Avg_Daily_Usage_Hours": data.Avg_Daily_Usage_Hours,
        "Daily_Unlocks": data.Daily_Unlocks,
        "Study_Hours": data.Study_Hours,
        "Physical_Activity_Hours": data.Physical_Activity_Hours,
        "Sleep_Hours_Per_Night": data.Sleep_Hours_Per_Night,
        "Stress_Level": data.Stress_Level,
        "Grouped_country": country_group
    }])

    # Generate prediction
    try:
        prediction = model.predict(input_row)[0]
    except Exception as exc:
        logger.exception("Prediction failed in the trained model pipeline")
        raise HTTPException(status_code=500, detail="The prediction model could not process this submission.") from exc

    # Return the predicted score
    return PredictionResponse(
        predicted_mental_health_score=round(float(prediction), 2)
    )
