
"""
FastAPI endpoint for MachineInnovator.
"""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from train import ModelTrainer
import logging


# Load env variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# FastAPI Setup
app = FastAPI(title="MachineInnovator API Backend")


# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data structures setup

class TrainData(BaseModel):
    pass


class TrainResult(BaseModel):
    pass


class AnalysisRequest(BaseModel):
    input_string: str


class AnalysisResponse(BaseModel):
    label: str
    score: float


@app.post("/inference", response_model=AnalysisResponse)
async def inference(request: AnalysisRequest):
    """
    """

    try:
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        classification = pipeline(
            "text-classification",
            model=MODEL
        )

        result = classification(request.input_string)[0]

        return AnalysisResponse(
            label=result['label'],
            score=result['score']
        )

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(request: TrainData):
    """
    """

    try:
        trainer = ModelTrainer()
        trainer.retrain()
        return TrainResult()

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns API status and version.
    """
    return {"status": "ok"}


@app.get("/")
def main_page():
    """
    Redirects root endpoint to health check.
    """
    return RedirectResponse(url="/health", status_code=301)
