
"""
FastAPI endpoint for MachineInnovator.
"""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import Inference
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
    model_name: str
    dataset: str

class TrainResult(BaseModel):
    metrics: dict

class InferenceInput(BaseModel):
    input_string: str
    model: str

class InferenceOutput(BaseModel):
    label: str
    score: float

class BatchInferenceInput(InferenceInput):
    dataset: str #huggingface format user/dataset
    split: str = "train"

class BatchInferenceOutput(BaseModel):
    predictions: dict
    metrics: dict


@app.post("/inference", response_model=InferenceOutput)
def inference(request: InferenceInput):
    """
    Single string inference for sentiment analysis.
    """

    try:
        infer = Inference(model=request.model)
        label, score = infer.single_inference(request.input_string)

        return InferenceOutput(label=label, score=score)
        
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_inference", response_model=BatchInferenceOutput)
def batch_inference(request: BatchInferenceInput):
    """
    Batch inference for sentiment analysis.
    """

    try:
        infer = Inference(model=request.model)
        result = infer.batch_inference(request.dataset, request.split)

        return BatchInferenceOutput(predictions=result['results'],
                                    metrics=result['metrics']
                                    )

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
