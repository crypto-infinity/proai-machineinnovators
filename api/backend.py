
"""
FastAPI endpoint for MachineInnovator.
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from model.download_dataset import (
    download_and_extract_sentiment140,
    load_sentiment140_as_pandas_dataset,
)

from api.inference import Inference
import logging


# Load env variables
load_dotenv()

DATA_DIR = "../data/sentiment140_data"

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

class InferenceInput(BaseModel):
    input_string: str
    model: str


class InferenceOutput(BaseModel):
    label: str
    score: float


class PerformanceMetricsOutput(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float

# class BatchInferenceInput(InferenceInput):
#     dataset: str #huggingface format user/dataset
#     split: str = "train"

# class BatchInferenceOutput(BaseModel):
#     predictions: dict
#     metrics: dict


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


# Grafana Metrics

accuracy_gauge = Gauge('model_accuracy', 'Accuracy')
precision_gauge = Gauge('model_precision', 'Precision')
recall_gauge = Gauge('model_recall', 'Recall')
f1_gauge = Gauge('model_f1', 'F1-score')


@app.post("/performance", response_model=PerformanceMetricsOutput)
def performance():

    download_and_extract_sentiment140()
    dataset = load_sentiment140_as_pandas_dataset()['test']

    infer = Inference(model=os.getenv("HF_REPO"))

    y_true = np.array(dataset['label'])
    y_pred = np.array(
        infer.single_inference(text)[0] for text in dataset["text"]
    )

    # Calcola le metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    accuracy_gauge.set(float(accuracy))
    precision_gauge.set(float(precision))
    recall_gauge.set(float(recall))
    f1_gauge.set(float(f1))

    return PerformanceMetricsOutput(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
