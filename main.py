"""
FastAPI Server Entrypoint
Demonstrates modern AI engineering serving integrations specifically requested in the Tredence JD (Async/FastAPI/Deployment).
"""

import logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import SelfPruningNet
from src.config import config

logger = logging.getLogger("TredenceServer")

app = FastAPI(
    title="Self-Pruning Network API",
    description="Web inference layer for the dynamic prunable PyTorch mechanism developed for the Tredence case study.",
    version="1.0.0"
)

# Initialize global architecture (In reality, we load the trained .pth state_dict here)
ml_backend = SelfPruningNet()
ml_backend.eval()

class InferencePayload(BaseModel):
    image_tensor: list[float]

@app.get("/health")
async def health_check():
    """Confirms traffic functionality."""
    return {"status": "up", "framework": "PyTorch + FastAPI"}

@app.post("/api/v1/predict")
async def execute_prediction(payload: InferencePayload):
    """Executes a single structural forward pass asynchronously."""
    if len(payload.image_tensor) != config.input_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Mismatched context window: Expected inputs length of {config.input_size}"
        )
        
    tensor_input = torch.tensor(payload.image_tensor, dtype=torch.float32).unsqueeze(0)
    
    try:
        with torch.no_grad():
            raw_logits = ml_backend(tensor_input)
            prob_matrix = torch.softmax(raw_logits, dim=1)
            prediction = torch.argmax(prob_matrix, dim=1).item()
            
        return {
            "prediction_class": prediction,
            "confidence_scores": prob_matrix.tolist()[0]
        }
    except Exception as e:
        logger.error(f"Inference failure: {e}")
        raise HTTPException(status_code=500, detail="Matrix transform error!")
