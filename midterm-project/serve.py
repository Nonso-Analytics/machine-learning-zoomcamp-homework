# serve.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import uvicorn
from predict import WaterPotabilityPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability based on quality parameters",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loaded once at startup)
predictor = None


@app.on_event("startup")
async def load_model():
    """Load the model when the application starts"""
    global predictor
    try:
        predictor = WaterPotabilityPredictor('model.bin')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


# Pydantic models for request/response validation
class WaterQualityInput(BaseModel):
    """Input schema for water quality parameters"""
    ph: float = Field(..., description="pH value of water", ge=0, le=14)
    Hardness: float = Field(..., description="Water hardness in mg/L", ge=0)
    Solids: float = Field(..., description="Total dissolved solids in ppm", ge=0)
    Chloramines: float = Field(..., description="Chloramines in ppm", ge=0)
    Sulfate: float = Field(..., description="Sulfate in mg/L", ge=0)
    Conductivity: float = Field(..., description="Electrical conductivity in μS/cm", ge=0)
    Organic_carbon: float = Field(..., description="Organic carbon in ppm", ge=0)
    Trihalomethanes: float = Field(..., description="Trihalomethanes in μg/L", ge=0)
    Turbidity: float = Field(..., description="Turbidity in NTU", ge=0)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "ph": 7.5,
                "Hardness": 180.0,
                "Solids": 20000.0,
                "Chloramines": 7.5,
                "Sulfate": 350.0,
                "Conductivity": 400.0,
                "Organic_carbon": 15.0,
                "Trihalomethanes": 70.0,
                "Turbidity": 4.0
            }
        }
    }    


class BatchWaterQualityInput(BaseModel):
    """Input schema for batch predictions"""
    samples: List[WaterQualityInput] = Field(..., description="List of water quality samples")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "samples": [
                    {
                        "ph": 7.5,
                        "Hardness": 180.0,
                        "Solids": 20000.0,
                        "Chloramines": 7.5,
                        "Sulfate": 350.0,
                        "Conductivity": 400.0,
                        "Organic_carbon": 15.0,
                        "Trihalomethanes": 70.0,
                        "Turbidity": 4.0
                    }
                ]
            }
        }
    }

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int = Field(..., description="Predicted class (0=Not Potable, 1=Potable)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    potability_probability: float = Field(..., description="Probability of water being potable")
    safety_status: str = Field(..., description="Safety status message")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: float = Field(..., description="Model confidence (0-1)")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[Dict] = Field(..., description="List of prediction results")
    total_samples: int = Field(..., description="Total number of samples processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    feature_names: Optional[List[str]]


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Water Potability Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None,
        "feature_names": predictor.feature_names if predictor else None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: WaterQualityInput):
    """
    Predict water potability for a single sample
    
    - **ph**: pH value (0-14)
    - **Hardness**: Water hardness in mg/L
    - **Solids**: Total dissolved solids in ppm
    - **Chloramines**: Chloramines concentration in ppm
    - **Sulfate**: Sulfate concentration in mg/L
    - **Conductivity**: Electrical conductivity in μS/cm
    - **Organic_carbon**: Organic carbon in ppm
    - **Trihalomethanes**: Trihalomethanes in μg/L
    - **Turbidity**: Turbidity in NTU
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict
        data = input_data.dict()
        
        # Make prediction
        result = predictor.predict(data)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(input_data: BatchWaterQualityInput):
    """
    Predict water potability for multiple samples
    
    Accepts a list of water quality samples and returns predictions for each.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic models to list of dicts
        data_list = [sample.dict() for sample in input_data.samples]
        
        # Make batch prediction
        results = predictor.predict_batch(data_list)
        
        # Count successful predictions
        successful = sum(1 for r in results if 'error' not in r)
        
        return {
            "predictions": results,
            "total_samples": len(results),
            "successful_predictions": successful
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/features", tags=["General"])
async def get_features():
    """Get list of required input features"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": predictor.feature_names,
        "total_features": len(predictor.feature_names)
    }


# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )