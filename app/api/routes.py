"""
API routes for agricultural ML predictions
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging
from ..schemas.prediction import (
    PredictionInput,
    FertilizerPrediction,
    IrrigationPrediction,
    PestAlertPrediction,
    YieldPrediction,
    AllPredictions,
    HealthResponse,
    ModelStatusResponse
)
from ..services.prediction_service import prediction_service
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def get_model_status():
    """
    Get status of all ML models
    """
    try:
        status_info = prediction_service.get_model_status()
        
        return ModelStatusResponse(
            models_loaded=status_info['models_loaded'],
            total_models=status_info['total_models'],
            all_ready=status_info['all_ready']
        )
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model status: {str(e)}"
        )


@router.post("/predict/fertilizer", response_model=FertilizerPrediction, tags=["Predictions"])
async def predict_fertilizer(input_data: PredictionInput):
    """
    Predict NPK fertilizer requirements
    
    Returns nitrogen, phosphorus, and potassium fertilizer recommendations in kg/ha
    based on soil conditions, crop type, weather, and other factors.
    """
    try:
        prediction = prediction_service.predict_fertilizer(input_data)
        return prediction
    except RuntimeError as e:
        logger.error(f"Runtime error in fertilizer prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in fertilizer prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during fertilizer prediction"
        )


@router.post("/predict/irrigation", response_model=IrrigationPrediction, tags=["Predictions"])
async def predict_irrigation(input_data: PredictionInput):
    """
    Predict irrigation need
    
    Returns whether irrigation is needed (0/1) based on soil moisture,
    weather conditions, crop water requirements, and growth stage.
    """
    try:
        prediction = prediction_service.predict_irrigation(input_data)
        return prediction
    except RuntimeError as e:
        logger.error(f"Runtime error in irrigation prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in irrigation prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during irrigation prediction"
        )


@router.post("/predict/pest-alert", response_model=PestAlertPrediction, tags=["Predictions"])
async def predict_pest_alert(input_data: PredictionInput):
    """
    Predict pest alert
    
    Returns pest alert status (0/1) based on weather conditions,
    crop type, growth stage, and environmental factors.
    """
    try:
        prediction = prediction_service.predict_pest_alert(input_data)
        return prediction
    except RuntimeError as e:
        logger.error(f"Runtime error in pest alert prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in pest alert prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during pest alert prediction"
        )


@router.post("/predict/yield", response_model=YieldPrediction, tags=["Predictions"])
async def predict_yield(input_data: PredictionInput):
    """
    Predict crop yield (optional)
    
    Returns predicted crop yield in tons/ha based on soil conditions,
    weather, crop management, and other factors.
    """
    try:
        prediction = prediction_service.predict_yield(input_data)
        
        if prediction is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Yield prediction model not available"
            )
        
        return prediction
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except RuntimeError as e:
        logger.error(f"Runtime error in yield prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in yield prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during yield prediction"
        )


@router.post("/predict/all", response_model=AllPredictions, tags=["Predictions"])
async def predict_all(input_data: PredictionInput):
    """
    Make all predictions at once
    
    Returns comprehensive predictions including:
    - NPK fertilizer requirements
    - Irrigation need
    - Pest alert status
    - Crop yield (if available)
    """
    try:
        predictions = prediction_service.predict_all(input_data)
        return predictions
    except RuntimeError as e:
        logger.error(f"Runtime error in combined predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in combined predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during combined predictions"
        )