"""
Prediction service layer for agricultural ML models
"""
import logging
from typing import Dict, Any, Optional
from ..models.ml_models import ml_models
from ..schemas.prediction import (
    PredictionInput, 
    FertilizerPrediction,
    IrrigationPrediction, 
    PestAlertPrediction,
    YieldPrediction,
    AllPredictions
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Service layer for handling predictions"""
    
    def __init__(self):
        self.models = ml_models
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all ML models"""
        return self.models.get_model_status()
    
    def ensure_models_loaded(self):
        """Ensure models are loaded, raise exception if not"""
        if not self.models.models_loaded:
            raise RuntimeError("ML models are not loaded. Please check model training and loading.")
    
    def predict_fertilizer(self, input_data: PredictionInput) -> FertilizerPrediction:
        """Predict fertilizer requirements"""
        self.ensure_models_loaded()
        
        try:
            # Convert Pydantic model to dict
            input_dict = input_data.dict()
            
            # Make prediction
            result = self.models.predict_fertilizer(input_dict)
            
            return FertilizerPrediction(
                n_fertilizer=result['n_fertilizer'],
                p_fertilizer=result['p_fertilizer'],
                k_fertilizer=result['k_fertilizer'],
                confidence=result.get('confidence')
            )
            
        except Exception as e:
            logger.error(f"Error in fertilizer prediction: {e}")
            raise RuntimeError(f"Fertilizer prediction failed: {str(e)}")
    
    def predict_irrigation(self, input_data: PredictionInput) -> IrrigationPrediction:
        """Predict irrigation need"""
        self.ensure_models_loaded()
        
        try:
            # Convert Pydantic model to dict
            input_dict = input_data.dict()
            
            # Make prediction
            result = self.models.predict_irrigation(input_dict)
            
            return IrrigationPrediction(
                irrigation_needed=result['irrigation_needed'],
                probability=result.get('probability'),
                confidence=result.get('confidence')
            )
            
        except Exception as e:
            logger.error(f"Error in irrigation prediction: {e}")
            raise RuntimeError(f"Irrigation prediction failed: {str(e)}")
    
    def predict_pest_alert(self, input_data: PredictionInput) -> PestAlertPrediction:
        """Predict pest alert"""
        self.ensure_models_loaded()
        
        try:
            # Convert Pydantic model to dict
            input_dict = input_data.dict()
            
            # Make prediction
            result = self.models.predict_pest_alert(input_dict)
            
            return PestAlertPrediction(
                pest_alert=result['pest_alert'],
                probability=result.get('probability'),
                confidence=result.get('confidence')
            )
            
        except Exception as e:
            logger.error(f"Error in pest alert prediction: {e}")
            raise RuntimeError(f"Pest alert prediction failed: {str(e)}")
    
    def predict_yield(self, input_data: PredictionInput) -> Optional[YieldPrediction]:
        """Predict crop yield (optional model)"""
        self.ensure_models_loaded()
        
        try:
            # Convert Pydantic model to dict
            input_dict = input_data.dict()
            
            # Make prediction
            result = self.models.predict_yield(input_dict)
            
            if result is None:
                return None
            
            return YieldPrediction(
                yield_prediction=result['yield_prediction'],
                confidence=result.get('confidence')
            )
            
        except Exception as e:
            logger.error(f"Error in yield prediction: {e}")
            return None
    
    def predict_all(self, input_data: PredictionInput) -> AllPredictions:
        """Make all predictions at once"""
        self.ensure_models_loaded()
        
        try:
            # Get individual predictions
            fertilizer = self.predict_fertilizer(input_data)
            irrigation = self.predict_irrigation(input_data)
            pest_alert = self.predict_pest_alert(input_data)
            yield_pred = self.predict_yield(input_data)
            
            return AllPredictions(
                fertilizer=fertilizer,
                irrigation=irrigation,
                pest_alert=pest_alert,
                yield_prediction=yield_pred
            )
            
        except Exception as e:
            logger.error(f"Error in combined predictions: {e}")
            raise RuntimeError(f"Combined prediction failed: {str(e)}")


# Global service instance
prediction_service = PredictionService()