"""
ML Models module for loading and making predictions
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)


class AgriculturalMLModels:
    """Handles loading and prediction with trained ML models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.model_names = [
            'n_fertilizer', 'p_fertilizer', 'k_fertilizer',
            'irrigation_needed', 'pest_alert', 'yield'
        ]
        self.models_loaded = False
        
    def load_models(self) -> bool:
        """Load all trained models"""
        models_dir = settings.models_dir
        
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return False
        
        # Load preprocessor
        try:
            from ...scripts.data_preprocessor import AgriculturalDataPreprocessor
            self.preprocessor = AgriculturalDataPreprocessor()
            self.preprocessor.load_preprocessors(models_dir)
            logger.info("Loaded data preprocessor")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return False
        
        # Load individual models
        loaded_count = 0
        for model_name in self.model_names:
            model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
            
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    loaded_count += 1
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        self.models_loaded = loaded_count > 0
        logger.info(f"Successfully loaded {loaded_count}/{len(self.model_names)} models")
        
        return self.models_loaded
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        
        for model_name in self.model_names:
            status[model_name] = model_name in self.models
        
        return {
            'models_loaded': status,
            'total_models': len(self.model_names),
            'loaded_count': sum(status.values()),
            'all_ready': all(status.values())
        }
    
    def _preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        if not self.preprocessor:
            raise RuntimeError("Preprocessor not loaded")
        
        return self.preprocessor.transform_single_input(input_data)
    
    def predict_fertilizer(self, input_data: dict) -> Dict[str, Any]:
        """Predict NPK fertilizer requirements"""
        required_models = ['n_fertilizer', 'p_fertilizer', 'k_fertilizer']
        
        for model_name in required_models:
            if model_name not in self.models:
                raise RuntimeError(f"Model {model_name} not loaded")
        
        # Preprocess input
        X = self._preprocess_input(input_data)
        
        predictions = {}
        confidences = {}
        
        for nutrient in ['n', 'p', 'k']:
            model_name = f'{nutrient}_fertilizer'
            model = self.models[model_name]
            
            prediction = model.predict(X)[0]
            predictions[f'{nutrient}_fertilizer'] = max(0, round(prediction, 1))
            
            # Calculate confidence (simplified - based on prediction variance)
            if hasattr(model, 'estimators_'):
                # For ensemble methods, use prediction variance
                individual_preds = [est.predict(X)[0] for est in model.estimators_]
                variance = np.var(individual_preds)
                confidence = max(0.5, 1 - min(variance / prediction, 0.5))
                confidences[f'{nutrient}_fertilizer'] = round(confidence, 3)
        
        return {
            **predictions,
            'confidence': round(np.mean(list(confidences.values())), 3) if confidences else None
        }
    
    def predict_irrigation(self, input_data: dict) -> Dict[str, Any]:
        """Predict irrigation need"""
        model_name = 'irrigation_needed'
        
        if model_name not in self.models:
            raise RuntimeError(f"Model {model_name} not loaded")
        
        # Preprocess input
        X = self._preprocess_input(input_data)
        model = self.models[model_name]
        
        prediction = model.predict(X)[0]
        
        result = {
            'irrigation_needed': int(prediction)
        }
        
        # Add probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            result['probability'] = round(probabilities[1], 3)  # Probability of needing irrigation
            result['confidence'] = round(max(probabilities), 3)
        
        return result
    
    def predict_pest_alert(self, input_data: dict) -> Dict[str, Any]:
        """Predict pest alert"""
        model_name = 'pest_alert'
        
        if model_name not in self.models:
            raise RuntimeError(f"Model {model_name} not loaded")
        
        # Preprocess input
        X = self._preprocess_input(input_data)
        model = self.models[model_name]
        
        prediction = model.predict(X)[0]
        
        result = {
            'pest_alert': int(prediction)
        }
        
        # Add probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            result['probability'] = round(probabilities[1], 3)  # Probability of pest occurrence
            result['confidence'] = round(max(probabilities), 3)
        
        return result
    
    def predict_yield(self, input_data: dict) -> Optional[Dict[str, Any]]:
        """Predict crop yield"""
        model_name = 'yield'
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not available")
            return None
        
        # Preprocess input
        X = self._preprocess_input(input_data)
        model = self.models[model_name]
        
        prediction = model.predict(X)[0]
        
        result = {
            'yield_prediction': max(0, round(prediction, 2))
        }
        
        # Calculate confidence for yield prediction
        if hasattr(model, 'estimators_'):
            individual_preds = [est.predict(X)[0] for est in model.estimators_]
            variance = np.var(individual_preds)
            confidence = max(0.5, 1 - min(variance / prediction, 0.5))
            result['confidence'] = round(confidence, 3)
        
        return result
    
    def predict_all(self, input_data: dict) -> Dict[str, Any]:
        """Make all predictions at once"""
        results = {}
        
        try:
            # Fertilizer predictions
            fertilizer_result = self.predict_fertilizer(input_data)
            results['fertilizer'] = fertilizer_result
        except Exception as e:
            logger.error(f"Error in fertilizer prediction: {e}")
            results['fertilizer'] = {'error': str(e)}
        
        try:
            # Irrigation prediction
            irrigation_result = self.predict_irrigation(input_data)
            results['irrigation'] = irrigation_result
        except Exception as e:
            logger.error(f"Error in irrigation prediction: {e}")
            results['irrigation'] = {'error': str(e)}
        
        try:
            # Pest alert prediction
            pest_result = self.predict_pest_alert(input_data)
            results['pest_alert'] = pest_result
        except Exception as e:
            logger.error(f"Error in pest alert prediction: {e}")
            results['pest_alert'] = {'error': str(e)}
        
        try:
            # Yield prediction (optional)
            yield_result = self.predict_yield(input_data)
            if yield_result:
                results['yield_prediction'] = yield_result
        except Exception as e:
            logger.error(f"Error in yield prediction: {e}")
            results['yield_prediction'] = {'error': str(e)}
        
        return results


# Global instance
ml_models = AgriculturalMLModels()