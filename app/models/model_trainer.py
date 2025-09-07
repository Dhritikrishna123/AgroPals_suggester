"""
ML Model training module for agricultural predictions
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import joblib
import os
import logging
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AgriculturalModelTrainer:
    """Trains and evaluates ML models for agricultural predictions"""
    
    def __init__(self, random_state: int = 42, n_estimators: int = 100, max_depth: int = 10):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = {}
        self.model_configs = {
            'n_fertilizer': {
                'type': 'regression',
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state,
                    'n_jobs': -1
                }
            },
            'p_fertilizer': {
                'type': 'regression',
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state,
                    'n_jobs': -1
                }
            },
            'k_fertilizer': {
                'type': 'regression',
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state,
                    'n_jobs': -1
                }
            },
            'irrigation_needed': {
                'type': 'classification',
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
            },
            'pest_alert': {
                'type': 'classification',
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
            },
            'yield': {
                'type': 'regression',
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state
                }
            }
        }
    
    def train_regression_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              model_name: str) -> Dict[str, Any]:
        """Train and evaluate a regression model"""
        logger.info(f"Training regression model: {model_name}")
        
        config = self.model_configs[model_name]
        model = config['model'](**config['params'])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1)
        metrics['cv_rmse_mean'] = -cv_scores.mean()
        metrics['cv_rmse_std'] = cv_scores.std()
        
        logger.info(f"{model_name} - Test RMSE: {metrics['test_rmse']:.3f}, R2: {metrics['test_r2']:.3f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
    
    def train_classification_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 model_name: str) -> Dict[str, Any]:
        """Train and evaluate a classification model"""
        logger.info(f"Training classification model: {model_name}")
        
        config = self.model_configs[model_name]
        model = config['model'](**config['params'])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='accuracy', n_jobs=-1)
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        # Classification report and confusion matrix
        metrics['classification_report'] = classification_report(y_test, y_pred_test)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
        
        logger.info(f"{model_name} - Test Accuracy: {metrics['test_accuracy']:.3f}, F1: {metrics['test_f1']:.3f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'test_proba': y_pred_proba_test
            }
        }
    
    def train_all_models(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Train all models using the provided data splits"""
        logger.info("Starting training of all models")
        
        X_train = data_splits['X_train']
        X_test = data_splits['X_test']
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            target_key = f'y_train_{model_name}'
            test_target_key = f'y_test_{model_name}'
            
            if target_key not in data_splits or test_target_key not in data_splits:
                logger.warning(f"Skipping {model_name} - target data not found")
                continue
            
            y_train = data_splits[target_key]
            y_test = data_splits[test_target_key]
            
            try:
                if config['type'] == 'regression':
                    result = self.train_regression_model(
                        X_train, y_train, X_test, y_test, model_name
                    )
                else:  # classification
                    result = self.train_classification_model(
                        X_train, y_train, X_test, y_test, model_name
                    )
                
                self.models[model_name] = result['model']
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Completed training {len(self.models)} models")
        return results
    
    def save_models(self, save_dir: str):
        """Save trained models to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save model metadata
        metadata = {
            'model_names': list(self.models.keys()),
            'model_configs': self.model_configs,
            'random_state': self.random_state
        }
        
        metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        logger.info(f"Saved model metadata to {metadata_path}")
    
    def load_models(self, load_dir: str) -> bool:
        """Load trained models from disk"""
        try:
            # Load metadata
            metadata_path = os.path.join(load_dir, 'model_metadata.pkl')
            if not os.path.exists(metadata_path):
                logger.warning(f"Model metadata not found at {metadata_path}")
                return False
            
            metadata = joblib.load(metadata_path)
            model_names = metadata.get('model_names', [])
            
            # Load individual models
            loaded_count = 0
            for model_name in model_names:
                model_path = os.path.join(load_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    loaded_count += 1
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            logger.info(f"Successfully loaded {loaded_count}/{len(model_names)} models")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> Optional[Dict[str, float]]:
        """Get feature importance for a trained model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def predict_single(self, model_name: str, X: np.ndarray) -> Dict[str, Any]:
        """Make prediction with a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        prediction = model.predict(X)
        
        result = {'prediction': prediction[0] if len(prediction) == 1 else prediction}
        
        # Add probability for classification models
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            result['probabilities'] = probabilities[0] if len(probabilities) == 1 else probabilities
            result['confidence'] = np.max(probabilities)
        
        return result
    
    def generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive training report"""
        report = []
        report.append("=" * 60)
        report.append("AGRICULTURAL ML MODELS TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Total Models Trained: {len(results)}")
        report.append(f"Random State: {self.random_state}")
        report.append("")
        
        for model_name, result in results.items():
            report.append(f"Model: {model_name.upper()}")
            report.append("-" * 40)
            
            metrics = result['metrics']
            config = self.model_configs[model_name]
            
            if config['type'] == 'regression':
                report.append(f"  Type: Regression ({config['model'].__name__})")
                report.append(f"  Train RMSE: {metrics['train_rmse']:.4f}")
                report.append(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
                report.append(f"  Train R²:   {metrics['train_r2']:.4f}")
                report.append(f"  Test R²:    {metrics['test_r2']:.4f}")
                report.append(f"  CV RMSE:    {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
            else:
                report.append(f"  Type: Classification ({config['model'].__name__})")
                report.append(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
                report.append(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
                report.append(f"  Test Precision: {metrics['test_precision']:.4f}")
                report.append(f"  Test Recall:    {metrics['test_recall']:.4f}")
                report.append(f"  Test F1:        {metrics['test_f1']:.4f}")
                report.append(f"  CV Accuracy:    {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
            
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)