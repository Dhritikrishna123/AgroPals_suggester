"""
Data preprocessing utilities for agricultural ML models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class AgriculturalDataPreprocessor:
    """Handles data preprocessing for agricultural ML models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = {
            'n_fertilizer': 'regression',
            'p_fertilizer': 'regression', 
            'k_fertilizer': 'regression',
            'irrigation_needed': 'classification',
            'pest_alert': 'classification',
            'yield': 'regression'  # Optional target
        }
        
    def _get_categorical_columns(self) -> list:
        """Get list of categorical columns to encode"""
        return [
            'crop', 'growth_stage', 'state', 'soil_type', 
            'variety', 'farmer_type', 'irrigation_system'
        ]
    
    def _get_numerical_columns(self) -> list:
        """Get list of numerical columns"""
        return [
            'soil_ph', 'soil_n', 'soil_p', 'soil_k', 'soil_moisture',
            'temperature', 'rainfall', 'humidity', 'month'
        ]
    
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and perform basic validation on the dataset"""
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Check for required columns
        required_columns = (
            self._get_categorical_columns() + 
            self._get_numerical_columns()
        )
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic data validation
        logger.info("Performing data validation...")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        
        # Check data ranges
        self._validate_data_ranges(df)
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame):
        """Validate numerical data ranges"""
        validations = {
            'soil_ph': (3.0, 10.0),
            'soil_n': (0, 300),
            'soil_p': (0, 200), 
            'soil_k': (0, 400),
            'soil_moisture': (0, 100),
            'temperature': (-10, 60),
            'rainfall': (0, 500),
            'humidity': (0, 100),
            'month': (1, 12)
        }
        
        for col, (min_val, max_val) in validations.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    logger.warning(f"Column {col}: {out_of_range} values out of range [{min_val}, {max_val}]")
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using label encoding"""
        df = df.copy()
        categorical_cols = self._get_categorical_columns()
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    logger.info(f"Fitted encoder for {col}: {len(self.label_encoders[col].classes_)} classes")
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        mask = df[col].isin(le.classes_)
                        df.loc[mask, col] = le.transform(df.loc[mask, col])
                        df.loc[~mask, col] = -1  # Assign -1 to unseen categories
                        logger.info(f"Transformed {col}: {mask.sum()} known, {(~mask).sum()} unknown categories")
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True, method: str = 'robust') -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()
        numerical_cols = self._get_numerical_columns()
        
        # Choose scaler
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        for col in numerical_cols:
            if col in df.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = scaler_class()
                    df[[col]] = self.scalers[col].fit_transform(df[[col]])
                    logger.info(f"Fitted {method} scaler for {col}")
                else:
                    if col in self.scalers:
                        df[[col]] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Separate features and targets"""
        # Get feature columns (everything except targets)
        target_cols = list(self.target_columns.keys())
        feature_cols = [col for col in df.columns if col not in target_cols]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = {}
        
        for target_col in target_cols:
            if target_col in df.columns:
                y[target_col] = df[target_col].copy()
                logger.info(f"Target {target_col}: shape {y[target_col].shape}")
            else:
                logger.warning(f"Target column {target_col} not found in data")
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: Dict[str, pd.Series], 
                   test_size: float = 0.2) -> Dict[str, Any]:
        """Split data into train/test sets"""
        splits = {}
        
        # Use the first target for stratification if it's classification
        stratify_target = None
        for target_name, target_type in self.target_columns.items():
            if target_name in y and target_type == 'classification':
                stratify_target = y[target_name]
                break
        
        # Split features
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=self.random_state,
            stratify=stratify_target
        )
        
        splits['X_train'] = X_train
        splits['X_test'] = X_test
        
        # Split each target
        for target_name, target_values in y.items():
            y_train, y_test = train_test_split(
                target_values, test_size=test_size, random_state=self.random_state,
                stratify=stratify_target
            )
            splits[f'y_train_{target_name}'] = y_train
            splits[f'y_test_{target_name}'] = y_test
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        return splits
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True, 
                           scale_features: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        if scale_features:
            df_encoded = self.scale_numerical_features(df_encoded, fit=fit)
        
        # Separate features and targets
        X, y = self.prepare_features_and_targets(df_encoded)
        
        logger.info("Preprocessing pipeline completed")
        
        return X, y
    
    def save_preprocessors(self, save_dir: str):
        """Save fitted preprocessors"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
            logger.info(f"Saved label encoders to {save_dir}")
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.pkl'))
            logger.info(f"Saved scalers to {save_dir}")
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(save_dir, 'feature_columns.pkl'))
        logger.info(f"Saved feature columns to {save_dir}")
    
    def load_preprocessors(self, load_dir: str):
        """Load fitted preprocessors"""
        # Load label encoders
        encoders_path = os.path.join(load_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
            logger.info(f"Loaded label encoders from {load_dir}")
        
        # Load scalers
        scalers_path = os.path.join(load_dir, 'scalers.pkl')
        if os.path.exists(scalers_path):
            self.scalers = joblib.load(scalers_path)
            logger.info(f"Loaded scalers from {load_dir}")
        
        # Load feature columns
        features_path = os.path.join(load_dir, 'feature_columns.pkl')
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
            logger.info(f"Loaded feature columns from {load_dir}")
    
    def transform_single_input(self, input_data: dict) -> pd.DataFrame:
        """Transform a single input for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply preprocessing (without fitting)
        df_encoded = self.encode_categorical_features(df, fit=False)
        df_scaled = self.scale_numerical_features(df_encoded, fit=False)
        
        # Select only feature columns
        if self.feature_columns:
            df_final = df_scaled[self.feature_columns]
        else:
            df_final = df_scaled
        
        return df_final