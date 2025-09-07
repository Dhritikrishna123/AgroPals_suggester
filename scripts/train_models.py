#!/usr/bin/env python3
"""
Training script for Agricultural ML models
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_preprocessor import AgriculturalDataPreprocessor
from app.models.model_trainer import AgriculturalModelTrainer
from app.core.config import settings
from app.core.logging import setup_logging


def main():
    """Main training function"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Agricultural ML Models Training")
    
    # Define paths
    data_file = os.path.join(settings.data_dir, settings.data_file)
    models_dir = settings.models_dir
    
    # Check if data file exists
    if not os.path.exists(data_file):
        logger.error(f"Training data not found at: {data_file}")
        logger.error("Please place your agricultural_data.csv in the data/ directory")
        return False
    
    try:
        # Initialize preprocessor
        logger.info("Initializing data preprocessor...")
        preprocessor = AgriculturalDataPreprocessor(random_state=settings.random_state)
        
        # Load and validate data
        logger.info("Loading and validating data...")
        df = preprocessor.load_and_validate_data(data_file)
        logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X, y = preprocessor.preprocess_pipeline(df, fit=True, scale_features=True)
        
        # Split data
        logger.info("Splitting data into train/test sets...")
        data_splits = preprocessor.split_data(X, y, test_size=settings.test_size)
        
        # Save preprocessor
        logger.info("Saving preprocessor...")
        preprocessor.save_preprocessors(models_dir)
        
        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer = AgriculturalModelTrainer(
            random_state=settings.random_state,
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth
        )
        
        # Train all models
        logger.info("Starting model training...")
        results = trainer.train_all_models(data_splits)
        
        # Save trained models
        logger.info("Saving trained models...")
        trainer.save_models(models_dir)
        
        # Generate and display training report
        report = trainer.generate_training_report(results)
        print("\n" + report)
        
        # Save report to file
        report_file = os.path.join(models_dir, "training_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Training report saved to: {report_file}")
        
        logger.info("Model training completed successfully!")
        logger.info(f"Models saved to: {models_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)