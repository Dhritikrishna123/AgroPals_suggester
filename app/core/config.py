"""
Configuration settings for the Agricultural ML API
"""
import os
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings"""
    
    # API Settings
    app_name: str = "Agricultural ML API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings
    allowed_origins: List[str] = ["*"]
    
    # Data and Model Paths
    data_dir: str = "data"
    models_dir: str = "trained_models"
    data_file: str = "agricultural_data.csv"
    
    # Model Settings
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Model Parameters
    n_estimators: int = 100
    max_depth: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override with environment variables if they exist
        self.app_name = os.getenv("APP_NAME", self.app_name)
        self.app_version = os.getenv("APP_VERSION", self.app_version)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", str(self.port)))
        
        self.data_dir = os.getenv("DATA_DIR", self.data_dir)
        self.models_dir = os.getenv("MODELS_DIR", self.models_dir)
        self.data_file = os.getenv("DATA_FILE", self.data_file)
        
        self.random_state = int(os.getenv("RANDOM_STATE", str(self.random_state)))
        self.test_size = float(os.getenv("TEST_SIZE", str(self.test_size)))
        
        self.n_estimators = int(os.getenv("N_ESTIMATORS", str(self.n_estimators)))
        self.max_depth = int(os.getenv("MAX_DEPTH", str(self.max_depth)))
        
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)


# Global settings instance
settings = Settings()