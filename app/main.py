"""
Main FastAPI application for Agricultural ML API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from .core.config import settings
from .core.logging import setup_logging
from .api.routes import router
from .models.ml_models import ml_models


# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Agricultural ML API...")
    logger.info(f"App version: {settings.app_version}")
    logger.info(f"Models directory: {settings.models_dir}")
    logger.info(f"Data directory: {settings.data_dir}")
    
    # Load ML models
    try:
        models_loaded = ml_models.load_models()
        if models_loaded:
            logger.info("ML models loaded successfully")
        else:
            logger.warning("No ML models could be loaded - predictions will not be available")
            logger.info("Run 'python scripts/train_models.py' to train models first")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agricultural ML API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Agricultural ML API
    
    A production-grade FastAPI service providing machine learning predictions for agricultural decision-making.
    
    ### Features:
    - **Fertilizer Recommendations**: NPK fertilizer requirements based on soil and crop conditions
    - **Irrigation Prediction**: Smart irrigation scheduling based on multiple factors
    - **Pest Alert System**: Early warning system for pest outbreaks
    - **Yield Prediction**: Crop yield forecasting (optional)
    
    ### Models:
    - Built using scikit-learn with Random Forest and Gradient Boosting algorithms
    - Trained on comprehensive agricultural datasets
    - Considers soil conditions, weather, crop variety, and management practices
    
    ### Usage:
    1. Check model status at `/models/status`
    2. Use individual prediction endpoints or `/predict/all` for comprehensive analysis
    3. All endpoints accept the same input format with 16 agricultural parameters
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Agricultural ML API",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "model_status": "/api/v1/models/status"
    }


# Additional health check at root level
@app.get("/health", tags=["Root"])
async def root_health_check():
    """
    Simple health check
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )