"""
Pydantic schemas for request and response validation
"""
from typing import Optional
from pydantic import BaseModel, Field, validator


class PredictionInput(BaseModel):
    """Input schema for agricultural predictions"""
    
    crop: str = Field(..., description="Crop type", example="rice")
    growth_stage: str = Field(..., description="Growth stage", example="flowering")
    soil_ph: float = Field(..., ge=3.0, le=10.0, description="Soil pH level", example=6.5)
    soil_n: float = Field(..., ge=0, le=300, description="Soil nitrogen (kg/ha)", example=85.2)
    soil_p: float = Field(..., ge=0, le=200, description="Soil phosphorus (kg/ha)", example=42.1)
    soil_k: float = Field(..., ge=0, le=400, description="Soil potassium (kg/ha)", example=120.5)
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture (%)", example=75.3)
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)", example=28.5)
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)", example=12.3)
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)", example=78.2)
    state: str = Field(..., description="State", example="punjab")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)", example=7)
    soil_type: str = Field(..., description="Soil type", example="loam")
    variety: str = Field(..., description="Crop variety", example="basmati")
    farmer_type: str = Field(..., description="Farmer type", example="progressive")
    irrigation_system: str = Field(..., description="Irrigation system", example="drip")

    @validator('crop')
    def validate_crop(cls, v):
        valid_crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
            'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
        ]
        if v.lower() not in valid_crops:
            raise ValueError(f'Crop must be one of: {valid_crops}')
        return v.lower()

    @validator('growth_stage')
    def validate_growth_stage(cls, v):
        valid_stages = ['germination', 'vegetative', 'flowering', 'harvest']
        if v.lower() not in valid_stages:
            raise ValueError(f'Growth stage must be one of: {valid_stages}')
        return v.lower()

    @validator('soil_type')
    def validate_soil_type(cls, v):
        valid_types = ['clay', 'loam', 'sandy_loam', 'clay_loam', 'sandy', 'silt']
        if v.lower() not in valid_types:
            raise ValueError(f'Soil type must be one of: {valid_types}')
        return v.lower()

    @validator('farmer_type')
    def validate_farmer_type(cls, v):
        valid_types = ['progressive', 'moderate', 'conservative', 'resource_poor']
        if v.lower() not in valid_types:
            raise ValueError(f'Farmer type must be one of: {valid_types}')
        return v.lower()

    @validator('irrigation_system')
    def validate_irrigation_system(cls, v):
        valid_systems = ['flood', 'sprinkler', 'drip', 'furrow']
        if v.lower() not in valid_systems:
            raise ValueError(f'Irrigation system must be one of: {valid_systems}')
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "crop": "rice",
                "growth_stage": "flowering",
                "soil_ph": 6.5,
                "soil_n": 85.2,
                "soil_p": 42.1,
                "soil_k": 120.5,
                "soil_moisture": 75.3,
                "temperature": 28.5,
                "rainfall": 12.3,
                "humidity": 78.2,
                "state": "punjab",
                "month": 7,
                "soil_type": "loam",
                "variety": "basmati",
                "farmer_type": "progressive",
                "irrigation_system": "drip"
            }
        }


class FertilizerPrediction(BaseModel):
    """Fertilizer prediction response"""
    n_fertilizer: float = Field(..., description="Nitrogen fertilizer (kg/ha)")
    p_fertilizer: float = Field(..., description="Phosphorus fertilizer (kg/ha)")
    k_fertilizer: float = Field(..., description="Potassium fertilizer (kg/ha)")
    confidence: Optional[float] = Field(None, description="Prediction confidence")


class IrrigationPrediction(BaseModel):
    """Irrigation prediction response"""
    irrigation_needed: int = Field(..., description="Irrigation needed (0/1)")
    probability: Optional[float] = Field(None, description="Probability of needing irrigation")
    confidence: Optional[float] = Field(None, description="Prediction confidence")


class PestAlertPrediction(BaseModel):
    """Pest alert prediction response"""
    pest_alert: int = Field(..., description="Pest alert (0/1)")
    probability: Optional[float] = Field(None, description="Probability of pest occurrence")
    confidence: Optional[float] = Field(None, description="Prediction confidence")


class YieldPrediction(BaseModel):
    """Yield prediction response"""
    yield_prediction: float = Field(..., description="Predicted crop yield (tons/ha)")
    confidence: Optional[float] = Field(None, description="Prediction confidence")


class AllPredictions(BaseModel):
    """Combined predictions response"""
    fertilizer: FertilizerPrediction
    irrigation: IrrigationPrediction
    pest_alert: PestAlertPrediction
    yield_prediction: Optional[YieldPrediction] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class ModelStatusResponse(BaseModel):
    """Model status response"""
    models_loaded: dict = Field(..., description="Status of each model")
    total_models: int = Field(..., description="Total number of models")
    all_ready: bool = Field(..., description="Whether all models are ready")