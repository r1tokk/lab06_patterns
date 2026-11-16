from fastapi import FastAPI, Depends
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, Field

from src.prediction_service import PredictionService
from src.local import LocalModelRepository

class HouseFeatures(BaseModel):
    OverallQual: int = Field(ge=1, le=10)
    GrLivArea: float = Field(gt=0)
    FirstFlrSF: float = Field(ge=0)
    TotalBsmtSF: float = Field(ge=0)
    GarageCars: float = Field(ge=0)
    YearBuilt: int = Field(ge=1800, le=2010)
    YrSold: int = Field(ge=2006, le=2010)
    FullBath: int = Field(ge=0)
    TotRmsAbvGrd: int = Field(ge=0)
    Neighborhood: str
    KitchenQual: Literal["Ex", "Gd", "TA", "Fa", "Po"]
    CentralAir: Literal["Y", "N"]

class PredictionResponse(BaseModel):
    prediction: float
    currency: str = "USD"
    model_version: str


app = FastAPI()

@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService(LocalModelRepository(), version="v1")

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HouseFeatures, service: PredictionService = Depends(get_service)):
    value = service.predict_one(payload.model_dump())
    return PredictionResponse(prediction=round(value, 2), model_version=service.version)