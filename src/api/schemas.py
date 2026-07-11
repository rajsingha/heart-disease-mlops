"""Pydantic request/response schemas for the prediction API."""

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """One patient record, mirroring the UCI Heart Disease feature schema."""

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(
        ..., ge=50, le=300, description="Resting blood pressure (mm Hg)"
    )
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(
        ..., ge=50, le=250, description="Maximum heart rate achieved"
    )
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(
        ..., ge=-1, le=10, description="ST depression induced by exercise"
    )
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST (1-3)")
    ca: float = Field(
        ..., ge=0, le=3, description="Number of major vessels colored (0-3)"
    )
    thal: float = Field(
        ..., description="Thalassemia (3 = normal, 6 = fixed, 7 = reversible defect)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 57,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 140,
                    "chol": 241,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 123,
                    "exang": 1,
                    "oldpeak": 0.2,
                    "slope": 2,
                    "ca": 0,
                    "thal": 7,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(description="1 = heart disease predicted, 0 = not")
    label: str = Field(description="Human-readable class label")
    probability: float = Field(description="P(heart disease), between 0 and 1")
    model_family: str = Field(description="Model family that produced the prediction")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_family: str | None = None
