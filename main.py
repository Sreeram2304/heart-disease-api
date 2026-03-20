"""
Heart Disease Prediction API — Full Stack
Endpoints: auth, predict, history, stats
Run: uvicorn main:app --reload
UI:  http://localhost:8000
"""
 
import json
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
 
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse
 
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.orm import Session
 
from database.db import init_db, get_db
from database.models import User, Prediction
from auth import hash_password, verify_password, create_access_token, decode_token
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
 
MODEL = {}
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    for path in ["model.pkl", "features.json"]:
        if not Path(path).exists():
            raise RuntimeError(f"{path} not found — run train.py first.")
    MODEL["pipeline"] = joblib.load("model.pkl")
    with open("features.json") as f:
        meta = json.load(f)
    MODEL["features"] = meta["features"]
    MODEL["metrics"]  = meta.get("metrics", {})
    logger.info("Model loaded | DB initialised")
    yield
    MODEL.clear()
 
app = FastAPI(title="Heart Disease API", version="2.0.0", lifespan=lifespan)
 
templates = Jinja2Templates(directory="templates")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
 
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s → %d | %.1f ms", request.method, request.url.path, response.status_code, ms)
    response.headers["X-Latency-Ms"] = f"{ms:.1f}"
    return response
 
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_token(token) if token else None
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = db.query(User).filter(User.username == payload.get("sub")).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
 
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email:    EmailStr
    password: str = Field(..., min_length=6)
 
class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
 
class PatientFeatures(BaseModel):
    age:      float = Field(..., ge=1,   le=120, example=54)
    sex:      float = Field(..., ge=0,   le=1,   example=1)
    cp:       float = Field(..., ge=0,   le=3,   example=0)
    trestbps: float = Field(..., ge=50,  le=250, example=122)
    chol:     float = Field(..., ge=50,  le=600, example=286)
    fbs:      float = Field(..., ge=0,   le=1,   example=0)
    restecg:  float = Field(..., ge=0,   le=2,   example=0)
    thalach:  float = Field(..., ge=50,  le=250, example=116)
    exang:    float = Field(..., ge=0,   le=1,   example=1)
    oldpeak:  float = Field(..., ge=0.0, le=10,  example=3.2)
    slope:    float = Field(..., ge=0,   le=2,   example=1)
    ca:       float = Field(..., ge=0,   le=4,   example=2)
    thal:     float = Field(..., ge=0,   le=3,   example=2)
 
class PredictionResponse(BaseModel):
    prediction:          int
    label:               str
    probability_disease: float
    confidence:          str
 
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
 
@app.post("/api/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="Username already taken.")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    user = User(username=body.username, email=body.email, hashed_password=hash_password(body.password))
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "username": user.username}
 
@app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "username": user.username}
 
@app.get("/api/auth/me", tags=["Auth"])
def me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email, "created_at": current_user.created_at}
 
@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientFeatures, db: Session = Depends(get_db),
            current_user: User = Depends(get_current_user)):
    if "pipeline" not in MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    X = np.array([getattr(patient, f) for f in MODEL["features"]]).reshape(1, -1)
    try:
        pred  = int(MODEL["pipeline"].predict(X)[0])
        proba = float(MODEL["pipeline"].predict_proba(X)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    confidence = (
        "High"   if proba >= 0.75 or proba <= 0.25 else
        "Medium" if proba >= 0.60 or proba <= 0.40 else "Low"
    )
    label = "Heart disease detected" if pred == 1 else "No heart disease detected"
    record = Prediction(user_id=current_user.id, **patient.model_dump(),
                        prediction=pred, probability_disease=round(proba, 4),
                        confidence=confidence, label=label)
    db.add(record); db.commit()
    logger.info("user=%s pred=%d P=%.3f", current_user.username, pred, proba)
    return {"prediction": pred, "label": label, "probability_disease": round(proba, 4), "confidence": confidence}
 
@app.get("/api/history", tags=["Data"])
def history(limit: int = 20, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(Prediction).filter(Prediction.user_id == current_user.id)
              .order_by(Prediction.created_at.desc()).limit(limit).all())
    return [{"id": r.id, "prediction": r.prediction, "label": r.label,
             "probability_disease": r.probability_disease, "confidence": r.confidence,
             "age": r.age, "created_at": r.created_at.isoformat()} for r in rows]
 
@app.get("/api/stats", tags=["Data"])
def stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Prediction).filter(Prediction.user_id == current_user.id).all()
    if not rows:
        return {"total": 0, "disease": 0, "no_disease": 0, "avg_probability": 0, "model_metrics": MODEL.get("metrics", {})}
    total = len(rows)
    disease = sum(1 for r in rows if r.prediction == 1)
    return {"total": total, "disease": disease, "no_disease": total - disease,
            "avg_probability": round(sum(r.probability_disease for r in rows) / total, 4),
            "model_metrics": MODEL.get("metrics", {})}
 
@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "features": len(MODEL.get("features", []))}