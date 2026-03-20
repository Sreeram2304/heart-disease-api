"""
Database models — SQLAlchemy ORM
Tables: users, predictions
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), unique=True, index=True, nullable=False)
    email         = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    predictions   = relationship("Prediction", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


class Prediction(Base):
    __tablename__ = "predictions"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Input features
    age         = Column(Float, nullable=False)
    sex         = Column(Float, nullable=False)
    cp          = Column(Float, nullable=False)
    trestbps    = Column(Float, nullable=False)
    chol        = Column(Float, nullable=False)
    fbs         = Column(Float, nullable=False)
    restecg     = Column(Float, nullable=False)
    thalach     = Column(Float, nullable=False)
    exang       = Column(Float, nullable=False)
    oldpeak     = Column(Float, nullable=False)
    slope       = Column(Float, nullable=False)
    ca          = Column(Float, nullable=False)
    thal        = Column(Float, nullable=False)

    # Results
    prediction          = Column(Integer, nullable=False)   # 0 or 1
    probability_disease = Column(Float,   nullable=False)
    confidence          = Column(String(10), nullable=False)
    label               = Column(Text, nullable=False)

    created_at  = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction id={self.id} user_id={self.user_id} result={self.prediction}>"
