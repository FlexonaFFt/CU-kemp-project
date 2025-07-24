from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

SQLALCHEMY_PREDICTION_DATABASE_URL = "sqlite:///./prediction_data.db"

engine = create_engine(
    SQLALCHEMY_PREDICTION_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionPrediction = sessionmaker(autocommit=False, autoflush=False, bind=engine)
BasePrediction = declarative_base()

class PredictionData(BasePrediction):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True, index=True)
    message_id = Column(String, index=True)
    dialog_id = Column(String, index=True)
    participant_index = Column(Integer)
    is_bot_probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

BasePrediction.metadata.create_all(bind=engine)