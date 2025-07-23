from app.core.logging import setup_clean_environment
setup_clean_environment()

from fastapi import FastAPI, HTTPException, Depends 
from app.core.logging import app_logger
from app.ml.model import get_model_response
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction
from random import random
from uuid import uuid4
from sqlalchemy.orm import Session 
from app.core.database import SessionLocal, Message 
from typing import List

import joblib 
import numpy as np

app = FastAPI()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel, db: Session = Depends(get_db)):
    model_response = get_model_response(body.last_msg_text)
    incoming_msg = Message(
        id=str(body.last_message_id),
        dialog_id=str(body.dialog_id),
        role="user",
        content=body.last_msg_text
    )
    db.add(incoming_msg)

    bot_msg_id = str(uuid4())
    bot_msg = Message(
        id=bot_msg_id,
        dialog_id=str(body.dialog_id),
        role="assistant",
        content=model_response
    )
    db.add(bot_msg)
    db.commit()

    app_logger.info(f"Saved messages for dialog_id: {body.dialog_id}")
    return GetMessageResponseModel(new_msg_text=model_response, dialog_id=body.dialog_id)

@app.get("/get_history", response_model=List[GetMessageResponseModel])
async def get_history(dialog_id: str, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.dialog_id == dialog_id).order_by(Message.timestamp).all()
    return [GetMessageResponseModel(new_msg_text=msg.content, dialog_id=msg.dialog_id) for msg in messages]

def get_bot_probability(text: str) -> float:
    vec = vectorizer.transform([text])
    probas = clf.predict_proba(vec)[0]
    return float(probas[0])  


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Endpoint to save a message and get the probability
    that this message is from bot.

    Returns a `Prediction` object.
    """
    is_bot_probability = get_bot_probability(msg.text)  
    prediction_id = uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )
