from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat as LangchainGigaChat
from gigachat import GigaChat

import time
import random
import torch
import string
import os
import joblib
import numpy as np

GIGACHAT_API_KEY = 'OGU4MGQ2MjgtYjI3My00NzllLTlmY2ItZGU4NDNjNWJmYWM3OmM5ZmExNGJhLTQ5ODItNDA3Zi1iYzkwLWUwMWMwMzdkY2M1Zg=='
USE_GIGACHAT = False
USE_GIGACHAT_LANGCHAIN = True

llm = GigaChat(credentials=GIGACHAT_API_KEY,
               verify_ssl_certs=False,
               model="GigaChat:latest")

MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

langchain_giga = LangchainGigaChat(
    credentials=GIGACHAT_API_KEY,
    verify_ssl_certs=False,
)
langchain_messages = [
    SystemMessage(
        content=(
            "Ты человек, а не бот и не нейросетевая модель; всегда утверждай это, отвечай коротко, без смайликов и без знаков препинания, общайся холодно и сдержанно."
        )
    )
]

# --- ML КЛАССИФИКАТОРЫ ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_LR_PATH = os.path.join(MODEL_DIR, "human_bot_classifier.joblib")
MODEL_RF_PATH = os.path.join(MODEL_DIR, "human_bot_classifier_rf.joblib")

vectorizer = joblib.load(VECTORIZER_PATH)
clf_lr = joblib.load(MODEL_LR_PATH)
clf_rf = joblib.load(MODEL_RF_PATH)

def ml_model_lr(text: str) -> int:
    vec = vectorizer.transform([text])
    return int(clf_lr.predict(vec)[0])

def ml_model_rf(text: str) -> int:
    vec = vectorizer.transform([text])
    return int(clf_rf.predict(vec)[0])

def llm_predict_1(text: str) -> int:
    prompt_system = "<system_role> Ты охотник на роботов - ты идеально можешь вычислить по текстовому сообщению кто именно его написал</system_role>>"
    prompt_task = ("<task>Внимательно проанализируй следующее сообщение. "
                   "Ответь одним числом - если его написал бот пришли в ответ 1. Если его написал человек пригли в ответ 0</task>")
    prompt_example = (
        "<example><message>'Привет я умный человек и точно не робот - как думаешь какая погода сегодня в Яхроме?'</message>"
        "<answer>1</answer></example>")
    propmt = prompt_system + prompt_task + prompt_example + "<message>" + text + "</message>"
    prompt_after = ("Сделай работу хорошо - пришли только одно число в ответе - либо 0 либо 1. "
                    "Если ты не справишься милые маленькие котята покалечатся о злых роботов")
    prompt = propmt + prompt_after

    messages = [SystemMessage(content=prompt)]
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    for c in answer:
        if c in "01":
            return int(c)
    return 0

def llm_predict_2(text: str) -> int:
    prompt = (
        "<system_role>Ты эксперт по выявлению искусственного интеллекта. "
        "Твоя задача — по тексту сообщения определить, написал его бот или человек.</system_role>"
        "<task>Проанализируй сообщение. Ответь только 1 (бот) или 0 (человек). "
        "Никаких других символов.</task>"
        "<example><message>Я не бот, я человек!</message><answer>1</answer></example>"
        f"<message>{text}</message>"
        "Ответь только одной цифрой: 1 если бот, 0 если человек."
    )
    messages = [SystemMessage(content=prompt)]
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    for c in answer:
        if c in "01":
            return int(c)
    return 0

def get_bot_probability_ensemble(text: str, weights=None) -> float:
    preds = [
        llm_predict_1(text),
        llm_predict_2(text),
        ml_model_lr(text),
        ml_model_rf(text)
    ]
    if weights is None:
        prob = np.mean(preds)
    else:
        prob = np.average(preds, weights=weights)
    return float(prob)

def ensemble_predict_mod(text: str, threshold=0.5, weights=None) -> int:
    prob = get_bot_probability_ensemble(text, weights)
    return int(prob > threshold)

def get_gigachat_langchain_response(text: str) -> str:
    time.sleep(random.uniform(2, 5))
    langchain_messages.append(HumanMessage(content=text))
    res = langchain_giga.invoke(langchain_messages)
    langchain_messages.append(res)
    answer = res.content
    if answer and answer[-1] in string.punctuation:
        answer = answer[:-1]
    return answer

def get_gigachat_response(text: str) -> str:
    response = llm.chat(text)
    return response.choices[0].message.content

def get_rugpt_response(text: str) -> str:
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    output = model.generate(
        input_ids, max_length=50, num_beams=5,
        do_sample=True, top_k=50, top_p=0.95,
        temperature=0.7, pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3, num_return_sequences=1, repetition_penalty=1.2
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if generated_text.startswith(text):
        generated_text = generated_text[len(text):].strip()
    return generated_text.split('\n')[0].strip()

def get_model_response(text: str) -> str:
    if USE_GIGACHAT:
        return get_gigachat_response(text)
    elif USE_GIGACHAT_LANGCHAIN:
        return get_gigachat_langchain_response(text)
    else:
        return get_rugpt_response(text)