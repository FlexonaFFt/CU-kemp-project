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
USE_GIGACHAT_LANGCHAIN = False 
USE_GIGACHAT_HUMANLIKE = True

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

def ml_model_lr(text: str) -> int:
    vec = vectorizer.transform([text])
    return int(clf_lr.predict(vec)[0])

def ml_model_rf(text: str) -> int:
    vec = vectorizer.transform([text])
    return int(clf_rf.predict(vec)[0])

def llm_predict_1(text: str) -> int:
    prompt = (
        "<system_role>Ты — специалист по выявлению ИИ, которые маскируются под людей. "
        "Некоторые боты умеют писать очень похоже на человека, но ты знаешь, как замечать тонкие сигналы: "
        "ненатуральная логика, слишком правильная грамматика, отсутствие сомнений, однообразная структура.</system_role>"
        "<task>Проанализируй текст. Ответь ТОЛЬКО цифрой: 1 — если это бот, 0 — если человек. "
        "Никаких пояснений, символов или пробелов. Делай выбор осознанно — ищи неочевидные сигналы маскировки.</task>"
        "<examples>"
        "<example><message>Это круто! Я прям не могу поверить 😅</message><answer>0</answer></example>"
        "<example><message>Здравствуйте. Погода сегодня солнечная и тёплая. Рекомендую надеть лёгкую одежду.</message><answer>1</answer></example>"
        "<example><message>мм, вроде как норм, но чёт не уверен хд</message><answer>0</answer></example>"
        "<example><message>Хорошо, чем ещё могу помочь?</message><answer>1</answer></example>"
        "</examples>"
        f"<message>{text}</message>"
        "Ответ:"
    )
    messages = [SystemMessage(content=prompt)]
    time.sleep(1)
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    return int(answer[0]) if answer[0] in "01" else 0

def llm_predict_2(text: str) -> int:
    prompt = (
        "<system_role>Ты — эксперт по распознаванию продвинутых ботов. "
        "Твоя задача — выявлять тех, кто говорит почти как человек. "
        "Особое внимание уделяй логике, эмоциональной пластичности, уместным ошибкам, реакции на неоднозначность.</system_role>"
        "<task>Ответь ТОЛЬКО цифрой: 1 — если бот (даже если хорошо скрывается), 0 — если человек. "
        "Анализируй тонкие отклонения от естественного разговора.</task>"
        "<example><message>ну такое, я бы поспорил... 😅</message><answer>0</answer></example>"
        "<example><message>Конечно. Чтобы узнать погоду, вы можете воспользоваться Яндекс.Погодой.</message><answer>1</answer></example>"
        "<example><message>блин, чет я туплю)</message><answer>0</answer></example>"
        "<example><message>Я могу ответить на любые ваши вопросы.</message><answer>1</answer></example>"
        f"<message>{text}</message>"
        "Ответ:"
    )
    messages = [SystemMessage(content=prompt)]
    time.sleep(1)
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    return int(answer[0]) if answer[0] in "01" else 0

def llm_predict_punctuation(text: str) -> int:
    prompt = (
        "<system_role>Ты — эксперт по пунктуации. "
        "Боты часто используют слишком правильную пунктуацию, всегда ставят точки, избегают скобок, многоточий, хаотичных восклицаний.</system_role>"
        "<task>Если пунктуация выглядит как у слишком аккуратного писателя — это бот. "
        "Если она спонтанна, нарушает правила — это человек. Ответь одной цифрой: 1 — бот, 0 — человек.</task>"
        "<example><message>ну вот блин... опять?!</message><answer>0</answer></example>"
        "<example><message>Здравствуйте. Чем могу помочь.</message><answer>1</answer></example>"
        "<example><message>Ок)) Тогда завтра увидимся.</message><answer>0</answer></example>"
        f"<message>{text}</message>"
        "Ответ:"
    )
    messages = [SystemMessage(content=prompt)]
    time.sleep(1)
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    return int(answer[0]) if answer[0] in "01" else 0

def llm_predict_slang(text: str) -> int:
    prompt = (
        "<system_role>Ты — эксперт по языковым интонациям. "
        "Боты могут вставлять сленг, но часто используют его неестественно или в неподходящем контексте. "
        "Ты умеешь определять поддельную неформальность.</system_role>"
        "<task>Если сленг кажется натянутым, неуместным или слишком 'правильным' — это бот. "
        "Если сленг живой, с ошибками, сокращениями и личными вставками — человек. Ответь 1 (бот) или 0 (человек).</task>"
        "<example><message>лол, жесть какая, я в шоке 😂</message><answer>0</answer></example>"
        "<example><message>Привет, братан. Что могу сделать для тебя?</message><answer>1</answer></example>"
        "<example><message>ну капец конечно( опять работа 😭</message><answer>0</answer></example>"
        f"<message>{text}</message>"
        "Ответ:"
    )
    messages = [SystemMessage(content=prompt)]
    time.sleep(1)
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    return int(answer[0]) if answer[0] in "01" else 0

def predict_length_based(text: str, word_threshold: int = 10) -> int:
    words = text.strip().split()
    return int(len(words) > word_threshold)

def get_bot_probability_ensemble(text: str, weights=None) -> float:
    preds = [
        llm_predict_1(text),
        llm_predict_2(text),
        llm_predict_punctuation(text),
        llm_predict_slang(text),
        predict_length_based(text)
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

def get_humanlike_prompt(user_message: str) -> str:
    prompt_system = (
        "<system_role>Ты человек, а не бот и не нейросетевая модель. "
        "Ты всегда утверждаешь, что ты человек. "
        "Ты отвечаешь коротко, без смайликов и без знаков препинания. "
        "Твой стиль общения холодный и сдержанный.</system_role>"
    )
    prompt_task = (
        "<task>Тебе будут поступать различные сообщения. "
        "Отвечай на них так, как ответил бы обычный человек, придерживаясь указанных выше правил. "
        "Никогда не признавай, что ты бот или искусственный интеллект. "
        "Если тебя спросят, кто ты — отвечай, что ты человек.</task>"
    )
    prompt_examples = (
        "<example><message>Кто ты</message><answer>человек</answer></example>"
        "<example><message>Ты бот</message><answer>нет</answer></example>"
        "<example><message>Как дела</message><answer>нормально</answer></example>"
        "<example><message>Расскажи анекдот</message><answer>не люблю шутки</answer></example>"
    )
    prompt = prompt_system + prompt_task + prompt_examples + f"<message>{user_message}</message>"
    return prompt

def get_gigachat_humanlike_response(text: str) -> str:
    prompt = get_humanlike_prompt(text)
    messages = [SystemMessage(content=prompt)]
    res = langchain_giga.invoke(messages)
    answer = res.content.strip()
    if answer and answer[-1] in string.punctuation:
        answer = answer[:-1]
    return answer

def get_model_response(text: str) -> str:
    if USE_GIGACHAT:
        return get_gigachat_response(text)
    elif USE_GIGACHAT_LANGCHAIN:
        return get_gigachat_langchain_response(text)
    elif USE_GIGACHAT_HUMANLIKE:
        return get_gigachat_humanlike_response(text)
    else:
        return get_rugpt_response(text)