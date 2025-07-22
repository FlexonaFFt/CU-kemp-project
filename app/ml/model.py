from transformers import AutoTokenizer, AutoModelForCausalLM
from gigachat import GigaChat
import torch 
import os

GIGACHAT_API_KEY = 'OGU4MGQ2MjgtYjI3My00NzllLTlmY2ItZGU4NDNjNWJmYWM3OmM5ZmExNGJhLTQ5ODItNDA3Zi1iYzkwLWUwMWMwMzdkY2M1Zg=='
USE_GIGACHAT = True   

llm = GigaChat(credentials=GIGACHAT_API_KEY,
               verify_ssl_certs=False,
               model="GigaChat:latest") 

MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

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
    else:
        return get_rugpt_response(text)
