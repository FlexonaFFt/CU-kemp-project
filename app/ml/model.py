from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def get_model_response(text: str) -> str:
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    output = model.generate(
        input_ids, max_length=50, num_beams=5,
        do_sample=True, top_k=50, top_p=0.95,
        temperature=0.7, pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3, num_return_sequences=1, repetition_penalty=1.2
    )

    generated_text = tokenizer.decode(output[0], skip_spicial_tokens=True)
    if generated_text.startswith(text):
        generated_text = generated_text[len(text):].strip()
    generated_text = generated_text.split('\n')[0].strip()
    return generated_text
