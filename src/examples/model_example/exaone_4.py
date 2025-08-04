from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.cuda.empty_cache()

model_name = "LGAI-EXAONE/EXAONE-4.0-32B"
#model_name= "LGAI-EXAONE/EXAONE-4.0-1.2B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# choose your prompt
#prompt = "Explain how wonderful you are"
#prompt = "Explica lo increíble que eres"
prompt = "너 김수미 배우 알아?"

messages = [
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))