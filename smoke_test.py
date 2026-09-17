import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import resource

# MPS available
print(torch.backends.mps.is_available())

# Weights load successfully
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto", device_map="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Generate returns token successfully
n=3

model_inputs = tokenizer(["Baker made 133 cakes. If he sold 51 of them.How many more cakes did baker make than those he sold?"], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=100, num_return_sequences=n)
text = tokenizer.batch_decode(generated_ids)
for i in range(n):
    print(text[i])

# Peak memory is fine
#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)