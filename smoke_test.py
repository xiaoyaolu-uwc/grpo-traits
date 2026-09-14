import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import resource

# MPS available
print(torch.backends.mps.is_available())

# Weights load successfully
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto", device_map="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Generate returns token successfully
model_inputs = tokenizer(["Daniel Kokotajlo is"], return_tensors="pt").to(model.device)
print(model_inputs)
generated_ids = model.generate(**model_inputs, max_new_tokens=30)
text = tokenizer.batch_decode(generated_ids)[0]
print(text)

# Peak memory is fine
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)