import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="cpu", torch_dtype=torch.bfloat16)
print("Layers:", len(model.model.layers))
layer0 = model.model.layers[0]
print("Layer 0 modules:")
for name, module in layer0.named_modules():
    if '.' not in name and name != '':
        print(f"  {name}")
