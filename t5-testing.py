from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

text = "put carrot on plate"

input_ids = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

## Get encoder

# inputs = tokenizer("i got permission to begin a start up company by my own..</s>",return_tensors='tf')
# attn = inputs['attention_mask']

encoder_outputs = model.encoder(input_ids)
# encoder_outputs = model.encoder(input_ids, encoder_outputs=True)
np_e = np.array(encoder_outputs.last_hidden_state.detach().numpy())
print (np.array(np_e))