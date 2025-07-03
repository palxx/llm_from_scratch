import torch
import tiktoken
from gpt_architecture import GPTModel, generate_text_simple
torch.manual_seed(123)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

CONFIG_124_SMALL_PARAM = {
    "vocab_size":50257,
    "context_length":256,
    "emd_dim":768,
    "n_heads" :12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

start_context = "Every effort moves"
start_context2 = "I really like"
tokenizer = tiktoken.get_encoding('gpt2')
inputs = torch.stack((text_to_token_ids(start_context, tokenizer), 
                      text_to_token_ids(start_context2, tokenizer)), dim=0)

inputs = inputs.squeeze(1) 

targets = torch.tensor([[3626, 6100, 345],
                        [1107, 588, 11311]])

model = GPTModel(CONFIG_124_SMALL_PARAM)
with torch.no_grad():
    logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("Token Ids: ", token_ids)
    # print(f"targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    # print(f"Outputs batch 1: " 
    #       f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    # print(probas[0, [0, 1, 2], targets[0]])
    # print(probas[1, [0, 1, 2], targets[1]])

# model.eval()
# token_ids = generate_text_simple(model, 
#                                  text_to_token_ids(start_context, tokenizer),
#                                  max_new_tokns=10,
#                                  context_size=CONFIG_124_SMALL_PARAM["context_length"]
#                                  )
# print("Output text: ", token_ids_to_text(token_ids, tokenizer))

