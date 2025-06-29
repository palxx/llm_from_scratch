from importlib.metadata import version
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

context_size = 4

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# tokenizer = tiktoken.get_encoding("gpt2") 
# enc_text = tokenizer.encode(raw_text)
# enc_text = enc_text[50:]

# for i in range(1, context_size+1):
#     context = enc_text[:i]
#     desired = enc_text[i]
#     print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.inputids = []
        self. target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.inputids.append(torch.tensor((input_chunk)))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
            return len(self.inputids)
        
    def __getitem__(self, idx):
            return self.inputids[idx], self.target_ids[idx]
        

def create_dataloader_v1(txt, batch_size=4, max_length = 256, stride=128, 
                        shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=2, shuffle=False)
data_iter = iter(dataloader)
input, targets = next(data_iter)
print(input.shape)
vocab_size=50237
embedding_dim = 256
token_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
token_embedding = token_embedding_layer(input)
print(token_embedding_layer(input).shape)

context_length = 4
pos_embedding_layer = nn.Embedding(context_length, embedding_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings())
print(token_embedding)
print(token_embedding+pos_embeddings)
#@

