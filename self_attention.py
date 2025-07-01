import torch
import torch.nn as nn

class self_attention_v1(nn.Module):
    def __init__(self, wei_q, wei_k, wei_v):
        super().__init__()
        self.W_query = wei_q.T
        self.W_key = wei_k.T
        self.W_value = wei_v.T

    def forward (self, x):
        query = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        atten_scores = (query @ torch.transpose(keys, 0, 1))/(keys.shape[-1]**0.5)
        atten_scores = torch.softmax(atten_scores, -1)
        context_vec = atten_scores @ values
        return context_vec
    
class self_attention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward (self, x):
        query = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        atten_scores = (query @ torch.transpose(keys, 0, 1))/(keys.shape[-1]**0.5)
        atten_scores = torch.softmax(atten_scores, -1)
        context_vec = atten_scores @ values
        return context_vec, self.W_query.weight, self.W_key.weight, self.W_value.weight
    
class causal_attention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        value = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill(self.mask.bool()[:tokens, :tokens], -torch.inf)
        atten_wei = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        atten_wei = self.dropout(atten_wei)
        context_vec = atten_wei @ value
        #print(context_vec)
        return context_vec
    
class multi_head_v1(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([causal_attention(d_in, d_out, context_length, dropout, qkv_bias=False) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

class Multihead_Attention_Main(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out%num_heads == 0), "Num heads should be divisible by d_out!"
        self.d_out=d_out
        self.num_heads = num_heads
        self.out_dim = d_out/num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        value = self.W_value(x)

        keys = keys.view(b, tokens, self.num_heads, self.out_dim)
        queries = queries.view(b, tokens, self.num_heads, self.out_dim)
        values = values.view(b, tokens, self.num_heads, self.out_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:tokens, :tokens]

        attn_wei = torch.softmax((attn_scores/keys.shape[-1]**0.5), dim = -1)
        context_vec = attn_wei @ values
        context_vec = context_vec.transpose(1,2)
        context_vec = context_vec.contiguous().view(b, tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.23, 0.56, 0.78],
        [0.63, 0.27, 0.94],
        [0.46, 0.59, 0.37],
        [0.54, 0.69, 0.47],
        [0/19, 0.39, 0.38]
    ]
)
torch.manual_seed(123)
d_in = 3
d_out = 1

g = self_attention_v2(d_in, d_out)
vec, wei_q, wei_k, wei_v = g(inputs)
f = self_attention_v1(wei_q, wei_k, wei_v)
f = f(inputs)

queries = g.W_query(inputs)
keys = g.W_key(inputs)
atten_scores = queries @ keys.T
atten_scores = torch.softmax((atten_scores/keys.shape[-1]),dim=-1)

    
context_length = atten_scores.shape[0]
mask_s = torch.tril(torch.ones(context_length, context_length))

masked_attn_wei = mask_s * atten_scores
#print(masked_attn_wei)

masked_sum = masked_attn_wei.sum(dim=1, keepdim=True)
masked_sum_norm = masked_attn_wei/masked_sum
#print(masked_sum_norm)

# new

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
mask =  atten_scores.masked_fill(mask.bool(), -torch.inf)
mask = torch.softmax(mask/keys.shape[-1]**0.5, dim=1)
# print(mask)

dropout = nn.Dropout(0.5)
# print(dropout(mask))

batch = torch.stack((inputs, inputs))
#print(batch)

ca = causal_attention(d_in, d_out, batch.shape[1], 0.0)
ca = ca(batch)

mha = multi_head_v1(d_in, d_out, batch.shape[1], 0.0, num_heads=2)
mha_vec = mha(batch)
print(mha_vec)
#print(ca)

