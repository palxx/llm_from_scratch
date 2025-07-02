import torch
import torch.nn as nn
import tiktoken
torch.manual_seed(123)

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embd = nn.Embedding(cfg["vocab_size"], cfg["emd_dim"])
        self.pos_emd = nn.Embedding(cfg["context_length"], cfg["emd_dim"])
        self.drop_emd = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[transformerBlock(cfg) for _ in range(cfg["n_heads"])]
        )
        self.final_norm = LayerNorm(cfg["emd_dim"])
        self.out_head = nn.Linear(cfg["emd_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_emd = self.token_embd(x)
        pos_emd = self.pos_emd(torch.arange(seq_len, device=x.device))
        all_tok = tok_emd+pos_emd
        all_tok = self.drop_emd(all_tok)
        all_tok = self.trf_blocks(all_tok)
        all_tok = self.final_norm(all_tok)
        logits = self.out_head(all_tok)
        return logits
    
class transformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = LayerNorm(cfg["emd_dim"])
        self.masked_multi_attn = Multihead_Attention_Main(cfg["emd_dim"], cfg["emd_dim"], cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.layer_norm_2 = LayerNorm(cfg["emd_dim"])
        self.ffw = FeedForward(cfg)

    def forward(self,x):
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.masked_multi_attn(x)
        x = self.dropout(x)
        x = x+shortcut

        shortcut =x 
        x= self.layer_norm_2(x)
        x = self.ffw(x)
        x =self.dropout(x)
        x = x+shortcut
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emd_dim))
        self.shift = nn.Parameter(torch.zeros(emd_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x+0.044715 * torch.pow(x,3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emd_dim"], 4*cfg["emd_dim"]), 
                                    GELU(),
                                    nn.Linear(4*cfg["emd_dim"],cfg["emd_dim"]))
        
    def forward(self, x):
        return self.layers(x)

class Multihead_Attention_Main(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "Num heads should be divisible by d_out!"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.out_dim = int(d_out/num_heads)
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
        values = value.view(b, tokens, self.num_heads, self.out_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:tokens, :tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_wei = torch.softmax((attn_scores/keys.shape[-1]**0.5), dim = -1)
        context_vec = attn_wei @ values
        context_vec = context_vec.transpose(1,2)
        context_vec = context_vec.contiguous().view(b, tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    
class ExDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(3, layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # print(f'this is x shape {x.shape} and layer output {layer_output.shape}')
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x+layer_output
            else:
                x = layer_output
        return x

def print_grad(model, x):
    output = model(x)
    #print(output)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} haa a gradient mean of {param.grad.abs().mean()}")

CONFIG_124PARAM = {
    "vocab_size":50257,
    "context_length":1024,
    "emd_dim":768,
    "n_heads" :12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}   

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1), dtype=float))
batch.append(torch.tensor(tokenizer.encode(txt2), dtype=float))

batch = torch.stack(batch, dim=0)

# model = GPTModel(CONFIG_124PARAM)
# logits = model(batch)
# print(logits.shape)

# ln = LayerNorm(emd_dim=4)
# out = ln(batch)
# print(out.mean(dim=-1, keepdim=True))
# print(out.var(dim=-1,unbiased=False, keepdim=True))

# ffn = FeedForward(CONFIG_124PARAM)
# x = torch.rand(2,3,768)
# out = ffn(x)
# print(out.shape)

# layer_sizes = [3,3,3,3,3,1]
# sam_in = torch.tensor([[1., 0.,-1.]])
# torch.manual_seed(123)
# model_wot_sc = ExDeepNeuralNetwork(layer_sizes, use_shortcut=True)
# print_grad(model_wot_sc, sam_in)

x = torch.rand(2,4,768)
block = transformerBlock(CONFIG_124PARAM)
outt = block(x)
print("Input shape: ", x.shape)
print("Output : ", outt.shape)