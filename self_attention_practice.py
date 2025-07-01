import torch

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
d_out = 2
embd_dim = d_in
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

queries = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value

attention_scores = queries @ torch.transpose(keys, 0, 1)
attention_scores = torch.softmax((attention_scores/embd_dim**0.5), 1)
print(attention_scores.shape)
context_vec = attention_scores @ values
print(context_vec)


# attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])

# for j in range(len(inputs)):
#      for i in range(len(inputs)):
#         row = attention_scores[j]
#         row[i] = torch.dot(inputs[j], inputs[i])
# print(attention_scores)
# print(inputs @ torch.transpose(inputs, 0, 1))


# def softmax(x):
#     return torch.exp(x)/torch.exp(x).sum(dim=0)


# attention_scores = torch.softmax(attention_scores, dim=1)
# #print(attention_scores)

# input_len = range(len(inputs))

# context_vec = torch.zeros(inputs.shape)
# #print(context_vec)
# input_len = range(len(inputs))
# for j in input_len:
#     for i in input_len:
#         context_vec[j] += attention_scores[j][i] * inputs[i]
#         #print(context_vec)

# print(context_vec)