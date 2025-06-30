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

attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])

for j in range(len(inputs)):
     for i in range(len(inputs)):
        row = attention_scores[j]
        row[i] = torch.dot(inputs[j], inputs[i])

#print(attention_scores)


def softmax(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)


attention_scores = torch.softmax(attention_scores, dim=1)
print(attention_scores)

input_len = range(len(inputs))

context_vec = torch.zeros(inputs.shape)
print(context_vec)
input_len = range(len(inputs))
for j in input_len:
    for i in input_len:
        context_vec[j] += attention_scores[j][i] * inputs[i]
        print(context_vec)

print(context_vec)