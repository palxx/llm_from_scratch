import re
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#print("Total number of characters: " , len(raw_text))
p = raw_text[:50]

result = re.split(r'([:;!?_()",.\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
all_words = sorted(set(result))
all_words.extend(["<|endoftext|>", "<|unk|>"])
print(len(all_words))
#print(all_words[:20])

vocab = {token:integer for integer, token in enumerate(all_words)}
#print(vocab)

class tokenizer():
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprossed = re.split(r'([:;!?_()",.\']|--|\s)', text)
        preprossed = [item.strip() for item in preprossed if item.strip()]
        preprossed = [item if item in self.str_to_int else "<|unk|>" for item in preprossed]
        ids = [self.str_to_int[s] for s in preprossed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[s] for s in ids])
        text = re.sub(r'\s+([:;!?_()",.\']|--|\s)', r'\1', text)
        return text
    
tokenizerIns = tokenizer(vocab)
text = """hello how are you doing"""
ids = tokenizerIns.encode(text)
dids = tokenizerIns.decode(ids)
text1= "hey its me!"
text2 = "i am text2 don't know what to do?"
text_total = " <|endoftext|> ".join((text1, text2))
ids1= tokenizerIns.encode(text_total)
dids1 = tokenizerIns.decode(ids1)
print(ids1)
print(dids1)
    





