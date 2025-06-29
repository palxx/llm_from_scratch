#import urllib.request
#url = ("https://raw.githubusercontent.com/rasbt/"
#       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#       "the-verdict.txt")
#file_path = "the-verdict.txt"
#urllib.request.urlretrieve(url, file_path)

from importlib.metadata import version
import tiktoken
#print("tiktoken v", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Akwirw ier")
inte = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(inte)
d = tokenizer.decode(inte)
print(d)