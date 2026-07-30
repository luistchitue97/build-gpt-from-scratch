import tiktoken

text = "hello, encode this text"
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(text)

print(enc_text)