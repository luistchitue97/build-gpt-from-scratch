import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-veredict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-veredict.txt", "r", encoding="utf-8") as f:
    RAW_TEXT = f.read()