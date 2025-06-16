import json
from collections import Counter

with open("word_freq.json", "r", encoding="utf-8") as f:
    word_freq = json.load(f)

print("Number of distinct words in dataset without separating by comma: ", word_freq.get("total_number_of_words"))

with open("word_freq_comma_separated.json", "r", encoding="utf-8") as f:
    word_freq_comma_separated = json.load(f)

print("Number of distinct words in dataset with separating by comma: ", word_freq_comma_separated.get("total_number_of_words"))

print(word_freq.get("words").get("fraser"))
print(word_freq_comma_separated.get("words").get("fraser"))
