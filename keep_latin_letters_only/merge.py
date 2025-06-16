import sys
import nltk
from pathlib import Path

base_dir = Path(__file__).parent / "cleaned_data"

with open("cleaned_merged_data", "w", encoding="utf-8") as output_file:
    i = 1
    for file_path in base_dir.glob("*.train"):
        print(str(i) + ": " + str(file_path))
        i += 1
        with open(file_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                for sentence in nltk.sent_tokenize(line):
                    processed_text = " ".join(nltk.word_tokenize(sentence)).lower()
                    output_file.write(processed_text + "\n")
