import re
import json
from collections import Counter
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-name")
    args = parser.parse_args()


    base_dir = Path(__file__).parent / args.directory_name
    word_freq_comma_separated = Counter()
    word_freq = Counter()

    for file in base_dir.rglob("*"):
        with file.open("r", encoding="utf-8") as f:
            text = f.read()

        words_comma_separated = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())
        words = re.findall(r"\b[a-zA-ZÀ-ÿ']+\b", text.lower())

        word_freq_comma_separated.update(words_comma_separated)
        word_freq.update(words)

    sorted_word_freq_comma_separated = dict(word_freq_comma_separated.most_common())
    sorted_word_freq = dict(word_freq.most_common())

    word_frequency_json = {
        "total_number_of_words": sum(word_freq_comma_separated.values()),
        "words": sorted_word_freq_comma_separated,
    }
    with open("new_word_freq_comma_separated_100M.json", "w", encoding="utf-8") as f:
        json.dump(word_frequency_json, f, ensure_ascii=False, indent=4)

    word_frequency_json = {
        "total_number_of_words": sum(word_freq.values()),
        "words": sorted_word_freq,
    }

    with open("new_word_freq.json_100M", "w", encoding="utf-8") as f:
        json.dump(word_frequency_json, f, ensure_ascii=False, indent=4)

    print("Word frequencies saved")
