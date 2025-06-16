import unicode_script_data
import unicodedata

import re
import argparse
import os

# import the lru_cache decorator from functools
from functools import lru_cache


emoji_punctuation = [
    # VARIATION SELECTORS
    u"\uFE00", u"\uFE01", u"\uFE02", u"\uFE03", u"\uFE04", u"\uFE05", u"\uFE06", u"\uFE07", u"\uFE08", u"\uFE09", u"\uFE0A", u"\uFE0B", u"\uFE0C", u"\uFE0D", u"\uFE0E", u"\uFE0F", 
    # ZERO WIDTH JOINER (ZWJ)
    u"\u200D",
]

@lru_cache(maxsize=1024)
def is_latin_or_numeric_or_punctuation(character):
    script, cat = unicode_script_data.script_cat(character)
    return script == 'Common' or script == 'Latin' or cat[0] in 'NSPZC' or character in emoji_punctuation

text = """
Here is a text with some non-Latin characters: 你好, мир, नमस्ते, العالم
Here is some text with only Latin characters (including accented versions of Latin characters): abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ àâäéèêëîïôöùûüÿ
Here are voyels with tilde which do not pass the regular expression: ãẽĩõũỹ
Here are some emojis, too: 😀🙃😉😍🤔🤷🏻‍♀️👩🏻‍💻
Here are some complex emojis which use ZWJ (like family emojis or skin tones): 👩🏻‍👩🏾‍👦🏿👨🏻‍👩🏻‍👧🏼👩🏻‍👩🏻‍👧🏼👨🏼‍👩🏼‍👧🏽‍👦🏿
"""

def remove_non_latin(text, filename):
    # normalize the text to NFKC form, which will reunite any decomposed characters into their precomposed form
    text = unicodedata.normalize('NFKC', text)

    if (filename == "childes.train"):
        text = re.sub(r'^\*\w+:\s*', '', text, flags=re.MULTILINE)

        text = re.sub(r'^=+\s+.*$', '', text, flags=re.MULTILINE)

        text = re.sub(r'\[(.*?)\]', r'\1', text)

        text = re.sub(r'\]', '', text)

        text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)

    if (filename == "gutenberg.train"):
        text = re.sub(r'^\*\s+\*.*$', '', text, flags=re.MULTILINE)

        text = re.sub(r'^\*+\s*(.*?)\s*\**$', r'\1', text, flags=re.MULTILINE)

        text = re.sub(r'^=+\s+.*$', '', text, flags=re.MULTILINE)

        text = re.sub(r'^\s*_+\s*$', '', text, flags=re.MULTILINE)

        text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)

    if (filename == "simple_wiki.train"):
        text = re.sub(r'^\s*=[=\s]*(.+?)\s*[=\s]*$', r'\1', text, flags=re.MULTILINE)

        text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)

    if (filename == "switchboard.train"):
        text = re.sub(r'^\w:\s*', '', text, flags=re.MULTILINE)

        text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)

    if (filename == "open_subtitles.train"):
        text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)

        text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)

    # this regular expression will match any character that is not Latin, numeric, or punctuation
    non_latin_regex = re.compile(r'[^-a-zàâäéèêëîïôöùûüÿ0-9,;.:!?<>\(\)\[\]\{\}\s]', re.UNICODE | re.IGNORECASE)

    # replace every non-Latin character with the output of the "replace_char_if_non_latin" function
    return non_latin_regex.sub(replace_char_if_non_latin, text)

def replace_char_if_non_latin(match):
    # match.group(0) will return the matched character
    match_char = match.group(0)

    # if the character is Latin or numeric or punctuation, return it as-is
    # otherwise, return an the replacement character of unicode U+FFFD (REPLACEMENT CHARACTER)

    # # debug code:
    # if is_latin_or_numeric_or_punctuation(match.group(0)):
    #     print("ALLOWING: " + match.group(0) + " " + str(ord(match.group(0))))
    # else:
    #     print("CENSURING: " + match.group(0) + " " + str(ord(match.group(0))))

    return match_char if is_latin_or_numeric_or_punctuation(match_char) else '\uFFFD'

# print(text)
# print(remove_non_latin(text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help="The file to process")
    parser.add_argument('--output_file', help="The output file name")

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    cleaned_text = remove_non_latin(text, os.path.basename(args.input_file))

    output_folder = os.path.dirname(args.output_file)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

#print('')
#print(unicode_script_data._compile_scripts_txt())