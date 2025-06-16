with open("cleaned_data/gutenberg.train", 'r', encoding='utf-8') as f:
    text = f.read().splitlines()

filtered_lines = []
for line in text:
    if line.startswith("*"):
        filtered_lines.append(line)

with open("test/gutenberg_after_filter_anomalies.txt", 'w', encoding='utf-8') as f:
    for line in filtered_lines:
        f.write(line + '\n')