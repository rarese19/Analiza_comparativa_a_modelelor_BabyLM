import random

input_file = "babylm_raw_data.jsonl" 

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

n_valid = int(len(lines) * 0.1 + 0.5)
valid, train = lines[:n_valid], lines[n_valid:]

with open("babylm_raw_data_validation.jsonl", "w", encoding="utf-8") as f:
    f.writelines(valid)
with open("babylm_raw_data_train.jsonl", "w", encoding="utf-8") as f:
    f.writelines(train)

print(f"Total records:     {len(lines)}")
print(f" → Train:          {len(train)}")
print(f" → Validation:     {len(valid)}")
