# !/bin/bash

# MODEL_PATH=$1
# MODEL_BASENAME=$(basename $MODEL_PATH)

# python -m lm_eval --model hf \
#     --model_args pretrained=$MODEL_PATH,backend="causal" \
#     --tasks blimp_supplement,blimp_filtered \
#     --device cuda:0 \
#     --batch_size 1 \
#     --log_samples \
#     --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json \
#     --trust_remote_code

# python -m lm_eval --model zipf_baseline \
# --model_args language=en,score_type=sum \
# --tasks blimp_supplement,blimp_filtered \
# --device cpu \
# --batch_size 1 \
# --log_samples \
# --output_path results/blimp/zipf_baseline/blimp_results.json

python -m lm_eval --model my_word_freq \
--model_args language=en,score_type=sum,comma_separation=f,strict_small=t,zipf=t \
--tasks blimp_supplement,blimp_filtered \
--device cpu \
--batch_size 1 \
--log_samples \
--output_path results/blimp/my_new_word_freq_strict_ablation/blimp_results.json

# ORDER=6
# STRICT="t"

# python3 -m lm_eval --model kenlm \
# --model_args order=$ORDER,strict=$STRICT \
# --tasks blimp_supplement,blimp_filtered \
# --device cpu \
# --batch_size 1 \
# --log_samples \
# --output_path results/blimp/kenlm-${ORDER}-gram-100M/blimp_results.json

# python3 -m lm_eval --model rnnlm \
# --tasks blimp_supplement,blimp_filtered \
# --device cpu \
# --batch_size 1 \
# --log_samples \
# --output_path results/blimp/rnn-superbpe-pretrained/blimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.

# blimp supplement este mai mic decat blimp filtered

# python -m lm_eval --model constant_model \
# --tasks blimp_supplement,blimp_filtered \
# --device cpu \
# --batch_size 1 \
# --log_samples \
# --output_path results/blimp/constant_model/blimp_results.json