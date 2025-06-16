python blimp.py \
    --input_path="./data_supplemental" \
    --output_dir="blimp_supplemental_results/gptbert-remake-clean-slate" \
    --tokenizer_path="../../tokenizers/tokenizer_unigram_original_data_10M.json" \
    --model_path_or_name="../../model_checkpoints/original-data-unigram-gptremake-7ksteps-256local-32768global-4reduction-original-params_15_16.bin" \
    --config_file="../../configs/small.json" \
    --backend="mlm_shift" \
    --batch_size=64 \
    --architecture="extra" \
    --predict
