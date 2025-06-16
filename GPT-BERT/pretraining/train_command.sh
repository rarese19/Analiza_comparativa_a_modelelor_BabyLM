python train_single_gpu.py \
  --train_path="../data/babylm_original_data_unigram_train_tokenized.bin" \
  --valid_path="../data/babylm_original_data_unigram_validation_tokenized.bin" \
  --name="original-data-unigram-gptremake-7ksteps-256local-32768global-4reduction-original-params" \
  --wandb_project="gpt-bert-sp-bpe" \
  --wandb_entity="rarese19-proiect" \
  --config_file="../configs/small.json" \
  --tokenizer_path="../tokenizers/tokenizer_unigram_original_data_10M.json" \
  --output_dir="../model_checkpoints" \
  --optimizer="lamb" \
  --hybrid_numerator=15 \
  --hybrid_denominator=16 \
  --seq_length=128 \
  --local_batch_size=256 \
  --global_batch_size=32768 \
  --batch_reduction=4 \
  --learning_rate=1.41e-2 \
  --max_steps=7812 \
  --ema_decay=0.999 \
  --validate_every=1000 \
  --validation_steps=1 \
  --log_stats_every=100 \
  --save_every=1000 \
  --warmup_proportion=0.016 \
  --cooldown_proportion=0.016 \
  --mask_p_start=0.3 \
  --mask_p_end=0.15 \
  --mask_random_p=0.1 \
  --mask_keep_p=0.1 \
  --weight_decay=0.1 \
  --optimizer_eps=1e-8 \
  --optimizer_beta1=0.9 \
  --optimizer_beta2=0.98 \
  --max_gradient=2.0 \
  --mixed_precision \
  --seed=42 \
  --n_special_tokens=16 \
  --z_loss_weight=1e-4




# python train_single_gpu.py \
#   --train_path="../data/babycosmofine_10M_train_tokenized.bin" \
#   --valid_path="../data/babycosmofine_10M_validation_tokenized.bin" \
#   --name="test-test-test" \
#   --wandb_project="gpt-bert-sp-bpe" \
#   --wandb_entity="rarese19-proiect" \
#   --config_file="../configs/small.json" \
#   --tokenizer_path="../tokenizers/tokenizer_10M.json" \
#   --output_dir="../model_checkpoints" \
#   --optimizer="lamb" \
#   --hybrid_numerator=15 \
#   --hybrid_denominator=16 \
#   --seq_length=64 \
#   --local_batch_size=256 \
#   --global_batch_size=4096 \
#   --batch_reduction=4 \
#   --learning_rate=1.75e-3 \
#   --max_steps=125000 \
#   --ema_decay=0.999 \
#   --validate_every=16000 \
#   --validation_steps=1 \
#   --log_stats_every=100 \
#   --save_every=16000 \
#   --warmup_proportion=0.016 \
#   --cooldown_proportion=0.016 \
#   --mask_p_start=0.3 \
#   --mask_p_end=0.15 \
#   --mask_random_p=0.1 \
#   --mask_keep_p=0.1 \
#   --weight_decay=0.1 \
#   --optimizer_eps=1e-8 \
#   --optimizer_beta1=0.9 \
#   --optimizer_beta2=0.98 \
#   --max_gradient=2.0 \
#   --mixed_precision \
#   --seed=42 \
#   --n_special_tokens=16 \
#   --z_loss_weight=1e-4

# -  --seq_length=128 \
# -  --global_batch_size=8192 \
# -  --learning_rate=3.5e-3 \
# -  --max_steps=31250 \
# -  --warmup_proportion=0.004 \
# -  --cooldown_proportion=0.004 \
# -  --validate_every=4000 \
# -  --save_every=4000 \
# +  --seq_length=64 \                    # ↘ half seq length  
# +  --global_batch_size=4096 \           # ↘ half global batch  
# +  --learning_rate=1.75e-3 \            # ↘ scale LR by ½ with batch  
# +  --max_steps=125000 \                 # ↗ ×4 steps to keep token budget  
# +  --warmup_proportion=0.001 \          # =125/125000 warmup steps  
# +  --cooldown_proportion=0.001 \        # =125/125000 cooldown steps  
# +  --validate_every=16000 \             # ≃12.8% of 125k  
# +  --save_every=16000 \                 # same cadence  


# python train_single_gpu.py \
#   --train_path="../data/babycosmofine_10M_train_tokenized.bin" \
#   --valid_path="../data/babycosmofine_10M_validation_tokenized.bin" \
#   --name="test-gptremake-7ksteps-256local-1024global-4reduction-original-params" \
#   --wandb_project="gpt-bert-sp-bpe" \
#   --wandb_entity="rarese19-proiect" \
#   --config_file="../configs/small.json" \
#   --tokenizer_path="../tokenizers/tokenizer_10M.json" \
#   --output_dir="../model_checkpoints" \
#   --optimizer="lamb" \
#   --hybrid_numerator=8 \
#   --hybrid_denominator=16 \
#   --seq_length=128 \
#   --local_batch_size=256 \
#   --global_batch_size=1024 \
#   --batch_reduction=4 \
#   --learning_rate=0.003 \
#   --max_steps=7812 \
#   --ema_decay=0.999 \
#   --validate_every=1000 \
#   --validation_steps=1 \
#   --log_stats_every=100 \
#   --save_every=1000 \
#   --warmup_proportion=0.05 \
#   --cooldown_proportion=0.05 \
#   --mask_p_start=0.25 \
#   --mask_p_end=0.15 \
#   --mask_random_p=0.1 \
#   --mask_keep_p=0.1 \
#   --weight_decay=0.05 \
#   --optimizer_eps=1e-8 \
#   --optimizer_beta1=0.9 \
#   --optimizer_beta2=0.98 \
#   --max_gradient=2.0 \
#   --mixed_precision \
#   --seed=42 \
#   --n_special_tokens=16 \
#   --z_loss_weight=1e-4