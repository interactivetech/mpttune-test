export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=1234  mpttune/run.py finetune  \
  --model=mpt-7b  \
  --weights=mosaicml/mpt-7b  \
  --dataset=./alpaca_data_cleaned.json  \
  --data_type=alpaca  \
  --lora_out_dir=./mpt-7b-alpaca/  \
  --mbatch_size=32  \
  --batch_size=32   \
  --epochs=3   \
  --lr=3e-4   \
  --cutoff_len=256  \
  --lora_r=8   \
  --lora_alpha=16  \
  --lora_dropout=0.05  \
  --warmup_steps=5  \
  --save_steps=50   \
  --save_total_limit=3  \
  --logging_steps=5  \
  --target_modules='["Wqkv"]'