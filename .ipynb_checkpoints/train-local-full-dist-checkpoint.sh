export WANDB_MODE=offline 
export WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 mpttune/run.py finetune  \
  --model=mpt-7b  \
  --weights=mosaicml/mpt-7b  \
  --dataset=./alpaca_data_cleaned.json  \
  --data_type=alpaca  \
  --lora_out_dir=./mpt-7b-alpaca-dist/  \
  --mbatch_size=16  \
  --batch_size=128   \
  --epochs=1   \
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