export WANDB_MODE=offline 
python mpttune/run.py generate --model mpt-7b \
    --weights mosaicml/mpt-7b \
    --lora_apply_dir mpt-7b-alpaca-dist \
    --max_new_tokens=200 \
    --use_cache \
    --do_sample \
    --prompt "List 5 elements in the periodic table:"