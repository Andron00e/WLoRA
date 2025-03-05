clear
#--dataset_config "3.0.0" \
for r in 1 2 4 8
do
    for seed in 18 52 1917
    do
        CUDA_VISIBLE_DEVICES=1 python ./nlg_experiment/run_nlg.py \
        --dataset_name xsum \
        --model_name_or_path facebook/bart-large \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32  \
        --gradient_accumulation_steps 6 \
        --learning_rate tuned \
        --lr_scheduler_type linear \
        --warmup_steps 100 \
        --num_train_epochs 20 \
        --eval_strategy epoch \
        --save_strategy no \
        --ft_strategy LoRA \
        --lora_r $r \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --seed $seed \
        --predict_with_generate true \
        --max_source_length 128 \
        --max_target_length 64 \
        --val_max_target_length 128 \
        --max_train_samples 7000 \
        --max_val_samples 1000 \
        --num_beams 8 \
        --do_eval true \
        --do_predict false \
        --report_to wandb # none or wandb 
    done
done
done