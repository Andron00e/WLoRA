clear
#--dataset_config "3.0.0" \
for r in 1 2 4
do
    for lr in 5e-5 8e-5 1e-4
    do
        for seed in 18
        do
            CUDA_VISIBLE_DEVICES=0 python ./nlg_experiment/run_nlg.py \
            --dataset_name xsum \
            --model_name_or_path facebook/bart-large \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 6 \
            --learning_rate $lr \
            --lr_scheduler_type linear \
            --warmup_steps 0 \
            --learning_rate_w 5e0 \
            --num_train_epochs 15 \
            --eval_strategy epoch \
            --save_strategy no \
            --ft_strategy WeightLoRA \
            --lora_r $r \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --k 20 \
            --predict_with_generate true \
            --max_source_length 256 \
            --max_target_length 160 \
            --val_max_target_length 256 \
            --max_train_samples 5000 \
            --max_val_samples 1000 \
            --num_beams 8 \
            --seed $seed \
            --report_to wandb # none or wandb
        done
    done
done