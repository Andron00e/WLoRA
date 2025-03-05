clear
#for task_name in cola mnli mrpc qnli qqp rte sst2 stsb
for r in 4 8
do
    for lr in 8e-6 3e-5 5e-5 8e-5
    # for lr in 3e-5 5e-5 8e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    do
        for seed in 18
        do
            CUDA_VISIBLE_DEVICES=6 python ./glue_experiment/run_glue.py \
                --dataset_name glue \
                --task_name mrpc \
                --model_name_or_path microsoft/deberta-v3-base \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 6 \
                --learning_rate $lr \
                --warmup_steps 50 \
                --learning_rate_w 5e0 \
                --lr_scheduler_type linear \
                --num_train_epochs 30 \
                --eval_strategy epoch \
                --save_strategy no \
                --ft_strategy WeightLoRA \
                --lora_r $r \
                --lora_dropout 0.05 \
                --lora_alpha 32 \
                --use_fat true \
                --fat_step 10 \
                --max_fat_steps 2 \
                --lora_extention smart \
                --seed $seed \
                --do_eval true \
                --do_predict false \
                --report_to wandb # none or wandb
        done
    done
done