clear
#for task_name in cola mnli mrpc qnli qqp rte sst2 stsb
for task_name in mrpc
do
    for r in 1 2 4 8
    do
        for seed in 18 52 1917
        do
            CUDA_VISIBLE_DEVICES=7 python ./glue_experiment/run_glue.py \
                --dataset_name glue \
                --task_name $task_name \
                --model_name_or_path microsoft/deberta-v3-base \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 6 \
                --learning_rate tuned \
                --lr_scheduler_type linear \
                --warmup_steps 100 \
                --num_train_epochs 30 \
                --eval_strategy epoch \
                --save_strategy no \
                --ft_strategy LoRA \
                --lora_r $r \
                --lora_alpha 32 \
                --lora_dropout 0.05 \
                --seed $seed \
                --do_eval true \
                --do_predict false \
                --report_to wandb # none or wandb
        done
    done
done