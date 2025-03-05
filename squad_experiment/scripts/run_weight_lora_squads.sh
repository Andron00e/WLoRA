clear
#for dataset_name in squad squad_v2
for r in 4 8
do
    #for lr in 8e-5 5e-5 3e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    for lr in 1e-4
    do
        for seed in 52
        do
            CUDA_VISIBLE_DEVICES=6 python ./squad_experiment/run_squad.py \
            --dataset_name squad_v2 \
            --model_name_or_path microsoft/deberta-v3-base \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 6 \
            --learning_rate $lr \
            --learning_rate_w 5e0 \
            --lr_scheduler_type linear \
            --num_train_epochs 10 \
            --eval_strategy epoch \
            --save_strategy no \
            --ft_strategy WeightLoRA \
            --lora_r $r \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --k 20 \
            --fat_step 1 \
            --seed $seed \
            --do_eval true \
            --do_predict false \
            --report_to wandb # none or wandb
        done
    done
done