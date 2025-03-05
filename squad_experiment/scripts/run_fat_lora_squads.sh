clear
#for dataset_name in squad squad_v2
for r in 4 8
do
    #for lr in 8e-5 5e-5 3e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    # for lr in 1e-4 3e-4 5e-4
    for lr in 8e-5
    do
        for seed in 52
        do
            CUDA_VISIBLE_DEVICES=7 python ./squad_experiment/run_squad.py \
                --dataset_name squad_v2 \
                --model_name_or_path microsoft/deberta-v3-base \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 6 \
                --learning_rate $lr \
                --warmup_steps 50 \
                --learning_rate_w 5e0 \
                --lr_scheduler_type linear \
                --num_train_epochs 10 \
                --eval_strategy epoch \
                --save_strategy no \
                --ft_strategy WeightLoRA \
                --lora_r $r \
                --lora_dropout 0.05 \
                --lora_alpha 32 \
                --use_fat true \
                --fat_step 5 \
                --max_fat_steps 2 \
                --lora_extention smart \
                --seed $seed \
                --do_eval true \
                --do_predict false \
                --report_to wandb # none or wandb
        done
    done
done