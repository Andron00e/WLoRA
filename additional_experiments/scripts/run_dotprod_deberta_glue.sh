for ((k=0;k<=35;k++));
do
    CUDA_VISIBLE_DEVICES=1 python ./additional_experiments/run_dotprod.py \
        --dataset_name glue \
        --task_name mnli \
        --model_name_or_path microsoft/deberta-v3-base \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 6 \
        --learning_rate 8e-4 \
        --lr_scheduler_type linear \
        --max_steps 256 \
        --lora_r 8 \
        --eval_strategy no \
        --save_strategy no \
        --do_eval false \
        --seed 52 \
        --data_seed 52 \
        --k $k \
        --report_to none
done