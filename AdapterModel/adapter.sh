#!/bin/bash


#TASKS=(crisis_8 crisis_9 crisis_13)
TASKS=(crisis_8_multi)
LANGUAGES=(en en en)
#TASKS=(crisis_8_multi)
#TASKS=(crisis_9_multi)
#TASKS=(crisis_13_multi)


SEEDS=(42 30 100 3407 0)

for j in "${!SEEDS[@]}"; do
  for ((i=0; i<${#TASKS[@]}; i++)); do

    seed=${SEEDS[j]}
    echo "SEED: $seed"
    echo "TASK: ${TASKS[i]}"
    echo "LANGUAGE: ${LANGUAGES[i]}"


    CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29506 run_seq2seq.py \
        --task_name ${TASKS[i]} \
        --eval_dataset_name ${TASKS[i]} \
        --test_dataset_name "ndtv1,ndtv2,ndtv3,${TASKS[i]}" \
        --output_dir ${TASKS[i]}/$seed/ \
        --do_train true \
        --do_eval true \
        --do_test true \
        --warmup_ratio 0.1 \
        --model_name_or_path t5-base \
        --tokenizer_name t5-base \
        --save_total_limit 1 \
        --load_best_model_at_end true \
        --metric_for_best_model f1_weighted \
        --greater_is_better true \
        --evaluation_strategy epoch \
        --non_linearity gelu_new \
        --overwrite_output_dir true \
        --max_source_length 256 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --split_validation_test false \
        --num_train_epochs 200 \
        --dataset_config_name ${LANGUAGES[i]} \
        --eval_dataset_config_name ${LANGUAGES[i]} \
        --test_dataset_config_name "${LANGUAGES[i]},${LANGUAGES[i]},${LANGUAGES[i]},${LANGUAGES[i]}" \
        --predict_with_generate true \
        --compute_memory true \
        --seed $seed \
        --data_seed $seed \
        --compute_time true \
        --print_num_parameters true \
        --learning_rate 3e-4 \
        --lr_schedule default \
        --early_stopping_patience 0\
        --cache_dir cache/ \
        --logging_dir runs/ \
        --dataloader_num_workers 2\
        --save_strategy epoch \
        --logging_strategy epoch \
        --save_steps 2000 \
        --logging_steps 2000 \
        --add_layer_norm_before_adapter false\
        --add_layer_norm_after_adapter false\
        --adapter_config_name adapter\
        --train_task_adapters true\
        --task_reduction_factor 32\
        --unfreeze_lm_head false\
        --unfreeze_layer_norms true;
  done;
done;
