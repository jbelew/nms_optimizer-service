#!/bin/bash

while true; do
    python generate_data.py --category Hyperdrive
    rsync -avz 192.168.0.15:/volume1/media/projects/nms_optimizer/training/* /home/jbelew/projects/nms_optimizer/nms_optimizer-service/training/generated_batches
    python train_model.py \
        --category Hyperdrive \
        --ship standard \
        --width 4 \
        --height 3 \
        --scheduler_gamma=0.7 \
        --wd 1e-5 \
        --epochs 200 \
        --batch_size 64 \
        --val_split 0.15 \
        --es_patience 12 \
        --es_metric val_loss \
        --log_dir runs_early_stopping \
        --model_dir trained_models \
        --data_source_dir generated_batches 
    echo "Program exited. Restarting..."
    sleep 1
done
