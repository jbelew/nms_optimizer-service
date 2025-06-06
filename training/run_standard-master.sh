#!/bin/bash

while true; do
    python generate_data.py --ship solar --category Hyperdrive --tech pulse --experimental &
    python generate_data.py --ship solar --category Hyperdrive --tech photonix --experimental &
    python generate_data.py --ship sentinel --category Hyperdrive --tech photonix --experimental &
    # Wait for all background jobs to finish
    wait
    # rsync -avz 192.168.0.15:/volume1/media/projects/nms_optimizer/training/* /home/jbelew/projects/nms_optimizer/nms_optimizer-service/training/generated_batches
    echo "Batch complete. Restarting..."
    sleep 5
done
