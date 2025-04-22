#!/bin/bash

while true; do
    python generate_data.py --ship standard --category Weaponry
    # python generate_data.py --ship standard --category Hyperdrive
    # python generate_data.py --ship standard --category "Defensive Systems"
    # python generate_data.py --ship living --category Weaponry
    # python generate_data.py --ship living --category Hyperdrive
    # python generate_data.py --ship living --category "Defensive Systems"
    # python generate_data.py --ship standard-mt --category Weaponry
    rsync -avz 192.168.0.15:/volume1/media/projects/nms_optimizer/training/* /home/jbelew/projects/nms_optimizer/nms_optimizer-service/training/generated_batches
    echo "Program exited. Restarting..."
    sleep 1
done