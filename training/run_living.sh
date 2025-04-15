#!/bin/bash

rsync -vrt --progress generated_batches/ /mnt/media/projects/nms_optimizer/training/
rm -rf generated_batches/

while true; do
    python generate_data.py --ship living --category Weaponry
    python generate_data.py --ship living --category Hyperdrive
    python generate_data.py --ship living --category "Defensive Systems"
    rsync -vrt --progress generated_batches/ /mnt/media/projects/nms_optimizer/training/
    echo "Program exited. Restarting..."
    sleep 1
done