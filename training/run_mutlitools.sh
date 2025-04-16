#!/bin/bash

# File bump
rsync -vrt --progress generated_batches/ /mnt/media/projects/nms_optimizer/training/
rm -rf generated_batches/

while true; do
    python generate_data.py --ship standard-mt --category Mining
    python generate_data.py --ship standard-mt --category Scanners
    python generate_data.py --ship standard-mt --category Weaponry
    python generate_data.py --ship atlantid --category Mining
    rsync -vrt --progress generated_batches/ /mnt/media/projects/nms_optimizer/training/
    echo "Program exited. Restarting..."
    sleep 1
done