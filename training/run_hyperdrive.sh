#!/bin/bash

rm -rf generated_batches/

while true; do
    python generate_data.py --category Hyperdrive
    rsync -vrt --progress generated_batches/ /mnt/media/projects/nms_optimizer/training/
    echo "Program exited. Restarting..."
    sleep 1
done