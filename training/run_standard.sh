#!/bin/bash

rm -rf generated_batches/

while true; do
    # python generate_data.py --ship standard --category Weaponry
    python generate_data.py --ship standard --category Hyperdrive
    python generate_data.py --ship solar --category Hyperdrive
    # python generate_data.py --ship living --category Weaponry
    # python generate_data.py --ship living --category Hyperdrive
    # python generate_data.py --ship living --category "Defensive Systems"
    # python generate_data.py --ship standard-mt --category "Weaponry"
    # python generate_data.py --ship standard-mt --category "Scanners"
    # python generate_data.py --ship standard-mt --category "Mining"
    rsync -vrt --progress generated_batches/ /mnt/media/projects/nms_optimizer/training/
    echo "Program exited. Restarting..."
    sleep 1
done