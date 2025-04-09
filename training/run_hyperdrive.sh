#!/bin/bash

while true; do
    python generate_data.py --category Hyperdrive
    echo "Program exited. Restarting..."
    sleep 1
done