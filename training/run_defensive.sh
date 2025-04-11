#!/bin/bash

while true; do
    python generate_data.py --category "Defensive Systems"
    echo "Program exited. Restarting..."
    sleep 1
done