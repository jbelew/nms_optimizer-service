#!/bin/bash

SESSION_NAME="nms_solvers"
BASE_CMD="cd /home/jbelew/projects/nms_optimizer-service && ./venv/bin/python -m scripts.training.generate_data --ship corvette --category \"Hyperdrive\" --tech shield"
NUM_SOLVERS=4

/usr/bin/tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create the first window and run the command
/usr/bin/tmux new-session -d -s $SESSION_NAME -n "solver-0" "while true; do $BASE_CMD; echo \"Restarting...\"; sleep 5; done"

# Create additional windows for the remaining solvers
for i in $(seq 1 $((NUM_SOLVERS-1))); do
    /usr/bin/tmux new-window -t $SESSION_NAME:$i -n "solver-$i" "while true; do $BASE_CMD; echo \"Restarting...\"; sleep 5; done"
done

echo "Tmux session '$SESSION_NAME' created with $NUM_SOLVERS windows."
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Switch between windows using Ctrl+b <window-number> or Ctrl+b n/p."
