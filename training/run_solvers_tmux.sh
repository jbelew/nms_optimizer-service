#!/bin/bash

SESSION_NAME="nms_solvers"

/usr/bin/tmux kill-session -t $SESSION_NAME 2>/dev/null
/usr/bin/tmux new-session -d -s $SESSION_NAME

# Pane 0: pulse
/usr/bin/tmux send-keys -t $SESSION_NAME 'while true; do python generate_data.py --ship solar --category Hyperdrive --tech pulse --experimental; echo "Restarting..."; sleep 5; done' C-m

# Pane 1: solar photonix
/usr/bin/tmux split-window -h -t $SESSION_NAME
/usr/bin/tmux send-keys -t $SESSION_NAME 'while true; do python generate_data.py --ship solar --category Hyperdrive --tech photonix --experimental; echo "Restarting..."; sleep 5; done' C-m

# Pane 2: sentinel photonix
/usr/bin/tmux select-pane -t 0
/usr/bin/tmux split-window -v -t $SESSION_NAME
/usr/bin/tmux send-keys -t $SESSION_NAME 'while true; do python generate_data.py --ship sentinel --category Hyperdrive --tech photonix --experimental; echo "Restarting..."; sleep 5; done' C-m
