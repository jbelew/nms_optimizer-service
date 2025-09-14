#!/bin/bash

SESSION_NAME="nms_solvers"
BASE_CMD="cd /home/jbelew/projects/nms_optimizer-service && ./venv/bin/python -m scripts.training.generate_data --ship corvette --category \"Defensive Systems\" --tech shield --solve_type max"

/usr/bin/tmux kill-session -t $SESSION_NAME 2>/dev/null
/usr/bin/tmux new-session -d -s $SESSION_NAME

# Setup a 3-pane layout: three stacked rows
/usr/bin/tmux split-window -v
/usr/bin/tmux split-window -v

# Send commands to each pane explicitly by pane index
# Pane 0: pulse
/usr/bin/tmux send-keys -t $SESSION_NAME:0.0 "while true; do $BASE_CMD; echo \"Restarting...\"; sleep 5; done" C-m

# Pane 1: solar photonix
/usr/bin/tmux send-keys -t $SESSION_NAME:0.1 "while true; do $BASE_CMD; echo \"Restarting...\"; sleep 5; done" C-m

# Pane 2: sentinel photonix
/usr/bin/tmux send-keys -t $SESSION_NAME:0.2 "while true; do $BASE_CMD; echo \"Restarting...\"; sleep 5; done" C-m
