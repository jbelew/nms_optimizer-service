#!/bin/bash

SESSION_NAME="nms_solvers"
PROJECT_DIR="/home/jbelew/projects/nms_optimizer-service"

# Configurable options with sensible defaults:
# Usage:
#   ./run_solvers_tmux.sh [recommended|targets] [target_sets] [num_solvers]
#   ./run_solvers_tmux.sh [ship] [techs/category] [target_sets] [num_solvers]

ARG1="${1:-recommended}"

if [ "$ARG1" = "recommended" ] || [[ "$ARG1" == *"/"* ]]; then
    TARGET_SPEC="$ARG1"
    TARGET_SETS="${2:-128}"
    NUM_SOLVERS="${3:-4}"
    LEGACY=false
else
    SHIP="$ARG1"
    TECH="${2:-pulse}"
    TARGET_SETS="${3:-64}"
    NUM_SOLVERS="${4:-4}"
    LEGACY=true
fi

WORKER_SCRIPT="$PROJECT_DIR/scripts/training/solver_worker.sh"
chmod +x "$WORKER_SCRIPT"

/usr/bin/tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Construct worker invocation command for a given worker index $i
get_worker_cmd() {
    local worker_id="$1"
    if [ "$LEGACY" = true ]; then
        echo "$WORKER_SCRIPT \"$SHIP\" \"$TECH\" \"$TARGET_SETS\" \"$worker_id\""
    else
        echo "$WORKER_SCRIPT \"$TARGET_SPEC\" \"$TARGET_SETS\" \"$worker_id\""
    fi
}

# Create first window
first_cmd=$(get_worker_cmd 0)
/usr/bin/tmux new-session -d -s "$SESSION_NAME" -n "solver-0" "$first_cmd"

# Create additional windows for the remaining solvers
for i in $(seq 1 $((NUM_SOLVERS-1))); do
    next_cmd=$(get_worker_cmd "$i")
    /usr/bin/tmux new-window -t "$SESSION_NAME:$i" -n "solver-$i" "$next_cmd"
done

echo "Tmux session '$SESSION_NAME' created with $NUM_SOLVERS parallel solvers."
if [ "$LEGACY" = true ]; then
    echo "Generating $TARGET_SETS data sets per tech for ship='$SHIP', target='$TECH'."
else
    echo "Generating $TARGET_SETS data sets per target for: $TARGET_SPEC."
fi
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Switch between windows using Ctrl+b <window-number> or Ctrl+b n/p."
