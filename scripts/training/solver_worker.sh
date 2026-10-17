#!/bin/bash
PROJECT_DIR="/home/jbelew/projects/nms_optimizer-service"
cd "$PROJECT_DIR" || exit 1

ARG1="${1:-recommended}"
ARG2="${2:-}"
ARG3="${3:-}"
ARG4="${4:-}"

# Parse parameters based on whether ARG1 is a preset / ship-tech pair, or legacy (ship + target)
if [ "$ARG1" = "recommended" ] || [[ "$ARG1" == *"/"* ]]; then
    TARGET_SPEC="$ARG1"
    TARGET_SETS="${ARG2:-128}"
    WORKER_ID="${ARG3:-0}"
else
    TARGET_SPEC="$ARG1"
    TARGET_ARG2="$ARG2"
    TARGET_SETS="${ARG3:-64}"
    WORKER_ID="${ARG4:-0}"
fi

# Resolve targets into space-separated "ship/tech" pairs
RESOLVED_TARGETS=$(PYTHONPATH=. ./venv/bin/python scripts/training/resolve_targets.py "$TARGET_SPEC" "${TARGET_ARG2:-}")

if [ -z "$RESOLVED_TARGETS" ]; then
    echo "Error: No targets resolved from '$TARGET_SPEC' '${TARGET_ARG2:-}'."
    exit 1
fi

echo "Worker $WORKER_ID monitoring targets: $RESOLVED_TARGETS"
echo "Target sets per technology: $TARGET_SETS"

while true; do
    all_done=true
    active_targets=()
    summary=""

    for item in $RESOLVED_TARGETS; do
        ship="${item%%/*}"
        tech="${item##*/}"
        tech_dir="$PROJECT_DIR/scripts/training/generated_batches/$ship/$tech"
        mkdir -p "$tech_dir"
        count=$(find "$tech_dir" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l)
        summary="$summary $ship/$tech:$count"
        if [ "$count" -lt "$TARGET_SETS" ]; then
            all_done=false
            active_targets+=("$item")
        fi
    done

    if [ "$all_done" = true ]; then
        echo "Target of $TARGET_SETS data sets reached for all requested targets ($summary). Stopping solver."
        sleep 5
        break
    fi

    num_active=${#active_targets[@]}
    # Offset target processing order by WORKER_ID to balance work across targets
    offset=$(( WORKER_ID % num_active ))

    for idx in $(seq 0 $((num_active - 1))); do
        target_idx=$(( (offset + idx) % num_active ))
        item="${active_targets[$target_idx]}"
        ship="${item%%/*}"
        tech="${item##*/}"
        tech_dir="$PROJECT_DIR/scripts/training/generated_batches/$ship/$tech"
        count=$(find "$tech_dir" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l)
        if [ "$count" -lt "$TARGET_SETS" ]; then
            echo "[Worker $WORKER_ID | Progress: $count/$TARGET_SETS sets for $ship/$tech |$summary] Generating batch..."
            ./venv/bin/python -m scripts.training.generate_data --ship "$ship" --tech "$tech"
        fi
    done
    sleep 2
done
