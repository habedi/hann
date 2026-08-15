#!/bin/bash
set -euo pipefail

# This script checks the recall values an example run printed against recorded minimums.
# Usage: check_recall.sh <thresholds-file> <index-name> <log-file>
# The thresholds file holds lines of the form: <index-name> <dataset> <minimum>.
# The exit code is 1 when any dataset is below its minimum or has no threshold.

THRESHOLDS="$1"
INDEX="$2"
LOG="$3"

fail=0
found=0
while read -r dataset recall; do
    found=1
    min=$(awk -v i="$INDEX" -v d="$dataset" '$1 == i && $2 == d {print $3}' "$THRESHOLDS")
    if [ -z "$min" ]; then
        echo "MISSING $INDEX $dataset recall=$recall (no threshold recorded)"
        fail=1
        continue
    fi
    if awk -v r="$recall" -v m="$min" 'BEGIN { exit !(r >= m) }'; then
        echo "PASS $INDEX $dataset recall=$recall min=$min"
    else
        echo "FAIL $INDEX $dataset recall=$recall min=$min"
        fail=1
    fi
done < <(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | awk '/Loading dataset:/ {d = $NF} /Average Recall@/ {print d, $NF}')

if [ "$found" = "0" ]; then
    echo "FAIL $INDEX: no recall lines found in $LOG"
    fail=1
fi
exit $fail
