#!/bin/bash
# Usage: bash hpc/submit.sh hpc/all_instances.txt 8 [sbatch resource overrides]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PARAMS="${1:-hpc/all_instances.txt}"
MAX_PARALLEL="${2:-8}"
CHUNK_SIZE="${HPC_ARRAY_SIZE:-1000}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid concurrency" >&2; exit 1; }
[[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid HPC_ARRAY_SIZE" >&2; exit 1; }
[[ -f "$PARAMS" ]] || { echo "Missing params: $PARAMS" >&2; exit 1; }
[[ "$PARAMS" = /* ]] || PARAMS="$ROOT/$PARAMS"
if grep -q '^[[:space:]]*$' "$PARAMS"; then
    echo "Blank parameter rows are not allowed" >&2; exit 1
fi
COUNT=$(awk 'END {print NR}' "$PARAMS")
(( COUNT > 0 )) || { echo "Empty parameter file" >&2; exit 1; }
mkdir -p "$ROOT/hpc/logs" "$ROOT/hpc/submissions"
# Freeze params so a queued array does not see later edits to the source file.
SNAPSHOT=$(mktemp "$ROOT/hpc/submissions/params.XXXXXXXX.txt")
cp -- "$PARAMS" "$SNAPSHOT"
chmod a-w "$SNAPSHOT"
echo "Submitting $COUNT tasks, at most $MAX_PARALLEL concurrently; snapshot=$SNAPSHOT"
PREVIOUS=""
for (( OFFSET=0; OFFSET<COUNT; OFFSET+=CHUNK_SIZE )); do
    LENGTH=$((COUNT-OFFSET))
    (( LENGTH <= CHUNK_SIZE )) || LENGTH="$CHUNK_SIZE"
    DEPENDENCY=()
    if [[ -n "$PREVIOUS" ]]; then
        DEPENDENCY=(--dependency="afterany:$PREVIOUS")
    fi
    JOB=$(sbatch --parsable --array="0-$((LENGTH-1))%$MAX_PARALLEL" \
        --output="$ROOT/hpc/logs/%x_%A_%a.out" \
        --error="$ROOT/hpc/logs/%x_%A_%a.err" \
        "${DEPENDENCY[@]}" "$@" "$ROOT/hpc/job_submit.sh" "$ROOT" "$SNAPSHOT" "$OFFSET")
    PREVIOUS="${JOB%%;*}"
    echo "Job $PREVIOUS: global tasks $OFFSET-$((OFFSET+LENGTH-1))"
    printf '%s\t%s\t%s\n' "$PREVIOUS" "$OFFSET" "$LENGTH" >> "$SNAPSHOT.jobs.tsv"
done
