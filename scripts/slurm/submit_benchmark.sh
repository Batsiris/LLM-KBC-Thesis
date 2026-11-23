#!/bin/bash
# Run on the login node from the repository root; never downloads or runs inference.
# Usage: bash scripts/slurm/submit_benchmark.sh ministral|granite|gemma smoke|full [tag]
# BENCH_SHOTS: unset/all selects 0,3,5%1; 0, 3 or 5 selects only that task.
set -Eeuo pipefail
trap 'rc=$?; printf "ERROR file=%s line=%s exit=%s command=%q\n" "${BASH_SOURCE[0]}" "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR
[[ $# -ge 2 && $# -le 3 ]] || { echo "Expected ministral|granite|gemma, smoke|full, optional run tag" >&2; exit 2; }
[[ "$1" = ministral || "$1" = granite || "$1" = gemma ]] || { echo "Unknown model; choose ministral, granite or gemma" >&2; exit 2; }
[[ "$2" = smoke || "$2" = full ]] || { echo "Choose smoke or full" >&2; exit 2; }
case "${BENCH_SHOTS-all}" in
  all) shot_array='0,3,5%1' ;;
  0|3|5) shot_array="$BENCH_SHOTS" ;;
  *) echo "BENCH_SHOTS must be all, 0, 3 or 5 (unset defaults to all)" >&2; exit 2 ;;
esac
: "${BENCH_SCRATCH_ROOT:?Set BENCH_SCRATCH_ROOT to an existing writable scratch directory}"
[[ "$BENCH_SCRATCH_ROOT" = /* && -d "$BENCH_SCRATCH_ROOT" ]] || { echo "Scratch root must be an existing absolute directory" >&2; exit 2; }
home_path=$(realpath -e -- "${HOME:?}")
export BENCH_SCRATCH_ROOT
export BENCH_LOG_DIR="${BENCH_LOG_DIR:-$BENCH_SCRATCH_ROOT/logs/slurm}"
for name in BENCH_SCRATCH_ROOT BENCH_LOG_DIR; do
  [[ "${!name}" = /* ]] || { echo "$name must be absolute" >&2; exit 2; }
  resolved=$(realpath -m -- "${!name}")
  [[ "$resolved" != "$home_path" && "$resolved" != "$home_path/"* ]] || { echo "$name must be outside HOME" >&2; exit 2; }
  printf -v "$name" '%s' "$resolved"
done
mkdir -p -- "$BENCH_LOG_DIR"
probe=$(mktemp "$BENCH_LOG_DIR/.benchmark-write-check.XXXXXXXX")
unlink -- "$probe"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
printf 'Slurm stdout: %s/%%x-%%A_%%a.out\nSlurm stderr: %s/%%x-%%A_%%a.err\n' "$BENCH_LOG_DIR" "$BENCH_LOG_DIR"
# CLI options override #SBATCH and inherited SBATCH_OUTPUT/SBATCH_ERROR values.
# Explicit --array also overrides SBATCH_ARRAY_INX: BENCH_SHOTS is authoritative.
# The shell expands paths now, before Slurm opens either log on the compute node.
sbatch --export=ALL --open-mode=append --array="$shot_array" \
  --output="$BENCH_LOG_DIR/%x-%A_%a.out" --error="$BENCH_LOG_DIR/%x-%A_%a.err" \
  "$script_dir/benchmark.sbatch" "$@"
