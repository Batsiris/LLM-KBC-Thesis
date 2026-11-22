#!/bin/bash
#SBATCH --job-name=kbc-benchmark
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:55:00
#SBATCH --array=0,3,5%1
#SBATCH --signal=INT@120
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --open-mode=append

# Submit from the repository root after creating results/slurm.
# One parameterized template covers both models and smoke/full configurations.
set -euo pipefail
[[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: sbatch $0 ministral|granite smoke|full [run-tag]" >&2; exit 2; }
: "${BENCH_PYTHON:?Set BENCH_PYTHON to the isolated benchmark-modern Python executable}"
[[ "$BENCH_PYTHON" = /* && -x "$BENCH_PYTHON" ]] || { echo "BENCH_PYTHON must be an absolute executable path" >&2; exit 2; }
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
case "$1" in
  ministral)
    model='mistralai/Ministral-3-14B-Instruct-2512'
    revision='29439f81c2be264d8d393273f99e7db9c0961120'
    quantization='fp8-dequantize'
    precision='bf16-from-fp8'
    ;;
  granite)
    model='ibm-granite/granite-4.1-8b'
    revision='1504002f650e656a0a3789d99574df12e3e94ed0'
    quantization='none'
    precision='bf16'
    ;;
  *) echo "Unknown model selection" >&2; exit 2 ;;
esac
extra=()
case "$2" in
  smoke) extra+=(--max-examples 5) ;;
  full) ;;
  *) echo "Choose smoke or full" >&2; exit 2 ;;
esac
shots="${SLURM_ARRAY_TASK_ID:?Submit as the supplied 0/3/5-shot array}"
[[ "$shots" =~ ^(0|3|5)$ ]] || { echo "Unexpected shot count" >&2; exit 2; }
tag="${3:-v1}"
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$ ]] || { echo "Invalid run tag" >&2; exit 2; }
if [[ -n "${BENCH_CACHE_DIR:-}" ]]; then extra+=(--cache-dir "$BENCH_CACHE_DIR"); fi
export PYTHONUNBUFFERED=1

# Require the reviewed local snapshots; no downloads or source substitution here.
printf '%s  %s\n' \
  ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e data/lm_kbc_2026/train.jsonl \
  ba86b53ac38eb4b23b80391b291e5987ff4bbfe79827596fc09751b1bb0ce2be data/lm_kbc_2026/val.jsonl | sha256sum --check --strict
git rev-parse HEAD
"$BENCH_PYTHON" -m pip check
"$BENCH_PYTHON" -m pip freeze
nvidia-smi

# No resume, CPU offload, precision fallback, parameter changes or automatic scoring.
# srun receives Slurm's pre-timeout INT; hard kills still cannot publish partial predictions.
exec srun --ntasks=1 "$BENCH_PYTHON" -m llm_kbc.benchmark run \
  --model "$model" --revision "$revision" --shots "$shots" \
  --dtype bfloat16 --quantization "$quantization" --thinking default \
  --seed 0 --determinism strict --max-new-tokens 512 --context-limit 4096 \
  --dataset data/lm_kbc_2026/val.jsonl --data-dir data/lm_kbc_2026 --expected-records 475 \
  --output-root "${BENCH_OUTPUT_ROOT:-results/model_benchmark}" \
  --run-id "${precision}-${2}-${tag}" "${extra[@]}"
