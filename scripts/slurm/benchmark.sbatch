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
#SBATCH --open-mode=append

# Submit through submit_benchmark.sh so Slurm opens logs outside /home.
# --output/--error are submission options: shell variables in #SBATCH do not expand.
# One parameterized template covers both models and smoke/full configurations.
set -Eeuo pipefail
trap 'rc=$?; printf "ERROR file=%s line=%s exit=%s command=%q\n" "${BASH_SOURCE[0]}" "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR
trap 'rc=$?; printf "EXIT job=%s task=%s host=%s cwd=%q exit=%s\n" "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${HOSTNAME:-unknown}" "$PWD" "$rc" >&2' EXIT
printf 'START job=%s task=%s host=%s submit_dir=%q\n' "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "${HOSTNAME:-unknown}" "${SLURM_SUBMIT_DIR:-unset}"
[[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: sbatch $0 ministral|granite smoke|full [run-tag]" >&2; exit 2; }
: "${BENCH_SCRATCH_ROOT:?Set BENCH_SCRATCH_ROOT to an existing writable scratch directory}"
: "${BENCH_PYTHON:?Set BENCH_PYTHON to the isolated benchmark-modern Python executable}"
[[ "$BENCH_PYTHON" = /* && -x "$BENCH_PYTHON" ]] || { echo "BENCH_PYTHON must be an absolute executable path" >&2; exit 2; }

# Resolve symlinks before checking home-quota safety. The repository may remain in
# home, but all new writable artifacts/caches and the interpreter must be outside it.
home_path=$(realpath -e -- "${HOME:?HOME must identify the home directory}")
outside_home() {
  local name="$1" resolved
  [[ "${!name}" = /* ]] || { echo "$name must be an absolute path" >&2; return 2; }
  resolved=$(realpath -m -- "${!name}")
  [[ "$resolved" != "$home_path" && "$resolved" != "$home_path/"* ]] || {
    echo "$name resolves inside HOME; refusing a home-quota write/fallback" >&2; return 2;
  }
  printf -v "$name" '%s' "$resolved"
}
outside_home BENCH_SCRATCH_ROOT
[[ -d "$BENCH_SCRATCH_ROOT" && -w "$BENCH_SCRATCH_ROOT" ]] || { echo "Scratch root must already exist and be writable" >&2; exit 2; }
# Validate both parent and target, but retain the venv executable's original path:
# resolving its python symlink to /usr/bin/python would bypass the environment.
interpreter_dir=$(dirname -- "$BENCH_PYTHON")
outside_home interpreter_dir
interpreter_target="$BENCH_PYTHON"
outside_home interpreter_target
export HF_HOME="${HF_HOME:-$BENCH_SCRATCH_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${BENCH_CACHE_DIR:-$HF_HOME/hub}}"
export BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-$HF_HUB_CACHE}"
export BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-$BENCH_SCRATCH_ROOT/results/model_benchmark}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BENCH_SCRATCH_ROOT/cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/cuda}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export TMPDIR="${TMPDIR:-$BENCH_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}" TEMP="${TEMP:-$TMPDIR}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$XDG_CACHE_HOME/pycache}"
storage=(HF_HOME HF_HUB_CACHE BENCH_CACHE_DIR BENCH_OUTPUT_ROOT XDG_CACHE_HOME
         HF_XET_CACHE HF_ASSETS_CACHE HF_MODULES_CACHE TORCH_HOME TRITON_CACHE_DIR
         CUDA_CACHE_PATH PIP_CACHE_DIR TMPDIR TMP TEMP PYTHONPYCACHEPREFIX)
for name in "${storage[@]}"; do outside_home "$name"; done
[[ "$BENCH_CACHE_DIR" = "$HF_HUB_CACHE" ]] || { echo "BENCH_CACHE_DIR and HF_HUB_CACHE must resolve to the same model cache" >&2; exit 2; }
# Older library overrides can defeat the explicit cache settings; fail rather than ignore them.
for name in TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE; do
  if [[ -n "${!name:-}" ]]; then
    outside_home "$name"
    [[ "${!name}" = "$HF_HUB_CACHE" ]] || { echo "$name conflicts with HF_HUB_CACHE; unset it or use the same directory" >&2; exit 2; }
  fi
done
for name in "${storage[@]}"; do
  mkdir -p -- "${!name}"
  probe=$(mktemp "${!name}/.benchmark-write-check.XXXXXXXX")
  # Remove only the empty probe just created here; never clean existing artifacts.
  unlink -- "$probe"
  printf 'STORAGE %s=%q\n' "$name" "${!name}"
done
df -h -- "$BENCH_SCRATCH_ROOT" "$HOME"
df -i -- "$BENCH_SCRATCH_ROOT" "$HOME"
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
extra+=(--cache-dir "$BENCH_CACHE_DIR")
export PYTHONUNBUFFERED=1
# Source provenance uses read-only Git commands; disable optional index refresh writes.
export GIT_OPTIONAL_LOCKS=0

# Trace only the fixed preflight/run commands, never dump credential environment variables.
PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
set -x
# Require the reviewed local snapshots; no downloads or source substitution here.
printf '%s  %s\n' \
  ad37cd30d1ff4b9f1ef2579b25e64093b202c40da11e8c412e13386f1e5d332e data/lm_kbc_2026/train.jsonl \
  ba86b53ac38eb4b23b80391b291e5987ff4bbfe79827596fc09751b1bb0ce2be data/lm_kbc_2026/val.jsonl | sha256sum --check --strict
printf 'PROVENANCE cwd=%q git=%q\n' "$PWD" "$(command -v git)"
git --version
# Do not change ownership settings, suppress Git errors or substitute another commit.
git rev-parse --show-toplevel
git rev-parse --absolute-git-dir
git rev-parse HEAD
"$BENCH_PYTHON" -m pip check
"$BENCH_PYTHON" -m pip freeze
nvidia-smi

# No resume, CPU offload, precision fallback, parameter changes or automatic scoring.
# srun receives Slurm's pre-timeout INT; hard kills still cannot publish partial predictions.
srun --ntasks=1 "$BENCH_PYTHON" -m llm_kbc.benchmark run \
  --model "$model" --revision "$revision" --shots "$shots" \
  --dtype bfloat16 --quantization "$quantization" --thinking default \
  --seed 0 --determinism strict --max-new-tokens 512 --context-limit 4096 \
  --dataset data/lm_kbc_2026/val.jsonl --data-dir data/lm_kbc_2026 --expected-records 475 \
  --output-root "$BENCH_OUTPUT_ROOT" \
  --run-id "${precision}-${2}-${tag}" "${extra[@]}"
