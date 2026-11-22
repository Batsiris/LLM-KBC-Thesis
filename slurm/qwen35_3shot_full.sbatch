#!/bin/bash
#SBATCH --job-name=qwen35-3shot-full
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/qwen35-3shot-full-%j.out
#SBATCH --error=logs/qwen35-3shot-full-%j.err

set -euo pipefail

cd ~/LLM-KBC-Thesis
source /scratch/m/mbatsir/llm-kbc/venv-benchmark-modern/bin/activate

export HF_HOME=/scratch/m/mbatsir/llm-kbc/hf-cache
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
nvidia-smi

python -m llm_kbc.benchmark run \
  --model Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --shots 3 \
  --dtype bfloat16 \
  --quantization none \
  --thinking disabled \
  --determinism strict \
  --seed 0 \
  --max-new-tokens 512 \
  --context-limit 4096 \
  --dataset data/lm_kbc_2026/val.jsonl \
  --run-id qwen35-bf16-3shot-full-001 \


echo "3-SHOT FULL COMPLETED"
