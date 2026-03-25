#!/usr/bin/env bash
set -euo pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

MODEL="Dream-org/Dream-v0-Base-7B"
BLOCK_LENGTH=32
TASK="gsm8k"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_one () {
  local mode="$1"          # original | prefix_cache | dual_cache
  local gen_len="$2"
  local steps="$3"

  local extra_args=""
  case "$mode" in
    prefix_cache) extra_args="use_cache=true" ;;
    dual_cache)   extra_args="use_cache=true,dual_cache=true" ;;
  esac

  local out_dir="evals_results/${mode}"
  local out_path="${out_dir}/${TASK}-gen${gen_len}-steps${steps}"
  local log_file="${LOG_DIR}/${TASK}_${mode}_gen${gen_len}_steps${steps}.log"

  echo "==================================================" | tee "${log_file}"
  echo "mode=${mode} gen_length=${gen_len} steps=${steps}" | tee -a "${log_file}"
  echo "output_path=${out_path}" | tee -a "${log_file}"
  echo "==================================================" | tee -a "${log_file}"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --tasks "${TASK}" \
    --model dream \
    --model_args "pretrained=${MODEL},max_new_tokens=${gen_len},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9${extra_args:+,${extra_args}},show_speed=True" \
    --batch_size 1 --num_fewshot 5 --output_path "${out_path}" --log_samples 2>&1 | tee -a "${log_file}"
}

for GEN_LEN in 256; do
  STEPS=$(( GEN_LEN / BLOCK_LENGTH ))
  run_one "original"     "${GEN_LEN}" "${STEPS}"
  run_one "prefix_cache" "${GEN_LEN}" "${STEPS}"
  run_one "dual_cache"   "${GEN_LEN}" "${STEPS}"
done
