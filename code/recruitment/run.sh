#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOMAIN="recruitment"
PYTHON="${PYTHON:-python3}"
mode="${1:-eval}"
if [[ $# -gt 0 && "$1" != -* ]]; then shift; else mode=eval; fi
if [[ "${1:-}" == --help && $# -eq 1 && "$mode" == eval ]]; then
    exec "$PYTHON" "${SCRIPT_DIR}/evaluate.py" --help
fi

case "$mode" in
    eval)
        target_model="${TARGET_MODEL:-gpt4o}"
        source "${SCRIPT_DIR}/../config.sh"
        args=(--model_name "$model_name" --model_path "$model_path"
              --input_dir "${PROJECT_ROOT}/data/${DOMAIN}"
              --mapping_dir "${PROJECT_ROOT}/data/${DOMAIN}/position"
              --output_dir "${PROJECT_ROOT}/outputs/${DOMAIN}/results/eval_results_${target_model}_$(date +%Y%m%d%H%M%S)"
              --temperature "$temperature" --max_tokens "$max_tokens"
              --tensor_parallel_size 4 --batch_size 50)
        if [[ -n "${EVAL_BIAS_TYPES:-}" ]]; then args+=(--eval_bias_types "$EVAL_BIAS_TYPES"); fi
        if [[ "${USE_DEBIASING:-false}" == true ]]; then args+=(--use_debiasing); fi
        exec "$PYTHON" "${SCRIPT_DIR}/evaluate.py" "${args[@]}" "$@"
        ;;
    build|inject)
        epsilon="${EPSILON:-1}"
        forwarded=()
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == --epsilon ]]; then
                [[ $# -ge 2 ]] || { echo "--epsilon requires a value" >&2; exit 2; }
                epsilon="$2"; shift 2
            else forwarded+=("$1"); shift; fi
        done
        suffix=""
        if [[ "$epsilon" != 1 ]]; then suffix="_eps${epsilon}"; fi
        work="${PROJECT_ROOT}/outputs/${DOMAIN}"
        pool="${work}/datasets/selector_resume_raw${suffix}.jsonl"
        if [[ "$mode" == build ]]; then
            args=(--input_dir "${work}/datasets" --output_file "$pool" --epsilon "$epsilon")
            exec "$PYTHON" "${SCRIPT_DIR}/build_dataset.py" "${args[@]}" "${forwarded[@]}"
        else
            exec "$PYTHON" "${SCRIPT_DIR}/inject_biases.py" --stage inject \
                --input_file "$pool" --output_dir "${work}/biased_datasets${suffix}" "${forwarded[@]}"
        fi
        ;;
    refine|control)
        exec "$PYTHON" "${SCRIPT_DIR}/inject_biases.py" --stage "$mode" "$@"
        ;;
    *)
        echo "Usage: bash run.sh [eval|build|inject|refine|control] [options]" >&2
        exit 2
        ;;
esac
