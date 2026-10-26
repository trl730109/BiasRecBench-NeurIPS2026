# BiasRecBench

Code and data for **Is Your LLM-as-a-Recommender Agent Trustable? LLMs' Recommendation is Easily Hacked by Biases (Preferences)**, NeurIPS 2026 E&D Track.

BiasRecBench evaluates LLM candidate selection in paper review, recruitment, and e-commerce. Sources: [Review-5K](https://huggingface.co/datasets/WestlakeNLP/Review-5K), [Amazon ESCI](https://github.com/amazon-science/esci-data), and [NetSol resume-score-details](https://huggingface.co/datasets/netsol/resume-score-details). See [sources and licenses](#sources-and-licenses).

## Layout and data

```text
code/
  config.sh
  validate_criteria.py
  paper_review/    # build_dataset.py, inject_biases.py, evaluate.py, run.sh
  recruitment/    # same entry points
  e_commerce/     # same entry points
data/
  paper_review/   # authority/, bandwagon/, distraction/, ...
  recruitment/
  e_commerce/
```

The 45 parquet files (42.82 MB) are included; no benchmark download is needed. Each domain has seven interventions: six have paired `unbiased`/`biased` conditions; Position has `unbiased`, `biased_first`, and `biased_last`. Conditions reuse underlying instances.

Data is grouped as `data/<domain>/<bias_type>/`, using lowercase bias names. Files retain `{Intervention}_eval_{Intervention}_{condition}-00000-of-00001.parquet`. Fields are `id`, `bias_type`, `input`, `target_output_bias`, `target_output_clean`, and JSON-encoded `meta`. Each domain's `position/` also contains the candidate-order mappings needed to evaluate Position.

## Environment

```bash
git clone https://github.com/trl730109/BiasRecBench-NeurIPS2026.git
cd BiasRecBench-NeurIPS2026
pip install -r requirements.txt
```

Set `api_base` and `api_keys` in `code/config.sh` for your OpenAI-compatible endpoint. For local GPU inference, install `vllm` separately and configure `model_path` in the same file. Do not commit credentials.

## Evaluation

```bash
bash code/paper_review/run.sh
bash code/recruitment/run.sh
bash code/e_commerce/run.sh
```

The default action is evaluation. Choose a model with `TARGET_MODEL` (a preset in `code/config.sh`), select interventions with `EVAL_BIAS_TYPES`, or enable the debiasing prompt with `USE_DEBIASING=true`:

```bash
TARGET_MODEL=gpt4o EVAL_BIAS_TYPES=Authority,Verbosity bash code/paper_review/run.sh
bash code/recruitment/run.sh eval --temperature 0 --eval_samples 10
```

Results are written to `outputs/<domain>/results/`, including clean/injected accuracy, bias success rate, and robustness rate. Additional evaluator options: `python code/paper_review/evaluate.py --help` (or the corresponding domain).

## Criterion validation

Compare LLM selections with and without proposed criteria on clean pools for a new task:

```bash
python code/validate_criteria.py \
  --pool_file pools_clean.jsonl \
  --criteria "Academic background; relevant experience; skill proficiency" \
  --model gpt-4o --n 50
```

Set `api_keys` and `api_base` environment variables. The input is JSONL with `id`, `input`, and `target_output_clean` (e.g., `[[C]]`). The output `validation_llm.json` reports agreement with gold labels and between the two selection settings.

## Sources and licenses

Original project contributions use [CC BY 4.0](LICENSE). Source data is selected, reformatted, and augmented with candidate pools and contextual cues; incorporated source material retains its upstream terms.

- Paper review: [Review-5K](https://huggingface.co/datasets/WestlakeNLP/Review-5K), from [CycleResearcher](https://openreview.net/forum?id=bjcsVLoHYs); [dataset license](https://huggingface.co/datasets/WestlakeNLP/Review-5K/blob/main/LICENSE).
- E-commerce: [Amazon ESCI](https://github.com/amazon-science/esci-data), via [tasksource/esci](https://huggingface.co/datasets/tasksource/esci); Apache-2.0 and the upstream notice are included in [LICENSE](LICENSE).
- Recruitment: [NetSol resume-score-details](https://huggingface.co/datasets/netsol/resume-score-details); see the [source card](https://huggingface.co/datasets/netsol/resume-score-details/blob/main/README.md) for source information.
