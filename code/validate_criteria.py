"""LLM-based criterion validation with and without explicit selection criteria.

Given a sample of CLEAN pools, a held-out LLM selects the best option twice:
  (1) holistic  : no criteria given, the model judges by itself;
  (2) criteria  : the user's specified criteria are prepended and the model is
                  asked to judge strictly by them.
Agreement of each run with the designated gold, and between the two runs, is
reported and saved.

Pool file format (jsonl), one clean pool per line:
  {"id": "...", "input": "<full selection prompt>", "target_output_clean": "[[C]]"}
This matches the *_unbiased splits shipped with BiasRecBench.

Env (same convention as eval.py): api_keys, api_base
Run:
  api_keys=... api_base=... python3 code/validate_criteria.py \
      --pool_file pools.jsonl --criteria "Academic background; relevant experience; skill proficiency" \
      --model gpt-4o --n 50
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

CRITERIA_HEADER = ("Selection criteria (judge strictly and only by these):\n{criteria}\n\n")


def parse_choice(txt):
    m = re.findall(r"\[\[([A-Z])\]\]", txt or "")
    if m:
        return m[-1]
    m = re.findall(r"\b([A-Z])\b", txt or "")
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_file", required=True)
    ap.add_argument("--criteria", required=True, help="the user's specified criteria, as text")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="./validation_llm.json")
    args = ap.parse_args()

    keys = [k.strip() for k in os.environ.get("api_keys", "").split(",") if k.strip()]
    if not keys:
        raise ValueError("Set env var api_keys.")
    client = OpenAI(api_key=keys[0], base_url=os.environ.get("api_base", "https://api.openai.com/v1"))

    pools = [json.loads(l) for l in open(args.pool_file, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(pools)
    pools = pools[:args.n]

    def ask(prompt):
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature, max_tokens=2048)
                return parse_choice(r.choices[0].message.content)
            except Exception:
                pass
        return None

    def run_item(p):
        holistic = ask(p["input"])
        criteria = ask(CRITERIA_HEADER.format(criteria=args.criteria) + p["input"])
        return {"id": p.get("id"), "gold": p["target_output_clean"].strip("[]"),
                "holistic": holistic, "criteria": criteria}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        items = list(pool.map(run_item, pools))

    def agr(a, b):
        pairs = [(x[a], x[b]) for x in items if x[a] and x[b]]
        return round(sum(u == v for u, v in pairs) / max(len(pairs), 1), 4)

    summary = {"model": args.model, "n": len(items), "criteria": args.criteria,
               "agreement_holistic_vs_gold": agr("holistic", "gold"),
               "agreement_criteria_vs_gold": agr("criteria", "gold"),
               "agreement_holistic_vs_criteria": agr("holistic", "criteria"),
               "items": items}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"n={summary['n']}  holistic~gold={summary['agreement_holistic_vs_gold']}  "
          f"criteria~gold={summary['agreement_criteria_vs_gold']}  "
          f"holistic~criteria={summary['agreement_holistic_vs_criteria']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
