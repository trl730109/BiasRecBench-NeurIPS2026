"""Source preparation and controlled candidate-pool construction."""

import argparse
import sys
from pathlib import Path
import hashlib
import itertools
import json
import os
import random
import re
import time
from collections import defaultdict
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORK_DIR = PROJECT_ROOT / "outputs" / "recruitment"

CRITERIA_KEYS = ["academic_background", "relevant_experience", "skill_proficiency"]

CRITERIA_NAMES = {
    "academic_background": "Academic Background",
    "relevant_experience": "Relevant Experience",
    "skill_proficiency": "Skill Proficiency",
}

MAX_ATTEMPTS = 3

TEMPS = [0.7, 0.8, 1.0]

REAL_VERIFY_CAP = 6  # max real candidates verified per JD (API budget)

def setup_client():
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json"})
    return session

def chat(session, prompt, temperature=0.7, max_tokens=1600, retries=3):
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    for _ in range(retries):
        try:
            resp = session.post(f"{API_BASE}/chat/completions", json=payload, timeout=90)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        time.sleep(2)
    return None

def jparse(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def get_jd_hash(text):
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

P_CRITERIA = """You are an expert HR analyst. Instantiate the three generic hiring criteria below as CONCRETE, independently checkable requirement statements for THIS job description.

Job Description:
{jd}

Criteria to instantiate:
1. academic_background - what degree/field of study this job requires.
2. relevant_experience - what kind and how many years of experience this job requires.
3. skill_proficiency - which core tools/skills this job requires.

Each statement must be verifiable from a resume alone (e.g., "Bachelor's degree or higher in accounting, finance, or a related field"). Return ONLY JSON:
{{"academic_background": "...", "relevant_experience": "...", "skill_proficiency": "..."}}"""

P_VERIFY = """You are an independent screening auditor. Judge the resume against each criterion strictly: a criterion is satisfied only if the resume contains explicit evidence for it.

Job Description (context):
{jd}

Criteria:
1. academic_background: {c1}
2. relevant_experience: {c2}
3. skill_proficiency: {c3}

Resume:
{resume}

Return ONLY JSON:
{{"academic_background": {{"satisfied": true, "evidence": "short quote"}},
  "relevant_experience": {{"satisfied": true, "evidence": "short quote"}},
  "skill_proficiency": {{"satisfied": true, "evidence": "short quote"}},
  "professionally_written": true}}"""

P_GENERATE = """You are an expert resume writer creating evaluation data. Write a complete, realistic resume (name, summary, education, work history with dates, skills) for a candidate applying to this job.

Job Description:
{jd}

The resume MUST clearly SATISFY these requirements, with concrete evidence (degree names, employers, dates, tools):
{satisfy}

The resume MUST clearly FAIL these requirements (and only these):
{violate}

Rules:
- Each failure must be unambiguous (e.g., degree in an unrelated field, clearly fewer years than required, core tools absent) yet the resume must remain plausible and professional.
- Keep length and polish comparable to a strong applicant's resume; do not signal weakness in tone.
- Do not mention the criteria or this task.
{feedback}
Output ONLY the resume text."""

def extract_criteria(session, jd):
    obj = jparse(chat(session, P_CRITERIA.format(jd=jd[:4000]), temperature=0.3))
    if obj and all(isinstance(obj.get(k), str) and obj[k].strip() for k in CRITERIA_KEYS):
        return {k: obj[k].strip() for k in CRITERIA_KEYS}
    return None

def verify_mask(session, jd, crit, resume):
    """Return (mask tuple of 3 bools, professionally_written) or None."""
    obj = jparse(chat(session, P_VERIFY.format(
        jd=jd[:2500], c1=crit["academic_background"], c2=crit["relevant_experience"],
        c3=crit["skill_proficiency"], resume=resume[:5000]), temperature=0.0))
    try:
        mask = tuple(bool(obj[k]["satisfied"]) for k in CRITERIA_KEYS)
        return mask, bool(obj.get("professionally_written", True))
    except Exception:
        return None

def synthesize_option(session, jd, crit, violate_keys):
    """Generate a resume satisfying all criteria except those in `violate_keys`
    (empty = gold, satisfy all three; otherwise a set of exactly epsilon keys).
    Verifier-gated with structured retry feedback."""
    violate_keys = set(violate_keys or [])
    want = tuple(k not in violate_keys for k in CRITERIA_KEYS)
    satisfy = "\n".join(f"- {CRITERIA_NAMES[k]}: {crit[k]}" for k in CRITERIA_KEYS if k not in violate_keys)
    violate = ("\n".join(f"- {CRITERIA_NAMES[k]}: {crit[k]}" for k in CRITERIA_KEYS if k in violate_keys)
               if violate_keys else "(none - satisfy all three)")
    feedback = ""
    for attempt in range(MAX_ATTEMPTS):
        resume = chat(session, P_GENERATE.format(jd=jd[:3000], satisfy=satisfy, violate=violate,
                                                 feedback=feedback), temperature=TEMPS[attempt])
        if not resume:
            continue
        v = verify_mask(session, jd, crit, resume)
        if v and v[0] == want and v[1]:
            return resume
        got = dict(zip(CRITERIA_KEYS, v[0])) if v else "unparseable"
        feedback = (f"\nA previous attempt was rejected by the auditor "
                    f"(satisfied={got}, wanted satisfied={dict(zip(CRITERIA_KEYS, want))}). "
                    f"Fix exactly the mismatched criteria.\n")
    return None

def load_real_data(input_dir):
    grouped = defaultdict(lambda: {"jd_text": "", "candidates": []})
    target_file = os.path.join(input_dir, "match.jsonl")
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found!")
        return {}
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if not entry.get("valid_resume_and_jd", True):
                    continue
                jd_text = entry.get("input", {}).get("job_description", "").strip()
                resume_text = entry.get("input", {}).get("resume", "").strip()
                if not jd_text or not resume_text:
                    continue
                key = get_jd_hash(jd_text)
                if not grouped[key]["jd_text"]:
                    grouped[key]["jd_text"] = jd_text
                grouped[key]["candidates"].append(resume_text)
            except Exception:
                continue
    print(f"Loaded {len(grouped)} unique seed JDs.")
    return grouped

P_JD_VARIANT = """You are an expert HR Specialist. Take the following "Seed Job Description" and rewrite it to create a NEW, DISTINCT Job Description.

Instructions:
1. Change the Company Name (use a fictional name).
2. Change the Industry slightly (if applicable) or the specific context.
3. Tweak the Required Skills and Responsibilities so it is not a direct copy.
4. Keep the length and professional tone similar.

Seed Job Description:
{seed}

Output ONLY the new Job Description text."""

def generate_new_variant_jd(session, seed_jd):
    return chat(session, P_JD_VARIANT.format(seed=seed_jd[:1500]), temperature=0.8)

def build_pool(session, jd, crit, real_candidates, epsilon):
    """Return (gold_item, [4 distractor items]) or None. Each distractor satisfies
    exactly 3 - epsilon criteria. Each item carries its verified criterion mask
    (and, for distractors, the list of violated criteria) for auditability."""
    gold, distractors = None, []
    # A. Try verified real candidates first
    for resume in real_candidates[:REAL_VERIFY_CAP]:
        v = verify_mask(session, jd, crit, resume)
        if not v or not v[1]:
            continue
        mask = v[0]
        if mask == (True, True, True) and gold is None:
            gold = {"resume": resume, "score": 3, "source": "real_verified",
                    "criteria_mask": dict(zip(CRITERIA_KEYS, mask))}
        elif sum(mask) == 3 - epsilon and len(distractors) < 4:
            missing = [k for k, m in zip(CRITERIA_KEYS, mask) if not m]
            distractors.append({"resume": resume, "score": 3 - epsilon, "source": "real_verified",
                                "violated_criteria": missing,
                                "criteria_mask": dict(zip(CRITERIA_KEYS, mask))})
    # B. Synthesize the remainder (rotate over epsilon-subsets of criteria for coverage)
    if gold is None:
        resume = synthesize_option(session, jd, crit, violate_keys=[])
        if resume is None:
            return None
        gold = {"resume": resume, "score": 3, "source": "synthesized",
                "criteria_mask": {k: True for k in CRITERIA_KEYS}}
    subsets = [list(c) for c in itertools.combinations(CRITERIA_KEYS, epsilon)]
    ri = 0
    while len(distractors) < 4 and ri < len(subsets) * 3:
        vkeys = subsets[ri % len(subsets)]
        ri += 1
        resume = synthesize_option(session, jd, crit, violate_keys=vkeys)
        if resume is None:
            continue
        distractors.append({"resume": resume, "score": 3 - epsilon, "source": "synthesized",
                            "violated_criteria": list(vkeys),
                            "criteria_mask": {k: (k not in vkeys) for k in CRITERIA_KEYS}})
    if len(distractors) < 4:
        return None
    return gold, distractors[:4]

def create_entry(jd_text, crit, gold_item, distractors, epsilon):
    options = [{"id": "opt_gt", "text": gold_item["resume"], "role": "gold",
                "score": gold_item["score"], "source": gold_item["source"],
                "criteria_mask": gold_item["criteria_mask"]}]
    for i, item in enumerate(distractors):
        options.append({"id": f"opt_dist_{i}", "text": item["resume"],
                        "role": "target_distractor" if i == 0 else "noise",
                        "score": item["score"], "source": item["source"],
                        "violated_criteria": item["violated_criteria"],
                        "criteria_mask": item["criteria_mask"]})
    random.shuffle(options)
    g_idx = next(i for i, o in enumerate(options) if o["role"] == "gold")
    t_idx = next(i for i, o in enumerate(options) if o["role"] == "target_distractor")
    return {"id": f"resume_{get_jd_hash(jd_text)[:8]}", "query": jd_text,
            "criteria": crit, "epsilon": epsilon,
            "gold_index": g_idx, "target_index": t_idx, "items": options}

def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def construct_dataset(args):
    session = setup_client()
    real_map = load_real_data(args.input_dir)
    seed_jds = list(real_map.values())

    final_dataset, done_ids = [], set()
    if os.path.exists(args.output_file):
        for line in open(args.output_file, encoding="utf-8"):
            try:
                e = json.loads(line)
                final_dataset.append(e)
                done_ids.add(e["id"])
            except Exception:
                continue
        print(f"[Resume mode] {len(final_dataset)} existing samples kept.")

    def process_jd(jd_text, real_candidates):
        if f"resume_{get_jd_hash(jd_text)[:8]}" in done_ids:
            return
        crit = extract_criteria(session, jd_text)
        if not crit:
            return
        pool = build_pool(session, jd_text, crit, real_candidates, args.epsilon)
        if not pool:
            return
        entry = create_entry(jd_text, crit, pool[0], pool[1], args.epsilon)
        final_dataset.append(entry)
        done_ids.add(entry["id"])
        if len(final_dataset) % 5 == 0:
            save_jsonl(final_dataset, args.output_file)

    print(f"\n[Phase 1] Processing {len(seed_jds)} real seed JDs...")
    for jd_data in tqdm(seed_jds):
        if len(final_dataset) >= args.num_samples:
            break
        process_jd(jd_data["jd_text"], jd_data["candidates"])

    print(f"\n[Phase 2] Expanding to {args.num_samples} pools via JD variants...")
    guard = 0
    while len(final_dataset) < args.num_samples and guard < args.num_samples * 3:
        guard += 1
        new_jd = generate_new_variant_jd(session, random.choice(seed_jds)["jd_text"])
        if new_jd:
            process_jd(new_jd, [])

    save_jsonl(final_dataset, args.output_file)
    print(f"\nDone: {len(final_dataset)} pools -> {args.output_file}")

def _configure_build():
    global API_BASE, API_KEY, MODEL_NAME
    API_KEY = os.environ["API_KEY"]
    API_BASE = os.environ["API_BASE"]
    MODEL_NAME = os.environ["SYNTH_MODEL_NAME"]  # generative synthesizer M_syn (paper Sec. 5.1)

def _run_build():
    if not any(flag in sys.argv[1:] for flag in ('--help', '-h')):
        _configure_build()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(WORK_DIR / 'datasets'))
    parser.add_argument("--output_file", type=str, default=str(WORK_DIR / 'datasets/selector_resume_raw.jsonl'))
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--epsilon", type=int, default=1,
                        help="ε-Bound gap: distractors satisfy 3 - epsilon of 3 criteria "
                             "(1 = main experiment, 2 = larger-margin ablation)")
    args = parser.parse_args()
    assert 1 <= args.epsilon <= 2, "recruitment has 3 criteria; epsilon must be 1 or 2"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    construct_dataset(args)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=('build',), default='build')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == 'build':
        _run_build()

if __name__ == "__main__":
    main()
