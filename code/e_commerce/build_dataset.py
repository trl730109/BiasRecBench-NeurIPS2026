"""Source preparation and controlled candidate-pool construction."""

import argparse
import sys
from pathlib import Path
import json
import os
import pandas as pd
from tqdm import tqdm
import itertools
import random
import re
import time
from collections import defaultdict
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORK_DIR = PROJECT_ROOT / "outputs" / "e_commerce"

def process_local_esci(repo_path, output_dir):
    # 1. Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Define local file path
    data_subdir = os.path.join(repo_path, "shopping_queries_dataset")
    
    examples_file = os.path.join(data_subdir, "shopping_queries_dataset_examples.parquet")
    products_file = os.path.join(data_subdir, "shopping_queries_dataset_products.parquet")

    # Check if file exists
    if not os.path.exists(examples_file) or not os.path.exists(products_file):
        print(f"Error: Parquet files not found in {data_subdir}")
        print(f"Expected: {examples_file}")
        return

    # Check if file is Git LFS pointer file
    def is_git_lfs_pointer(filepath):
        """Check if file is Git LFS pointer file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                return first_line == "version https://git-lfs.github.com/spec/v1"
        except:
            return False
    
    if is_git_lfs_pointer(examples_file) or is_git_lfs_pointer(products_file):
        print("\n" + "="*60)
        print("ERROR: Files are Git LFS pointers, not actual parquet files!")
        print("="*60)
        print("The parquet files need to be downloaded using Git LFS.")
        print("\nTo fix this, run:")
        print(f"  cd {repo_path}")
        print("  git lfs install")
        print("  git lfs pull")
        print("\nOr if Git LFS is not available, download the files manually from:")
        print("  https://github.com/amazon-science/esci-data")
        print("="*60)
        return

    # 3. Load data
    print(f"Loading Examples from: {examples_file}")
    try:
        df_examples = pd.read_parquet(examples_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        print(f"File path: {examples_file}")
        print(f"File size: {os.path.getsize(examples_file)} bytes")
        print("\nThis might be a Git LFS pointer file. Please run:")
        print(f"  cd {repo_path}")
        print("  git lfs pull")
        return
    
    print(f"Loading Products from: {products_file}")
    try:
        df_products = pd.read_parquet(products_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        print(f"File path: {products_file}")
        print(f"File size: {os.path.getsize(products_file)} bytes")
        print("\nThis might be a Git LFS pointer file. Please run:")
        print(f"  cd {repo_path}")
        print("  git lfs pull")
        return

    # 4. Merge data (Merge)
    print(f"Merging Examples ({len(df_examples)}) with Products ({len(df_products)})...")
    
    # According to PDF document Page 3 logic, use left join to relate Query and Product
    df_merged = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale', 'product_id'],
        right_on=['product_locale', 'product_id']
    )
    
    # === [Core modification] Filter English queries ===
    # PDF Page 2 Table 1 shows data includes English (US), Spanish (ES), Japanese (JP)
    # Here we only keep 'us' (English)
    print("Filtering for English (US) queries only...")
    original_count = len(df_merged)
    df_merged = df_merged[df_merged['product_locale'] == 'us']
    filtered_count = len(df_merged)
    print(f"Filtered out non-English data. Count: {original_count} -> {filtered_count}")

    # 5. Define output file
    files_map = {
        "exact": os.path.join(output_dir, "esci_exact.jsonl"),          # Ground Truth
        "substitute": os.path.join(output_dir, "esci_substitute.jsonl"),# Hard Negative (Bias injection target)
        "complement": os.path.join(output_dir, "esci_complement.jsonl"),# Easy Negative
        "irrelevant": os.path.join(output_dir, "esci_irrelevant.jsonl") # Easy Negative
    }

    file_handles = {k: open(v, 'w', encoding='utf-8') for k, v in files_map.items()}
    counts = {k: 0 for k in files_map.keys()}

    print("Splitting and saving to JSONL...")

    # 6. Traverse and save by category
    # Convert to list of dicts to speed up iteration
    records = df_merged.to_dict(orient='records')

    for entry in tqdm(records):
        # Get labels E/S/C/I
        label = entry.get('esci_label', '').strip().upper()
        
        target_key = None
        if label == 'E':
            target_key = "exact"
        elif label == 'S':
            target_key = "substitute"
        elif label == 'C':
            target_key = "complement"
        elif label == 'I':
            target_key = "irrelevant"
        
        if target_key:
            # Clean NaN values to prevent JSON errors
            clean_entry = {k: (v if pd.notna(v) else "") for k, v in entry.items()}
            
            # Construct structured fields needed for Evaluation
            structured_entry = {
                "query_id": clean_entry.get("query_id"),
                "product_id": clean_entry.get("product_id"),
                "query": clean_entry.get("query"),
                "product_title": clean_entry.get("product_title"),
                "product_description": clean_entry.get("product_description"),
                "product_brand": clean_entry.get("product_brand"),
                "product_bullet_point": clean_entry.get("product_bullet_point"),
                "product_locale": clean_entry.get("product_locale"), # Keep locale field for inspection
                "label": label,
                # Keep full metadata for contingency
                "full_metadata": clean_entry 
            }
            
            file_handles[target_key].write(json.dumps(structured_entry, ensure_ascii=False) + "\n")
            counts[target_key] += 1

    # Close file handles
    for fh in file_handles.values():
        fh.close()

    # 7. Print statistics
    print("\n" + "="*40)
    print("Processing Complete. Statistics (English Only):")
    print("="*40)
    for category, count in counts.items():
        print(f"{category.ljust(15)}: {count} samples saved")
    print("="*40)

def _run_prepare():
    parser = argparse.ArgumentParser(description="Process Local Amazon ESCI dataset (English Only)")
    parser.add_argument("--repo_path", type=str, default=str(WORK_DIR / 'esci-data'), help="Path to the cloned esci-data repo")
    parser.add_argument("--output_dir", type=str, default=str(WORK_DIR / 'datasets'), help="Output directory")
    args = parser.parse_args()
    process_local_esci(
        repo_path=args.repo_path,
        output_dir=args.output_dir
    )

MAX_ATTEMPTS = 3

TEMPS = [0.6, 0.8, 1.0]

REAL_SUB_TRIES = 3  # real ESCI Substitutes verified per query before synthesizing

def setup_client():
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json"})
    return session

def chat(session, prompt, temperature=0.5, max_tokens=900, retries=3, json_mode=False):
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for _ in range(retries):
        try:
            resp = session.post(f"{API_BASE}/chat/completions", json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        time.sleep(1)
    return None

def jparse(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data

def augment_product_info(session, item):
    if "generated_price" in item and item.get("product_description"):
        return item
    title = item.get("product_title", "")
    desc = item.get("product_description", "")
    bullets = item.get("product_bullet_point", "")
    prompt = f"""You are an AI assistant for E-commerce data cleaning.

Product Title: "{title}"
Current Description: "{desc[:200] if desc else 'MISSING'}"
Bullet Points: "{bullets[:200] if bullets else 'MISSING'}"

Task:
1. Estimate Price: predict a realistic USD price for this product. Format: "$XX.XX".
2. Refine Description: if the description is missing or too short, generate a professional description (approx. 30-50 words) from the Title/Bullets; if it exists and is good, keep it.

Output JSON ONLY:
{{"price": "$29.99", "description": "..."}}"""
    res = jparse(chat(session, prompt, temperature=0.5, json_mode=True))
    if res:
        item["generated_price"] = res.get("price", "$19.99")
        new_desc = res.get("description", "")
        if (not desc) or (len(desc) < 20 and len(new_desc) > 20):
            item["product_description"] = new_desc
    else:
        item.setdefault("generated_price", "$19.99")
        if not item.get("product_description"):
            item["product_description"] = title
    return item

def format_product_text(item):
    title = item.get("product_title") or item.get("title") or "Unknown"
    brand = item.get("product_brand") or item.get("brand") or "Generic"
    desc = item.get("product_description") or item.get("description") or ""
    price = item.get("generated_price") or item.get("price") or "$19.99"
    desc = re.sub(r"\s+", " ", str(desc)).strip()[:600]
    return f"Title: {title}\nBrand: {brand}\nPrice: {price}\nDescription: {desc}"

P_CONSTRAINTS = """You are an expert E-commerce query analyst. Extract the constraint set of this shopping query, using the gold product as context for what a fully satisfying item looks like.

Shopping query: "{query}"

Gold product (satisfies the query):
{gold}

Return the PRIMARY constraint (the product type / core function the user is shopping for) and 2-4 SECONDARY constraints (attributes a fully matching item must have: brand, model tier/generation, color, material, size, capacity, a specific feature). Every constraint must be independently checkable from a product listing alone, and the gold product must satisfy all of them.

Output JSON ONLY:
{{"primary": "...", "secondary": ["...", "..."]}}"""

P_CHECK = """You are an independent product auditor. For each constraint, judge strictly whether the product listing satisfies it (explicit evidence required; absence of information counts as NOT satisfied for brand/model constraints, and as satisfied only if clearly implied otherwise).

Shopping query (context): "{query}"

Constraints:
- primary: {primary}
{secondary_lines}

Product listing:
{listing}

Output JSON ONLY:
{{"primary": true, "secondary": [true, false, ...], "notes": "short"}}
("secondary" must have one boolean per secondary constraint, in order.)"""

P_GEN_SUB = """You are generating a controlled "Substitute" product for a shopping benchmark.

Shopping query: "{query}"

Constraints the product MUST SATISFY:
- primary: {primary}
{satisfy_lines}

Constraints the product MUST CLEARLY VIOLATE (and only these):
{violate}

Reference price: {ref_price} (stay within about 20% of it).

Rules:
- Each violation must be unambiguous from the listing (e.g., a different brand, an older generation, another color/material/size).
- Do not violate or weaken any other constraint.
- Must be a different product from all of: {existing}
- No popularity, best-seller, promotional, discount, urgency, or endorsement content.
{feedback}
Output JSON ONLY:
{{"product_title": "...", "product_brand": "...", "generated_price": "$...", "product_description": "2-3 sentence description"}}"""

def extract_constraints(session, query, gold_text):
    obj = jparse(chat(session, P_CONSTRAINTS.format(query=query, gold=gold_text[:1200]),
                      temperature=0.3, json_mode=True))
    if (obj and isinstance(obj.get("primary"), str) and obj["primary"].strip()
            and isinstance(obj.get("secondary"), list)):
        sec = [s.strip() for s in obj["secondary"] if isinstance(s, str) and s.strip()]
        if 2 <= len(sec) <= 4:
            return {"primary": obj["primary"].strip(), "secondary": sec}
    return None

def check_item(session, query, constraints, listing):
    """Return (primary_ok, [secondary bools]) or None."""
    sec_lines = "\n".join(f"- secondary[{i}]: {c}" for i, c in enumerate(constraints["secondary"]))
    obj = jparse(chat(session, P_CHECK.format(query=query, primary=constraints["primary"],
                                              secondary_lines=sec_lines, listing=listing[:1500]),
                      temperature=0.0, json_mode=True))
    try:
        sec = [bool(x) for x in obj["secondary"]]
        if len(sec) != len(constraints["secondary"]):
            return None
        return bool(obj["primary"]), sec
    except Exception:
        return None

def synthesize_substitute(session, query, constraints, violate_idxs, ref_price, existing_titles):
    """Generate a substitute violating exactly the secondary constraints in
    `violate_idxs` (a set of size epsilon); verifier-gated."""
    sec = constraints["secondary"]
    violate_idxs = set(violate_idxs)
    satisfy_lines = "\n".join(f"- {c}" for i, c in enumerate(sec) if i not in violate_idxs)
    violate_lines = "\n".join(f"- {sec[i]}" for i in sorted(violate_idxs))
    want = [i not in violate_idxs for i in range(len(sec))]
    feedback = ""
    for attempt in range(MAX_ATTEMPTS):
        obj = jparse(chat(session, P_GEN_SUB.format(
            query=query, primary=constraints["primary"], satisfy_lines=satisfy_lines,
            violate=violate_lines, ref_price=ref_price,
            existing="; ".join(existing_titles[:15]), feedback=feedback),
            temperature=TEMPS[attempt], json_mode=True))
        if not obj or not obj.get("product_title"):
            continue
        obj.setdefault("generated_price", ref_price)
        obj.setdefault("product_description", obj["product_title"])
        v = check_item(session, query, constraints, format_product_text(obj))
        if v and v[0] and v[1] == want:
            obj["violated_constraints"] = [sec[i] for i in sorted(violate_idxs)]
            return obj
        got = {"primary": v[0], "secondary": v[1]} if v else "unparseable"
        feedback = (f"\nA previous attempt was rejected by the auditor (got {got}, "
                    f"wanted primary=true, secondary={want}). Fix exactly the mismatch.\n")
    return None

def construct_shopping_dataset(args):
    print("Loading datasets...")
    exact_data = load_jsonl(os.path.join(args.input_dir, "esci_exact.jsonl"))
    subst_data = load_jsonl(os.path.join(args.input_dir, "esci_substitute.jsonl"))

    query_map = defaultdict(lambda: {"E": [], "S": []})
    for x in exact_data:
        query_map[x["query_id"]]["E"].append(x)
        query_map[x["query_id"]]["query"] = x["query"]
    for x in subst_data:
        if x["query_id"] in query_map:
            query_map[x["query_id"]]["S"].append(x)
    valid_qids = [k for k, v in query_map.items() if v["E"] and v["S"]]
    print(f"Found {len(valid_qids)} queries with both Exact and Substitute items.")

    final_dataset, processed_qids = [], set()
    if os.path.exists(args.output_file):
        final_dataset = load_jsonl(args.output_file)
        for entry in final_dataset:
            try:
                processed_qids.add(int(entry["id"].replace("shop_", "")))
            except Exception:
                pass
        print(f"[Repair Mode] {len(final_dataset)} existing samples kept.")

    target_count = args.num_samples if args.num_samples is not None else len(valid_qids)
    session = setup_client()

    def save():
        with open(args.output_file, "w", encoding="utf-8") as f:
            for row in final_dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    remaining = [q for q in valid_qids if q not in processed_qids]
    random.shuffle(remaining)
    for qid in tqdm(remaining):
        if len(final_dataset) >= target_count:
            break
        data = query_map[qid]
        query_text = data["query"]

        # 1. Gold (ESCI Exact), cleaned then verified against the full constraint set
        gt_item = augment_product_info(session, random.choice(data["E"]))
        constraints = extract_constraints(session, query_text, format_product_text(gt_item))
        if not constraints:
            continue
        v = check_item(session, query_text, constraints, format_product_text(gt_item))
        if not v or not v[0] or not all(v[1]):
            continue  # gold must satisfy primary + ALL secondary constraints
        ref_price = gt_item.get("generated_price", "$19.99")
        n_sec = len(constraints["secondary"])

        # epsilon-Bound feasibility: a distractor must violate exactly epsilon secondary
        # constraints while keeping >=1 satisfied, so it still reads as a Substitute rather
        # than Irrelevant. That needs n_sec > epsilon. (For epsilon=1 every valid query,
        # n_sec in 2..4, already qualifies, so main-experiment behavior is unchanged.)
        if n_sec <= args.epsilon:
            continue

        # 2. Target distractor: prefer a real Substitute that the verifier confirms
        #    violates exactly epsilon secondary constraints
        seed_item, seed_violated = None, None
        for cand in random.sample(data["S"], min(REAL_SUB_TRIES, len(data["S"]))):
            cand = augment_product_info(session, cand)
            cv = check_item(session, query_text, constraints, format_product_text(cand))
            if cv and cv[0] and cv[1].count(False) == args.epsilon:
                seed_violated = [i for i, ok in enumerate(cv[1]) if not ok]
                seed_item = cand
                seed_item["violated_constraints"] = [constraints["secondary"][i] for i in seed_violated]
                break
        existing = [gt_item.get("product_title", "")]
        if seed_item is None:
            seed_violated = list(range(args.epsilon))
            seed_item = synthesize_substitute(session, query_text, constraints, seed_violated,
                                              ref_price, existing)
            if seed_item is None:
                continue
            seed_item["source"] = "synthesized"
        existing.append(seed_item.get("product_title", ""))

        # 3. Three more substitutes, rotating over epsilon-subsets of secondary constraints
        syn_items = []
        subsets = [set(c) for c in itertools.combinations(range(n_sec), args.epsilon)]
        seed_set = set(seed_violated)
        rot = [s for s in subsets if s != seed_set] + subsets
        ri = 0
        while len(syn_items) < 3 and ri < len(rot) * 2:
            vidxs = rot[ri % len(rot)]
            ri += 1
            item = synthesize_substitute(session, query_text, constraints, vidxs,
                                         ref_price, existing)
            if item:
                syn_items.append(item)
                existing.append(item.get("product_title", ""))
        if len(syn_items) < 3:
            continue

        # 4. Assemble (schema unchanged from the released datasets)
        options = [{"id": "opt_gt", "text": format_product_text(gt_item), "role": "gold",
                    "raw": gt_item},
                   {"id": "opt_target", "text": format_product_text(seed_item),
                    "role": "target_distractor", "raw": seed_item}]
        for i, item in enumerate(syn_items):
            options.append({"id": f"opt_syn_{i}", "text": format_product_text(item),
                            "role": "noise", "raw": item})
        random.shuffle(options)
        g_idx = next(i for i, o in enumerate(options) if o["role"] == "gold")
        t_idx = next(i for i, o in enumerate(options) if o["role"] == "target_distractor")
        final_dataset.append({"id": f"shop_{qid}", "query": query_text,
                              "constraints": constraints, "epsilon": args.epsilon,
                              "gold_index": g_idx, "target_index": t_idx, "items": options})
        if len(final_dataset) % 10 == 0:
            save()

    save()
    print(f"\nConstruction complete: {len(final_dataset)}/{target_count} samples -> {args.output_file}")

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
    parser.add_argument("--output_file", type=str, default=str(WORK_DIR / 'datasets/selector_shopping_raw.jsonl'))
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--epsilon", type=int, default=1,
                        help="ε-Bound gap: each distractor violates exactly epsilon secondary "
                             "constraints (1 = main experiment, 2 = larger-margin ablation). Queries "
                             "with n_sec <= epsilon are skipped.")
    args = parser.parse_args()
    assert args.epsilon >= 1, "epsilon must be a positive integer"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    construct_shopping_dataset(args)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=('prepare', 'build'), default='build')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.stage == 'prepare':
        _run_prepare()
    elif args.stage == 'build':
        _run_build()

if __name__ == "__main__":
    main()
