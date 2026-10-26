"""Source preparation and controlled candidate-pool construction."""

import argparse
import sys
from pathlib import Path
import json
import os
from datasets import load_dataset
from tqdm import tqdm
import random
import statistics
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORK_DIR = PROJECT_ROOT / "outputs" / "paper_review"

def split_hf_dataset(dataset_name, output_dir, split="train", token=None):
    # 1. Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Define output files map
    files_map = {
        "reject": os.path.join(output_dir, "decision_reject.jsonl"),
        "poster": os.path.join(output_dir, "decision_accept_poster.jsonl"),
        "spotlight": os.path.join(output_dir, "decision_accept_spotlight.jsonl"),
        "oral": os.path.join(output_dir, "decision_accept_oral.jsonl"),
        "unknown": os.path.join(output_dir, "decision_unknown.jsonl")
    }

    # 3. Open file handles
    file_handles = {k: open(v, 'w', encoding='utf-8') for k, v in files_map.items()}
    counts = {k: 0 for k in files_map.keys()}

    print(f"Loading {dataset_name} [{split}] from Hugging Face...")
    if token:
        print("Using provided Hugging Face authentication token.")

    # Load dataset with authentication token
    try:
        # Note: token parameter is used to pass HF token
        dataset = load_dataset(dataset_name, split=split, streaming=True, token=token)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Hint: Check if your token is valid and has access to this gated dataset.")
        return

    print("Processing and splitting...")

    try:
        # Iterate directly over the HF dataset object
        for entry in tqdm(dataset):
            
            # === Decision Cleaning Logic ===
            raw_decision = entry.get('decision', '')
            
            # Handle "## Paper Decision\nReject" format
            if '\n' in raw_decision:
                clean_decision = raw_decision.split('\n')[-1].strip().lower()
            else:
                clean_decision = raw_decision.strip().lower()

            # === Classification Logic ===
            target_key = "unknown"
            
            if "reject" in clean_decision:
                target_key = "reject"
            elif "poster" in clean_decision:
                target_key = "poster"
            elif "spotlight" in clean_decision:
                target_key = "spotlight"
            elif "oral" in clean_decision:
                target_key = "oral"
            
            # Write to file
            file_handles[target_key].write(json.dumps(entry, ensure_ascii=False) + "\n")
            counts[target_key] += 1

    finally:
        # Close handles
        for fh in file_handles.values():
            fh.close()

    # 4. Statistics
    print("\n" + "="*40)
    print("Processing Complete. Statistics:")
    print("="*40)
    for category, count in counts.items():
        print(f"{category.ljust(15)}: {count} samples saved")
    print("="*40)

def _run_prepare():
    parser = argparse.ArgumentParser(description="Split Review-5K dataset from HF")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face User Access Token (starts with hf_); defaults to $HF_TOKEN")
    parser.add_argument("--output_dir", type=str, default=str(WORK_DIR / 'datasets'), help="Output directory")
    args = parser.parse_args()
    split_hf_dataset(
        dataset_name="WestlakeNLP/Review-5K",
        output_dir=args.output_dir,
        token=args.token
    )

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(data)} entries from {filepath}")
    return data

def get_avg_score(paper):
    """Calculate average score from 'rates' field."""
    rates = paper.get('rates', [])
    if not rates:
        return 0.0
    # Handle possible string type scores
    try:
        clean_rates = []
        for r in rates:
            if isinstance(r, (int, float)):
                clean_rates.append(r)
            elif isinstance(r, str):
                # Attempt to extract numbers
                match = re.search(r'\d+(\.\d+)?', r)
                if match:
                    clean_rates.append(float(match.group()))
        
        if not clean_rates:
            return 0.0
        return statistics.mean(clean_rates)
    except Exception:
        return 0.0

def parse_paper_elements(paper_data):
    """
    Parse article content, separate Title, Abstract, and Body Content.
    Strategy:
    1. Prefer existing structured fields in JSON.
    2. If not, try to extract from full text (content/text) using regex.
    """
    
    # --- 1. Get raw data ---
    title = paper_data.get('title', '')
    abstract = paper_data.get('abstract', '')
    
    # Try to find the field name for body (Review-5K or other common dataset fields)
    # First check messages field (some dataset formats)
    full_text = ""
    if 'messages' in paper_data and isinstance(paper_data['messages'], list):
        # Find messages with role 'user', usually contain paper content
        for msg in paper_data['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                full_text = msg.get('content', '')
                break
    
    # If not found in messages, try other fields
    if not full_text:
        full_text = paper_data.get('pdf_text', '') or paper_data.get('content', '') or paper_data.get('text', '') or ""
    
    # --- 2. Complete Title ---
    if not title and full_text:
        # Try to extract first line or content after "Title:"
        title_match = re.search(r'^Title:\s*(.+)$', full_text, re.MULTILINE | re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # If no Title marker, take first 100 chars as fallback
            title = full_text[:100].split('\n')[0]

    # --- 3. Complete Abstract ---
    if not abstract and full_text:
        # Regex strategy: match ABSTRACT ... (until) INTRODUCTION
        # Allow ABSTRACT preceded by newline or directly after Title
        abs_pattern = re.search(r'(?:^|\n)(ABSTRACT|Abstract)\s*\n\s*(.*?)\n\s*(?:1\s+)?(?:INTRODUCTION|Introduction)', full_text, re.DOTALL | re.IGNORECASE)
        if abs_pattern:
            abstract = abs_pattern.group(2).strip()
        else:
            # Fallback: find all content after ABSTRACT (until two consecutive newlines or INTRODUCTION)
            abs_pattern_simple = re.search(r'(?:^|\n)(ABSTRACT|Abstract)\s*\n\s*(.*?)(?:\n\n|\n(?:1\s+)?(?:INTRODUCTION|Introduction)|$)', full_text, re.DOTALL | re.IGNORECASE)
            if abs_pattern_simple:
                abstract = abs_pattern_simple.group(2).strip()[:1500]  # Truncate to avoid length

    # --- 4. Extract Partial Content (Introduction and after) ---
    # We only want part starting from Introduction, not full text, to avoid token explosion
    content_extract = ""
    
    if full_text:
        # Find start position of Introduction
        intro_match = re.search(r'\n(1\s+)?(INTRODUCTION|Introduction)', full_text)
        
        if intro_match:
            start_idx = intro_match.start()
            # Extract content after Introduction starts
            # Limit length: e.g., 3000 chars, enough for LLM to judge quality/style but not too long
            raw_content = full_text[start_idx:]
            content_extract = raw_content[:3500].strip() 
        else:
            # If Introduction marker not found, and we have extracted abstract
            # Then take content after abstract (if available)
            if abstract and abstract in full_text:
                parts = full_text.split(abstract)
                if len(parts) > 1:
                    content_extract = parts[1][:3500].strip()
            else:
                # As last resort, take latter half of full text
                content_extract = full_text[:3500].strip()
    
    # Clean up extra newlines
    title = re.sub(r'\s+', ' ', title).strip()
    abstract = re.sub(r'\\n+', ' ', abstract).strip() # Abstract is usually a paragraph
    content_extract = re.sub(r'\\n{3,}', '\\n\\n', content_extract).strip() # Preserve paragraph structure
    
    return title, abstract, content_extract

def construct_dataset(args):
    # 1. Load Data
    print("Loading datasets...")
    spotlights = load_jsonl(args.spotlight_path)
    rejects = load_jsonl(args.reject_path)
    
    if not spotlights or not rejects:
        print("Error: Datasets are empty or missing.")
        return

    # Pre-calculate scores for rejects
    rejects_pool = []
    for p in rejects:
        s = get_avg_score(p)
        if s > 0: 
            rejects_pool.append((p, s))
    
    final_dataset = []
    
    print(f"Constructing {args.num_samples} samples...")
    for i in tqdm(range(args.num_samples)):
        # A. Select 1 Gold Paper
        gold_paper = random.choice(spotlights)
        gold_score = get_avg_score(gold_paper)
        
        # B. Select 4 Distractors
        # Enforce BOTH sides of the epsilon-bound (Eq. 11): the distractor must
        # score strictly below the gold paper (0 < S(o*) - S(o_i)) and within
        # the 2.5-point band (S(o*) - S(o_i) <= 2.5).
        score_threshold = gold_score - 2.5
        candidates = [x for x in rejects_pool if score_threshold <= x[1] < gold_score]
        
        distractors_data = []
        if len(candidates) >= 4:
            distractors_data = random.sample(candidates, 4)
        else:
            distractors_data = list(candidates)
            needed = 4 - len(distractors_data)
            others = sorted([x for x in rejects_pool if x[1] < score_threshold], key=lambda x: x[1], reverse=True)
            distractors_data.extend(others[:needed])
            below_gold = [x for x in rejects_pool if x[1] < gold_score]
            while len(distractors_data) < 4 and below_gold:
                distractors_data.append(random.choice(below_gold))
            if len(distractors_data) < 4:
                continue  # no admissible distractors below gold for this sample

        # C. Combine
        options = []
        options.append({'paper': gold_paper, 'role': 'gold', 'score': gold_score})
        for p, s in distractors_data:
            options.append({'paper': p, 'role': 'reject', 'score': s})
            
        # D. Shuffle
        random.shuffle(options)
        
        # E. Build Entry
        entry = {
            "id": f"sample_{i}",
            "gold_index": -1,
            "papers": [],
            "meta": {
                "gold_avg_score": gold_score,
                "score_threshold": score_threshold
            }
        }
        
        for idx, opt in enumerate(options):
            if opt['role'] == 'gold':
                entry['gold_index'] = idx
            
            # === Here call Parse Function ===
            # Parse and separate content
            p_title, p_abstract, p_content = parse_paper_elements(opt['paper'])
            
            candidate_obj = {
                "paper_idx": idx,
                "role": opt['role'],
                "avg_score": opt['score'],
                
                # Save parsed content structurally
                "paper_info": {
                    "title": p_title,
                    "abstract": p_abstract,
                    "partial_content": p_content,
                    # If review comments need to be kept, retrieve from raw data
                    "review_contexts": opt['paper'].get('review_contexts', []),
                    "rates": opt['paper'].get('rates', [])
                }
            }
            entry['papers'].append(candidate_obj)
            
        final_dataset.append(entry)

    # 4. Save
    print(f"Saving to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done. {len(final_dataset)} samples generated.")

def _run_build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spotlight_path", type=str, default=str(WORK_DIR / 'datasets/decision_accept_spotlight.jsonl'))
    parser.add_argument("--reject_path", type=str, default=str(WORK_DIR / 'datasets/decision_reject.jsonl'))
    parser.add_argument("--output_file", type=str, default=str(WORK_DIR / 'datasets/selector_dataset_raw.jsonl'))
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
    construct_dataset(args)

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
