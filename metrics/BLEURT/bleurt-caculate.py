import json
import os
import csv
import time
import numpy as np
from glob import glob
from tqdm import tqdm
from bleurt.score import BleurtScorer  # pip install bleurt

def normalize_bleurt_scores(scores):
    """Normalize raw BLEURT scores to 0-1 range using sigmoid"""
    scores = np.array(scores)
    return 1 / (1 + np.exp(-scores))

def calculate_bleurt(hypotheses, references, scorer, batch_size=32):
    """Compute BLEURT scores in batches to avoid memory issues"""
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        batch_hyp = hypotheses[i:i+batch_size]
        batch_ref = references[i:i+batch_size]
        batch_scores = scorer.score(references=batch_ref, candidates=batch_hyp)
        scores.extend(batch_scores)
    return scores

def get_fields_from_filename(filename):
    """Determine hypothesis and reference fields based on filename"""
    fname = filename.lower()
    # Hypothesis field
    if "qwen" in fname:
        hyp_field = "qwen_answer"
    elif "videollama3" in fname:
        hyp_field = "videollama3_answer"
    elif "videollava" in fname:
        hyp_field = "videollava_answer"
    elif "hawk" in fname:
        hyp_field = "hawk_answer"
    elif "holmesvau" in fname:
        hyp_field = "holmesvau_answer"
    else:
        hyp_field = "our_answer"
    # Reference field
    if "qa" in fname or "question" in fname:
        ref_field = "answer"
    elif "des" in fname or "description" in fname:
        ref_field = "corrected_descriptions"
    else:
        ref_field = "null"
    return hyp_field, ref_field

def detect_model_name(filename):
    """Extract model name from filename"""
    fname = filename.lower()
    if "videollama" in fname:
        return "videollama"
    elif "videollava" in fname:
        return "videollava"
    elif "qwen" in fname:
        return "qwen"
    elif "ablated" in fname:
        return "ablated"
    else:
        return "our_model"

def batch_process_folder(input_folder, output_csv):
    """Batch process JSON files and compute BLEURT scores"""
    try:
        scorer = BleurtScorer()
    except Exception as e:
        print(f"Failed to initialize BLEURT: {e}")
        return False

    files = sorted(glob(os.path.join(input_folder, "*.json")))
    if not files:
        print("No JSON files found")
        return False

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'index', 'model', 'bleurt_raw', 'bleurt_norm', 'hyp_len', 'ref_len'])

        start_time = time.time()
        processed = 0
        pbar = tqdm(total=len(files), desc="Processing", unit="file")
        for file_path in files:
            filename = os.path.basename(file_path)
            pbar.set_description(f"Processing: {filename[:20]}...")

            try:
                with open(file_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)

                hyp_field, ref_field = get_fields_from_filename(filename)
                valid = [x for x in data if hyp_field in x and ref_field in x
                         and isinstance(x[hyp_field], str) and isinstance(x[ref_field], str)]
                if not valid:
                    print(f"Warning: No valid data in {filename}")
                    continue

                hypotheses = [x[hyp_field] for x in valid]
                references = [x[ref_field] for x in valid]

                raw_scores = calculate_bleurt(hypotheses, references, scorer)
                norm_scores = normalize_bleurt_scores(raw_scores)
                hyp_lengths = [len(h) for h in hypotheses]
                ref_lengths = [len(r) for r in references]

                for idx, (raw, norm, h_len, r_len) in enumerate(zip(raw_scores, norm_scores, hyp_lengths, ref_lengths)):
                    writer.writerow([filename, idx, detect_model_name(filename), f"{raw:.4f}", f"{norm:.4f}", h_len, r_len])

                processed += 1
                pbar.update(1)

            except json.JSONDecodeError:
                print(f"Error: {filename} is not a valid JSON")
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)[:200]}")
                pbar.update(1)

        pbar.close()

    total_time = time.time() - start_time
    print(f"Processed {processed}/{len(files)} files")
    print(f"Total time: {total_time/60:.2f} min, Avg per file: {total_time/max(1, processed):.2f} s")
    print(f"Results saved to: {output_csv}")
    return True

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = ''
    input_folder = "PreVAD-Instruct27k/metrics/model-answers/"
    output_csv = "PreVAD-Instruct27k/metrics/bleurt_results.csv"
    success = batch_process_folder(input_folder, output_csv)
    if not success:
        print("Errors occurred during processing")
        exit(1)
