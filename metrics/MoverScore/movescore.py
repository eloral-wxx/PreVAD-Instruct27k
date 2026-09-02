import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv
import time
from glob import glob
from tqdm import tqdm
from moverscore import get_idf_dict, word_mover_score  # pip install moverscore

# ====================
# MoverScore calculation
# ====================
def calculate_moverscore(hypotheses, references):
    """
    Compute MoverScore for a list of hypotheses and references
    Returns scores normalized to 0-1
    """
    idf_hyp = get_idf_dict(hypotheses)
    idf_ref = get_idf_dict(references)

    scores = word_mover_score(
        references, hypotheses,
        idf_ref, idf_hyp,
        stop_words=[], n_gram=1, remove_subwords=True
    )

    scores = np.array(scores)
    return ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8)).tolist()

# ====================
# Field detection
# ====================
def get_fields_from_filename(filename):
    """Detect hypothesis and reference fields based on filename"""
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

# ====================
# Model name detection
# ====================
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

# ====================
# Batch processing
# ====================
def batch_process_folder(input_folder, output_csv):
    """Process all JSON files in a folder and save MoverScore results to CSV"""
    files = glob(os.path.join(input_folder, "*.json"))
    total_files = len(files)

    if total_files == 0:
        print("No JSON files found")
        return

    print(f"Found {total_files} JSON files to process")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'index', 'model', 'moverscore'])

        pbar = tqdm(total=total_files, desc="Processing", unit="file")
        processed = 0
        start_time = time.time()

        for file_path in files:
            filename = os.path.basename(file_path)
            pbar.set_description(f"Processing: {filename[:20]}...")

            hyp_field, ref_field = get_fields_from_filename(filename)
            model_name = detect_model_name(filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)

                # Filter valid data
                valid = [x for x in data if hyp_field in x and ref_field in x]
                if not valid:
                    print(f"Warning: No valid data in {filename}")
                    pbar.update(1)
                    continue

                hypotheses = [x[hyp_field] for x in valid]
                references = [x[ref_field] for x in valid]

                scores = calculate_moverscore(hypotheses, references)

                for idx, score in enumerate(scores):
                    writer.writerow([filename, idx, model_name, score])

                processed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / processed
                remaining = avg_time * (total_files - processed)

                pbar.update(1)
                pbar.set_postfix({
                    "Processed": f"{processed}/{total_files}",
                    "ETA": f"{remaining/60:.1f} min"
                })

            except Exception as e:
                print(f"Error processing {filename}: {str(e)[:200]}")
                pbar.update(1)
                continue

    total_time = time.time() - start_time
    print(f"\nFinished processing {processed}/{total_files} files")
    print(f"Total time: {total_time/60:.2f} min, Avg per file: {total_time/processed:.2f} s")
    print(f"Results saved to: {output_csv}")

# ====================
# Example usage
# ====================
if __name__ == "__main__":
    input_folder = "PreVAD-Instruct27k/metrics/model-answers/"
    output_csv = "PreVAD-Instruct27k/metrics/moverscore_results.csv"
    batch_process_folder(input_folder, output_csv)
