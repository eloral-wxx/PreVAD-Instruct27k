import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv
import time
import sys
from glob import glob
from tqdm import tqdm
import nltk

# ---------------- Path setup ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, 'external', 'UniEval'))

NLTK_DATA_DIR = os.path.join(BASE_DIR, 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR)

# Transformers offline cache
HF_CACHE_DIR = os.path.join(BASE_DIR, 'huggingface_hub')
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ---------------- UniEval imports ----------------
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

DESCRIPTION_SOURCE = "Briefly describe the anomalies occurring in the video."

# ---------------- Helper functions ----------------
def detect_model_name(filename):
    """Extract model name from filename"""
    filename_lower = filename.lower()
    if "videollama" in filename_lower:
        return "videollama"
    elif "videollava" in filename_lower:
        return "videollava"
    elif "qwen" in filename_lower:
        return "qwen"
    elif "ablated" in filename_lower or "消融" in filename_lower:
        return "ablated"
    else:
        return "our_model"

def get_fields_from_filename(filename):
    """Determine hypothesis and reference fields from filename"""
    filename_lower = filename.lower()
    if "qwen" in filename_lower:
        hyp_field = "qwen_answer"
    elif "videollama3" in filename_lower:
        hyp_field = "videollama3_answer"
    elif "videollava" in filename_lower:
        hyp_field = "videollava_answer"
    elif "hawk" in filename_lower:
        hyp_field = "hawk_answer"
    elif "holmesvau" in filename_lower:
        hyp_field = "holmesvau_answer"
    else:
        hyp_field = "our_answer"

    is_des_task = "des" in filename_lower
    if is_des_task:
        ref_field = "corrected_descriptions"
        src_field = DESCRIPTION_SOURCE
    elif "qa" in filename_lower:
        ref_field = "answer"
        src_field = "question"
    else:
        ref_field = "null"
        src_field = "null"
    return hyp_field, ref_field, src_field, is_des_task

# ---------------- UniEval evaluation ----------------
def calculate_unieval_multi(hypotheses, sources, references, dimensions):
    data = convert_to_json(
        output_list=hypotheses,
        src_list=sources,
        ref_list=references
    )
    evaluator = get_evaluator(task='summarization')
    eval_scores = evaluator.evaluate(data, dims=dimensions, overall=True, print_result=False)
    return eval_scores

def batch_process_folder(input_folder, output_csv, dimensions):
    """Process all JSON files in a folder and save UniEval scores"""
    file_paths = glob(os.path.join(input_folder, "*.json"))
    total_files = len(file_paths)
    if total_files == 0:
        print("No JSON files found")
        return

    print(f"Processing {total_files} JSON files | Dimensions: {dimensions + ['overall']}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as csv_file:
        header = ['filename', 'index', 'model'] + dimensions + ['overall']
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        progress_bar = tqdm(total=total_files, desc="Progress", unit="file")
        start_time = time.time()
        processed_count = 0
        error_count = 0

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            progress_bar.set_description(f"Processing: {filename[:20]}{'...' if len(filename) > 20 else ''}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)

                hyp_field, ref_field, src_field, is_des_task = get_fields_from_filename(filename)

                hypotheses, sources, references = [], [], []
                for item in dataset:
                    hyp = item.get(hyp_field, "")
                    ref = item.get(ref_field, "")
                    if not hyp or not ref:
                        continue
                    hypotheses.append(str(hyp).strip())
                    references.append(str(ref).strip())
                    sources.append(src_field if is_des_task else str(item.get(src_field, "")).strip())

                if not hypotheses:
                    progress_bar.write(f"Warning: {filename} contains no valid data")
                    error_count += 1
                    progress_bar.update(1)
                    continue

                eval_scores = calculate_unieval_multi(hypotheses, sources, references, dimensions)

                for i, score_dict in enumerate(eval_scores):
                    model_name = detect_model_name(filename)
                    row = [filename, i, model_name] + [score_dict.get(dim, "") for dim in dimensions] + [score_dict.get("overall", "")]
                    csv_writer.writerow(row)

                processed_count += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "Completed": f"{processed_count + error_count}/{total_files}",
                    "Success": processed_count,
                    "Fail": error_count,
                    "Last file overall": f"{score_dict.get('overall', 'N/A'):.3f}" if score_dict.get('overall') else "N/A"
                })

            except Exception as e:
                error_count += 1
                progress_bar.write(f"Error processing {filename}: {str(e)}")
                progress_bar.update(1)
                continue

    progress_bar.close()
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Processing completed! Total files: {total_files}")
    print(f"Success: {processed_count} | Fail: {error_count}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {output_csv}")
    print("="*50)

# ---------------- Run example ----------------
if __name__ == "__main__":
    input_folder = "PreVAD-Instruct27k/metrics/model-answers/"
    output_csv = "PreVAD-Instruct27k/metrics/unieval_results.csv"
    evaluation_dimensions = ['consistency', 'relevance', 'fluency', 'coherence']
    batch_process_folder(input_folder, output_csv, dimensions=evaluation_dimensions)
