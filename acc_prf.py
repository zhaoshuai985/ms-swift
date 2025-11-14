import json
import argparse
import re
from collections import Counter
import os
import glob

def load_jsonl(file_path):
    """Load and parse a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        # print(f"Loaded {len(data)} samples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSONL format in {file_path}.")
        exit(1)

def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def extract_answer_from_content(content):
    """
    智能提取答案，兼容多种格式。
    
    支持格式：
    1. <answer>...</answer> 标签格式（GRPO训练输出）
    2. 纯文本格式（旧格式）
    
    Args:
        content: assistant的回复内容
    
    Returns:
        提取的答案字符串
    
    Examples:
        >>> extract_answer_from_content("Yes")
        "Yes"
        >>> extract_answer_from_content("<answer>Yes</answer>")
        "Yes"
        >>> extract_answer_from_content("<think>...</think>\\n<answer>Yes</answer>")
        "Yes"
    """
    # 1. 尝试从 <answer> 标签提取
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # 如果答案是多行的，只取第一行（通常答案应该是简短的）
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
        return answer
    
    # 2. 回退：没有标签，使用第一行（兼容旧格式）
    return content.split('\n')[0].strip()

def extract_answers(sample):
    """Extract ground truth and predicted answers from a sample."""
    true_answer = sample.get("answer", "").strip()
    messages = sample.get("messages", [])
    pred_answer = ""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "").strip()
            # 智能提取答案（支持标签格式和纯文本格式）
            pred_answer = extract_answer_from_content(content)
            break
    if not true_answer or not pred_answer:
        print(f"Warning: Missing answer in sample qid={sample.get('qid', 'unknown')}")
    return true_answer, pred_answer

def preprocess_text(text, return_words=False):
    """Preprocess text: lowercase, remove punctuation, replace multiple spaces with single space.
    If return_words is True, return list of words; else, return normalized text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text.strip())  # Replace multiple spaces with single space
    if return_words:
        return text.split()
    return text

def is_correct(pred, true):
    """Check if predicted answer matches true answer exactly after normalization."""
    norm_pred = preprocess_text(pred)
    norm_true = preprocess_text(true)
    return norm_pred == norm_true

def compute_prf1(pred, true):
    """Compute Precision, Recall, and F1 based on word-level matching after normalization."""
    pred_words = set(preprocess_text(pred, return_words=True))
    true_words = set(preprocess_text(true, return_words=True))
    
    if not pred_words and not true_words:
        return 1.0, 1.0, 1.0  # Both empty, consider perfect match
    if not pred_words:
        return 0.0, 0.0, 0.0  # Predicted empty, no match
    if not true_words:
        return 0.0, 0.0, 0.0  # True empty, no match
    
    common_words = pred_words & true_words
    precision = len(common_words) / len(pred_words)
    recall = len(common_words) / len(true_words)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def collect_jsonl_files(input_path):
    """Collect all jsonl files from input path (file or directory)."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Look for checkpoint-*/infer_result/*.jsonl pattern
        pattern = os.path.join(input_path, "checkpoint-*", "infer_result", "*.jsonl")
        files = glob.glob(pattern)
        if not files:
            # Fallback: look for any jsonl files in the directory
            pattern = os.path.join(input_path, "*.jsonl")
            files = glob.glob(pattern)
            return sorted(files)
        else:
            # Sort by checkpoint number naturally
            def extract_checkpoint_number(file_path):
                """Extract checkpoint number from file path for sorting."""
                try:
                    # Extract checkpoint-XXX part from path
                    checkpoint_match = re.search(r'checkpoint-(\d+)', file_path)
                    if checkpoint_match:
                        return int(checkpoint_match.group(1))
                    return 0
                except:
                    return 0
            
            return sorted(files, key=extract_checkpoint_number)
    else:
        print(f"Error: Path {input_path} does not exist.")
        exit(1)

def process_single_file(jsonl_file, save_splits=False):
    """Process a single jsonl file and return results."""
    # Load data
    data = load_jsonl(jsonl_file)

    # Separate closed and open questions
    closed_samples = [s for s in data if s.get("answer_type") == "CLOSED"]
    open_samples = [s for s in data if s.get("answer_type") == "OPEN"]
    # print(f"Found {len(closed_samples)} CLOSED samples and {len(open_samples)} OPEN samples.")

    # Lists to store correct and incorrect samples if saving splits
    if save_splits:
        correct_samples = []
        incorrect_samples = []

    # Compute accuracy for closed questions
    if closed_samples:
        closed_correct = 0
        for sample in closed_samples:
            if is_correct(*extract_answers(sample)):
                closed_correct += 1
                if save_splits:
                    correct_samples.append(sample)
            else:
                if save_splits:
                    incorrect_samples.append(sample)
        closed_accuracy = closed_correct / len(closed_samples)
    else:
        # print("No CLOSED samples found.")
        closed_accuracy = 0.0
        closed_correct = 0

    # Compute metrics for open questions
    if open_samples:
        open_correct = 0
        precisions, recalls, f1s = [], [], []
        for sample in open_samples:
            true, pred = extract_answers(sample)
            if is_correct(pred, true):
                open_correct += 1
                if save_splits:
                    correct_samples.append(sample)
            else:
                if save_splits:
                    incorrect_samples.append(sample)
            p, r, f = compute_prf1(pred, true)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        open_accuracy = open_correct / len(open_samples)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
    else:
        # print("No OPEN samples found.")
        open_accuracy = avg_precision = avg_recall = avg_f1 = 0.0
        open_correct = 0

    # Overall accuracy
    total_samples = len(closed_samples) + len(open_samples)
    if total_samples > 0:
        total_correct = closed_correct + open_correct
        overall_accuracy = total_correct / total_samples
    else:
        overall_accuracy = 0.0
        total_correct = 0

    # Save splits if requested
    if save_splits:
        input_dir = os.path.dirname(jsonl_file)
        input_basename = os.path.splitext(os.path.basename(jsonl_file))[0]
        
        correct_file = os.path.join(input_dir, f"{input_basename}_correct.jsonl")
        incorrect_file = os.path.join(input_dir, f"{input_basename}_incorrect.jsonl")
        
        save_jsonl(correct_samples, correct_file)
        save_jsonl(incorrect_samples, incorrect_file)
        print(f"Saved {len(correct_samples)} correct samples to {correct_file}")
        print(f"Saved {len(incorrect_samples)} incorrect samples to {incorrect_file}")

    return {
        'file': jsonl_file,
        'open_accuracy': open_accuracy,
        'closed_accuracy': closed_accuracy,
        'overall_accuracy': overall_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'open_correct': open_correct,
        'open_total': len(open_samples),
        'closed_correct': closed_correct,
        'closed_total': len(closed_samples),
        'total_correct': total_correct,
        'total_samples': total_samples
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute performance metrics for VQA predictions.")
    parser.add_argument("input_path", help="Path to the JSONL file or directory containing JSONL files.")
    parser.add_argument("--save_splits", action="store_true", help="Save correct and incorrect samples to separate files.")
    args = parser.parse_args()

    # Collect all jsonl files to process
    jsonl_files = collect_jsonl_files(args.input_path)
    
    if not jsonl_files:
        print(f"No JSONL files found in {args.input_path}")
        exit(1)

    # Process each file
    results = []
    for jsonl_file in jsonl_files:
        print(f"Processing: {jsonl_file}")
        result = process_single_file(jsonl_file, args.save_splits)
        results.append(result)
        
        # Print result for this file
        print(f"Acc[{result['open_accuracy']:.4f}_{result['closed_accuracy']:.4f}_{result['overall_accuracy']:.4f}]_PRF[{result['avg_precision']:.4f}_{result['avg_recall']:.4f}_{result['avg_f1']:.4f}]_Count[{result['open_correct']}/{result['open_total']}_{result['closed_correct']}/{result['closed_total']}_{result['total_correct']}/{result['total_samples']}]")

    # If processing multiple files, show summary
    if len(results) > 1:
        print("\n" + "="*50)
        print("SUMMARY:")
        
        # Calculate overall metrics
        total_open_correct = sum(r['open_correct'] for r in results)
        total_open_samples = sum(r['open_total'] for r in results)
        total_closed_correct = sum(r['closed_correct'] for r in results)
        total_closed_samples = sum(r['closed_total'] for r in results)
        total_correct = sum(r['total_correct'] for r in results)
        total_samples = sum(r['total_samples'] for r in results)
        
        overall_open_acc = total_open_correct / total_open_samples if total_open_samples > 0 else 0.0
        overall_closed_acc = total_closed_correct / total_closed_samples if total_closed_samples > 0 else 0.0
        overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Calculate average PRF metrics
        valid_results = [r for r in results if r['open_total'] > 0]
        avg_precision = sum(r['avg_precision'] for r in valid_results) / len(valid_results) if valid_results else 0.0
        avg_recall = sum(r['avg_recall'] for r in valid_results) / len(valid_results) if valid_results else 0.0
        avg_f1 = sum(r['avg_f1'] for r in valid_results) / len(valid_results) if valid_results else 0.0
        
        print(f"Overall_Acc[{overall_open_acc:.4f}_{overall_closed_acc:.4f}_{overall_acc:.4f}]_PRF[{avg_precision:.4f}_{avg_recall:.4f}_{avg_f1:.4f}]_Count[{total_open_correct}/{total_open_samples}_{total_closed_correct}/{total_closed_samples}_{total_correct}/{total_samples}]")
        print(f"Processed {len(results)} files.")

if __name__ == "__main__":
    main()