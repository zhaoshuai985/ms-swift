import json
import re
import sys
from pathlib import Path

DEFAULT_FILE = '/data/workspace/swift/output/v113-20251127-180718/completions.jsonl'

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "[No <answer> tag found]"

def analyze(file_path, sample_limit=3, verbose=True):
    count_all_zeros = 0
    total_samples = 0
    
    if verbose:
        print(f"Scanning {file_path} for samples with all-zero AnswerMatchCosine...")
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                total_samples += 1
                try:
                    data = json.loads(line)
                    scores = data.get('AnswerMatchCosine', [])
                    
                    # Check if all scores are 0
                    if scores and all(s == 0.0 for s in scores):
                        count_all_zeros += 1
                        
                        # Print details for the first 3 such samples
                        if verbose and count_all_zeros <= sample_limit:
                            print(f"\n=== Sample {i+1} (All Zero Scores) ===")
                            
                            # Prompt (Question)
                            prompt = data.get('prompt', [''])[0] # Assuming list of same prompts
                            # Try to extract just the question part from the prompt (usually last user message)
                            # Or just print the last few chars if it's long
                            print(f"Prompt (preview): ...{prompt[-300:] if len(prompt)>300 else prompt}")
                            
                            # Ground Truth
                            answers = data.get('answer', [])
                            gt = answers[0] if answers else "N/A"
                            print(f"Ground Truth Answer: '{gt}'")
                            
                            # Generations (show first 3 out of 32 to save space)
                            completions = data.get('completion', [])
                            print(f"Generations (showing 3/{len(completions)}):")
                            for j, comp in enumerate(completions[:3]):
                                extracted = extract_answer(comp)
                                print(f"  Gen {j+1}: Extracted='{extracted}' | Full Len={len(comp)}")
                                print(f"         Full content preview: {comp[:100]}...")
                            
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print("File not found.")
        return

    if verbose:
        print(f"\nTotal samples scanned: {total_samples}")
        print(f"Samples with all-zero AnswerMatchCosine: {count_all_zeros}")

    return total_samples, count_all_zeros

if __name__ == "__main__":
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_FILE)
    analyze(str(file_path))

