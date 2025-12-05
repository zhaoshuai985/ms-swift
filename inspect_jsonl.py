import json
import sys

file_path = '/data/workspace/swift/output/v113-20251127-180718/completions.jsonl'

try:
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3: break # Check first 3 lines
            try:
                data = json.loads(line)
                print(f"--- Line {i+1} Keys ---")
                print(list(data.keys()))
                
                # Check for answer_match_cosine recursively or in known fields
                # Adjust based on what we see. For now, just dump top level.
                if 'reward_components' in data:
                    print("Found 'reward_components':", data['reward_components'])
                elif 'rewards' in data:
                    print("Found 'rewards':", data['rewards'])
                
                # Check if it's in some other field
                print(f"Sample of data: {str(data)[:200]}...")

            except json.JSONDecodeError:
                print(f"Line {i+1} is not valid JSON")
except FileNotFoundError:
    print(f"File not found: {file_path}")

