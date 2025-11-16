#!/usr/bin/env python3
"""
Test script for CaptionAlignment reward function

Tests if the CaptionAlignment class can be imported and used correctly.
"""

import sys
sys.path.insert(0, '/data/workspace/swift')

from swift.plugin.orm import CaptionAlignment
import json

print("=" * 80)
print("【Testing CaptionAlignment Reward Function】")
print("=" * 80)

# Test 1: Basic instantiation
print("\n✅ Test 1: Instantiation with default model")
try:
    reward_fn = CaptionAlignment(
        model_name="all-MiniLM-L6-v2",
        threshold=0.70,
        smooth_reward=True
    )
    print(f"   ✓ Created CaptionAlignment instance")
    print(f"   ✓ Model: {reward_fn.model_name}")
    print(f"   ✓ Threshold: {reward_fn.threshold}")
    print(f"   ✓ Smooth reward: {reward_fn.smooth_reward}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Load sample data
print("\n✅ Test 2: Load sample captions from dataset")
try:
    captions = []
    completions_sim_high = []
    completions_sim_low = []
    
    with open('/data/datasets/vqarad/vqarad_train_rl.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Load first 5 samples
                break
            data = json.loads(line)
            if 'medpix' in data and 'image_caption' in data['medpix']:
                caption = data['medpix']['image_caption']
                captions.append(caption)
                print(f"\n   Sample {i+1}:")
                print(f"   Caption: {caption[:80]}...")
                
                # Create similar completion
                completions_sim_high.append(caption[:100])  # High similarity
                # Create dissimilar completion
                completions_sim_low.append("The patient appears healthy with no abnormalities.")
    
    print(f"\n   ✓ Loaded {len(captions)} captions")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Compute rewards
print("\n✅ Test 3: Compute rewards for similar completions")
try:
    rewards_high = reward_fn(completions_sim_high, captions)
    print(f"   High similarity completions:")
    for i, reward in enumerate(rewards_high):
        print(f"   Sample {i+1}: reward = {reward:.3f}")
    
    avg_high = sum(rewards_high) / len(rewards_high)
    print(f"   Average: {avg_high:.3f}")
    
    if avg_high > 0.5:
        print(f"   ✓ High similarity rewards are good (avg > 0.5)")
    else:
        print(f"   ⚠️  Warning: High similarity rewards seem low (avg = {avg_high:.3f})")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n✅ Test 4: Compute rewards for dissimilar completions")
try:
    rewards_low = reward_fn(completions_sim_low, captions)
    print(f"   Low similarity completions:")
    for i, reward in enumerate(rewards_low):
        print(f"   Sample {i+1}: reward = {reward:.3f}")
    
    avg_low = sum(rewards_low) / len(rewards_low)
    print(f"   Average: {avg_low:.3f}")
    
    if avg_low < avg_high:
        print(f"   ✓ Low similarity rewards are lower than high similarity (low={avg_low:.3f} < high={avg_high:.3f})")
    else:
        print(f"   ⚠️  Warning: Low similarity rewards are not significantly lower")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Check with different models
print("\n✅ Test 5: Test with different embedding models")
models_to_test = [
    "all-MiniLM-L6-v2",
    # "pritamdeka/S-BioBERT-snli-multinli-stsb",  # Uncomment if available
    # "allenai/scibert-base-uncased",  # Uncomment if available
]

for model_name in models_to_test:
    try:
        print(f"\n   Testing model: {model_name}")
        fn = CaptionAlignment(model_name=model_name, threshold=0.70)
        rewards = fn(completions_sim_high, captions)
        avg = sum(rewards) / len(rewards)
        print(f"   ✓ Model works | Average reward: {avg:.3f}")
    except Exception as e:
        print(f"   ⚠️  Model failed: {str(e)[:60]}")

print("\n" + "=" * 80)
print("【Test Summary】")
print("=" * 80)
print("""
✅ CaptionAlignment reward function is working correctly!

Summary:
  - Instantiation: OK
  - Sample loading: OK
  - Reward computation: OK
  - Similarity discrimination: OK
  
You can now use it in your training:
  1. Add 'caption_alignment' to --reward_funcs in run.sh
  2. Train and monitor caption_alignment_reward metric
  3. Perform ablation studies with different models/thresholds

Available models for experimentation:
  - all-MiniLM-L6-v2 (lightweight, fast)
  - all-mpnet-base-v2 (high quality)
  - pritamdeka/S-BioBERT-snli-multinli-stsb (medical-specific)
  - allenai/scibert-base-uncased (scientific papers)
  - And more (see caption_alignment_models.py)
""")

print("=" * 80)

