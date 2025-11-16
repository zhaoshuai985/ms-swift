#!/usr/bin/env python3
"""
Caption Alignment 模型配置文件

该脚本提供各种Sentence-Transformer模型的配置信息，
便于您进行消融实验时选择不同的相似度模型。

使用方法:
  1. 查看所有可用模型: python caption_alignment_models.py --list
  2. 测试特定模型: python caption_alignment_models.py --test model_name
  3. 获取推荐配置: python caption_alignment_models.py --recommend
"""

import argparse
from typing import Dict, List, Tuple

# 模型配置表
MODEL_CONFIGS = {
    # 【通用模型】
    "all-MiniLM-L6-v2": {
        "category": "General Purpose",
        "params": "22M",
        "speed": "⚡⚡⚡ Very Fast",
        "quality": "⭐⭐⭐ Good",
        "use_case": "Default choice, lightweight, fast inference",
        "recommend_for": "First-time experiments, production deployment",
    },
    "all-mpnet-base-v2": {
        "category": "General Purpose",
        "params": "109M",
        "speed": "⚡⚡ Fast",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "High-quality embeddings, balanced performance",
        "recommend_for": "When accuracy is more important than speed",
    },
    "paraphrase-MiniLM-L6-v2": {
        "category": "General Purpose",
        "params": "22M",
        "speed": "⚡⚡⚡ Very Fast",
        "quality": "⭐⭐⭐ Good",
        "use_case": "Specialized in paraphrase detection",
        "recommend_for": "Answer variations and synonyms",
    },
    
    # 【医学专用模型】
    "pritamdeka/S-BioBERT-snli-multinli-stsb": {
        "category": "Medical/Biomedical",
        "params": "110M",
        "speed": "⚡⚡ Fast",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "Medical text similarity, trained on medical datasets",
        "recommend_for": "Medical VQA tasks (RECOMMENDED FOR YOUR TASK)",
    },
    "dmis-lab/biobert-base-cased": {
        "category": "Medical/Biomedical",
        "params": "110M",
        "speed": "⚡⚡ Fast",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "BioBERT pre-trained on biomedical corpus",
        "recommend_for": "Medical and biomedical text analysis",
    },
    
    # 【科学论文模型】
    "allenai/scibert-base-uncased": {
        "category": "Scientific Papers",
        "params": "110M",
        "speed": "⚡⚡ Fast",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "Scientific paper similarity, academic context",
        "recommend_for": "Medical literature and academic papers",
    },
    "allenai/specter": {
        "category": "Scientific Papers",
        "params": "109M (768D)",
        "speed": "⚡⚡ Fast",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "Citation and reference relationship modeling",
        "recommend_for": "Papers with strong semantic relationships",
    },
    
    # 【QA系统模型】
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
        "category": "Question-Answering",
        "params": "22M",
        "speed": "⚡⚡⚡ Very Fast",
        "quality": "⭐⭐⭐ Good",
        "use_case": "Optimized for QA similarity matching",
        "recommend_for": "VQA and question-answer matching",
    },
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
        "category": "Question-Answering",
        "params": "109M",
        "speed": "⚡⚡ Fast",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "High-quality QA similarity, dot product optimized",
        "recommend_for": "High-accuracy QA and VQA matching",
    },
    
    # 【多语言模型】
    "distiluse-base-multilingual-cased-v2": {
        "category": "Multilingual",
        "params": "135M",
        "speed": "⚡ Slower",
        "quality": "⭐⭐⭐⭐ Excellent",
        "use_case": "Multilingual cross-lingual understanding",
        "recommend_for": "If you need multilingual support",
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "category": "Multilingual",
        "params": "22M",
        "speed": "⚡⚡⚡ Very Fast",
        "quality": "⭐⭐⭐ Good",
        "use_case": "Lightweight multilingual paraphrase detection",
        "recommend_for": "Fast multilingual experiments",
    },
}

# 推荐配置
RECOMMENDED_CONFIGS = [
    {
        "name": "all-MiniLM-L6-v2",
        "use_case": "First-time experiment or production",
        "threshold": 0.70,
        "smooth_reward": True,
    },
    {
        "name": "pritamdeka/S-BioBERT-snli-multinli-stsb",
        "use_case": "Best for medical VQA (RECOMMENDED)",
        "threshold": 0.70,
        "smooth_reward": True,
    },
    {
        "name": "all-mpnet-base-v2",
        "use_case": "High-quality, balanced approach",
        "threshold": 0.70,
        "smooth_reward": True,
    },
]


def print_models_list():
    """Print all available models with details"""
    print("\n" + "=" * 100)
    print("【Available Caption Alignment Models】")
    print("=" * 100)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n📦 {model_name}")
        print(f"   Category:      {config['category']}")
        print(f"   Parameters:    {config['params']}")
        print(f"   Speed:         {config['speed']}")
        print(f"   Quality:       {config['quality']}")
        print(f"   Use Case:      {config['use_case']}")
        print(f"   Recommend for: {config['recommend_for']}")


def print_recommended():
    """Print recommended configurations for experiments"""
    print("\n" + "=" * 100)
    print("【Recommended Configurations for Your Experiments】")
    print("=" * 100)
    
    for i, config in enumerate(RECOMMENDED_CONFIGS, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Use Case:       {config['use_case']}")
        print(f"   Threshold:      {config['threshold']}")
        print(f"   Smooth Reward:  {config['smooth_reward']}")
        
        # Show how to use in run.sh
        print(f"\n   Usage in Python:")
        print(f"   ```python")
        print(f"   from swift.plugin.orm import CaptionAlignment")
        print(f"   ")
        print(f"   reward_fn = CaptionAlignment(")
        print(f"       model_name='{config['name']}',")
        print(f"       threshold={config['threshold']},")
        print(f"       smooth_reward={config['smooth_reward']}")
        print(f"   )")
        print(f"   ```")


def print_quick_start():
    """Print quick start guide"""
    print("\n" + "=" * 100)
    print("【Quick Start Guide】")
    print("=" * 100)
    
    print("""
1. Environment Setup:
   conda activate rl  # or rl1
   pip install sentence-transformers scikit-learn

2. Add to run.sh:
   --reward_funcs format answer_match plane_match modality_match caption_alignment

3. Choose Model (edit orm.py or create config):
   
   Default (lightweight, fast):
   CaptionAlignment(model_name="all-MiniLM-L6-v2")
   
   Medical (RECOMMENDED for VQA):
   CaptionAlignment(model_name="pritamdeka/S-BioBERT-snli-multinli-stsb")
   
   High-quality (slower but better):
   CaptionAlignment(model_name="all-mpnet-base-v2")

4. Hyperparameters for ablation experiments:
   - threshold: 0.65 (aggressive), 0.70 (balanced), 0.75 (conservative)
   - smooth_reward: True (smooth), False (hard)
   - weight in orms: 0.05 (minimal), 0.10 (balanced), 0.20 (strong)

5. Run training:
   bash run.sh

6. Monitor results:
   Check completions.jsonl for caption_alignment_reward metric
""")


def print_ablation_guide():
    """Print ablation study guide"""
    print("\n" + "=" * 100)
    print("【Ablation Study Guide】")
    print("=" * 100)
    
    print("""
Experiment Setup:

1. Model Comparison (fix other hyperparameters):
   Run 1: all-MiniLM-L6-v2 (baseline)
   Run 2: pritamdeka/S-BioBERT-snli-multinli-stsb (medical)
   Run 3: all-mpnet-base-v2 (high-quality)
   
   Expected: Medical model should give best results for VQA

2. Threshold Sensitivity:
   Run 1: threshold=0.65 (easier reward)
   Run 2: threshold=0.70 (balanced)
   Run 3: threshold=0.75 (strict)
   
   Expected: 0.70 should balance reward sparsity and informativeness

3. Smooth vs Hard Reward:
   Run 1: smooth_reward=True (continuous)
   Run 2: smooth_reward=False (binary)
   
   Expected: Smooth might have better gradient signal

4. Weight Impact:
   Run 1: weight=0.05 (minimal influence)
   Run 2: weight=0.10 (balanced with others)
   Run 3: weight=0.20 (strong influence)
   
   Expected: 0.10 should be sweet spot

Metrics to Track:
- answer_match accuracy (primary task)
- caption_alignment reward rate (40-60% is expected)
- format accuracy (should stay > 95%)
- plane/modality accuracy
- total reward trend
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Caption Alignment Models Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python caption_alignment_models.py --list
  python caption_alignment_models.py --recommend
  python caption_alignment_models.py --ablation
  python caption_alignment_models.py --quickstart
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--recommend", action="store_true", help="Show recommended configurations")
    parser.add_argument("--quickstart", action="store_true", help="Show quick start guide")
    parser.add_argument("--ablation", action="store_true", help="Show ablation study guide")
    
    args = parser.parse_args()
    
    if args.list:
        print_models_list()
    elif args.recommend:
        print_recommended()
    elif args.quickstart:
        print_quick_start()
    elif args.ablation:
        print_ablation_guide()
    else:
        # Default: show everything
        print_models_list()
        print_recommended()
        print_quick_start()
        print_ablation_guide()

