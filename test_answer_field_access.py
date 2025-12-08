#!/usr/bin/env python3
"""
测试脚本：验证在 GRPO trainer 中能否访问 answer 字段
"""
import json
import sys
from pathlib import Path

def test_log_structure():
    """测试日志文件中的数据结构"""
    log_file = Path("/data/workspace/swift/output/v128-20251204-154030/completions.jsonl")
    
    print("=" * 80)
    print("测试1: 检查日志文件中的字段")
    print("=" * 80)
    
    with open(log_file, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    print(f"✅ answer 字段存在: {'answer' in data}")
    if 'answer' in data:
        answer = data['answer']
        print(f"   answer 类型: {type(answer)}")
        print(f"   answer 值: {answer}")
        if isinstance(answer, list):
            print(f"   answer 长度: {len(answer)}")
            print(f"   answer 唯一值: {set(answer) if len(set(answer)) < 10 else 'too many'}")
    
    print(f"\n✅ 其他相关字段:")
    for field in ['image_plane', 'image_modality', 'image_caption', 'image_title', 'medpix']:
        exists = field in data
        print(f"   {field}: {'存在' if exists else '不存在'}")
        if exists and field == 'medpix' and isinstance(data[field], dict):
            print(f"      medpix keys: {list(data[field].keys())[:10]}")

def test_dataset_structure():
    """测试原始数据集的结构"""
    dataset_file = Path("/data/datasets/vqarad/vqarad_train_rl.jsonl")
    
    print("\n" + "=" * 80)
    print("测试2: 检查原始数据集中的字段")
    print("=" * 80)
    
    with open(dataset_file, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    print(f"✅ answer 字段存在: {'answer' in data}")
    if 'answer' in data:
        print(f"   answer 值: {data['answer']}")
        print(f"   answer 类型: {type(data['answer'])}")
    
    print(f"\n✅ 数据集所有字段: {list(data.keys())}")
    
    if 'medpix' in data:
        medpix = data['medpix']
        if isinstance(medpix, dict):
            print(f"\n✅ medpix 字段包含:")
            for key in ['image_plane', 'image_modality', 'image_caption', 'image_title']:
                if key in medpix:
                    print(f"   {key}: {medpix[key][:50] if isinstance(medpix[key], str) else medpix[key]}")

def analyze_answer_consistency():
    """分析answer字段在日志中的一致性"""
    log_file = Path("/data/workspace/swift/output/v128-20251204-154030/completions.jsonl")
    
    print("\n" + "=" * 80)
    print("测试3: 分析answer字段的一致性")
    print("=" * 80)
    
    answer_types = {}
    answer_list_lengths = {}
    sample_count = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            if sample_count >= 100:  # 只检查前100个样本
                break
            data = json.loads(line)
            if 'answer' in data:
                answer = data['answer']
                answer_type = type(answer).__name__
                answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
                
                if isinstance(answer, list):
                    length = len(answer)
                    answer_list_lengths[length] = answer_list_lengths.get(length, 0) + 1
                    # 检查是否所有值相同
                    if len(set(answer)) == 1:
                        print(f"   样本 {sample_count}: answer是列表，所有值相同: {answer[0]}")
            sample_count += 1
    
    print(f"\n✅ answer字段类型分布: {answer_types}")
    print(f"✅ answer列表长度分布: {answer_list_lengths}")

def main():
    test_log_structure()
    test_dataset_structure()
    analyze_answer_consistency()
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
基于以上测试，我们可以确认：

1. ✅ 日志文件中确实包含 answer 字段
2. ✅ 原始数据集中也包含 answer 字段
3. ✅ 在 _generate_and_score_completions 方法中，inputs 应该包含 answer 字段
   （因为代码中有检查：if all('answer' in inp for inp in inputs)）

⚠️  注意事项：
- 日志中的 answer 可能是列表格式（8个值），对应8个generations
- 在实施注入时，需要从原始数据集中获取单个 answer 值
- 需要确认在 _generate_and_score_completions 开始时，inputs 中的 answer 是什么格式
    """)

if __name__ == '__main__':
    main()

