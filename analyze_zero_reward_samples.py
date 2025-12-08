#!/usr/bin/env python3
"""
分析 GRPO 训练日志，统计全零奖励样本的分布情况
"""
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def load_completions(file_path: str) -> List[Dict]:
    """加载completions.jsonl文件"""
    completions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                completions.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
    return completions

def extract_reward_info(completion: Dict) -> Dict[str, Any]:
    """从completion中提取奖励信息"""
    # 检查是否有answer_match_cosine字段（这是主要的奖励函数）
    rewards = None
    
    if 'AnswerMatchCosine' in completion:
        rewards = completion['AnswerMatchCosine']
        if not isinstance(rewards, list):
            rewards = [rewards]
    elif 'answer_match_cosine' in completion:
        rewards = completion['answer_match_cosine']
        if not isinstance(rewards, list):
            rewards = [rewards]
    elif 'rewards' in completion:
        rewards = completion['rewards']
        if not isinstance(rewards, list):
            rewards = [rewards]
    elif 'reward' in completion:
        rewards = [completion['reward']]
    
    # 提取prompt（可能是list格式）
    prompt = completion.get('prompt', '')
    question_text = ''
    if isinstance(prompt, list):
        # 如果是列表，提取第一个元素（通常是完整的对话字符串）
        if len(prompt) > 0:
            prompt_str = prompt[0] if isinstance(prompt[0], str) else str(prompt[0])
            # 尝试从prompt字符串中提取user的问题部分
            # 查找 <|im_start|>user 之后的内容
            if '<|im_start|>user' in prompt_str:
                user_start = prompt_str.find('<|im_start|>user')
                user_content = prompt_str[user_start:]
                # 提取user消息的内容（到下一个<|im_start|>或<|im_end|>之前）
                if '<|im_end|>' in user_content:
                    user_end = user_content.find('<|im_end|>')
                    question_text = user_content[:user_end].replace('<|im_start|>user', '').strip()
                else:
                    question_text = user_content.replace('<|im_start|>user', '').strip()
            else:
                question_text = prompt_str[:300]  # 如果找不到，取前300字符
        prompt = question_text if question_text else (prompt[0] if prompt else '')
    elif not isinstance(prompt, str):
        prompt = str(prompt)
    else:
        # 如果是字符串，也尝试提取user部分
        if '<|im_start|>user' in prompt:
            user_start = prompt.find('<|im_start|>user')
            user_content = prompt[user_start:]
            if '<|im_end|>' in user_content:
                user_end = user_content.find('<|im_end|>')
                question_text = user_content[:user_end].replace('<|im_start|>user', '').strip()
            else:
                question_text = user_content.replace('<|im_start|>user', '').strip()
            prompt = question_text if question_text else prompt
    
    # 提取completion（可能是list格式）
    completion_text = completion.get('completion', '')
    if isinstance(completion_text, list):
        # 如果是列表，取第一个作为代表
        completion_text = completion_text[0] if completion_text else ''
    elif not isinstance(completion_text, str):
        completion_text = str(completion_text)
    
    return {
        'rewards': rewards,
        'prompt': prompt,
        'completion': completion_text,
        'answer': completion.get('answer', ''),
        'solution': completion.get('solution', ''),
        'question': completion.get('question', ''),
        'image_plane': completion.get('image_plane', ''),
        'image_modality': completion.get('image_modality', ''),
        'image_caption': completion.get('image_caption', ''),
        'image_title': completion.get('image_title', ''),
        'step': completion.get('step', -1),
    }

def process_completions(completions: List[Dict]) -> List[Dict]:
    """处理completions，每条记录已经是一个完整的组（包含8个generations）"""
    processed = []
    for comp in completions:
        reward_info = extract_reward_info(comp)
        processed.append({
            'raw': comp,
            'reward_info': reward_info,
        })
    return processed

def analyze_zero_reward_groups(processed: List[Dict]) -> Dict[str, Any]:
    """分析全零奖励的组"""
    stats = {
        'total_groups': len(processed),
        'zero_reward_groups': 0,
        'zero_reward_samples': [],
        'partial_zero_groups': 0,
        'reward_distribution': [],
        'group_sizes': [],
    }
    
    for idx, item in enumerate(processed):
        reward_info = item['reward_info']
        raw_comp = item['raw']
        
        # 提取这组的所有奖励（应该是一个包含8个值的列表）
        group_rewards = reward_info['rewards']
        if not group_rewards:
            group_rewards = [0.0] * 8  # 默认8个
        elif not isinstance(group_rewards, list):
            group_rewards = [group_rewards]
        
        # 确保是float类型
        group_rewards = [float(r) if r is not None else 0.0 for r in group_rewards]
        group_size = len(group_rewards)
        
        stats['group_sizes'].append(group_size)
        stats['reward_distribution'].extend(group_rewards)
        
        # 检查是否全零
        if all(r == 0.0 for r in group_rewards):
            stats['zero_reward_groups'] += 1
            # 保存这个组的详细信息
            sample_info = {
                'index': idx,
                'group_size': group_size,
                'rewards': group_rewards,
                'sample': raw_comp,  # 保存原始样本
                'reward_info': reward_info,
            }
            stats['zero_reward_samples'].append(sample_info)
        elif any(r == 0.0 for r in group_rewards):
            stats['partial_zero_groups'] += 1
    
    return stats

def analyze_zero_sample_characteristics(zero_samples: List[Dict]) -> Dict[str, Any]:
    """分析全零样本的特征"""
    characteristics = {
        'question_lengths': [],
        'completion_lengths': [],
        'answer_lengths': [],
        'has_solution': 0,
        'has_answer': 0,
        'image_planes': Counter(),
        'image_modalities': Counter(),
        'sample_questions': [],  # 保存前10个问题作为示例
    }
    
    for sample_info in zero_samples[:100]:  # 只分析前100个，避免内存问题
        reward_info = sample_info.get('reward_info')
        if not reward_info:
            continue
        
        # 问题长度
        question = reward_info.get('question', '') or reward_info.get('prompt', '')
        if question:
            characteristics['question_lengths'].append(len(question))
            if len(characteristics['sample_questions']) < 10:
                characteristics['sample_questions'].append(question[:200])  # 前200字符
        
        # 答案长度
        completion = reward_info.get('completion', '')
        if completion:
            characteristics['completion_lengths'].append(len(completion))
        
        answer = reward_info.get('answer', '')
        if answer:
            characteristics['answer_lengths'].append(len(answer))
            characteristics['has_answer'] += 1
        
        solution = reward_info.get('solution', '')
        if solution:
            characteristics['has_solution'] += 1
        
        # 图像特征
        plane = reward_info.get('image_plane', '')
        if plane:
            # 如果是列表，取第一个
            if isinstance(plane, list):
                plane = plane[0] if plane else ''
            if plane and isinstance(plane, str):
                characteristics['image_planes'][plane] += 1
        
        modality = reward_info.get('image_modality', '')
        if modality:
            # 如果是列表，取第一个
            if isinstance(modality, list):
                modality = modality[0] if modality else ''
            if modality and isinstance(modality, str):
                characteristics['image_modalities'][modality] += 1
    
    return characteristics

def print_statistics(stats: Dict, characteristics: Dict):
    """打印统计结果"""
    print("=" * 80)
    print("GRPO 全零奖励样本分析报告")
    print("=" * 80)
    print()
    
    # 基本统计
    total_groups = stats['total_groups']
    zero_groups = stats['zero_reward_groups']
    zero_ratio = (zero_groups / total_groups * 100) if total_groups > 0 else 0
    
    print(f"📊 基本统计:")
    print(f"  总组数: {total_groups}")
    print(f"  全零奖励组数: {zero_groups}")
    print(f"  全零比例: {zero_ratio:.2f}%")
    print(f"  部分零奖励组数: {stats['partial_zero_groups']}")
    print()
    
    # 组大小分布
    if stats['group_sizes']:
        group_sizes = stats['group_sizes']
        print(f"📦 组大小分布:")
        print(f"  平均组大小: {np.mean(group_sizes):.2f}")
        print(f"  组大小范围: {min(group_sizes)} - {max(group_sizes)}")
        print(f"  最常见的组大小: {Counter(group_sizes).most_common(3)}")
        print()
    
    # 奖励分布
    if stats['reward_distribution']:
        rewards = np.array(stats['reward_distribution'])
        print(f"🎯 奖励分数分布:")
        print(f"  平均奖励: {np.mean(rewards):.4f}")
        print(f"  标准差: {np.std(rewards):.4f}")
        print(f"  最小值: {np.min(rewards):.4f}")
        print(f"  最大值: {np.max(rewards):.4f}")
        print(f"  零奖励样本数: {np.sum(rewards == 0)}")
        print(f"  零奖励比例: {np.sum(rewards == 0) / len(rewards) * 100:.2f}%")
        print()
    
    # 全零样本特征
    if characteristics:
        print(f"🔍 全零样本特征分析:")
        
        if characteristics['question_lengths']:
            qlens = characteristics['question_lengths']
            print(f"  问题长度: 平均={np.mean(qlens):.1f}, 中位数={np.median(qlens):.1f}, 范围=[{min(qlens)}, {max(qlens)}]")
        
        if characteristics['completion_lengths']:
            clens = characteristics['completion_lengths']
            print(f"  生成答案长度: 平均={np.mean(clens):.1f}, 中位数={np.median(clens):.1f}, 范围=[{min(clens)}, {max(clens)}]")
        
        if characteristics['answer_lengths']:
            alens = characteristics['answer_lengths']
            print(f"  标准答案长度: 平均={np.mean(alens):.1f}, 中位数={np.median(alens):.1f}, 范围=[{min(alens)}, {max(alens)}]")
        
        print(f"  有标准答案的样本: {characteristics['has_answer']}/{len(stats['zero_reward_samples'])}")
        print(f"  有solution的样本: {characteristics['has_solution']}/{len(stats['zero_reward_samples'])}")
        print()
        
        if characteristics['image_planes']:
            print(f"  图像平面分布:")
            for plane, count in characteristics['image_planes'].most_common(5):
                print(f"    {plane}: {count}")
            print()
        
        if characteristics['image_modalities']:
            print(f"  图像模态分布:")
            for modality, count in characteristics['image_modalities'].most_common(5):
                print(f"    {modality}: {count}")
            print()
        
        if characteristics['sample_questions']:
            print(f"📝 示例问题（前10个全零样本）:")
            for i, q in enumerate(characteristics['sample_questions'][:10], 1):
                print(f"  {i}. {q[:150]}...")
            print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_zero_reward_samples.py <completions.jsonl>")
        sys.exit(1)
    
    completions_file = sys.argv[1]
    
    if not Path(completions_file).exists():
        print(f"Error: File not found: {completions_file}")
        sys.exit(1)
    
    print(f"正在加载文件: {completions_file}")
    completions = load_completions(completions_file)
    print(f"加载了 {len(completions)} 条记录")
    print()
    
    print("正在处理数据...")
    processed = process_completions(completions)
    print(f"处理了 {len(processed)} 条记录")
    print()
    
    print("正在分析全零奖励组...")
    stats = analyze_zero_reward_groups(processed)
    
    print("正在分析全零样本特征...")
    characteristics = analyze_zero_sample_characteristics(stats['zero_reward_samples'])
    
    print_statistics(stats, characteristics)
    
    # 保存详细结果到文件
    output_file = Path(completions_file).parent / "zero_reward_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': {
                'total_groups': stats['total_groups'],
                'zero_reward_groups': stats['zero_reward_groups'],
                'zero_ratio': stats['zero_reward_groups'] / stats['total_groups'] * 100 if stats['total_groups'] > 0 else 0,
            },
            'characteristics': {
                'avg_question_length': np.mean(characteristics['question_lengths']) if characteristics['question_lengths'] else 0,
                'avg_completion_length': np.mean(characteristics['completion_lengths']) if characteristics['completion_lengths'] else 0,
                'has_answer_ratio': characteristics['has_answer'] / len(stats['zero_reward_samples']) if stats['zero_reward_samples'] else 0,
                'top_image_planes': dict(characteristics['image_planes'].most_common(5)),
                'top_image_modalities': dict(characteristics['image_modalities'].most_common(5)),
            },
            'sample_questions': characteristics['sample_questions'][:20],
        }, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

