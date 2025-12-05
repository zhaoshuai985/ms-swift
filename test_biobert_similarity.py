#!/usr/bin/env python3
"""
测试脚本：使用 BioBERT 计算 yes 和 no 的相似度分数
"""

import sys
import os

# 添加 swift 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'swift'))

from swift.plugin.orm import AnswerMatchCosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_similarity():
    """测试 yes 和 no 的相似度"""
    
    print("=" * 60)
    print("BioBERT 相似度测试")
    print("=" * 60)
    print(f"模型: pritamdeka/S-BioBERT-snli-multinli-stsb")
    print()
    
    # 初始化 AnswerMatchCosine（使用当前配置）
    matcher = AnswerMatchCosine(
        model_name="pritamdeka/S-BioBERT-snli-multinli-stsb",
        threshold=0.70,
        smooth_reward=True
    )
    
    # 加载模型
    print("正在加载模型...")
    matcher._load_model()
    print("模型加载完成！\n")
    
    # 测试用例
    test_cases = [
        ("yes", "yes", "相同答案"),
        ("no", "no", "相同答案"),
        ("yes", "no", "不同答案（关键测试）"),
        ("no", "yes", "不同答案（反向）"),
        ("Yes", "yes", "大小写变化"),
        ("NO", "no", "大小写变化"),
    ]
    
    print("测试结果：")
    print("-" * 60)
    
    similarities = []
    for text1, text2, description in test_cases:
        # 归一化
        norm1 = matcher._normalize_answer(text1)
        norm2 = matcher._normalize_answer(text2)
        
        # 编码
        emb1 = matcher.model.encode([norm1], convert_to_tensor=False)[0]
        emb2 = matcher.model.encode([norm2], convert_to_tensor=False)[0]
        
        # 计算相似度
        sim = cosine_similarity([emb1], [emb2])[0, 0]
        sim_float = float(sim)
        similarities.append(sim_float)
        
        print(f"{description:20s} | '{text1}' vs '{text2}' | 相似度: {sim_float:.4f}")
    
    print("-" * 60)
    print()
    
    # 关键结果
    yes_yes_sim = similarities[0]
    no_no_sim = similarities[1]
    yes_no_sim = similarities[2]
    no_yes_sim = similarities[3]
    
    print("=" * 60)
    print("关键结果分析：")
    print("=" * 60)
    print(f"'yes' vs 'yes' 相似度: {yes_yes_sim:.4f}")
    print(f"'no' vs 'no' 相似度:   {no_no_sim:.4f}")
    print(f"'yes' vs 'no' 相似度:  {yes_no_sim:.4f}")
    print(f"'no' vs 'yes' 相似度:  {no_yes_sim:.4f}")
    print()
    
    # 阈值建议
    print("=" * 60)
    print("阈值建议：")
    print("=" * 60)
    
    # 计算 yes 和 no 之间的相似度
    cross_similarity = (yes_no_sim + no_yes_sim) / 2
    
    # 相同答案的相似度（应该是1.0）
    same_similarity = (yes_yes_sim + no_no_sim) / 2
    
    print(f"相同答案平均相似度: {same_similarity:.4f}")
    print(f"不同答案平均相似度: {cross_similarity:.4f}")
    print(f"相似度差距: {same_similarity - cross_similarity:.4f}")
    print()
    
    # 建议阈值
    if cross_similarity < 0.5:
        # yes 和 no 相似度很低，可以使用较高的阈值
        recommended_threshold = (cross_similarity + same_similarity) / 2
        print(f"✅ 推荐阈值: {recommended_threshold:.3f}")
        print(f"   理由: yes 和 no 相似度较低 ({cross_similarity:.3f})，")
        print(f"         可以使用 {(cross_similarity + same_similarity) / 2:.3f} 作为阈值")
        print(f"         这样可以准确区分 yes 和 no")
    elif cross_similarity < 0.7:
        # yes 和 no 相似度中等
        recommended_threshold = cross_similarity + 0.1
        print(f"⚠️  推荐阈值: {recommended_threshold:.3f}")
        print(f"   理由: yes 和 no 相似度中等 ({cross_similarity:.3f})，")
        print(f"         建议阈值设置在 {recommended_threshold:.3f} 以上")
        print(f"         当前阈值 0.70 可能不够严格")
    else:
        # yes 和 no 相似度较高，需要非常高的阈值
        recommended_threshold = cross_similarity + 0.15
        print(f"❌ 推荐阈值: {recommended_threshold:.3f}")
        print(f"   警告: yes 和 no 相似度较高 ({cross_similarity:.3f})，")
        print(f"         可能需要使用更高的阈值 ({recommended_threshold:.3f})")
        print(f"         或者考虑使用 AnswerMatchString 进行精确匹配")
    
    print()
    print("=" * 60)
    print("不同阈值下的判断结果：")
    print("=" * 60)
    
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    print(f"{'阈值':<8} | 'yes'='yes' | 'no'='no' | 'yes'='no' | 'no'='yes'")
    print("-" * 60)
    
    for threshold in thresholds:
        yes_yes_match = "✓" if yes_yes_sim > threshold else "✗"
        no_no_match = "✓" if no_no_sim > threshold else "✗"
        yes_no_match = "✓" if yes_no_sim > threshold else "✗"
        no_yes_match = "✓" if no_yes_sim > threshold else "✗"
        
        # 理想情况：yes=yes 和 no=no 应该匹配，yes=no 和 no=yes 不应该匹配
        is_ideal = (yes_yes_match == "✓" and no_no_match == "✓" and 
                   yes_no_match == "✗" and no_yes_match == "✗")
        ideal_mark = " ✅" if is_ideal else ""
        
        print(f"{threshold:<8.2f} |     {yes_yes_match}     |     {no_no_match}     |     {yes_no_match}     |     {no_yes_match}     {ideal_mark}")
    
    print()
    print("说明: ✓ 表示相似度 > 阈值（会被判定为匹配）")
    print("     ✗ 表示相似度 <= 阈值（会被判定为不匹配）")
    print("     ✅ 表示该阈值能正确区分 yes 和 no")

if __name__ == "__main__":
    test_similarity()

