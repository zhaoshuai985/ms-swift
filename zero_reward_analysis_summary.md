# GRPO 全零奖励样本分析报告

## 📊 核心发现

### 基本统计
- **总样本组数**: 5,001 组
- **全零奖励组数**: 888 组
- **全零比例**: **17.76%** ⚠️
- **部分零奖励组数**: 2,306 组

### 奖励分布
- **平均奖励**: 0.5815
- **标准差**: 0.4859
- **零奖励样本数**: 15,908 个（占所有generations的 39.76%）
- **组大小**: 所有组都是 8 个 generations（符合预期）

## 🔍 全零样本特征分析

### 文本特征
- **问题长度**: 平均 851.2 字符，中位数 847 字符
- **生成答案长度**: 平均 594.3 字符，中位数 566 字符
- **标准答案长度**: 平均 8.0 字符（非常短，通常是单词或短语）
- **有标准答案的样本**: 100/888 (11.3%) ⚠️

### 图像特征分布

**图像平面 (Image Plane)**:
- Axial: 69 个 (77.5%)
- PA: 28 个 (31.5%)
- AP: 3 个 (3.4%)

**图像模态 (Image Modality)**:
- CT: 38 个 (42.7%)
- X-Ray: 31 个 (34.8%)
- FLAIR: 13 个 (14.6%)
- T1: 6 个 (6.7%)
- T2: 6 个 (6.7%)

### 示例问题（全零样本）
1. What are the hypoattenuated round structures surrounding the veterbral column?
2. What is the pathology seen in the spleen?
3. Any abnormal findings in the lower lung fields?
4. What is most likely causing these lesions?
5. Is there an acute bleed present?
6. What brain territory is the hemorrhage located?
7. Are the lungs normal?
8. How was this film taken?
9. What is the form of the mass?
10. Is there a cyst in the left kidney?

## ⚠️ 关键问题

1. **梯度消失风险**: 17.76% 的样本组全部为 0 分，导致 GRPO 算法在这些样本上无法计算有效的优势值（Advantage），梯度更新无效。

2. **数据可用性**: 只有 11.3% 的全零样本有 `answer` 字段，但原始数据集 (`vqarad_train_rl.jsonl`) 中所有样本都有 `answer` 字段。这可能是因为：
   - 日志记录时没有保存 `answer` 字段
   - 或者字段名不匹配

3. **困难样本特征**: 
   - 主要集中在 Axial 平面的 CT 和 X-Ray 图像
   - 问题类型多样，没有明显的单一模式

## 💡 解决方案建议

### 方案一：Golden Truth Injection（黄金答案注入）✅ 推荐

**原理**: 当检测到 8 个答案全为 0 时，将其中一个（通常是第 8 个）的 completion 替换为 Ground Truth，并给予满分奖励。

**实施步骤**:
1. 在 `_generate_and_score_completions` 方法中，在计算 advantages 之前
2. 检测全零组：`if all(rewards == 0)`
3. 替换第 8 个 completion 的文本为 `answer` 字段（从原始数据中获取）
4. 将对应的 reward 设为 1.0
5. 重置/清除旧的 log_probs（如果存在）

**数据可用性确认**:
- ✅ 原始数据集有 `answer` 字段
- ✅ 原始数据集有 `medpix.image_caption` 和 `medpix.image_title` 字段
- ⚠️ 需要确保在训练时能访问到原始数据集的这些字段

**风险与缓解**:
- 风险：可能过度偏向 SFT，掩盖 RL 信号
- 缓解：设置注入比例（如 50%），或仅在训练初期启用

### 方案二：Relaxed Reward Criteria（松弛奖励标准）

**原理**: 对全零组启用备用奖励函数，给格式正确、包含关键词的答案给予微小正分。

**实施**: 修改 reward 函数，在全零情况下降低阈值或启用格式奖励。

### 方案三：Curriculum Filtering（课程学习）

**原理**: 在第一个 epoch 记录全零样本，后续 epoch 暂时剔除或降低采样概率。

## 📝 下一步行动

1. ✅ **已完成**: 统计分析全零样本分布
2. ⏳ **待执行**: 实施 Golden Truth Injection 方案
3. ⏳ **待验证**: 确认训练时能访问到原始数据集的 `answer` 字段
4. ⏳ **待测试**: 小规模实验验证注入方案的有效性

## 📁 相关文件

- 分析脚本: `/data/workspace/swift/analyze_zero_reward_samples.py`
- 详细结果: `/data/workspace/swift/output/v128-20251204-154030/zero_reward_analysis.json`
- 原始日志: `/data/workspace/swift/output/v128-20251204-154030/completions.jsonl`
- 训练数据集: `/data/datasets/vqarad/vqarad_train_rl.jsonl`

