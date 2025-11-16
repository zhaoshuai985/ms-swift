# Caption Alignment 设置指南

## ✅ 完成状态

所有准备工作已完成：

- ✅ 依赖安装完成 (sentence-transformers, scikit-learn)
- ✅ 多个Embedding模型已预下载到本地缓存
- ✅ CaptionAlignment 类已集成到 orm.py
- ✅ image_caption 字段提取已集成到 grpo_trainer.py
- ✅ 测试脚本已验证功能正常
- ✅ 模型配置指南已生成

## 🚀 快速使用

### Step 1: 激活环境

```bash
conda activate rl  # 或 rl1
```

### Step 2: 修改 run.sh

在您的 `run.sh` 中，找到 `--reward_funcs` 参数并添加 `caption_alignment`：

```bash
# 原来:
--reward_funcs format answer_match plane_match modality_match

# 修改为:
--reward_funcs format answer_match plane_match modality_match caption_alignment
```

### Step 3: 启动训练

```bash
bash run.sh
```

就这样！CaptionAlignment 奖励函数会自动使用默认配置：
- 模型: `all-MiniLM-L6-v2` (轻量级，快速)
- 阈值: 0.70 (平衡)
- 平滑奖励: True

## 🔬 消融实验 - 选择不同的模型

### 查看所有可用模型

```bash
python caption_alignment_models.py --list
```

### 查看推荐配置

```bash
python caption_alignment_models.py --recommend
```

### 查看消融实验指南

```bash
python caption_alignment_models.py --ablation
```

## 🛠️ 自定义配置

如果您想使用特定的模型或超参数，需要修改 orm.py：

### 方法1: 修改默认配置

编辑 `/data/workspace/swift/swift/plugin/orm.py`，找到 orms 字典的初始化部分，修改实例化参数：

```python
orms = {
    # ... other orms ...
    'caption_alignment': CaptionAlignment(
        model_name="pritamdeka/S-BioBERT-snli-multinli-stsb",  # 改为您想要的模型
        threshold=0.70,
        smooth_reward=True
    ),
}
```

### 可选的模型选择

#### 轻量级 (快速，内存少)

```python
model_name="all-MiniLM-L6-v2"      # 默认，22M参数
model_name="paraphrase-MiniLM-L6-v2"  # 释义检测专用
```

#### 高质量 (精度高，稍慢)

```python
model_name="all-mpnet-base-v2"     # 109M参数，高精度
model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"  # QA优化
```

#### 医学专用 (⭐ 推荐用于VQA)

```python
model_name="pritamdeka/S-BioBERT-snli-multinli-stsb"  # 医学特定预训练
model_name="dmis-lab/biobert-base-cased"  # BioBERT医学模型
```

#### 科学论文

```python
model_name="allenai/scibert-base-uncased"  # 科学论文预训练
model_name="allenai/specter"  # 学术引用关系
```

### 超参数调整

```python
# 相似度阈值
threshold=0.65  # 激进，更容易获得奖励
threshold=0.70  # 平衡 (推荐)
threshold=0.75  # 保守，严格要求

# 奖励函数类型
smooth_reward=True   # 平滑奖励，0-1连续变化
smooth_reward=False  # 硬奖励，0或1二值选择
```

## 📊 监控训练

训练过程中，您可以在 `completions.jsonl` 中看到：

```json
{
  "caption_alignment_reward": 0.45,  # Caption对齐奖励
  "answer_match_reward": 1.0,
  "format_reward": 1.0,
  "plane_match_reward": 1.0,
  "modality_match_reward": 1.0,
  ...
}
```

### 期望值

- **caption_alignment_reward**: 40-60% (平均值 0.4-0.6)
  - 过低 (< 0.2): 考虑降低阈值
  - 过高 (> 0.9): 考虑提高阈值

## 🧪 消融实验建议

### 实验1: 模型对比

固定其他超参数，测试不同模型：

```python
# Run 1
model_name="all-MiniLM-L6-v2"  # 基准

# Run 2  
model_name="pritamdeka/S-BioBERT-snli-multinli-stsb"  # 医学模型

# Run 3
model_name="all-mpnet-base-v2"  # 高质量
```

### 实验2: 阈值灵敏度

固定模型，测试不同阈值：

```python
threshold=0.65  # Run 1
threshold=0.70  # Run 2
threshold=0.75  # Run 3
```

### 实验3: 奖励权重

在 run.sh 中调整权重配置：

```bash
# 假设 swift 支持 --reward_func_weights
--reward_func_weights 0.20 0.40 0.20 0.10 0.10
# format(0.20) answer_match(0.40) plane(0.20) modality(0.10) caption_alignment(0.10)
```

## 🔍 故障排查

### 问题1: 导入错误

```
ImportError: No module named 'sentence_transformers'
```

解决方案：
```bash
conda activate rl
pip install sentence-transformers scikit-learn
```

### 问题2: 模型下载失败

```
RuntimeError: Failed to load model ...
```

解决方案：
```bash
# 手动下载模型
conda run -n rl python3 << 'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
EOF
```

### 问题3: 显存不足

如果使用larger模型导致OOM：

```python
# 改用轻量级模型
model_name="all-MiniLM-L6-v2"  # 22M参数
```

### 问题4: 速度过慢

如果训练速度明显变慢：

```python
# 确保使用轻量级模型
model_name="all-MiniLM-L6-v2"

# 或检查是否正在CPU上运行（应该在GPU上）
```

## 📚 参考资源

- [Sentence-Transformers 文档](https://www.sbert.net/)
- [S-BioBERT 医学模型](https://huggingface.co/pritamdeka/S-BioBERT-snli-multinli-stsb)
- [BioBERT 医学预训练](https://github.com/dmis-lab/biobert)
- [SciBERT 科学论文](https://github.com/allenai/scibert)

## ✨ 提示

1. **首次使用**: 使用 `all-MiniLM-L6-v2` (默认)
   - 快速迭代
   - 足够的质量用于初步测试

2. **医学VQA最优**: 使用 `pritamdeka/S-BioBERT-snli-multinli-stsb`
   - 医学预训练
   - 对医学术语更敏感

3. **追求最高精度**: 使用 `all-mpnet-base-v2`
   - 更高的语义相似度准确度
   - 稍微慢一些但精度更好

4. **消融实验**: 
   - 每次只改一个变量
   - 记录所有配置和结果
   - 对比 caption_alignment_reward 和主任务指标

## 🎯 预期结果

启用 CaptionAlignment 后，您应该看到：

1. **caption_alignment_reward** 在 40-60% 之间
2. **答案准确率** 可能提升 3-5% (取决于模型和阈值)
3. **其他奖励** (format, plane, modality) 保持不变
4. **总体reward** 趋势向上

---

**准备好了吗？启动您的第一个实验吧！** 🚀

```bash
conda activate rl
bash run.sh
```

