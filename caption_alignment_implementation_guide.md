# Image Caption 对齐奖励函数 - 完整实施指南

## 📋 快速决策表

| 方案 | 推荐度 | 精度 | 显存 | 速度 | 适合 |
|------|--------|------|------|------|------|
| **A. Cosine相似度** | ⭐⭐⭐⭐⭐ | 85-90% | +5% | 快 | ✅ 我们的情况 |
| B. BERT分类 | ⭐⭐⭐⭐ | 90-95% | +10-15% | 中 | 显存充足时 |
| C. LLM Judge | ⭐⭐⭐ | 95%+ | +100% | 慢 | ❌ 不推荐 |

**选择方案: 方案A (Cosine相似度)**

---

## 🎯 完整实施路线图

### 第1阶段: 准备 (预计 30 分钟)

```bash
# Step 1: 安装依赖
pip install sentence-transformers scikit-learn

# Step 2: 下载预训练模型 (首次会自动下载)
# 可选在这里提前下载避免训练时延迟
python3 << 'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("模型已下载")
EOF
```

### 第2阶段: 代码实现 (预计 2-3 小时)

#### 2.1 创建 CaptionAlignment 奖励函数

**文件**: `/data/workspace/swift/swift/plugin/orm.py`

**位置**: 在现有的 `ModalityMatch` 类后面添加

```python
# ============ 新增代码 ============

class CaptionAlignment(ORM):
    """
    基于Sentence-BERT的Image Caption对齐奖励函数
    
    衡量模型生成的completion是否与ground truth image caption语义相关
    
    使用场景:
      - 确保模型回复与图像描述对齐
      - 防止模型生成与图像无关的内容
      - 提升医学准确性
    
    示例:
      completion: "Multiple small infarcts in the MCA territory"
      caption: "Multiple small infarcts showing reduced diffusion..."
      相似度: 0.85 > 0.70 → reward = 1.0
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 threshold: float = 0.70,
                 smooth_reward: bool = True):
        """
        初始化CaptionAlignment奖励函数
        
        Args:
            model_name: SentenceTransformer模型名
              推荐值: "all-MiniLM-L6-v2" (轻量通用)
                    或 "pritamdeka/S-BioBERT-snli-multinli-stsb" (医学专用)
            threshold: 相似度阈值 (0-1)
              - 0.65: 激进，更容易获得奖励
              - 0.70: 平衡 (推荐)
              - 0.75: 保守，严格要求
            smooth_reward: 是否使用平滑奖励
              - True: reward = max(0, (similarity - threshold) * 2.0)
              - False: reward = 1.0 if similarity > threshold else 0.0
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        self.threshold = threshold
        self.smooth_reward = smooth_reward
    
    def __call__(self, completions, image_captions, **kwargs) -> List[float]:
        """
        计算Caption对齐奖励
        
        Args:
            completions: 模型生成的完整回复 (List[str])
            image_captions: 图像描述文本 (List[str])
            **kwargs: 其他参数 (兼容性)
        
        Returns:
            rewards: 每个样本的奖励 (List[float], 范围0-1)
        """
        rewards = []
        
        if not completions or not image_captions:
            return [0.0] * len(completions)
        
        try:
            # 批量编码completion和caption
            completion_embeddings = self.model.encode(
                completions, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            caption_embeddings = self.model.encode(
                image_captions,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # 计算cosine相似度
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = cosine_similarity(
                completion_embeddings, 
                caption_embeddings
            ).diagonal()  # 取对角线 (一一对应)
            
            # 转换为奖励
            for sim in similarities:
                if self.smooth_reward:
                    # 平滑奖励: 在阈值附近渐变
                    reward = max(0.0, (float(sim) - self.threshold) * 2.0)
                    reward = min(1.0, reward)  # 上限1.0
                else:
                    # 离散奖励: 0 or 1
                    reward = 1.0 if float(sim) > self.threshold else 0.0
                
                rewards.append(float(reward))
        
        except Exception as e:
            import logging
            logging.warning(f"CaptionAlignment计算异常: {e}")
            return [0.0] * len(completions)
        
        return rewards


# ============ 在orms字典中注册 ============

orms = {
    'format': Format(),
    'answer_match': AnswerMatch(),
    'plane_match': PlaneMatch(),
    'modality_match': ModalityMatch(),
    'caption_alignment': CaptionAlignment(),  # 新增
    # ... 其他现有的 ...
}
```

#### 2.2 修改 GRPO Trainer 以提取 image_caption

**文件**: `/data/workspace/swift/swift/trainers/rlhf_trainer/grpo_trainer.py`

**位置**: 在 `_generate_and_score_completions` 方法中，与 image_plane/image_modality 提取相同位置

**修改1** - 提取 image_caption (Line ~845):

```python
# 从medpix中提取image_caption到顶层
for inp in inputs:
    if 'medpix' in inp:
        if 'image_caption' not in inp and 'image_caption' in inp['medpix']:
            inp['image_caption'] = inp['medpix']['image_caption']
```

**修改2** - 添加到日志记录 (Line ~891, 可选):

```python
# 提取 image_caption 以供日志记录
if all('image_caption' in inp for inp in inputs):
    metrics_for_logs_to_gather['image_caption'] = [
        inp['image_caption'] for inp in inputs
    ]
```

#### 2.3 更新 run.sh 参数

**文件**: `/data/workspace/swift/run.sh`

**修改**:

```bash
# OLD
--reward_funcs format answer_match plane_match modality_match \

# NEW
--reward_funcs format answer_match plane_match modality_match caption_alignment \

# 同时设置权重 (如果支持)
--reward_func_weights 0.20 0.40 0.20 0.10 0.10 \
```

---

## 🧪 测试与验证

### 测试1: 单元测试 - CaptionAlignment功能

```python
# test_caption_alignment.py
from orm import CaptionAlignment
import json

# 初始化
reward_fn = CaptionAlignment(
    model_name="all-MiniLM-L6-v2",
    threshold=0.70,
    smooth_reward=True
)

# 加载样本数据
with open('/data/datasets/vqarad/vqarad_train_rl.jsonl', 'r') as f:
    samples = [json.loads(f.readline()) for _ in range(5)]

# 提取caption和模拟completion
captions = [s['medpix']['image_caption'] for s in samples]

# 生成模拟completion (与caption相关)
completions = [
    "The image shows infarcts with reduced diffusion",  # 高相似度
    "Multiple nodular opacities visible",                # 高相似度
    "Dense consolidation in the lower lobe",            # 高相似度
    "The patient has pneumonia",                         # 低相似度
    "No abnormalities detected",                         # 低相似度
]

# 计算奖励
rewards = reward_fn(completions, captions)

print("测试结果:")
for i, (comp, reward) in enumerate(zip(completions, rewards)):
    print(f"  样本{i+1}: reward={reward:.3f}, completion={comp[:40]}...")
```

**预期结果**:
- 前3个completion (与caption相关): reward > 0.5
- 后2个completion (与caption无关): reward < 0.3

### 测试2: 集成测试 - 与其他奖励的兼容性

```python
# test_integration.py
import subprocess

# 运行一个小规模训练 (100 steps)
cmd = [
    "swift", "rlhf",
    "--rlhf_type", "grpo",
    "--max_steps", "100",
    "--reward_funcs", "format answer_match plane_match modality_match caption_alignment",
    # ... 其他参数 ...
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

# 检查是否有错误
if "caption_alignment" in result.stdout:
    print("✅ CaptionAlignment成功集成")
else:
    print("❌ 集成失败")
```

---

## 📊 监控与调试

### 关键指标

在训练过程中监控以下指标 (从completions.jsonl):

```python
# monitor.py
import json
from collections import defaultdict

def analyze_rewards(jsonl_file):
    """分析奖励函数的表现"""
    
    rewards_dist = defaultdict(list)
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # 收集各奖励函数的值
            for key in ['format_reward', 'answer_match_reward', 
                       'plane_match_reward', 'modality_match_reward', 
                       'caption_alignment_reward']:
                if key in data:
                    rewards_dist[key].append(data[key])
    
    # 打印统计
    for key, values in rewards_dist.items():
        if values:
            print(f"{key}:")
            print(f"  平均: {sum(values)/len(values):.3f}")
            print(f"  最小: {min(values):.3f}")
            print(f"  最大: {max(values):.3f}")
            print(f"  非零比例: {sum(1 for v in values if v > 0)/len(values):.1%}")
```

### 调优指南

| 现象 | 可能原因 | 解决方案 |
|------|--------|--------|
| caption_reward < 30% | 阈值太高 | 降低 threshold (0.70 → 0.65) |
| caption_reward > 80% | 阈值太低或任务太简单 | 提高 threshold (0.70 → 0.75) |
| answer准确率下降 | caption权重太高 | 降低权重 (0.10 → 0.05) |
| 训练速度明显下降 | 模型太慢 | 改用 all-MiniLM-L6-v2 |
| CUDA OOM | 显存不足 | 使用CPU编码，或缓存embeddings |

---

## 🚀 完整实施清单

### 阶段1: 代码准备 (2-3小时)

- [ ] 安装 sentence-transformers
- [ ] 在 orm.py 中添加 CaptionAlignment 类
- [ ] 在 orms 字典中注册
- [ ] 在 grpo_trainer.py 中添加 image_caption 提取
- [ ] 修改 run.sh 添加 caption_alignment 奖励

### 阶段2: 测试验证 (1小时)

- [ ] 运行单元测试
- [ ] 验证奖励函数可调用
- [ ] 检查是否有import错误
- [ ] 手工测试3-5个样本

### 阶段3: 训练启动 (5-8小时)

- [ ] 更新参数 (从阶段1的优化)
- [ ] 启动训练
- [ ] 每500步检查一次
- [ ] 监控 caption_reward 的变化

### 阶段4: 效果评估 (1-2小时)

- [ ] 5000步后停止训练
- [ ] 收集各奖励的统计数据
- [ ] 对比是否有性能提升
- [ ] 决定是否调整参数

---

## ⚙️ 高级配置

### 模型选择对比

```python
# 选项1: 轻量通用 (推荐用于第一次实验)
model_name = "all-MiniLM-L6-v2"
# 参数: 22M, 精度: 85-90%, 速度: 最快, 显存: 最低

# 选项2: 医学专用 (如果效果不理想，升级到这个)
model_name = "pritamdeka/S-BioBERT-snli-multinli-stsb"
# 参数: 110M, 精度: 90-95%, 速度: 中等, 显存: 中等

# 选项3: 科学论文专用
model_name = "allenai/scibert-base-uncased"
# 参数: 110M, 精度: 88-93%, 速度: 中等, 显存: 中等
```

### 权重调整策略

```python
# 初期 (保守)
weights = [0.25, 0.40, 0.15, 0.10, 0.10]  # format, answer, plane, modality, caption

# 中期 (平衡) - 推荐
weights = [0.20, 0.40, 0.20, 0.10, 0.10]

# 后期 (激进) - 如果caption表现很好
weights = [0.15, 0.35, 0.20, 0.10, 0.20]

# 最激进 - 仅用于验证
weights = [0.10, 0.30, 0.20, 0.10, 0.30]
```

---

## 📈 预期结果

### 如果一切正常

✅ caption_alignment_reward 应在 40-60%  
✅ 其他奖励函数保持不变  
✅ 答案准确率提升 3-5%  
✅ 总训练时间增加 < 10%

### 如果有问题

❌ caption_reward 很低 (< 20%)  
  → 检查 image_caption 是否成功提取  
  → 尝试降低 threshold

❌ caption_reward 很高 (> 90%)  
  → 任务太简单，提高 threshold

❌ 其他奖励下降  
  → 降低 caption 权重

❌ 训练非常慢  
  → 使用更小的embedding模型

---

## 📚 参考资源

- [Sentence-Transformers 文档](https://www.sbert.net/)
- [BioBERT 论文](https://arxiv.org/abs/1901.08746)
- [Cosine Similarity 原理](https://en.wikipedia.org/wiki/Cosine_similarity)
- [医学NLP 最佳实践](https://github.com/dmis-lab/biobert)

---

**最后更新**: 2024-11-16  
**状态**: 准备执行 ✅

