# Code Review Report

## 1. 总体评估
对比官方基准 (`121f8fa4`) 与当前版本 (`HEAD`)，修改主要集中在引入 Medical VQA 相关的 Reward Functions (基于规则和 BERT)、适配 GRPO 训练流程以及优化推理输出格式。

**结论**: 核心算法逻辑基本正确，实现了预期的功能。但是，在**资源管理（显存/内存）**和**工程架构**方面存在一处**高危风险**，必须在运行大规模训练前修复。

## 2. 关键风险与建议

### 🔴 High Priority (高危风险)

#### 1. `swift/plugin/orm.py` 模块级模型加载
**位置**: `swift/plugin/orm.py` 末尾 `orms` 字典定义处。
**问题**: `AnswerMatchCosine` 和 `CaptionMatchCosine` 被直接**实例化**。
```python
'answer_match_cosine': AnswerMatchCosine(model_name=...),
'caption_match_cosine': CaptionMatchCosine(model_name=...),
```
**风险**:
- **显存爆炸**: 任何 import `swift.plugin.orm` 的代码（包括 Trainer, Infer, 甚至只做简单处理的脚本）都会触发 `SentenceTransformer` 模型的加载。
- **GPU 冲突**: `SentenceTransformer` 默认可能占用 GPU。在多卡分布式训练（DDP/DeepSpeed）中，每个进程都会加载该模型，导致与 LLM 主模型争抢显存，极易引发 OOM (Out of Memory)。
- **多进程死锁**: 在 Python 多进程环境（fork start method）下初始化 CUDA 上下文可能导致死锁。

**建议方案**:
1. **改为类注册**: 在 `orms` 字典中只注册类，不要注册实例。
2. **延迟加载 (Lazy Loading)**: 修改 `AnswerMatchCosine` 和 `CaptionMatchCosine` 的 `__init__` 方法，不要在初始化时加载模型，而是在第一次调用 `__call__` 时加载。或者使用 `functools.partial`。

### 🟠 Medium Priority (中等风险)

#### 2. `swift/llm/dataset/preprocessor/core.py` 禁用列过滤
**位置**: `RowPreprocessor.remove_useless_columns` 方法。
**问题**: 该方法现在直接返回 `dataset`，注释掉了原有的列筛选逻辑。
**风险**:
- **内存激增**: 数据集对象将携带所有原始字段（可能包含未处理的 Base64 图片、巨大的 JSON 文本等）。传递给 `DataLoader` 时会占用大量 RAM。
- **Collator 报错**: 如果保留的字段包含 PyTorch `default_collate` 无法处理的数据类型（如嵌套字典、None 值），训练循环会崩溃。

**建议**:
- 仅在推理阶段保留所有列，或在训练配置中显式过滤。
- 至少过滤掉已知会导致 collation 失败的复杂对象。

#### 3. `swift/llm/infer/infer.py` 全量数据加载
**位置**: `SwiftInfer.infer_dataset` 方法。
**问题**: `val_dataset = list(val_dataset)` 以及新增的 `original_data_list` 列表。
**风险**: 强制将整个验证集加载到内存中。对于大规模验证集（>10万条），可能导致内存溢出。
**建议**: 评估验证集大小。如果数据量大，应保持 Dataset 的迭代器特性，避免全量 list 化。

### 🟡 Low Priority (逻辑确认)

#### 4. Reward Function 平滑系数
**位置**: `orm.py` 中的 `smooth_reward` 计算。
**问题**: `AnswerMatchCosine` 使用系数 `10.0`，`CaptionMatchCosine` 使用系数 `2.0`。
**分析**: `(sim - threshold) * 10` 非常陡峭。例如 threshold=0.8，sim=0.9 时 reward 已经是 1.0。这使得 reward 实际上接近 binary（0或1），平滑效果有限。
**建议**: 确认这是符合预期的设计。

## 3. 逐文件详细审查

### `swift/plugin/orm.py`
- **[正确]** 新增的 `SmartAccuracy`, `AnswerMatchString` 等类逻辑正确，且处理了大小写归一化。
- **[正确]** `Format` 正则表达式更新正确，涵盖了 `<plane>`, `<modality>`, `<caption>`。
- **[风险]** `AnswerMatchCosine` 依赖 `sentence-transformers`，需确保环境中已安装该库且能正确下载模型（网络问题）。

### `swift/trainers/rlhf_trainer/grpo_trainer.py`
- **[正确]** 修复了 `reward_funcs` 初始化的逻辑，增加了 `inspect.isclass` 检查，从而兼容了上述 `orm.py` 中直接传实例的写法（虽然这种写法本身有风险，但 Trainer 代码做了防御性编程，这点很好）。
- **[正确]** `medpix` 字段提取逻辑正确，确保了 Reward Function 能获取到必要的 ground truth。
- **[建议]** 日志记录部分增加了 `answer`, `image_plane` 等字段，这对调试非常有帮助。

### `swift/llm/infer/infer.py`
- **[正确]** `_extract_original_images` 处理了 dict/string/list 等多种图片格式，鲁棒性强。
- **[正确]** 输出结构重构（引入 `meta` 字段）使得最终的 JSONL 更整洁，且保留了原始信息以便后续评估。

## 4. 训练参数配置审查与调优建议 (New)

### 现状分析 (基于 `run.sh`)
```bash
swift rlhf \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_steps 10000 \
  --reward_weights 0.05 0.45 0.05 0.05 0.40 \
  --vllm_gpu_memory_utilization 0.50
```

### 关键问题与优化建议

#### 1. Group Size (`num_generations`) 偏小
- **现状**: `num_generations=4`。
- **分析**: GRPO 的核心优势在于通过组内对比 (Group Relative) 来减少 Advantage 估计的方差。DeepSeek-R1 论文推荐 Group Size=64。当前值 4 偏小，可能导致基准线 (Baseline) 估计不准，训练不稳。
- **建议**: 在显存允许的情况下，尽可能增大此值。建议尝试 **8 或 16**。如果遇到 OOM，可适当降低 `max_length` 或调整 `vllm_gpu_memory_utilization`。

#### 2. 有效 Batch Size 偏小
- **现状**: `devices(1) * batch(1) * group(4) * accum(4) = 16`。
- **分析**: 对于 RL 训练，Batch Size 过小会导致梯度方向震荡剧烈，难以收敛到最优策略。
- **建议**: 建议将有效 Batch Size 提升至 **64 或 128**。可以通过增加 `gradient_accumulation_steps` 来实现（例如设为 16）。

#### 3. 训练步数 (`max_steps`) 可能过长
- **现状**: `max_steps=10000`。
- **分析**: VQA-RAD 数据集较小（通常几千条）。10000 步可能会导致模型过拟合（记住答案而不是学会推理）。
- **建议**: 建议改为基于 Epoch 的设置，或者大幅减少步数（如 1000-2000 步），并配合 `eval_steps` 密切监控验证集 Reward。

#### 4. 奖励权重 (`reward_weights`) 策略
- **现状**: `Format(0.05)`, `Answer(0.45)`, `Caption(0.40)`。
- **分析**: Format 权重极低。如果模型初期生成格式混乱（无法解析出 xml tag），Answer 和 Caption 也将拿不到分，导致整个 Reward 接近 0，训练可能难以启动。
- **建议**:
    - **初期**: 建议适当提高 Format 权重（如 0.2-0.3），强迫模型先学会格式。
    - **后期**: 当前配置更注重语义正确性，这在模型已掌握格式后是合理的。

---
**总结建议**:
请优先修复 **`orm.py` 的模块级实例化问题**。这是目前代码中最大的隐患，可能导致训练任务无法启动或随机崩溃。修复后，该代码库逻辑扎实，适合作为论文实验的基础。
