# Qwen-MATH-RL

用于 Qwen 系列模型在 MATH-500 数据集上进行推理强化学习（RL）的评测和对比框架。提供高效的模型下载、批量推理、答案判定和多模型对比功能。

## 项目内容

### 核心脚本

1. **[download_qwen3_from_mirror.py](download_qwen3_from_mirror.py)** - 从 Hugging Face 镜像下载 Qwen3 模型
2. **[eval_math500_deepthink.py](eval_math500_deepthink.py)** - 在 MATH-500 数据集上评测模型，支持单模型或多模型对比
3. **[qwen30.5B.ipynb](qwen30.5B.ipynb)** - Jupyter Notebook，展示推理过程、数据分析和可视化

### 数据文件

- **compare.jsonl** - 两个模型的对比评测结果（JSONL 格式），包含问题、黄金答案、两个模型的预测和正确性判定

---

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install torch transformers vllm datasets tqdm

# 如需从 Hugging Face 镜像下载，设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=your_token  # 如果模型需要认证
```

### 下载模型

使用 `download_qwen3_from_mirror.py` 从镜像下载 Qwen 模型：

```bash
python download_qwen3_from_mirror.py \
  --repo Qwen/Qwen3-1.7B \
  --out ./models/Qwen3-1.7B
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--repo` | 模型 repo ID，如 `Qwen/Qwen3-1.7B` | 必填 |
| `--out` | 输出目录 | 必填 |
| `--revision` | 分支/标签/commit（可选） | 无 |
| `--allow` | 文件白名单（glob 模式，空格分隔） | 全部下载 |
| `--ignore` | 文件黑名单（glob 模式，空格分隔） | 无 |
| `--resume` | 允许断点续传 | True |
| `--local-dir-use-symlinks` | 符号链接策略 `auto`/`true`/`false` | `auto` |

### 评测模型

使用 `eval_math500_deepthink.py` 在 MATH-500 上评测模型：

```bash
# 单模型评测，保存结果到 JSON
python eval_math500_deepthink.py \
  --model ./models/Qwen3-1.7B \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --save_jsonl output.jsonl

# 两个模型对比
python eval_math500_deepthink.py \
  --model ./models/ModelA \
  --compare_model ./models/ModelB \
  --batch_size 8 \
  --save_jsonl output_a.jsonl \
  --save_jsonl_compare compare.jsonl
```

**关键参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 主评测模型路径（merged 模型推荐） | 必填 |
| `--compare_model` | 对比模型路径（可选） | 无 |
| `--dataset_id` | 数据集 ID | `HuggingFaceH4/MATH-500` |
| `--split` | 数据集分割 | `test` |
| `--limit` | 限制评测题数（-1 为全部） | -1 |
| `--batch_size` | 批处理大小 | 8 |
| `--max_new_tokens` | 最大生成 token 数 | 4096 |
| `--temperature` | 采样温度 | 0.0 |
| `--top_p` | 核采样参数 | 1.0 |
| `--dtype` | 模型数据类型 | `bf16` |
| `--force_prefix` | 强制在生成前添加 `<start_deepthink>` | False |
| `--strict_format` | 严格要求答案格式，无 `<SOLUTION>` 标签则判错 | False |
| `--gpu_memory_utilization` | GPU 显存利用率 | 0.90 |
| `--tensor_parallel_size` | 张量并行大小 | 1 |
| `--save_jsonl` | 保存评测结果到 JSONL 文件 | 无 |
| `--save_jsonl_compare` | 保存对比结果到 JSONL 文件 | 无 |

---

## 输出格式

### 单模型结果 (output.jsonl)

每行是一个 JSON 对象，包含以下字段：

```json
{
  "idx": 0,
  "unique_id": "test/precalculus/807.json",
  "subject": "Precalculus",
  "level": 2,
  "problem": "Convert the point $(0,3)$ in rectangular coordinates...",
  "gold": "\\left( 3, \\frac{\\pi}{2} \\right)",
  "pred": "(3,\\frac{\\pi}{2})",
  "pred_text": "<start_deepthink>\n...\n<SOLUTION>...",
  "correct": true,
  "format_ok": true
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `idx` | 题目索引 |
| `unique_id` | 题目唯一标识 |
| `subject` | 学科 |
| `level` | 难度等级 (1-5) |
| `problem` | 问题文本 |
| `gold` | 黄金答案 |
| `pred` | 提取的模型答案 |
| `pred_text` | 完整模型输出文本 |
| `correct` | 答案是否正确（bool） |
| `format_ok` | 答案格式是否满足要求（bool） |

### 对比结果 (compare.jsonl)

每行包含两个模型的对比结果：

```json
{
  "idx": 0,
  "unique_id": "test/precalculus/807.json",
  "subject": "Precalculus",
  "level": 2,
  "problem": "...",
  "gold": "...",
  "model_pred": "(3,\\frac{\\pi}{2})",
  "model_correct": true,
  "model_format_ok": true,
  "model_pred_text": "...",
  "compare_pred": "...",
  "compare_correct": false,
  "compare_format_ok": true,
  "compare_pred_text": "..."
}
```

**对比字段：** 每个模型的结果前缀为 `model_` 或 `compare_`。

---

## 推理格式

模型默认采用以下格式进行推理：

```
<start_deepthink>
[深度思考过程]
</end_deepthink>
<SOLUTION>
[最终答案]
</SOLUTION>
```

使用 `--force_prefix` 可在生成开始前强制插入 `<start_deepthink>`，以提高格式兼容性。

---

## Notebook 使用

[qwen30.5B.ipynb](qwen30.5B.ipynb) 包含完整的推理流程、数据加载、答案判定和可视化。建议用于：

- **探索性分析**：查看模型推理过程和答案分布
- **调试**：逐行执行推理和判定逻辑
- **可视化**：生成准确度、难度等级分析图表

在 Jupyter 中打开：

```bash
jupyter notebook qwen30.5B.ipynb
```

---

## 常见问题

### Q: 如何加速下载？
**A:** 使用 `--allow` 参数只下载必要文件（如 `.safetensors`、`config.json`、`tokenizer.*`），避免下载不需要的训练产物。

### Q: 如何使用本地镜像？
**A:** 设置 `HF_ENDPOINT` 环境变量，如：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 如何在多 GPU 上加速推理？
**A:** 使用 `--tensor_parallel_size` 参数，如：
```bash
python eval_math500_deepthink.py \
  --model ./model \
  --tensor_parallel_size 2
```

### Q: 为什么答案被判为错误？
**A:** 检查以下几点：
1. 是否启用了 `--strict_format`（严格要求 `<SOLUTION>` 标签）
2. 答案格式是否与黄金答案兼容（支持数值容差）
3. 是否正确提取了 `<SOLUTION>` 标签内的答案

### Q: 如何自定义系统提示？
**A:** 使用 `--system_prompt` 参数传入自定义提示文本，或修改脚本中的 `DEFAULT_SYSTEM_PROMPT`。

---

## 相关资源

- [Qwen 官方仓库](https://github.com/QwenLM/Qwen)
- [MATH-500 数据集](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [vLLM 文档](https://vllm.ai/)

---

**最后更新：2026-01-11**