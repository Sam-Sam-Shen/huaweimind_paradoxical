# AGENTS.md - 华为思维悖论分析项目

## 项目概述

本项目是基于**STM（结构主题模型）**和**BERT**的自然语言处理分析系统，用于量化论证任正非管理思想中的悖论性特征。

### 核心功能

- **主题建模**：使用LDA/NMF进行主题提取，分析主题与时间的相关性
- **矛盾检测**：使用多语言BERT NLI模型检测文本中的自相矛盾
- **词向量分析**：Word2Vec分析悖论词汇的语义距离
- **可视化**：生成主题趋势图、矛盾分布图等
- **报告生成**：自动生成Markdown和PDF分析报告

---

## 项目结构

```
huaweimind_paradoxical/
├── src/                        # 源代码
│   ├── __init__.py
│   ├── config.py              # 配置管理
│   ├── utils.py               # 工具函数
│   ├── data_loader.py         # 数据加载
│   ├── preprocess.py          # 中文预处理
│   ├── stm_analysis.py         # 主题建模
│   ├── contradiction_detect.py # 矛盾检测
│   ├── word_embedding.py      # 词向量分析
│   ├── visualize.py           # 可视化
│   └── report_generator.py    # 报告生成
├── data/                      # 数据目录（运行时生成）
├── results/                   # 结果目录
│   ├── topics/               # 主题模型结果
│   ├── contradictions/       # 矛盾检测结果
│   └── figures/             # 图表
├── notebooks/                # Jupyter笔记
├── config.yaml               # 配置文件
├── requirements.txt          # Python依赖
├── main.py                   # 主程序入口
└── AGENTS.md                 # 本文件
```

---

## 环境配置

### 方式一：使用Conda（推荐）

```bash
# 方法1：使用environment.yml创建环境
conda env create -f environment.yml

# 方法2：手动创建
conda create -n huawei_mind python=3.11 -y
conda activate huawei_mind

# 安装核心依赖
conda install -c conda-forge jieba numpy pandas scipy scikit-learn gensim matplotlib seaborn tqdm pyyaml -y

# 安装PyTorch (CPU版本)
conda install -c pytorch pytorch torchvision cpuonly -y

# 安装其他依赖 (pip)
pip install transformers tokenizers sentencepiece pyLDAvis wordcloud markdown jinja2 weasyprint

# 安装ModelScope (用于下载中文NLI模型)
pip install modelscope
```

### 方式二：使用Virtualenv（备选）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

注意：无GPU环境下，transformers会自动使用CPU，可能较慢。

---

## 运行命令

### 完整流程

```bash
# 激活conda环境
conda activate huawei_mind

# 运行分析
python main.py --all
```

### 分步执行

```bash
# 激活conda环境
conda activate huawei_mind

# 仅预处理（分词、去停用词）
python main.py --preprocess

# 主题建模
python main.py --topics --n-topics 25 --method lda

# 矛盾检测
python main.py --contradict

# 词向量分析
python main.py --embedding

# 可视化
python main.py --visualize

# 生成报告
python main.py --report
```

### 单个模块运行

```bash
# 导入模块使用
python -c "
from src.preprocess import run_preprocessing
from src.stm_analysis import run_topic_modeling

# 预处理
processed = run_preprocessing()

# 主题建模
result = run_topic_modeling(method='lda', n_topics=25)
"
```

---

## 代码规范

### 导入顺序

```python
# 1. 标准库
import os
import sys
import json
import logging
from pathlib import Path
from typing import Any

# 2. 第三方库
import numpy as np
import pandas as pd
from tqdm import tqdm

# 3. 本地模块
from src.config import get_config
from src.utils import ensure_dir
```

### 命名规范

| 类型 | 规则 | 示例 |
|------|------|------|
| 模块 | snake_case | `data_loader.py` |
| 类 | PascalCase | `class DataLoader:` |
| 函数 | snake_case | `def load_documents():` |
| 变量 | snake_case | `document_list` |
| 常量 | UPPER_SNAKE | `MAX_ITER = 500` |

### 类型标注

```python
# 函数必须使用类型提示
def process_document(doc_id: str, content: str) -> dict[str, Any]:
    """处理单个文档
    
    Args:
        doc_id: 文档ID
        content: 文档内容
    
    Returns:
        处理结果字典
    """
    ...
```

### 错误处理

```python
# 使用自定义异常
class DataLoadError(Exception):
    """数据加载异常"""
    pass

# 捕获特定异常
try:
    result = load_data()
except FileNotFoundError as e:
    logger.error(f"文件未找到: {e}")
    raise DataLoadError(f"无法加载数据: {e}") from e
except Exception as e:
    logger.exception("未知错误")
    raise
```

### 格式化

- 使用 **Black** 格式化代码
- 行长度限制：**88字符**
- 使用 **isort** 排序导入

```bash
# 格式化命令
black .
isort .
```

### 日志记录

```python
logger = logging.getLogger(__name__)

# 使用logger而非print
logger.info("处理完成")
logger.warning("注意: xxx")
logger.error("错误: xxx")
```

---

## 测试

### 运行所有测试

```bash
conda activate huawei_mind
pytest
```

### 运行单个测试

```bash
conda activate huawei_mind

# 运行指定测试文件
pytest tests/test_preprocess.py

# 运行指定测试函数
pytest tests/test_preprocess.py::test_tokenize

# 运行带标记的测试
pytest -m "not slow"
```

### 编写测试

```python
import pytest
from src.preprocess import ChinesePreprocessor

class TestPreprocessor:
    """预处理测试"""
    
    def test_tokenize(self):
        """测试分词"""
        preprocessor = ChinesePreprocessor()
        result = preprocessor.tokenize("华为技术有限公司")
        assert "华为" in result
        assert "技术" in result
    
    def test_clean_text(self):
        """测试文本清理"""
        preprocessor = ChinesePreprocessor()
        result = preprocessor._clean_text("# 标题\n内容")
        assert "#" not in result
```

---

## 配置说明

### config.yaml 关键配置

```yaml
# 主题建模
topic_modeling:
  num_topics: 25          # 主题数量
  model_type: "lda"      # lda 或 nmf

# 矛盾检测
contradiction_detection:
  model_name: "Fengshenbang/Erlangshen-RoBERTa-330M-NLI"  # ModelScope中文NLI模型
  confidence_threshold: 0.7  # 置信度阈值

# 悖论词对（用于矛盾检测和词向量分析）
paradox_pairs:
  - ["开放", "保守"]
  - ["激进", "稳健"]
  - ["集权", "放权"]
```

### 修改配置

修改 `config.yaml` 后，重新运行相应模块即可：

```bash
python main.py --topics  # 重新运行主题建模
```

---

## 常见问题

### Q: 内存不足怎么办？

A: 减少主题数量或分批处理：
```python
# 在config.yaml中修改
topic_modeling:
  num_topics: 10  # 减少到10个主题
```

### Q: 矛盾检测运行太慢？

A: 
1. 减少候选对数量
2. 使用更小的NLI模型
3. 先用规则筛选关键句对

### Q: 如何添加新的悖论词对？

A: 在 `config.yaml` 的 `paradox_pairs` 中添加：
```yaml
paradox_pairs:
  - ["新词A", "新词B"]
```

---

## 开发指南

### 添加新模块

1. 在 `src/` 下创建新模块文件
2. 在 `src/__init__.py` 中导入
3. 在 `main.py` 中添加命令行选项

### 添加新可视化

1. 在 `src/visualize.py` 中添加方法
2. 在 `generate_all_plots()` 中调用

### 添加新分析方法

1. 创建新模块
2. 在 `config.yaml` 添加配置
3. 在 `main.py` 中集成

---

## 输出结果说明

### results/topics/

| 文件 | 说明 |
|------|------|
| `topics.json` | 主题词汇列表 |
| `document_topics.json` | 每篇文档的主题分布 |
| `topic_time_trend.json` | 主题时间趋势 |
| `paradox_topics.json` | 悖论性主题 |
| `paradox_distances.json` | 悖论词对语义距离 |

### results/contradictions/

| 文件 | 说明 |
|------|------|
| `contradictions.json` | 矛盾句对列表 |
| `stats.json` | 统计信息 |

### results/figures/

| 文件 | 说明 |
|------|------|
| `year_distribution.png` | 文档年份分布 |
| `topic_time_trend.png` | 主题时间趋势 |
| `topic_heatmap.png` | 主题热力图 |
| `paradox_distances.png` | 悖论词对距离 |
| `contradiction_distribution.png` | 矛盾分布 |

---

## 性能提示

- 预处理结果会被缓存到 `data/processed_corpus.json`
- 再次运行时自动从缓存加载，加 `--force` 强制重新处理
- 使用 `tqdm` 可以查看处理进度

---

## 注意事项

1. **编码**：所有文件使用 UTF-8 编码
2. **中文**：确保系统有中文字体支持
3. **路径**：使用 `pathlib.Path` 处理路径
4. **日志**：重要操作使用日志记录
5. **异常**：不要吞掉异常，要适当处理和记录
