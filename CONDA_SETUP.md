# 华为思维悖论分析 - 环境配置

## 使用Conda创建环境

### 1. 创建环境

```bash
# 创建新环境
conda create -n huawei_mind python=3.11 -y

# 激活环境
conda activate huawei_mind
```

### 2. 安装依赖

```bash
# 核心依赖
conda install -c conda-forge jieba numpy pandas scipy -y

# 机器学习与NLP
conda install -c conda-forge scikit-learn gensim -y

# PyTorch (CPU版本)
conda install -c pytorch pytorch torchvision cpuonly -y

# Transformers
pip install transformers tokenizers sentencepiece

# 可视化
conda install -c conda-forge matplotlib seaborn -y
pip install pyLDAvis wordcloud

# 工具库
conda install -c conda-forge tqdm pyyaml -y
pip install python-dateutil regex

# 报告生成
pip install markdown jinja2 weasyprint pypandoc

# 测试与代码质量
conda install -c conda-forge pytest pytest-cov black flake8 isort mypy -y
```

### 3. 一键安装脚本

```bash
# 创建并运行安装脚本
bash install_conda_env.sh
```
