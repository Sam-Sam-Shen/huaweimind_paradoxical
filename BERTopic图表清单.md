# BERTopic生成的图表清单

## 交互式HTML图表（可用浏览器打开）

1. **bertopic_intertopic_distance.html** (4.7MB)
   - 主题距离图（降维后的2D/3D可视化）
   - 显示主题之间的相似性和聚类关系
   - 可交互缩放和查看主题详情

2. **bertopic_hierarchy.html** (4.7MB)
   - 主题层次结构图
   - 展示主题的父子关系和聚类层次
   - 帮助理解主题的粒度结构

3. **bertopic_barchart.html** (4.7MB)
   - 主题词条形图
   - 每个主题的前N个关键词及其权重
   - 便于快速浏览所有主题内容

4. **bertopic_heatmap.html** (4.7MB)
   - 主题相似度热力图
   - 显示主题之间的余弦相似度
   - 颜色越深表示主题越相似

5. **bertopic_distribution.html** (4.7MB)
   - 文档-主题分布图
   - 显示文档在不同主题上的概率分布
   - 帮助理解文档的主题构成

## 静态PNG图表

### 主题词云（wordclouds/文件夹）
共生成10个主题词云，每个词云展示该主题的关键词分布：
- topic_0_wordcloud.png (113KB)
- topic_1_wordcloud.png (133KB)
- topic_2_wordcloud.png (119KB)
- topic_3_wordcloud.png (136KB)
- topic_4_wordcloud.png (103KB)
- topic_5_wordcloud.png (124KB)
- topic_6_wordcloud.png (127KB)
- topic_7_wordcloud.png (114KB)
- topic_8_wordcloud.png (137KB)
- ...（更多主题词云）

### 主题时间趋势图（topic_trends/文件夹）
每个主题的时间演变趋势图，显示主题流行度随时间的变化。

## JSON数据文件

1. **bertopic_topics.json** (13KB)
   - 主题详细信息（ID、名称、关键词、权重）

2. **bertopic_document_topics.json** (378KB)
   - 每个文档的主题分布
   - 包含主导主题和主题权重

3. **bertopic_time_trend.json** (6.6KB)
   - 主题时间趋势数据
   - 用于生成时间趋势图

## 模型文件

- **bertopic_model/** 文件夹
  - 保存的BERTopic模型，可用于后续推理

## 图表总数统计

- 交互式HTML图表: 5个
- 主题词云图: 10+个
- 时间趋势图: 10+个
- JSON数据文件: 3个
- **总计: 28+个图表/文件**

## 使用方法

### 查看交互式图表
```bash
# 在浏览器中打开（以主题距离图为例）
python -m http.server 8000 --directory results/topics/
# 然后访问 http://localhost:8000/bertopic_intertopic_distance.html
```

### 查看词云图
```bash
# 直接打开PNG文件
open results/topics/wordclouds/topic_0_wordcloud.png
```

## 学术价值

这些图表提供了：
1. **主题结构可视化** - 直观理解主题间的关系
2. **时间演变追踪** - 观察主题的兴起和衰落
3. **文档-主题关联** - 理解文档的主题构成
4. **中文语义理解** - 基于Transformer的深度语义分析

---
生成时间: 2026-02-25
模型: iic/nlp_corom_sentence-embedding_chinese-base (本地中文模型)
文档数: 775篇
主题数: 动态确定（约10-15个）
