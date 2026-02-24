import json
import re
import sys
sys.path.insert(0, '/workspace/huaweimind_paradoxical')

from src.data_loader import DataLoader
from src.config import get_config

loader = DataLoader()
corpus = loader.load_all_documents()

documents = [
    {
        "id": doc.id,
        "title": doc.title,
        "content": doc.content,
        "year": doc.year,
    }
    for doc in corpus.documents
]

# 检查文档数量
print(f"总文档数: {len(documents)}")

# 检查悖论词对配置
config = get_config()
paradox_pairs = config.get("contradiction_detection.paradox_pairs", [])
print(f"\n配置的悖论词对: {paradox_pairs}")

# 提取所有悖论关键词
paradox_keywords = []
for pair in paradox_pairs:
    paradox_keywords.extend(pair)
print(f"\n所有悖论关键词: {paradox_keywords}")

# 检查第一篇文档
doc = documents[0]
print(f"\n第一个文档标题: {doc['title']}")
print(f"第一个文档内容前200字: {doc['content'][:200]}...")

# 提取句子
from src.utils import split_sentences
sentences = split_sentences(doc['content'])
print(f"\n第一个文档句子数: {len(sentences)}")

# 检查哪些句子包含悖论关键词
print("\n包含悖论关键词的句子:")
for i, sent in enumerate(sentences):
    for kw in paradox_keywords:
        if kw in sent:
            print(f"  句{i}: ...{sent[:60]}... (关键词: {kw})")
            break
