import json
import re
import sys
sys.path.insert(0, '/workspace/huaweimind_paradoxical')

from src.data_loader import DataLoader
from src.config import get_config
from src.utils import split_sentences

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

config = get_config()
paradox_pairs = config.get("contradiction_detection.paradox_pairs", [])

paradox_keywords = []
for pair in paradox_pairs:
    paradox_keywords.extend(pair)

# 检查前20篇文档中有多少包含悖论关键词的句子
stats = []
for doc in documents[:20]:
    sentences = split_sentences(doc['content'])
    paradox_sents = []
    for i, sent in enumerate(sentences):
        for kw in paradox_keywords:
            if kw in sent:
                paradox_sents.append(i)
                break
    stats.append({
        'title': doc['title'][:30],
        'total_sents': len(sentences),
        'paradox_sents': len(paradox_sents),
        'indices': paradox_sents
    })
    
print("前20篇文档的悖论句子统计:")
for s in stats:
    print(f"{s['title']}: {s['paradox_sents']}个悖论句 (索引: {s['indices']})")

total_paradox_sents = sum(s['paradox_sents'] for s in stats)
print(f"\n总计: {total_paradox_sents}个悖论句子")
print(f"可生成候选对: {sum(1 for s in stats if s['paradox_sents'] >= 2)} 篇")
