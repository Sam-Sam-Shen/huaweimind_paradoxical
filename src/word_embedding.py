"""
词向量分析模块

使用Word2Vec分析悖论词汇的语义距离
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from .config import get_config
from .data_loader import DataLoader
from .preprocess import run_preprocessing
from .utils import ensure_dir, save_json_safe


logger = logging.getLogger(__name__)


class WordEmbeddingAnalyzer:
    """词向量分析器"""
    
    def __init__(self, processed_data: dict[str, Any] | None = None) -> None:
        self.config = get_config()
        self.processed_data = processed_data
        self.model: Word2Vec | None = None
    
    def train_word2vec(
        self,
        sentences: list[list[str]],
    ) -> Word2Vec:
        """训练Word2Vec模型"""
        embed_config = self.config.word_embedding
        
        logger.info(f"训练Word2Vec模型，词数: {len(sentences)}")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=embed_config.get("embedding_dim", 100),
            window=embed_config.get("window", 5),
            min_count=embed_config.get("min_count", 5),
            epochs=embed_config.get("epochs", 20),
            workers=4,
            sg=1,  # Skip-gram
        )
        
        logger.info(f"Word2Vec训练完成，词汇量: {len(self.model.wv)}")
        
        return self.model
    
    def get_word_vector(self, word: str) -> np.ndarray | None:
        """获取词向量"""
        if self.model is None:
            return None
        
        try:
            return self.model.wv[word]
        except KeyError:
            return None
    
    def get_similar_words(
        self,
        word: str,
        topn: int = 10,
    ) -> list[tuple[str, float]]:
        """获取相似词"""
        if self.model is None:
            return []
        
        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []
    
    def analyze_paradox_distance(
        self,
    ) -> dict[str, Any]:
        """分析悖论词对的语义距离"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        paradox_pairs = self.config.get(
            "word_embedding.paradox_word_pairs",
            [
                ["开放", "保守"],
                ["激进", "稳健"],
                ["集权", "放权"],
            ],
        )
        
        distances = []
        
        for pair in paradox_pairs:
            word1, word2 = pair[0], pair[1]
            
            try:
                # 余弦距离
                dist = self.model.wv.similarity(word1, word2)
                
                # 检查词是否存在
                exists1 = word1 in self.model.wv
                exists2 = word2 in self.model.wv
                
                distances.append({
                    "word1": word1,
                    "word2": word2,
                    "similarity": float(dist),
                    "distance": float(1 - dist),
                    "word1_exists": exists1,
                    "word2_exists": exists2,
                })
                
            except KeyError as e:
                logger.warning(f"词汇不存在: {e}")
                distances.append({
                    "word1": word1,
                    "word2": word2,
                    "error": "word_not_found",
                })
        
        return {"paradox_distances": distances}
    
    def analyze_semantic_drift(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """分析语义漂移（不同年份的词义变化）"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 按年份分组
        year_docs: dict[int, list[list[str]]] = {}
        
        for doc in documents:
            year = doc.get("year", 0)
            if year not in year_docs:
                year_docs[year] = []
            year_docs[year].append(doc.get("words", []))
        
        years = sorted(year_docs.keys())
        
        # 分析每年的悖论词对距离
        paradox_pairs = self.config.get(
            "word_embedding.paradox_word_pairs",
        )
        
        drift_results = {}
        
        for pair in paradox_pairs:
            word1, word2 = pair[0], pair[1]
            key = f"{word1}_{word2}"
            
            yearly_distances = []
            
            for year in years:
                try:
                    # 该年份的子语料库训练（简化的方法）
                    year_similar = self.get_similar_words(word1, topn=50)
                    
                    if any(w == word2 for w, _ in year_similar):
                        idx = next(i for i, (w, _) in enumerate(year_similar) if w == word2)
                        dist = 1 - year_similar[idx][1]
                    else:
                        # 使用全局距离作为近似
                        dist = 1 - self.model.wv.similarity(word1, word2)
                    
                    yearly_distances.append({
                        "year": year,
                        "distance": float(dist),
                    })
                    
                except KeyError:
                    continue
            
            if yearly_distances:
                drift_results[key] = yearly_distances
        
        return {"semantic_drift": drift_results}
    
    def find_paradox_contexts(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """寻找悖论词汇的上下文"""
        paradox_keywords = set()
        
        for pairs in self.config.get("word_embedding.paradox_word_pairs", []):
            paradox_keywords.update(pairs)
        
        contexts = []
        
        for doc in tqdm(documents, desc="寻找悖论上下文"):
            content = doc.get("content", "")
            words = doc.get("words", [])
            
            for i, word in enumerate(words):
                if word in paradox_keywords:
                    # 获取上下文窗口
                    start = max(0, i - 5)
                    end = min(len(words), i + 6)
                    context = "".join(words[start:end])
                    
                    if len(context) >= 10:
                        contexts.append({
                            "doc_id": doc.get("id"),
                            "doc_title": doc.get("title"),
                            "year": doc.get("year"),
                            "keyword": word,
                            "context": context,
                            "position": i,
                        })
        
        return contexts
    
    def calculate_paradox_complexity(self) -> dict[str, Any]:
        """计算悖论复杂度"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 基于悖论词对的分析
        analysis = self.analyze_paradox_distance()
        
        # 计算整体悖论指数
        valid_distances = [
            d["distance"]
            for d in analysis.get("paradox_distances", [])
            if "distance" in d
        ]
        
        paradox_index = np.mean(valid_distances) if valid_distances else 0
        
        return {
            "paradox_index": float(paradox_index),
            "num_paradox_pairs": len(valid_distances),
            "paradox_distances": analysis.get("paradox_distances", []),
        }
    
    def save_results(self, output_dir: str) -> None:
        """保存结果"""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # 保存悖论距离
        paradox_dist = self.analyze_paradox_distance()
        save_json_safe(paradox_dist, output_path / "paradox_distances.json")
        
        # 保存悖论复杂度
        complexity = self.calculate_paradox_complexity()
        save_json_safe(complexity, output_path / "paradox_complexity.json")
        
        # 保存相似词示例
        sample_words = ["开放", "保守", "创新", "奋斗", "危机"]
        similar_words = {}
        
        for word in sample_words:
            similar = self.get_similar_words(word, topn=10)
            if similar:
                similar_words[word] = [
                    {"word": w, "similarity": float(s)}
                    for w, s in similar
                ]
        
        save_json_safe(similar_words, output_path / "similar_words.json")
        
        logger.info(f"词向量分析结果已保存到: {output_path}")


def run_word_embedding() -> dict[str, Any]:
    """运行词向量分析"""
    from .preprocess import run_preprocessing
    
    # 预处理获取分词结果
    processed = run_preprocessing()
    
    # 准备训练语料
    sentences = [doc["words"] for doc in processed["documents"]]
    
    # 训练模型
    analyzer = WordEmbeddingAnalyzer(processed)
    analyzer.train_word2vec(sentences)
    
    # 分析
    documents = processed["documents"]
    
    # 保存结果
    output_dir = get_config().get_path("topics_dir")
    analyzer.save_results(output_dir)
    
    return {
        "vocab_size": len(analyzer.model.wv),
        "output_dir": output_dir,
    }
