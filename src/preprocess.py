"""
中文文本预处理模块

包括分词、去停用词、构建词袋模型等
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from tqdm import tqdm

from .config import get_config
from .data_loader import Corpus, DataLoader
from .utils import ensure_dir, get_project_root, load_json_safe, save_json_safe


logger = logging.getLogger(__name__)


class ChinesePreprocessor:
    """中文文本预处理器"""
    
    def __init__(self) -> None:
        self.config = get_config()
        self._stopwords: set[str] = set()
        self._vocab: dict[str, int] = {}
        self._vocab_list: list[str] = []
        
    @property
    def stopwords(self) -> set[str]:
        """获取停用词表"""
        if not self._stopwords:
            self._load_stopwords()
        return self._stopwords
    
    def _load_stopwords(self) -> None:
        """加载停用词表"""
        stopword_files = self.config.get("preprocessing.stopwords", [])
        
        # 默认停用词
        default_stopwords = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他",
            "她", "它", "们", "什么", "怎么", "为什么", "如何", "可以",
            "这个", "那个", "但是", "因为", "所以", "如果", "或者", "而且",
            "并且", "虽然", "还是", "而是", "只是", "已经", "正在", "将要",
            "我们", "他们", "这些", "那些", "给", "对", "从", "把", "让",
            "与", "及", "等", "以", "为", "于", "来", "中", "大", "小",
            "多", "少", "更", "最", "很", "真", "比较", "非常", "特别",
            "需要", "应该", "能够", "可能", "必须", "一定", "当然", "其实",
            "当然", "确实", "真正", "主要", "重要", "一些", "某些", "各种",
            "每个", "其他", "另外", "还有", "还有", "包括", "关于", "通过",
            "根据", "按照", "为了", "由于", "对于", "由于", "基于",
        }
        
        self._stopwords = default_stopwords
        
        # 从文件加载额外停用词
        for stopfile in stopword_files:
            stopfile_path = get_project_root() / stopfile
            if stopfile_path.exists():
                with open(stopfile_path, "r", encoding="utf-8") as f:
                    extra = [line.strip() for line in f if line.strip()]
                    self._stopwords.update(extra)
        
        logger.info(f"加载停用词表: {len(self._stopwords)} 个词")
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> list[str]:
        """分词"""
        # 清理文本
        text = self._clean_text(text)
        
        # jieba分词
        words = jieba.lcut(text)
        
        # 过滤
        min_len = self.config.get("preprocessing.min_word_length", 2)
        words = [w for w in words if len(w) >= min_len]
        
        # 去停用词
        if remove_stopwords:
            words = [w for w in words if w not in self.stopwords]
        
        return words
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除Markdown格式
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"[#*`_~]", "", text)
        
        # 移除URL
        text = re.sub(r"http[s]?://\S+", "", text)
        
        # 移除邮箱
        text = re.sub(r"\S+@\S+", "", text)
        
        # 移除数字（可选）
        if self.config.get("preprocessing.remove_numbers", True):
            text = re.sub(r"\d+", "", text)
        
        # 移除多余空白
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def build_corpus(
        self,
        corpus: Corpus,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        """构建语料库"""
        processed_file = self.config.get_path("processed_file")
        
        # 尝试从缓存加载
        if not force_rebuild:
            cached = load_json_safe(processed_file)
            if cached:
                logger.info("从缓存加载预处理结果")
                return cached
        
        logger.info(f"开始预处理: {len(corpus.documents)} 篇文档")
        
        processed_docs = []
        all_words = []
        
        for doc in tqdm(corpus.documents, desc="预处理文档"):
            # 分词
            words = self.tokenize(doc.content)
            
            if len(words) < self.config.get("preprocessing.min_doc_words", 50):
                logger.warning(f"文档词数过少: {doc.title} ({len(words)} 词)")
                continue
            
            processed_docs.append({
                "id": doc.id,
                "title": doc.title,
                "year": doc.year,
                "doc_type": doc.doc_type,
                "words": words,
                "word_count": len(words),
            })
            
            all_words.extend(words)
        
        # 构建词汇表
        min_count = self.config.get("word_embedding.min_count", 5)
        word_counts = Counter(all_words)
        self._vocab = {
            word: idx
            for idx, (word, count) in enumerate(word_counts.most_common())
            if count >= min_count
        }
        self._vocab_list = list(self._vocab.keys())
        
        logger.info(f"词汇表大小: {len(self._vocab)}")
        
        # 构建词袋表示
        for doc in processed_docs:
            word_freq = Counter(doc["words"])
            bow = [self._vocab.get(w, -1) for w in doc["words"]]
            bow = [x for x in bow if x >= 0]
            doc["bow"] = bow
            doc["word_freq"] = dict(word_counts & Counter(self._vocab_list))
        
        result = {
            "metadata": {
                "vocab_size": len(self._vocab),
                "num_docs": len(processed_docs),
                "total_words": len(all_words),
            },
            "documents": processed_docs,
            "vocab": self._vocab,
        }
        
        # 保存
        save_json_safe(result, processed_file)
        logger.info(f"预处理完成，保存到: {processed_file}")
        
        return result
    
    def get_vocab(self) -> dict[str, int]:
        """获取词汇表"""
        return self._vocab
    
    def get_vocab_list(self) -> list[str]:
        """获取词汇表（列表形式）"""
        return self._vocab_list


class TopicModelInputPreparer:
    """为主题模型准备输入数据"""
    
    def __init__(self, processed_data: dict[str, Any]) -> None:
        self.data = processed_data
        self.vocab = processed_data.get("vocab", {})
    
    def get_document_term_matrix(self) -> tuple[np.ndarray, list[dict]]:
        """获取文档-词项矩阵"""
        from scipy import sparse
        
        docs = self.data["documents"]
        vocab_size = len(self.vocab)
        
        rows, cols, data = [], [], []
        
        for doc_idx, doc in enumerate(docs):
            word_freq = Counter(doc.get("bow", []))
            for word_idx, freq in word_freq.items():
                rows.append(doc_idx)
                cols.append(word_idx)
                data.append(freq)
        
        dtm = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(docs), vocab_size),
            dtype=np.float32,
        )
        
        return dtm, docs
    
    def get_metadata_matrix(self) -> np.ndarray:
        """获取元数据矩阵（年份编码）"""
        docs = self.data["documents"]
        years = [doc["year"] for doc in docs]
        
        # 年份归一化
        min_year = min(years)
        max_year = max(years)
        
        if max_year > min_year:
            normalized = [(y - min_year) / (max_year - min_year) for y in years]
        else:
            normalized = [0.5] * len(years)
        
        return np.array(normalized)
    
    def get_year_labels(self) -> list[int]:
        """获取年份标签"""
        return [doc["year"] for doc in self.data["documents"]]
    
    def get_doc_types(self) -> list[str]:
        """获取文档类型"""
        return [doc["doc_type"] for doc in self.data["documents"]]
    
    def get_topic_prevalence_by_year(
        self,
        doc_topic_dist: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """获取每年主题分布"""
        docs = self.data["documents"]
        years = sorted(set(doc["year"] for doc in docs))
        
        year_topics: dict[int, list] = {y: [] for y in years}
        
        for doc_idx, doc in enumerate(docs):
            year_topics[doc["year"]].append(doc_topic_dist[doc_idx])
        
        # 计算平均值
        return {
            year: np.mean(dists, axis=0)
            for year, dists in year_topics.items()
            if dists
        }


def run_preprocessing() -> dict[str, Any]:
    """运行完整预处理流程"""
    # 加载数据
    loader = DataLoader()
    corpus = loader.load_all_documents()
    
    # 预处理
    preprocessor = ChinesePreprocessor()
    processed = preprocessor.build_corpus(corpus)
    
    # 保存元数据
    metadata = loader.get_metadata()
    metadata_file = get_config().get_path("metadata_file")
    save_json_safe(metadata, metadata_file)
    
    return processed
