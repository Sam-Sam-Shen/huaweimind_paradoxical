"""
主题建模模块

使用LDA和NMF进行主题分析，并模拟STM的元数据协变量分析
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from .config import get_config
from .preprocess import TopicModelInputPreparer
from .utils import ensure_dir, save_json_safe


logger = logging.getLogger(__name__)


class TopicModeler:
    """主题模型训练器"""
    
    def __init__(self, processed_data: dict[str, Any]) -> None:
        self.config = get_config()
        self.data = processed_data
        self.preparer = TopicModelInputPreparer(processed_data)
        
        self.model = None
        self.doc_topic_dist: np.ndarray | None = None
        self.topic_term_dist: np.ndarray | None = None
        self.feature_names: list[str] = []
    
    def train_lda(
        self,
        n_topics: int = 25,
        max_iter: int = 500,
    ) -> dict[str, Any]:
        """训练LDA模型"""
        logger.info(f"训练LDA模型，主题数: {n_topics}")
        
        # 获取文档
        documents = [" ".join(doc["words"]) for doc in self.data["documents"]]
        
        # 创建词袋模型
        vectorizer = CountVectorizer(
            max_features=len(self.data.get("vocab", {})),
            min_df=5,
            max_df=0.8,
        )
        
        dtm = vectorizer.fit_transform(documents)
        self.feature_names = vectorizer.get_feature_names_out().tolist()
        
        # 训练LDA
        lda_config = self.config.topic_modeling
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="online",
            learning_offset=50.0,
            random_state=42,
            n_jobs=-1,
        )
        
        self.doc_topic_dist = self.model.fit_transform(dtm)
        self.topic_term_dist = self.model.components_
        
        # 计算困惑度
        perplexity = self.model.perplexity(dtm)
        
        logger.info(f"LDA训练完成，困惑度: {perplexity:.2f}")
        
        return {
            "n_topics": n_topics,
            "perplexity": perplexity,
            "log_likelihood": self.model.score(dtm),
        }
    
    def train_nmf(self, n_topics: int = 25) -> dict[str, Any]:
        """训练NMF模型"""
        logger.info(f"训练NMF模型，主题数: {n_topics}")
        
        documents = [" ".join(doc["words"]) for doc in self.data["documents"]]
        
        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=len(self.data.get("vocab", {})),
            min_df=5,
            max_df=0.8,
        )
        
        tfidf = vectorizer.fit_transform(documents)
        self.feature_names = vectorizer.get_feature_names_out().tolist()
        
        # 训练NMF
        self.model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=500,
            init="nndsvd",
        )
        
        self.doc_topic_dist = self.model.fit_transform(tfidf)
        self.topic_term_dist = self.model.components_
        
        reconstruction_error = self.model.reconstruction_err_
        
        logger.info(f"NMF训练完成，重构误差: {reconstruction_error:.2f}")
        
        return {
            "n_topics": n_topics,
            "reconstruction_error": reconstruction_error,
        }
    
    def get_topics(
        self,
        n_top_words: int = 15,
    ) -> list[dict[str, Any]]:
        """获取主题词列表"""
        if self.topic_term_dist is None:
            raise ValueError("模型未训练")
        
        topics = []
        
        for topic_idx, topic in enumerate(self.topic_term_dist):
            top_indices = topic.argsort()[::-1][:n_top_words]
            top_words = [
                {"word": self.feature_names[i], "weight": float(topic[i])}
                for i in top_indices
            ]
            
            topics.append({
                "topic_id": topic_idx,
                "words": top_words,
            })
        
        return topics
    
    def get_document_topics(self) -> list[dict[str, Any]]:
        """获取每个文档的主题分布"""
        if self.doc_topic_dist is None:
            raise ValueError("模型未训练")
        
        results = []
        
        for doc_idx, doc in enumerate(self.data["documents"]):
            topic_dist = self.doc_topic_dist[doc_idx]
            top_topics = np.argsort(topic_dist)[::-1][:5]
            
            results.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "year": doc["year"],
                "dominant_topic": int(top_topics[0]),
                "topic_weights": {
                    str(t): float(topic_dist[t]) for t in top_topics
                },
            })
        
        return results
    
    def analyze_topic_time_trend(
        self,
    ) -> dict[str, Any]:
        """分析主题随时间的演变趋势"""
        if self.doc_topic_dist is None:
            raise ValueError("模型未训练")
        
        years = self.preparer.get_year_labels()
        unique_years = sorted(set(years))
        
        # 计算每年的主题分布
        year_topic_dist: dict[int, np.ndarray] = {}
        
        for year in unique_years:
            year_docs = [
                self.doc_topic_dist[i]
                for i, y in enumerate(years)
                if y == year
            ]
            year_topic_dist[year] = np.mean(year_docs, axis=0)
        
        # 计算主题与时间的相关性
        correlations = []
        
        for topic_idx in range(self.doc_topic_dist.shape[1]):
            topic_scores = self.doc_topic_dist[:, topic_idx]
            
            # Pearson相关
            corr, p_value = pearsonr(years, topic_scores)
            
            correlations.append({
                "topic_id": topic_idx,
                "pearson_corr": float(corr),
                "p_value": float(p_value),
                "trend": "increasing" if corr > 0.1 else "decreasing" if corr < -0.1 else "stable",
            })
        
        return {
            "year_topic_distribution": {
                str(year): dist.tolist() for year, dist in year_topic_dist.items()
            },
            "topic_time_correlations": correlations,
        }
    
    def analyze_metadata_effect(
        self,
        covariate_name: str = "year",
    ) -> dict[str, Any]:
        """分析元数据对主题的影响（模拟STM的协变量分析）"""
        if self.doc_topic_dist is None:
            raise ValueError("模型未训练")
        
        if covariate_name == "year":
            covariates = np.array(self.preparer.get_year_labels())
            cov_name = "年份"
        else:
            return {}
        
        # 回归分析
        results = []
        
        for topic_idx in range(self.doc_topic_dist.shape[1]):
            topic_scores = self.doc_topic_dist[:, topic_idx]
            
            # 简单线性回归
            slope = np.polyfit(covariates, topic_scores, 1)[0]
            
            # 相关性
            corr, p_val = pearsonr(covariates, topic_scores)
            
            results.append({
                "topic_id": topic_idx,
                "slope": float(slope),
                "correlation": float(corr),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
            })
        
        return {
            "covariate": cov_name,
            "topic_effects": results,
        }
    
    def find_paradox_topics(self) -> list[dict[str, Any]]:
        """寻找可能体现悖论的主题"""
        topics = self.get_topics(n_top_words=20)
        
        # 悖论相关关键词
        paradox_keywords = [
            ["开放", "保守", "封闭"],
            ["进攻", "防守", "收缩"],
            ["集权", "放权", "授权"],
            ["创新", "守成", "稳定"],
            ["扩张", "收缩", "稳健"],
            ["利润", "规模", "增长"],
            ["个人", "集体", "团队"],
            ["激进", "稳健", "谨慎"],
            ["自主", "引进", "合作"],
            ["全球化", "本土化", "本地化"],
        ]
        
        paradox_topics = []
        
        for topic in topics:
            words = [w["word"] for w in topic["words"]]
            
            for pairs in paradox_keywords:
                # 检查是否同时包含对立词汇
                found = [w for w in words if w in pairs]
                
                if len(found) >= 2:
                    paradox_topics.append({
                        "topic_id": topic["topic_id"],
                        "paradox_pair": pairs,
                        "found_words": found,
                        "all_topic_words": words[:10],
                    })
                    break
        
        return paradox_topics
    
    def save_results(self, output_dir: str) -> None:
        """保存结果"""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # 保存主题
        topics = self.get_topics()
        save_json_safe(topics, output_path / "topics.json")
        
        # 保存文档主题
        doc_topics = self.get_document_topics()
        save_json_safe(doc_topics, output_path / "document_topics.json")
        
        # 保存时间趋势
        time_trend = self.analyze_topic_time_trend()
        save_json_safe(time_trend, output_path / "topic_time_trend.json")
        
        # 保存元数据效应
        meta_effect = self.analyze_metadata_effect()
        save_json_safe(meta_effect, output_path / "metadata_effect.json")
        
        # 保存悖论主题
        paradox = self.find_paradox_topics()
        save_json_safe(paradox, output_path / "paradox_topics.json")
        
        logger.info(f"结果已保存到: {output_path}")


def find_optimal_topics(
    processed_data: dict[str, Any],
    topic_range: range = range(10, 51, 5),
) -> dict[str, Any]:
    """寻找最优主题数"""
    logger.info("开始寻找最优主题数")
    
    modeler = TopicModeler(processed_data)
    results = []
    
    for n_topics in tqdm(topic_range, desc="测试主题数"):
        # 训练小规模迭代
        documents = [" ".join(doc["words"]) for doc in processed_data["documents"]]
        
        vectorizer = CountVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
        )
        
        dtm = vectorizer.fit_transform(documents)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=50,
            learning_method="online",
            random_state=42,
        )
        
        lda.fit(dtm)
        
        perplexity = lda.perplexity(dtm)
        
        results.append({
            "n_topics": n_topics,
            "perplexity": perplexity,
        })
    
    return {"topic_search": results}


def run_topic_modeling(
    method: str = "lda",
    n_topics: int = 25,
) -> dict[str, Any]:
    """运行主题建模"""
    from .preprocess import run_preprocessing
    
    # 预处理
    processed = run_preprocessing()
    
    # 训练模型
    modeler = TopicModeler(processed)
    
    if method == "lda":
        model_info = modeler.train_lda(n_topics=n_topics)
    else:
        model_info = modeler.train_nmf(n_topics=n_topics)
    
    # 保存结果
    output_dir = get_config().get_path("topics_dir")
    modeler.save_results(output_dir)
    
    return {
        "model_info": model_info,
        "output_dir": output_dir,
    }
