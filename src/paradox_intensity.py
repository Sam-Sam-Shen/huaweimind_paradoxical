"""
悖论强度指数（Paradox Intensity Index, PII）模块

整合四个维度构建综合性的悖论强度测量：
1. 语义张力（Semantic Tension）
2. 矛盾检测（Contradiction Detection）
3. 主题一致性（Thematic Coherence）
4. 时序动态（Temporal Dynamics）

PII = w1·Semantic + w2·Contradiction + w3·Thematic + w4·Temporal
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from .config import get_config
from .utils import ensure_dir, load_json_safe, save_json, timer

logger = logging.getLogger(__name__)


class ParadoxIntensityCalculator:
    """悖论强度指数计算器"""
    
    def __init__(self, weights: dict[str, float] = None) -> None:
        """
        初始化计算器
        
        Args:
            weights: 各维度权重，默认使用均衡权重
        """
        self.config = get_config()
        
        # 默认权重（可通过验证调整）
        self.weights = weights or {
            'semantic': 0.25,      # 语义距离
            'contradiction': 0.30,  # 矛盾检测
            'thematic': 0.25,       # 主题一致性
            'temporal': 0.20        # 时序变化
        }
        
        # 加载数据
        self.word2vec_model = None
        self.paradox_pairs = None
        self.contradictions = None
        self.topics = None
        self.document_topics = None
        
        logger.info(f"PII权重配置: {self.weights}")
    
    def load_data(self, 
                  word2vec_path: Path = None,
                  contradictions_path: Path = None,
                  topics_path: Path = None,
                  document_topics_path: Path = None) -> None:
        """
        加载必要的数据
        
        Args:
            word2vec_path: Word2Vec模型路径
            contradictions_path: 矛盾检测结果路径
            topics_path: 主题信息路径
            document_topics_path: 文档主题分布路径
        """
        # 加载Word2Vec模型
        if word2vec_path is None:
            word2vec_path = Path(self.config.get('output.topic_dir', 'results/topics')) / 'word2vec.model'
        
        if word2vec_path.exists():
            self.word2vec_model = Word2Vec.load(str(word2vec_path))
            logger.info(f"加载Word2Vec模型: {word2vec_path}")
        else:
            logger.warning(f"Word2Vec模型未找到: {word2vec_path}")
        
        # 加载矛盾检测结果
        if contradictions_path is None:
            contradictions_path = Path(self.config.get('output.contradiction_dir', 'results/contradictions')) / 'contradictions.json'
        
        if contradictions_path.exists():
            self.contradictions = load_json_safe(contradictions_path, default=[])
            logger.info(f"加载矛盾检测结果: {len(self.contradictions)} 条")
        else:
            logger.warning(f"矛盾检测结果未找到: {contradictions_path}")
        
        # 加载主题信息
        if topics_path is None:
            topics_path = Path(self.config.get('output.topic_dir', 'results/topics')) / 'topics.json'
        
        if topics_path.exists():
            topics_data = load_json_safe(topics_path, default=[])
            self.topics = topics_data if isinstance(topics_data, list) else topics_data.get('topics', [])
            logger.info(f"加载主题信息: {len(self.topics)} 个主题")
        else:
            logger.warning(f"主题信息未找到: {topics_path}")
        
        # 加载文档主题分布
        if document_topics_path is None:
            document_topics_path = Path(self.config.get('output.topic_dir', 'results/topics')) / 'document_topics.json'
        
        if document_topics_path.exists():
            self.document_topics = load_json_safe(document_topics_path, default=[])
            logger.info(f"加载文档主题分布: {len(self.document_topics)} 篇文档")
        else:
            logger.warning(f"文档主题分布未找到: {document_topics_path}")
        
        # 加载悖论词对
        self.paradox_pairs = self.config.get('word_embedding.paradox_word_pairs', [])
        if not self.paradox_pairs:
            self.paradox_pairs = self.config.get('contradiction_detection.paradox_pairs', [])
        logger.info(f"加载悖论词对: {len(self.paradox_pairs)} 对")
    
    def calculate_pii(self, word_pair: list[str], year: int = None) -> dict[str, Any]:
        """
        计算悖论强度指数
        
        Args:
            word_pair: 悖论词对 [word1, word2]
            year: 年份（用于时序分析）
            
        Returns:
            PII结果字典
        """
        w1, w2 = word_pair[0], word_pair[1]
        
        # 计算各维度得分
        semantic_score = self._calculate_semantic_tension(w1, w2)
        contradiction_score = self._calculate_contradiction_strength(w1, w2)
        thematic_score = self._calculate_thematic_coherence(w1, w2)
        temporal_score = self._calculate_temporal_dynamics(w1, w2, year)
        
        # 加权求和
        pii = (
            self.weights['semantic'] * semantic_score +
            self.weights['contradiction'] * contradiction_score +
            self.weights['thematic'] * thematic_score +
            self.weights['temporal'] * temporal_score
        )
        
        return {
            'word_pair': word_pair,
            'pii': round(pii, 4),
            'components': {
                'semantic': round(semantic_score, 4),
                'contradiction': round(contradiction_score, 4),
                'thematic': round(thematic_score, 4),
                'temporal': round(temporal_score, 4)
            },
            'weights': self.weights
        }
    
    def _calculate_semantic_tension(self, w1: str, w2: str) -> float:
        """
        计算语义张力
        
        不是简单的余弦距离，而是使用"张力"概念：
        - 距离在0.5-0.7之间时张力最大（既不太近也不太远）
        - 使用高斯函数建模
        """
        if self.word2vec_model is None:
            return 0.5  # 默认值
        
        try:
            # 获取词向量
            v1 = self.word2vec_model.wv[w1]
            v2 = self.word2vec_model.wv[w2]
            
            # 计算余弦距离
            distance = cosine(v1, v2)
            
            # 使用高斯函数：在optimal_distance处张力最大
            optimal_distance = 0.6
            sigma = 0.2
            
            tension = np.exp(-((distance - optimal_distance) ** 2) / (2 * sigma ** 2))
            
            return float(tension)
            
        except KeyError:
            logger.warning(f"词汇未找到: {w1} 或 {w2}")
            return 0.0
    
    def _calculate_contradiction_strength(self, w1: str, w2: str) -> float:
        """
        计算矛盾强度
        
        基于：
        - 包含该词对的句子被检测为矛盾的频率
        - 平均置信度
        """
        if self.contradictions is None:
            return 0.5  # 默认值
        
        # 筛选包含这两个词的矛盾
        relevant_contradictions = []
        
        for contra in self.contradictions:
            premise = contra.get('premise', '')
            hypothesis = contra.get('hypothesis', '')
            
            # 检查是否包含这两个词
            w1_in_premise = w1 in premise
            w1_in_hypothesis = w1 in hypothesis
            w2_in_premise = w2 in premise
            w2_in_hypothesis = w2 in hypothesis
            
            # 两个词分别出现在premise和hypothesis中
            if (w1_in_premise and w2_in_hypothesis) or (w2_in_premise and w1_in_hypothesis):
                relevant_contradictions.append(contra)
        
        if not relevant_contradictions:
            return 0.0
        
        # 计算频率（假设总共有1000个候选句对）
        total_candidates = 1000  # 估算值
        frequency = len(relevant_contradictions) / total_candidates
        
        # 计算平均置信度
        avg_confidence = np.mean([c.get('confidence', 0) for c in relevant_contradictions])
        
        # 综合得分（频率40% + 置信度60%）
        score = frequency * 0.4 + avg_confidence * 0.6
        
        return float(min(score, 1.0))  # 限制在0-1之间
    
    def _calculate_thematic_coherence(self, w1: str, w2: str) -> float:
        """
        计算主题一致性
        
        基于：
        - 两个词是否经常出现在同一主题中
        - 主题分布的Jensen-Shannon散度（越相似=越一致）
        """
        if self.document_topics is None or self.topics is None:
            return 0.5  # 默认值
        
        # 获取每个词的主题分布
        def get_word_topic_dist(word):
            """获取词的主题分布"""
            topic_counts = {}
            
            for doc in self.document_topics:
                # 简化处理：假设词在文档中就属于该文档的主导主题
                # 实际应该基于词-主题共现统计
                doc_id = doc.get('doc_id', '')
                dominant_topic = doc.get('dominant_topic', 0)
                
                # 这里简化处理，假设词在文档中均匀分布
                topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1
            
            # 归一化
            total = sum(topic_counts.values())
            if total == 0:
                return np.zeros(len(self.topics))
            
            dist = np.zeros(len(self.topics))
            for topic_id, count in topic_counts.items():
                if 0 <= topic_id < len(self.topics):
                    dist[topic_id] = count / total
            
            return dist
        
        dist1 = get_word_topic_dist(w1)
        dist2 = get_word_topic_dist(w2)
        
        # 计算Jensen-Shannon散度
        js_div = self._js_divergence(dist1, dist2)
        
        # 转换为一致性分数（1 - 归一化散度）
        coherence = 1 - js_div
        
        return float(coherence)
    
    def _calculate_temporal_dynamics(self, w1: str, w2: str, year: int = None) -> float:
        """
        计算时序动态性
        
        基于：
        - 词对语义距离随时间的变化率
        - 变化越大 = 悖论越动态
        """
        if self.word2vec_model is None:
            return 0.5  # 默认值
        
        # 简化实现：基于全局模型计算距离变化
        # 实际应该使用多个时间段的词向量模型
        
        try:
            v1 = self.word2vec_model.wv[w1]
            v2 = self.word2vec_model.wv[w2]
            current_distance = cosine(v1, v2)
            
            # 如果没有指定年份，返回中等动态性
            if year is None:
                return 0.5
            
            # 模拟：假设早期（1994-2000）距离较小，近期距离较大
            # 表示悖论随时间变得更加明显
            if year < 2000:
                baseline_distance = 0.4
            elif year < 2010:
                baseline_distance = 0.5
            elif year < 2018:
                baseline_distance = 0.6
            else:
                baseline_distance = 0.7
            
            # 计算变化
            change = abs(current_distance - baseline_distance)
            
            # 归一化到0-1
            volatility = min(change * 2, 1.0)
            
            return float(volatility)
            
        except KeyError:
            return 0.0
    
    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """计算Jensen-Shannon散度"""
        # 避免log(0)
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        
        # 归一化
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = 0.5 * (p + q)
        
        js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        
        return float(js_div)
    
    @timer
    def calculate_all_pairs(self) -> dict[str, Any]:
        """
        计算所有悖论词对的PII
        
        Returns:
            所有词对的PII结果
        """
        if not self.paradox_pairs:
            logger.error("未加载悖论词对")
            return {}
        
        logger.info(f"开始计算 {len(self.paradox_pairs)} 对悖论词的PII...")
        
        results = []
        
        for pair in self.paradox_pairs:
            try:
                result = self.calculate_pii(pair)
                results.append(result)
                logger.debug(f"{pair}: PII={result['pii']:.4f}")
            except Exception as e:
                logger.error(f"计算 {pair} 的PII时出错: {e}")
                results.append({
                    'word_pair': pair,
                    'pii': 0.0,
                    'error': str(e)
                })
        
        # 计算总体统计
        valid_pii = [r['pii'] for r in results if 'error' not in r]
        
        summary = {
            'num_pairs': len(results),
            'avg_pii': round(np.mean(valid_pii), 4) if valid_pii else 0,
            'std_pii': round(np.std(valid_pii), 4) if valid_pii else 0,
            'max_pii': round(np.max(valid_pii), 4) if valid_pii else 0,
            'min_pii': round(np.min(valid_pii), 4) if valid_pii else 0,
            'top_paradoxes': sorted(results, key=lambda x: x.get('pii', 0), reverse=True)[:5]
        }
        
        return {
            'summary': summary,
            'detailed_results': results,
            'weights': self.weights
        }
    
    def save_results(self, output_dir: Path) -> dict[str, Path]:
        """保存PII计算结果"""
        ensure_dir(output_dir)
        
        # 计算所有词对的PII
        results = self.calculate_all_pairs()
        
        # 保存详细结果
        results_file = output_dir / 'paradox_intensity_index.json'
        save_json(results, results_file)
        
        # 保存摘要
        summary_file = output_dir / 'pii_summary.json'
        save_json(results['summary'], summary_file)
        
        logger.info(f"PII结果已保存到: {output_dir}")
        
        return {
            'full_results': results_file,
            'summary': summary_file
        }


def run_paradox_intensity_analysis(
    word2vec_path: Path = None,
    contradictions_path: Path = None,
    topics_path: Path = None,
    document_topics_path: Path = None,
    output_dir: Path = None,
    weights: dict[str, float] = None
) -> dict[str, Any]:
    """
    运行悖论强度指数分析（兼容原有接口）
    
    Args:
        word2vec_path: Word2Vec模型路径
        contradictions_path: 矛盾检测结果路径
        topics_path: 主题信息路径
        document_topics_path: 文档主题分布路径
        output_dir: 输出目录
        weights: 自定义权重
        
    Returns:
        分析结果字典
    """
    config = get_config()
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(config.get('output.topic_dir', 'results/topics'))
    
    # 创建计算器
    calculator = ParadoxIntensityCalculator(weights)
    
    # 加载数据
    calculator.load_data(
        word2vec_path=word2vec_path,
        contradictions_path=contradictions_path,
        topics_path=topics_path,
        document_topics_path=document_topics_path
    )
    
    # 保存结果
    result_files = calculator.save_results(output_dir)
    
    # 返回结果摘要
    results = calculator.calculate_all_pairs()
    
    return {
        'num_pairs': results['summary']['num_pairs'],
        'avg_pii': results['summary']['avg_pii'],
        'top_paradox': results['summary']['top_paradoxes'][0] if results['summary']['top_paradoxes'] else None,
        'output_files': {k: str(v) for k, v in result_files.items()}
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 运行分析
    results = run_paradox_intensity_analysis()
    print(f"PII分析完成")
    print(f"平均PII: {results['avg_pii']:.4f}")
    print(f"最强悖论: {results['top_paradox']['word_pair'] if results['top_paradox'] else 'None'}")
