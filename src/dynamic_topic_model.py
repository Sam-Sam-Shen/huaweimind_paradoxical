"""
动态主题模型（Dynamic Topic Model, DTM）模块

追踪主题随时间的演变，识别转折点和新出现的悖论主题
简化实现：基于时间切片的多次LDA训练
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from gensim import corpora, models
from scipy.stats import linregress
from sklearn.feature_extraction.text import CountVectorizer

from .config import get_config
from .utils import ensure_dir, save_json, timer

logger = logging.getLogger(__name__)


class DynamicTopicAnalyzer:
    """动态主题分析器"""
    
    def __init__(self, n_topics: int = 25, time_slice_years: int = 5) -> None:
        """
        初始化分析器
        
        Args:
            n_topics: 主题数量
            time_slice_years: 每个时间片的年数
        """
        self.config = get_config()
        self.n_topics = n_topics
        self.time_slice_years = time_slice_years
        
        self.time_slices = []  # 每个时间片的文档数
        self.time_labels = []  # 时间标签（年份）
        self.models = {}  # 每个时间片的模型
        self.dictionaries = {}  # 每个时间片的词典
        
        logger.info(f"初始化DTM分析器: {n_topics} 主题, {time_slice_years} 年/时间片")
    
    def prepare_time_slices(self, documents: list[dict]) -> dict[int, list[str]]:
        """
        按年份准备时间切片
        
        Args:
            documents: 文档列表
            
        Returns:
            按年份分组的文档字典
        """
        logger.info("准备时间切片...")
        
        # 按年份分组
        year_groups = defaultdict(list)
        for doc in documents:
            year = doc.get('year', 0)
            if year > 0:
                processed = doc.get('processed_text', '')
                if processed:
                    year_groups[year].append(processed)
        
        # 按时间片合并（如每5年一个时间片）
        min_year = min(year_groups.keys())
        max_year = max(year_groups.keys())
        
        slice_groups = defaultdict(list)
        for year in range(min_year, max_year + 1):
            if year in year_groups:
                slice_id = (year - min_year) // self.time_slice_years
                slice_groups[slice_id].extend(year_groups[year])
                
                # 记录时间标签（使用起始年份）
                if slice_id not in [t['slice_id'] for t in self.time_labels]:
                    self.time_labels.append({
                        'slice_id': slice_id,
                        'start_year': min_year + slice_id * self.time_slice_years,
                        'end_year': min(min_year + (slice_id + 1) * self.time_slice_years - 1, max_year)
                    })
        
        logger.info(f"创建了 {len(slice_groups)} 个时间片")
        
        return dict(slice_groups)
    
    @timer
    def train_slice_models(self, slice_groups: dict[int, list[str]]) -> None:
        """
        为每个时间片训练LDA模型
        
        Args:
            slice_groups: 时间片分组文档
        """
        logger.info("训练时间片模型...")
        
        for slice_id, texts in sorted(slice_groups.items()):
            if len(texts) < 10:
                logger.warning(f"时间片 {slice_id} 文档数过少 ({len(texts)})，跳过")
                continue
            
            logger.info(f"训练时间片 {slice_id} ({len(texts)} 篇文档)...")
            
            # 创建词典和语料库
            texts_split = [t.split() for t in texts]
            dictionary = corpora.Dictionary(texts_split)
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in texts_split]
            
            # 训练LDA模型
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.n_topics,
                random_state=42,
                passes=10,
                iterations=100,
                alpha='auto'
            )
            
            self.models[slice_id] = lda_model
            self.dictionaries[slice_id] = dictionary
            self.time_slices.append(len(texts))
        
        logger.info(f"训练完成: {len(self.models)} 个时间片模型")
    
    def analyze_topic_evolution(self, topic_id: int) -> dict[str, Any]:
        """
        分析特定主题的演变
        
        Args:
            topic_id: 主题ID
            
        Returns:
            主题演变分析结果
        """
        evolution = {
            'topic_id': topic_id,
            'time_slices': [],
            'word_evolution': [],
            'popularity_trend': []
        }
        
        # 获取每个时间片的主题信息
        for slice_id in sorted(self.models.keys()):
            model = self.models[slice_id]
            
            if topic_id >= model.num_topics:
                continue
            
            # 获取主题词
            topic_words = model.show_topic(topic_id, topn=10)
            
            # 计算主题流行度
            corpus = model.id2word.doc2bow([])  # 空文档，仅用于获取词典大小
            topic_dist = model.get_topic_terms(topic_id, topn=model.num_topics)
            popularity = sum([prob for _, prob in topic_dist]) / model.num_topics
            
            time_info = next((t for t in self.time_labels if t['slice_id'] == slice_id), {})
            
            evolution['time_slices'].append(slice_id)
            evolution['word_evolution'].append({
                'slice_id': slice_id,
                'year_range': f"{time_info.get('start_year', 0)}-{time_info.get('end_year', 0)}",
                'top_words': [{'word': w, 'weight': float(p)} for w, p in topic_words]
            })
            evolution['popularity_trend'].append({
                'slice_id': slice_id,
                'year_range': f"{time_info.get('start_year', 0)}-{time_info.get('end_year', 0)}",
                'popularity': float(popularity)
            })
        
        # 检测趋势
        if len(evolution['popularity_trend']) >= 2:
            popularities = [p['popularity'] for p in evolution['popularity_trend']]
            slice_ids = list(range(len(popularities)))
            
            slope, intercept, r_value, p_value, std_err = linregress(slice_ids, popularities)
            
            if p_value < 0.05:
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            evolution['trend'] = {
                'type': trend,
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value)
            }
        
        return evolution
    
    def detect_turning_points(self, topic_id: int, threshold_std: float = 2.0) -> list[dict]:
        """
        检测主题的转折点
        
        Args:
            topic_id: 主题ID
            threshold_std: 变化率标准差阈值
            
        Returns:
            转折点列表
        """
        evolution = self.analyze_topic_evolution(topic_id)
        
        if len(evolution['popularity_trend']) < 3:
            return []
        
        popularities = [p['popularity'] for p in evolution['popularity_trend']]
        changes = np.diff(popularities)
        
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        turning_points = []
        for i, change in enumerate(changes):
            if abs(change - mean_change) > threshold_std * std_change:
                time_info = evolution['popularity_trend'][i + 1]
                turning_points.append({
                    'slice_id': time_info['slice_id'],
                    'year_range': time_info['year_range'],
                    'change': float(change),
                    'significance': 'high' if abs(change) > 3 * std_change else 'medium'
                })
        
        return turning_points
    
    def find_emerging_paradoxes(self, paradox_keywords: list[list[str]]) -> list[dict]:
        """
        发现新兴的悖论主题
        
        Args:
            paradox_keywords: 悖论关键词对
            
        Returns:
            新兴悖论主题列表
        """
        logger.info("寻找新兴悖论主题...")
        
        emerging = []
        
        for topic_id in range(self.n_topics):
            evolution = self.analyze_topic_evolution(topic_id)
            
            if 'trend' not in evolution or evolution['trend']['type'] != 'increasing':
                continue
            
            # 检查是否包含悖论关键词
            has_paradox = False
            paradox_words_found = []
            
            for word_info in evolution['word_evolution'][-1]['top_words']:
                word = word_info['word']
                for pair in paradox_keywords:
                    if word in pair:
                        has_paradox = True
                        paradox_words_found.append(word)
            
            if has_paradox:
                turning_points = self.detect_turning_points(topic_id)
                
                emerging.append({
                    'topic_id': topic_id,
                    'evolution': evolution,
                    'paradox_words': paradox_words_found,
                    'turning_points': turning_points,
                    'emergence_slice': evolution['time_slices'][0] if evolution['time_slices'] else 0
                })
        
        # 按趋势强度排序
        emerging.sort(key=lambda x: x['evolution']['trend']['slope'], reverse=True)
        
        return emerging
    
    def save_results(self, output_dir: Path) -> dict[str, Path]:
        """保存分析结果"""
        ensure_dir(output_dir)
        
        # 分析所有主题的演变
        all_evolutions = []
        for topic_id in range(self.n_topics):
            try:
                evolution = self.analyze_topic_evolution(topic_id)
                all_evolutions.append(evolution)
            except Exception as e:
                logger.error(f"分析主题 {topic_id} 时出错: {e}")
        
        # 保存主题演变
        evolution_file = output_dir / 'topic_evolution.json'
        save_json({'evolutions': all_evolutions}, evolution_file)
        
        # 保存转折点
        all_turning_points = {}
        for topic_id in range(self.n_topics):
            points = self.detect_turning_points(topic_id)
            if points:
                all_turning_points[f'topic_{topic_id}'] = points
        
        turning_points_file = output_dir / 'turning_points.json'
        save_json(all_turning_points, turning_points_file)
        
        # 保存新兴悖论主题
        paradox_keywords = self.config.get('contradiction_detection.paradox_pairs', [])
        emerging = self.find_emerging_paradoxes(paradox_keywords)
        
        emerging_file = output_dir / 'emerging_paradoxes.json'
        save_json({'emerging_paradoxes': emerging}, emerging_file)
        
        logger.info(f"DTM结果已保存到: {output_dir}")
        
        return {
            'evolution': evolution_file,
            'turning_points': turning_points_file,
            'emerging_paradoxes': emerging_file
        }


def run_dynamic_topic_analysis(
    documents: list[dict] = None,
    output_dir: Path = None,
    n_topics: int = 25,
    time_slice_years: int = 5
) -> dict[str, Any]:
    """
    运行动态主题分析（兼容原有接口）
    
    Args:
        documents: 文档列表
        output_dir: 输出目录
        n_topics: 主题数量
        time_slice_years: 时间片年数
        
    Returns:
        分析结果字典
    """
    config = get_config()
    
    # 加载文档
    if documents is None:
        from .data_loader import load_corpus
        documents = load_corpus()
        logger.info(f"从语料库加载了 {len(documents)} 篇文档")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(config.get('output.topic_dir', 'results/topics')) / 'dynamic'
    
    # 创建分析器
    analyzer = DynamicTopicAnalyzer(n_topics, time_slice_years)
    
    # 准备时间切片
    slice_groups = analyzer.prepare_time_slices(documents)
    
    # 训练模型
    analyzer.train_slice_models(slice_groups)
    
    # 保存结果
    result_files = analyzer.save_results(output_dir)
    
    # 返回摘要
    paradox_keywords = config.get('contradiction_detection.paradox_pairs', [])
    emerging = analyzer.find_emerging_paradoxes(paradox_keywords)
    
    return {
        'num_time_slices': len(analyzer.models),
        'num_topics': n_topics,
        'emerging_paradoxes': len(emerging),
        'top_emerging': emerging[:3] if emerging else [],
        'output_files': {k: str(v) for k, v in result_files.items()}
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 运行分析
    results = run_dynamic_topic_analysis()
    print(f"DTM分析完成")
    print(f"时间片数: {results['num_time_slices']}")
    print(f"新兴悖论主题: {results['emerging_paradoxes']} 个")
