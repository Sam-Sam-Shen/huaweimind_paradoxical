"""
BERTopic主题分析模块

使用Transformer嵌入和HDBSCAN聚类进行主题建模
替代传统的LDA方法，提供更好的语义理解能力
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from .config import get_config
from .utils import ensure_dir, load_json_safe, save_json, timer

logger = logging.getLogger(__name__)


class BERTopicAnalyzer:
    """BERTopic主题分析器"""
    
    def __init__(self, embedding_model: str = None) -> None:
        """
        初始化BERTopic分析器
        
        Args:
            embedding_model: Sentence Transformer模型名称或本地路径
        """
        self.config = get_config()
        self.model = None
        self.docs = None
        self.doc_ids = None
        
        # 设置嵌入模型（使用本地中文模型）
        if embedding_model:
            self.embedding_model_name = embedding_model
        else:
            # 使用本地中文sentence transformer模型
            local_model = Path('/workspace/huaweimind_paradoxical/models/iic/nlp_corom_sentence-embedding_chinese-base')
            if local_model.exists():
                self.embedding_model_name = str(local_model)
                logger.info(f"使用本地中文嵌入模型: {self.embedding_model_name}")
            else:
                self.embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
                logger.info(f"使用默认嵌入模型: {self.embedding_model_name}")
    
    @timer
    def fit_transform(self, documents: list[dict], min_topic_size: int = 15) -> tuple:
        """
        训练BERTopic模型并转换文档
        
        Args:
            documents: 文档列表，每个文档包含'doc_id', 'content', 'year'等
            min_topic_size: 最小主题大小
            
        Returns:
            (topics, probs, topic_model): 主题标签、概率和模型
        """
        logger.info(f"开始BERTopic分析，文档数: {len(documents)}")
        
        # 提取文本内容
        self.docs = documents
        texts = [doc.get('content', '') for doc in documents]
        self.doc_ids = [doc.get('doc_id', f'doc_{i}') for i, doc in enumerate(documents)]
        
        # 创建嵌入模型
        logger.info("加载Sentence Transformer模型...")
        embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # 配置UMAP降维
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # 配置HDBSCAN聚类
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # 创建BERTopic模型
        logger.info("创建BERTopic模型...")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True,
            calculate_probabilities=True
        )
        
        # 训练模型
        logger.info("训练BERTopic模型...")
        topics, probs = topic_model.fit_transform(texts)
        
        self.model = topic_model
        self.topics = topics
        self.probs = probs
        
        logger.info(f"BERTopic训练完成，发现 {len(set(topics)) - 1} 个主题")
        
        return topics, probs, topic_model
    
    def get_topic_info(self) -> dict[str, Any]:
        """获取主题信息"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        topic_info = self.model.get_topic_info()
        
        # 转换为字典格式
        topics_data = []
        for _, row in topic_info.iterrows():
            if row['Topic'] == -1:
                continue  # 跳过离群值
            
            topic_id = int(row['Topic'])
            topic_words = self.model.get_topic(topic_id)
            
            topics_data.append({
                'topic_id': topic_id,
                'topic_name': f"主题_{topic_id}",  # 可后续手动命名
                'document_count': int(row['Count']),
                'words': [
                    {'word': word, 'weight': float(weight), 'c_tf_idf': float(weight)}
                    for word, weight in topic_words
                ]
            })
        
        return {'topics': topics_data}
    
    def get_document_topics(self) -> list[dict]:
        """获取每个文档的主题分布"""
        if self.model is None or self.docs is None:
            raise ValueError("模型尚未训练")
        
        doc_topics = []
        for i, doc in enumerate(self.docs):
            topic_id = int(self.topics[i])
            
            # 获取该文档的主题概率分布
            if self.probs is not None and len(self.probs) > i:
                topic_probs = self.probs[i]
                # 找到概率最高的主题
                dominant_topic = int(np.argmax(topic_probs))
                topic_weights = {
                    str(j): float(prob) 
                    for j, prob in enumerate(topic_probs)
                }
            else:
                dominant_topic = topic_id
                topic_weights = {str(topic_id): 1.0}
            
            doc_topics.append({
                'doc_id': self.doc_ids[i],
                'title': doc.get('title', ''),
                'year': doc.get('year', 0),
                'type': doc.get('type', ''),
                'dominant_topic': dominant_topic,
                'topic_weights': topic_weights
            })
        
        return doc_topics
    
    def analyze_topic_time_trend(self) -> dict[str, Any]:
        """分析主题时间趋势"""
        if self.model is None or self.docs is None:
            raise ValueError("模型尚未训练")
        
        # 按年份聚合
        year_topic_dist = {}
        years = sorted(set(doc.get('year', 0) for doc in self.docs if doc.get('year', 0) > 0))
        
        for year in years:
            year_docs = [
                i for i, doc in enumerate(self.docs) 
                if doc.get('year', 0) == year
            ]
            
            if not year_docs:
                continue
            
            # 计算该年份的主题分布
            year_topics = [self.topics[i] for i in year_docs]
            topic_counts = {}
            for t in year_topics:
                topic_counts[str(t)] = topic_counts.get(str(t), 0) + 1
            
            # 归一化
            total = len(year_topics)
            year_topic_dist[str(year)] = {
                k: v/total for k, v in topic_counts.items()
            }
        
        # 计算时间相关性
        topic_time_corr = []
        unique_topics = set(self.topics)
        
        for topic_id in unique_topics:
            if topic_id == -1:
                continue
            
            topic_scores = []
            valid_years = []
            
            for year in years:
                year_dist = year_topic_dist.get(str(year), {})
                score = year_dist.get(str(topic_id), 0)
                topic_scores.append(score)
                valid_years.append(year)
            
            if len(valid_years) >= 3 and np.std(topic_scores) > 0:
                from scipy.stats import pearsonr
                corr, p_value = pearsonr(valid_years, topic_scores)
                
                # 判断趋势
                if p_value < 0.05:
                    if corr > 0.1:
                        trend = "increasing"
                    elif corr < -0.1:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"
                
                topic_time_corr.append({
                    'topic_id': int(topic_id),
                    'pearson_corr': float(corr),
                    'p_value': float(p_value),
                    'trend': trend
                })
        
        return {
            'year_topic_distribution': year_topic_dist,
            'topic_time_correlations': topic_time_corr
        }
    
    def find_similar_topics(self, topic_id: int, n_similar: int = 5) -> list[dict]:
        """找到相似的主题"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        similar_topics = self.model.find_topics(topic_id, top_n=n_similar)
        
        return [
            {
                'topic_id': int(tid),
                'similarity': float(sim)
            }
            for tid, sim in zip(similar_topics[0], similar_topics[1])
        ]
    
    def generate_visualizations(self, output_dir: Path) -> dict[str, Path]:
        """
        生成BERTopic可视化图表
        
        Args:
            output_dir: 输出目录
            
        Returns:
            图表文件路径字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        logger.info("生成BERTopic可视化图表...")
        ensure_dir(output_dir)
        
        viz_files = {}
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 主题距离图（Intertopic Distance Map）
        try:
            logger.info("生成主题距离图...")
            fig = self.model.visualize_topics()
            fig_path = output_dir / 'bertopic_intertopic_distance.html'
            fig.write_html(str(fig_path))
            viz_files['intertopic_distance'] = fig_path
            logger.info(f"主题距离图已保存: {fig_path}")
        except Exception as e:
            logger.warning(f"生成主题距离图失败: {e}")
        
        # 2. 主题层次结构图（Topic Hierarchy）
        try:
            logger.info("生成主题层次结构图...")
            fig = self.model.visualize_hierarchy()
            fig_path = output_dir / 'bertopic_hierarchy.html'
            fig.write_html(str(fig_path))
            viz_files['hierarchy'] = fig_path
            logger.info(f"主题层次结构图已保存: {fig_path}")
        except Exception as e:
            logger.warning(f"生成主题层次结构图失败: {e}")
        
        # 3. 主题词条形图（Top Topics）
        try:
            logger.info("生成主题词条形图...")
            fig = self.model.visualize_barchart(top_n_topics=20)
            fig_path = output_dir / 'bertopic_barchart.html'
            fig.write_html(str(fig_path))
            viz_files['barchart'] = fig_path
            logger.info(f"主题词条形图已保存: {fig_path}")
        except Exception as e:
            logger.warning(f"生成主题词条形图失败: {e}")
        
        # 4. 主题相似度热力图
        try:
            logger.info("生成主题相似度热力图...")
            fig = self.model.visualize_heatmap()
            fig_path = output_dir / 'bertopic_heatmap.html'
            fig.write_html(str(fig_path))
            viz_files['heatmap'] = fig_path
            logger.info(f"主题相似度热力图已保存: {fig_path}")
        except Exception as e:
            logger.warning(f"生成主题相似度热力图失败: {e}")
        
        # 5. 文档-主题分布图（Document-Topic Distribution）
        try:
            logger.info("生成文档-主题分布图...")
            # 选择前100个文档进行可视化
            n_docs = min(100, len(self.docs))
            fig = self.model.visualize_distribution(self.probs[0][:n_docs])
            fig_path = output_dir / 'bertopic_distribution.html'
            fig.write_html(str(fig_path))
            viz_files['distribution'] = fig_path
            logger.info(f"文档-主题分布图已保存: {fig_path}")
        except Exception as e:
            logger.warning(f"生成文档-主题分布图失败: {e}")
        
        # 6. 主题词云（静态图）
        try:
            logger.info("生成主题词云...")
            self._generate_topic_wordclouds(output_dir)
            viz_files['wordclouds'] = output_dir / 'wordclouds'
        except Exception as e:
            logger.warning(f"生成主题词云失败: {e}")
        
        # 7. 主题时间趋势图（静态图）
        try:
            logger.info("生成主题时间趋势图...")
            self._generate_topic_trend_plots(output_dir)
            viz_files['trend_plots'] = output_dir / 'topic_trends'
        except Exception as e:
            logger.warning(f"生成主题时间趋势图失败: {e}")
        
        logger.info(f"可视化图表生成完成，共 {len(viz_files)} 个")
        return viz_files
    
    def _generate_topic_wordclouds(self, output_dir: Path) -> None:
        """为每个主题生成词云"""
        from wordcloud import WordCloud
        
        wordcloud_dir = output_dir / 'wordclouds'
        ensure_dir(wordcloud_dir)
        
        # 获取所有主题
        topics = self.model.get_topic_info()
        
        for _, row in topics.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue
            
            # 获取主题词
            topic_words = self.model.get_topic(topic_id)
            
            # 创建词频字典
            word_freq = {word: weight for word, weight in topic_words}
            
            # 生成词云
            wc = WordCloud(
                font_path='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                width=800,
                height=400,
                background_color='white',
                max_words=50
            ).generate_from_frequencies(word_freq)
            
            # 保存
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'主题 {topic_id} 词云', fontsize=16, fontfamily='WenQuanYi Micro Hei')
            
            output_file = wordcloud_dir / f'topic_{topic_id}_wordcloud.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        
        logger.info(f"词云已保存到: {wordcloud_dir}")
    
    def _generate_topic_trend_plots(self, output_dir: Path) -> None:
        """生成主题时间趋势图"""
        trend_dir = output_dir / 'topic_trends'
        ensure_dir(trend_dir)
        
        # 获取时间趋势数据
        time_trend = self.analyze_topic_time_trend()
        year_dist = time_trend.get('year_topic_distribution', {})
        
        if not year_dist:
            logger.warning("没有时间趋势数据")
            return
        
        years = sorted([int(y) for y in year_dist.keys()])
        
        # 为每个主题生成趋势图
        unique_topics = set(self.topics)
        
        for topic_id in unique_topics:
            if topic_id == -1:
                continue
            
            # 获取该主题的时间序列
            topic_scores = []
            for year in years:
                year_data = year_dist.get(str(year), {})
                score = year_data.get(str(topic_id), 0)
                topic_scores.append(score)
            
            # 绘制趋势图
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(years, topic_scores, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('年份', fontsize=12, fontfamily='WenQuanYi Micro Hei')
            ax.set_ylabel('主题占比', fontsize=12, fontfamily='WenQuanYi Micro Hei')
            ax.set_title(f'主题 {topic_id} 时间趋势', fontsize=14, fontfamily='WenQuanYi Micro Hei')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(years) >= 3:
                z = np.polyfit(years, topic_scores, 1)
                p = np.poly1d(z)
                ax.plot(years, p(years), "r--", alpha=0.5, label='趋势线')
                ax.legend()
            
            plt.tight_layout()
            
            output_file = trend_dir / f'topic_{topic_id}_trend.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        
        logger.info(f"主题趋势图已保存到: {trend_dir}")
    
    def save_results(self, output_dir: Path) -> dict[str, Path]:
        """保存分析结果"""
        ensure_dir(output_dir)
        
        # 保存主题信息
        topic_info = self.get_topic_info()
        topic_file = output_dir / 'bertopic_topics.json'
        save_json(topic_info, topic_file)
        
        # 保存文档主题分布
        doc_topics = self.get_document_topics()
        doc_topic_file = output_dir / 'bertopic_document_topics.json'
        save_json(doc_topics, doc_topic_file)
        
        # 保存时间趋势
        time_trend = self.analyze_topic_time_trend()
        time_trend_file = output_dir / 'bertopic_time_trend.json'
        save_json(time_trend, time_trend_file)
        
        # 生成可视化
        viz_files = self.generate_visualizations(output_dir)
        
        # 保存模型
        model_file = output_dir / 'bertopic_model'
        self.model.save(str(model_file))
        
        logger.info(f"BERTopic结果已保存到: {output_dir}")
        
        result_files = {
            'topics': topic_file,
            'document_topics': doc_topic_file,
            'time_trend': time_trend_file,
            'model': model_file
        }
        result_files.update(viz_files)
        
        return result_files


def run_bertopic_analysis(
    documents: list[dict] = None,
    output_dir: Path = None,
    min_topic_size: int = 15
) -> dict[str, Any]:
    """
    运行BERTopic分析（兼容原有接口）
    
    Args:
        documents: 文档列表，如果为None则加载预处理数据
        output_dir: 输出目录
        min_topic_size: 最小主题大小
        
    Returns:
        分析结果字典
    """
    config = get_config()
    
    # 加载文档
    if documents is None:
        from .data_loader import DataLoader
        loader = DataLoader()
        corpus = loader.load_all_documents()
        documents = [
            {
                'doc_id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'year': doc.year,
                'type': doc.doc_type
            }
            for doc in corpus.documents
        ]
        logger.info(f"从语料库加载了 {len(documents)} 篇文档")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(config.get('output.topic_dir', 'results/topics'))
    
    # 创建分析器
    analyzer = BERTopicAnalyzer()
    
    # 运行分析
    topics, probs, model = analyzer.fit_transform(documents, min_topic_size)
    
    # 保存结果
    result_files = analyzer.save_results(output_dir)
    
    # 返回结果摘要
    topic_info = analyzer.get_topic_info()
    
    return {
        'num_topics': len(topic_info['topics']),
        'num_documents': len(documents),
        'output_files': {k: str(v) for k, v in result_files.items()},
        'topics': topic_info['topics'][:5]  # 只返回前5个主题作为预览
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 运行分析
    results = run_bertopic_analysis()
    print(f"分析完成，发现 {results['num_topics']} 个主题")
    print(f"结果保存在: {results['output_files']}")
