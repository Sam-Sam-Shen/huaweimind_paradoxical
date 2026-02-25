"""
可视化模块

生成分析结果的可视化图表
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from wordcloud import WordCloud

from .config import get_config
from .utils import ensure_dir, load_json_safe


logger = logging.getLogger(__name__)


class Visualizer:
    """可视化器"""
    
    def __init__(self) -> None:
        self.config = get_config()
        self._setup_style()
    
    def _setup_style(self) -> None:
        """设置绘图风格"""
        # 清除matplotlib字体缓存，强制重新加载
        import matplotlib.font_manager as fm
        fm._load_fontmanager(try_read_cache=False)
        
        # 设置样式（必须在字体设置之前，因为set_style会重置rcParams）
        style = self.config.get("visualization.style", "seaborn-v0_8-darkgrid")
        try:
            sns.set_style(style.split("-")[1] if "-" in style else "darkgrid")
        except Exception:
            sns.set_style("darkgrid")
        
        # 设置中文字体 - 优先使用文泉驿字体（必须在set_style之后设置）
        try:
            plt.rcParams["font.sans-serif"] = [
                "WenQuanYi Micro Hei",
                "WenQuanYi Zen Hei",
                "SimHei",
                "DejaVu Sans",
                "Arial Unicode MS",
            ]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("中文字体设置成功")
        except Exception as e:
            logger.warning(f"设置中文字体失败: {e}")
        
        # 设置尺寸
        self.fig_size = self.config.get("visualization.figure_size", [12, 8])
    
    def plot_year_distribution(
        self,
        metadata: dict[str, Any],
        output_path: Path,
    ) -> None:
        """绘制年份分布图"""
        year_dist = metadata.get("year_distribution", {})
        
        if not year_dist:
            logger.warning("无年份分布数据")
            return
        
        years = sorted(year_dist.keys())
        counts = [year_dist[y] for y in years]
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.bar(years, counts, color="steelblue", alpha=0.8)
        ax.set_xlabel("年份", fontsize=12)
        ax.set_ylabel("文档数量", fontsize=12)
        ax.set_title("任正非讲话文档年份分布", fontsize=14)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "year_distribution.png", dpi=300)
        plt.close()
        
        logger.info(f"年份分布图已保存: {output_path / 'year_distribution.png'}")
    
    def plot_topic_time_trend(
        self,
        time_trend: dict[str, Any],
        output_path: Path,
    ) -> None:
        """绘制主题时间趋势"""
        year_topic = time_trend.get("year_topic_distribution", {})
        
        if not year_topic:
            logger.warning("无时间趋势数据")
            return
        
        years = sorted(year_topic.keys())
        topic_dists = [year_topic[y] for y in years]
        
        # 选择前10个主题
        num_topics = len(topic_dists[0])
        top_topics = min(10, num_topics)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for topic_idx in range(top_topics):
            ax = axes[topic_idx]
            values = [dist[topic_idx] for dist in topic_dists]
            
            ax.plot(years, values, marker="o", linewidth=2)
            ax.set_title(f"主题 {topic_idx + 1}", fontsize=10)
            ax.set_xlabel("年份", fontsize=8)
            ax.set_ylabel("权重", fontsize=8)
            ax.tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "topic_time_trend.png", dpi=300)
        plt.close()
        
        logger.info(f"主题时间趋势图已保存: {output_path / 'topic_time_trend.png'}")
    
    def plot_topic_heatmap(
        self,
        time_trend: dict[str, Any],
        output_path: Path,
    ) -> None:
        """绘制主题热力图"""
        year_topic = time_trend.get("year_topic_distribution", {})
        
        if not year_topic:
            return
        
        years = sorted(year_topic.keys())
        topics = year_topic[years[0]]
        
        # 转换为矩阵
        matrix = np.array([year_topic[y] for y in years])
        
        # 选择前15个主题
        if matrix.shape[1] > 15:
            matrix = matrix[:, :15]
            topics = topics[:15]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sns.heatmap(
            matrix.T,
            xticklabels=years,
            yticklabels=[f"主题{i+1}" for i in range(matrix.shape[1])],
            cmap="YlOrRd",
            ax=ax,
        )
        
        ax.set_xlabel("年份", fontsize=12)
        ax.set_ylabel("主题", fontsize=12)
        ax.set_title("主题分布热力图", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path / "topic_heatmap.png", dpi=300)
        plt.close()
        
        logger.info(f"主题热力图已保存: {output_path / 'topic_heatmap.png'}")
    
    def plot_paradox_distances(
        self,
        distances: dict[str, Any],
        output_path: Path,
    ) -> None:
        """绘制悖论词对距离"""
        paradox_dist = distances.get("paradox_distances", [])
        
        if not paradox_dist:
            logger.warning("无悖论距离数据")
            return
        
        valid_dist = [
            d for d in paradox_dist
            if "distance" in d and "word1" in d
        ]
        
        if not valid_dist:
            return
        
        labels = [f"{d['word1']}-{d['word2']}" for d in valid_dist]
        values = [d["distance"] for d in valid_dist]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(labels, values, color="coral", alpha=0.8)
        
        ax.set_xlabel("语义距离", fontsize=12)
        ax.set_ylabel("悖论词对", fontsize=12)
        ax.set_title("悖论词对语义距离分析", fontsize=14)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center")
        
        plt.tight_layout()
        plt.savefig(output_path / "paradox_distances.png", dpi=300)
        plt.close()
        
        logger.info(f"悖论距离图已保存: {output_path / 'paradox_distances.png'}")
    
    def plot_contradiction_distribution(
        self,
        stats: dict[str, Any],
        output_path: Path,
    ) -> None:
        """绘制矛盾分布图"""
        year_dist = stats.get("year_distribution", {})
        
        if not year_dist:
            logger.warning("无矛盾分布数据")
            return
        
        years = sorted(year_dist.keys())
        counts = [year_dist[y] for y in years]
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.bar(years, counts, color="indianred", alpha=0.8)
        ax.set_xlabel("年份", fontsize=12)
        ax.set_ylabel("矛盾对数量", fontsize=12)
        ax.set_title("矛盾表达年份分布", fontsize=14)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "contradiction_distribution.png", dpi=300)
        plt.close()
        
        logger.info(f"矛盾分布图已保存: {output_path / 'contradiction_distribution.png'}")
    
    def plot_word_cloud(
        self,
        topics: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """绘制词云"""
        for topic in topics[:5]:  # 前5个主题
            topic_id = topic.get("topic_id", 0)
            words = topic.get("words", [])
            
            if not words:
                continue
            
            # 构建词频字典
            word_freq = {w["word"]: w["weight"] for w in words}
            
            # 生成词云
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                font_path=None,  # 使用默认字体
                max_words=50,
            ).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"主题 {topic_id + 1} 词云", fontsize=14)
            
            plt.savefig(
                output_path / f"topic_{topic_id}_wordcloud.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        
        logger.info("词云图已保存")
    
    def plot_correlation_network(
        self,
        topics: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """绘制主题相关性网络（简化版）"""
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx未安装，跳过网络图")
            return
        
        # 从主题词汇构建简单网络
        G = nx.Graph()
        
        for topic in topics[:10]:
            topic_id = topic.get("topic_id", 0)
            words = topic.get("words", [])[:10]
            
            G.add_node(f"主题{topic_id+1}")
            
            # 连接相关主题（基于词汇重叠）
            for other_topic in topics[:10]:
                if other_topic["topic_id"] <= topic_id:
                    continue
                
                other_words = set(w["word"] for w in other_topic.get("words", [])[:10])
                current_words = set(w["word"] for w in words)
                
                overlap = len(current_words & other_words)
                
                if overlap >= 3:
                    G.add_edge(
                        f"主题{topic_id+1}",
                        f"主题{other_topic['topic_id']+1}",
                        weight=overlap,
                    )
        
        if G.number_of_nodes() > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightblue",
                node_size=2000,
                font_size=10,
                font_family="sans-serif",
            )
            
            ax.set_title("主题相关性网络", fontsize=14)
            
            plt.savefig(output_path / "topic_network.png", dpi=300)
            plt.close()
            
            logger.info(f"主题网络图已保存: {output_path / 'topic_network.png'}")
    
    def generate_all_plots(
        self,
        results_dir: str,
        output_dir: str | None = None,
    ) -> None:
        """生成所有图表"""
        results_path = Path(results_dir)
        
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = results_path / "figures"
        
        ensure_dir(output_path)
        
        # 加载数据
        topics_file = results_path / "topics.json"
        time_trend_file = results_path / "topic_time_trend.json"
        paradox_file = results_path / "topics" / "paradox_distances.json"
        contradiction_file = results_path / "contradictions" / "stats.json"
        
        # 元数据
        metadata_file = get_config().get_path("metadata_file")
        metadata = load_json_safe(metadata_file) or {}
        
        if metadata:
            self.plot_year_distribution(metadata, output_path)
        
        # 主题趋势
        time_trend = load_json_safe(str(time_trend_file)) or {}
        if time_trend:
            self.plot_topic_time_trend(time_trend, output_path)
            self.plot_topic_heatmap(time_trend, output_path)
        
        # 主题列表
        topics = load_json_safe(str(topics_file)) or []
        if topics:
            self.plot_word_cloud(topics, output_path)
            self.plot_correlation_network(topics, output_path)
        
        # 悖论距离
        paradox = load_json_safe(str(paradox_file)) or {}
        if paradox:
            self.plot_paradox_distances(paradox, output_path)
        
        # 矛盾分布
        contradiction_stats = load_json_safe(str(contradiction_file)) or {}
        if contradiction_stats:
            self.plot_contradiction_distribution(contradiction_stats, output_path)
        
        logger.info(f"所有图表已生成: {output_path}")


def run_visualization() -> dict[str, Any]:
    """运行可视化"""
    results_dir = get_config().get_path("results_dir")
    
    visualizer = Visualizer()
    visualizer.generate_all_plots(results_dir)
    
    return {"output_dir": results_dir}
