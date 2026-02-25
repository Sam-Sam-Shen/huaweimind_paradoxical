"""
语义网络分析模块

构建悖论词汇的共现网络，分析网络结构和社区发现
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm

from .config import get_config
from .utils import ensure_dir, save_json, timer

logger = logging.getLogger(__name__)


class ParadoxNetworkAnalyzer:
    """悖论语义网络分析器"""
    
    def __init__(self, paradox_pairs: list[list[str]] = None) -> None:
        """
        初始化网络分析器
        
        Args:
            paradox_pairs: 悖论词对列表
        """
        self.config = get_config()
        self.paradox_pairs = paradox_pairs or self._load_paradox_pairs()
        self.G = nx.Graph()
        
        # 提取所有悖论词汇
        self.paradox_words = set()
        for pair in self.paradox_pairs:
            self.paradox_words.update(pair)
        
        logger.info(f"加载了 {len(self.paradox_pairs)} 对悖论词汇，共 {len(self.paradox_words)} 个词")
    
    def _load_paradox_pairs(self) -> list[list[str]]:
        """从配置加载悖论词对"""
        pairs = self.config.get('contradiction_detection.paradox_pairs', [])
        if not pairs:
            # 默认词对
            pairs = [
                ["开放", "保守"],
                ["激进", "稳健"],
                ["集权", "放权"],
                ["扩张", "收缩"],
                ["进攻", "防守"],
                ["创新", "守成"],
                ["自主", "引进"],
                ["全球化", "本土化"],
                ["利润", "规模"],
                ["奋斗", "享受"]
            ]
        return pairs
    
    @timer
    def build_cooccurrence_network(
        self, 
        documents: list[str], 
        window_size: int = 5,
        min_cooccurrence: int = 3
    ) -> nx.Graph:
        """
        构建悖论词汇共现网络
        
        Args:
            documents: 文档文本列表（已分词）
            window_size: 共现窗口大小
            min_cooccurrence: 最小共现次数阈值
            
        Returns:
            NetworkX图对象
        """
        logger.info(f"构建共现网络，窗口大小: {window_size}，文档数: {len(documents)}")
        
        # 统计共现
        cooccurrence = defaultdict(int)
        
        for doc_idx, doc in enumerate(documents):
            if doc_idx % 100 == 0:
                logger.debug(f"处理文档 {doc_idx}/{len(documents)}")
            
            words = doc.split()
            
            # 滑动窗口
            for i in range(len(words) - window_size + 1):
                window = words[i:i+window_size]
                
                # 提取窗口中的悖论词汇
                window_paradox = [w for w in window if w in self.paradox_words]
                
                # 统计共现（无序对）
                for j in range(len(window_paradox)):
                    for k in range(j + 1, len(window_paradox)):
                        w1, w2 = window_paradox[j], window_paradox[k]
                        if w1 != w2:
                            pair = tuple(sorted([w1, w2]))
                            cooccurrence[pair] += 1
        
        logger.info(f"发现 {len(cooccurrence)} 对共现关系")
        
        # 构建网络
        self.G = nx.Graph()
        
        for (w1, w2), weight in cooccurrence.items():
            if weight >= min_cooccurrence:
                self.G.add_edge(w1, w2, weight=weight)
        
        # 添加孤立节点（没有共现的悖论词汇）
        for word in self.paradox_words:
            if word not in self.G:
                self.G.add_node(word)
        
        logger.info(f"网络构建完成: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        return self.G
    
    def calculate_network_metrics(self) -> dict[str, Any]:
        """
        计算网络指标
        
        Returns:
            网络指标字典
        """
        if self.G.number_of_nodes() == 0:
            logger.warning("网络为空，无法计算指标")
            return {}
        
        logger.info("计算网络指标...")
        
        metrics = {
            'basic_stats': {
                'num_nodes': self.G.number_of_nodes(),
                'num_edges': self.G.number_of_edges(),
                'density': nx.density(self.G),
                'is_connected': nx.is_connected(self.G),
            }
        }
        
        # 连通分量
        if not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            metrics['basic_stats']['num_components'] = len(components)
            metrics['basic_stats']['largest_component_size'] = len(max(components, key=len))
        
        # 聚类系数
        metrics['clustering'] = {
            'average_clustering': nx.average_clustering(self.G),
            'transitivity': nx.transitivity(self.G)
        }
        
        # 中心性指标
        logger.info("计算中心性指标...")
        
        # 度中心性
        degree_cent = nx.degree_centrality(self.G)
        
        # 中介中心性（仅对连通分量计算）
        try:
            betweenness_cent = nx.betweenness_centrality(self.G, weight='weight')
        except:
            betweenness_cent = {node: 0.0 for node in self.G.nodes()}
        
        # 特征向量中心性
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.G, weight='weight', max_iter=1000)
        except:
            eigenvector_cent = {node: 0.0 for node in self.G.nodes()}
        
        # 接近中心性（仅对连通分量计算）
        try:
            closeness_cent = nx.closeness_centrality(self.G)
        except:
            closeness_cent = {node: 0.0 for node in self.G.nodes()}
        
        # 整理中心性数据
        centrality_data = []
        for node in self.G.nodes():
            centrality_data.append({
                'word': node,
                'degree': self.G.degree(node),
                'degree_centrality': round(degree_cent.get(node, 0), 4),
                'betweenness_centrality': round(betweenness_cent.get(node, 0), 4),
                'eigenvector_centrality': round(eigenvector_cent.get(node, 0), 4),
                'closeness_centrality': round(closeness_cent.get(node, 0), 4)
            })
        
        # 按度中心性排序
        centrality_data.sort(key=lambda x: x['degree_centrality'], reverse=True)
        
        metrics['centrality'] = centrality_data
        
        # 识别关键节点（top 5）
        metrics['key_nodes'] = {
            'highest_degree': centrality_data[:5],
            'highest_betweenness': sorted(centrality_data, key=lambda x: x['betweenness_centrality'], reverse=True)[:5]
        }
        
        return metrics
    
    def detect_communities(self) -> dict[str, Any]:
        """
        检测网络社区（悖论词汇群组）
        
        Returns:
            社区分析结果
        """
        if self.G.number_of_edges() == 0:
            logger.warning("网络无边，无法检测社区")
            return {}
        
        logger.info("检测网络社区...")
        
        # 使用Louvain算法
        partition = community_louvain.best_partition(self.G, weight='weight')
        
        # 按社区分组
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # 计算模块度
        modularity = community_louvain.modularity(partition, self.G, weight='weight')
        
        # 分析每个社区
        community_analysis = []
        for comm_id, nodes in communities.items():
            # 计算社区内部边权重
            internal_edges = []
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    if self.G.has_edge(n1, n2):
                        internal_edges.append(self.G[n1][n2]['weight'])
            
            # 检查是否包含悖论对
            paradox_pairs_in_comm = []
            for pair in self.paradox_pairs:
                if pair[0] in nodes and pair[1] in nodes:
                    paradox_pairs_in_comm.append(pair)
            
            community_analysis.append({
                'community_id': int(comm_id),
                'nodes': nodes,
                'size': len(nodes),
                'internal_edge_count': len(internal_edges),
                'avg_internal_weight': np.mean(internal_edges) if internal_edges else 0,
                'paradox_pairs': paradox_pairs_in_comm,
                'is_paradox_community': len(paradox_pairs_in_comm) > 0
            })
        
        # 按大小排序
        community_analysis.sort(key=lambda x: x['size'], reverse=True)
        
        return {
            'num_communities': len(communities),
            'modularity': round(modularity, 4),
            'communities': community_analysis
        }
    
    def analyze_paradox_pairs_in_network(self) -> dict[str, Any]:
        """
        分析悖论词对在网络中的关系
        
        Returns:
            悖论对分析结果
        """
        logger.info("分析悖论词对关系...")
        
        pair_analysis = []
        
        for pair in self.paradox_pairs:
            w1, w2 = pair[0], pair[1]
            
            analysis = {
                'word1': w1,
                'word2': w2,
                'both_in_network': w1 in self.G and w2 in self.G,
                'directly_connected': self.G.has_edge(w1, w2) if w1 in self.G and w2 in self.G else False
            }
            
            if analysis['both_in_network'] and analysis['directly_connected']:
                analysis['cooccurrence_weight'] = self.G[w1][w2]['weight']
                
                # 计算共同邻居（桥梁词）
                common_neighbors = list(nx.common_neighbors(self.G, w1, w2))
                analysis['common_neighbors'] = common_neighbors
                analysis['num_common_neighbors'] = len(common_neighbors)
                
                # 最短路径（如果不直接相连）
                try:
                    path = nx.shortest_path(self.G, w1, w2, weight='weight')
                    analysis['shortest_path_length'] = len(path) - 1
                    analysis['shortest_path'] = path
                except nx.NetworkXNoPath:
                    analysis['shortest_path_length'] = -1
            else:
                analysis['cooccurrence_weight'] = 0
                analysis['common_neighbors'] = []
                analysis['num_common_neighbors'] = 0
            
            pair_analysis.append(analysis)
        
        # 统计
        connected_pairs = [p for p in pair_analysis if p['directly_connected']]
        
        return {
            'total_pairs': len(pair_analysis),
            'pairs_in_network': len([p for p in pair_analysis if p['both_in_network']]),
            'directly_connected_pairs': len(connected_pairs),
            'avg_cooccurrence_weight': np.mean([p['cooccurrence_weight'] for p in connected_pairs]) if connected_pairs else 0,
            'pair_details': pair_analysis
        }
    
    def visualize_network(self, output_path: Path, figsize: tuple = (14, 14)) -> Path:
        """
        可视化网络
        
        Args:
            output_path: 输出文件路径
            figsize: 图像尺寸
            
        Returns:
            输出文件路径
        """
        if self.G.number_of_nodes() == 0:
            logger.warning("网络为空，无法可视化")
            return None
        
        logger.info("生成网络可视化...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 布局算法
        if self.G.number_of_nodes() < 30:
            pos = nx.spring_layout(self.G, k=3, iterations=100, seed=42)
        else:
            pos = nx.kamada_kawai_layout(self.G)
        
        # 检测社区用于着色
        if self.G.number_of_edges() > 0:
            partition = community_louvain.best_partition(self.G, weight='weight')
            communities = set(partition.values())
            colors = [partition[node] for node in self.G.nodes()]
        else:
            colors = 'lightblue'
        
        # 节点大小基于度中心性
        node_sizes = [self.G.degree(node) * 300 + 200 for node in self.G.nodes()]
        
        # 绘制节点
        nx.draw_networkx_nodes(
            self.G, pos, 
            node_color=colors,
            node_size=node_sizes,
            cmap=cm.Set3,
            alpha=0.8,
            ax=ax
        )
        
        # 绘制边（粗细基于权重）
        if self.G.number_of_edges() > 0:
            edges = self.G.edges()
            weights = [self.G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            
            nx.draw_networkx_edges(
                self.G, pos,
                width=[w/max_weight * 3 for w in weights],
                alpha=0.5,
                edge_color='gray',
                ax=ax
            )
        
        # 绘制标签
        nx.draw_networkx_labels(
            self.G, pos,
            font_size=11,
            font_family='WenQuanYi Micro Hei',
            ax=ax
        )
        
        # 标题和图例
        ax.set_title(
            f'悖论词汇共现网络\n({self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边)',
            fontsize=16,
            fontfamily='WenQuanYi Micro Hei',
            pad=20
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        # 保存
        output_file = output_path / 'paradox_network.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"网络可视化已保存: {output_file}")
        
        return output_file
    
    def save_results(self, output_dir: Path) -> dict[str, Path]:
        """保存分析结果"""
        ensure_dir(output_dir)
        
        # 保存网络指标
        metrics = self.calculate_network_metrics()
        metrics_file = output_dir / 'network_metrics.json'
        save_json(metrics, metrics_file)
        
        # 保存社区分析
        communities = self.detect_communities()
        communities_file = output_dir / 'network_communities.json'
        save_json(communities, communities_file)
        
        # 保存悖论对分析
        pair_analysis = self.analyze_paradox_pairs_in_network()
        pair_file = output_dir / 'paradox_pairs_network.json'
        save_json(pair_analysis, pair_file)
        
        # 保存网络数据（GEXF格式，可用Gephi打开）
        gexf_file = output_dir / 'paradox_network.gexf'
        nx.write_gexf(self.G, str(gexf_file))
        
        # 生成可视化
        viz_file = self.visualize_network(output_dir)
        
        logger.info(f"网络分析结果已保存到: {output_dir}")
        
        return {
            'metrics': metrics_file,
            'communities': communities_file,
            'pair_analysis': pair_file,
            'gexf': gexf_file,
            'visualization': viz_file
        }


def run_semantic_network_analysis(
    documents: list[str] = None,
    output_dir: Path = None,
    window_size: int = 5,
    min_cooccurrence: int = 3
) -> dict[str, Any]:
    """
    运行语义网络分析（兼容原有接口）
    
    Args:
        documents: 文档文本列表，如果为None则加载预处理数据
        output_dir: 输出目录
        window_size: 共现窗口大小
        min_cooccurrence: 最小共现次数
        
    Returns:
        分析结果字典
    """
    config = get_config()
    
    # 加载文档
    if documents is None:
        from .data_loader import DataLoader
        from .preprocess import ChinesePreprocessor
        
        loader = DataLoader()
        corpus = loader.load_all_documents()
        
        # 对文档进行分词处理
        logger.info("对文档进行分词处理...")
        preprocessor = ChinesePreprocessor()
        documents = []
        for doc in corpus.documents:
            # 只提取文本内容并分词
            tokens = preprocessor.tokenize(doc.content)
            if tokens:
                documents.append(' '.join(tokens))
        
        logger.info(f"从语料库加载并分词了 {len(documents)} 篇文档")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(config.get('output.topic_dir', 'results/topics')) / 'network'
    
    # 创建分析器
    analyzer = ParadoxNetworkAnalyzer()
    
    # 构建网络
    analyzer.build_cooccurrence_network(documents, window_size, min_cooccurrence)
    
    # 保存结果
    result_files = analyzer.save_results(output_dir)
    
    # 返回结果摘要
    metrics = analyzer.calculate_network_metrics()
    communities = analyzer.detect_communities()
    
    return {
        'num_nodes': metrics.get('basic_stats', {}).get('num_nodes', 0),
        'num_edges': metrics.get('basic_stats', {}).get('num_edges', 0),
        'density': metrics.get('basic_stats', {}).get('density', 0),
        'num_communities': communities.get('num_communities', 0),
        'modularity': communities.get('modularity', 0),
        'output_files': {k: str(v) for k, v in result_files.items()}
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 运行分析
    results = run_semantic_network_analysis()
    print(f"网络分析完成")
    print(f"节点数: {results['num_nodes']}, 边数: {results['num_edges']}")
    print(f"社区数: {results['num_communities']}, 模块度: {results['modularity']:.4f}")
