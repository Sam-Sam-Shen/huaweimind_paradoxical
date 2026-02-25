"""
方法论创新整合模块

整合所有新方法：
- BERTopic主题建模
- 语义网络分析
- 悖论强度指数（PII）
- 动态主题模型（DTM）

提供统一的分析接口
"""

import logging
from pathlib import Path
from typing import Any

from .bertopic_analysis import run_bertopic_analysis
from .config import get_config
from .dynamic_topic_model import run_dynamic_topic_analysis
from .paradox_intensity import run_paradox_intensity_analysis
from .semantic_network import run_semantic_network_analysis
from .utils import timer

logger = logging.getLogger(__name__)


class IntegratedAnalyzer:
    """整合分析器"""
    
    def __init__(self, use_new_methods: bool = True) -> None:
        """
        初始化整合分析器
        
        Args:
            use_new_methods: 是否使用新方法（BERTopic等）
        """
        self.config = get_config()
        self.use_new_methods = use_new_methods
        
        logger.info(f"初始化整合分析器 (新方法: {use_new_methods})")
    
    @timer
    def run_full_analysis(self, 
                         run_bertopic: bool = True,
                         run_network: bool = True,
                         run_pii: bool = True,
                         run_dtm: bool = True) -> dict[str, Any]:
        """
        运行完整的创新方法分析
        
        Args:
            run_bertopic: 是否运行BERTopic分析
            run_network: 是否运行语义网络分析
            run_pii: 是否运行悖论强度指数分析
            run_dtm: 是否运行动态主题模型分析
            
        Returns:
            所有分析结果的字典
        """
        results = {}
        
        # 1. BERTopic主题建模
        if run_bertopic and self.use_new_methods:
            logger.info("=" * 60)
            logger.info("开始 BERTopic 主题建模...")
            logger.info("=" * 60)
            try:
                results['bertopic'] = run_bertopic_analysis()
                logger.info(f"BERTopic完成: {results['bertopic']['num_topics']} 个主题")
            except Exception as e:
                logger.error(f"BERTopic分析失败: {e}")
                results['bertopic'] = {'error': str(e)}
        
        # 2. 语义网络分析
        if run_network and self.use_new_methods:
            logger.info("=" * 60)
            logger.info("开始 语义网络分析...")
            logger.info("=" * 60)
            try:
                results['network'] = run_semantic_network_analysis()
                logger.info(f"网络分析完成: {results['network']['num_nodes']} 节点, {results['network']['num_edges']} 边")
            except Exception as e:
                logger.error(f"语义网络分析失败: {e}")
                results['network'] = {'error': str(e)}
        
        # 3. 悖论强度指数
        if run_pii and self.use_new_methods:
            logger.info("=" * 60)
            logger.info("开始 悖论强度指数（PII）分析...")
            logger.info("=" * 60)
            try:
                results['pii'] = run_paradox_intensity_analysis()
                logger.info(f"PII分析完成: 平均PII={results['pii']['avg_pii']:.4f}")
            except Exception as e:
                logger.error(f"PII分析失败: {e}")
                results['pii'] = {'error': str(e)}
        
        # 4. 动态主题模型
        if run_dtm and self.use_new_methods:
            logger.info("=" * 60)
            logger.info("开始 动态主题模型（DTM）分析...")
            logger.info("=" * 60)
            try:
                results['dtm'] = run_dynamic_topic_analysis()
                logger.info(f"DTM分析完成: {results['dtm']['num_time_slices']} 个时间片")
            except Exception as e:
                logger.error(f"DTM分析失败: {e}")
                results['dtm'] = {'error': str(e)}
        
        return results
    
    def generate_comparison_report(self, results: dict[str, Any]) -> str:
        """
        生成方法对比报告
        
        Args:
            results: 分析结果字典
            
        Returns:
            Markdown格式的报告
        """
        report = []
        report.append("# 方法论创新分析报告\n")
        report.append("## 执行摘要\n")
        
        # BERTopic结果
        if 'bertopic' in results and 'error' not in results['bertopic']:
            report.append(f"### BERTopic主题建模\n")
            report.append(f"- 发现主题数: {results['bertopic']['num_topics']}\n")
            report.append(f"- 分析文档数: {results['bertopic']['num_documents']}\n")
            report.append(f"- 创新点: 使用Transformer嵌入，自动确定主题数，更好的语义理解\n\n")
        
        # 网络分析结果
        if 'network' in results and 'error' not in results['network']:
            report.append(f"### 语义网络分析\n")
            report.append(f"- 网络节点: {results['network']['num_nodes']}\n")
            report.append(f"- 网络边: {results['network']['num_edges']}\n")
            report.append(f"- 网络密度: {results['network']['density']:.4f}\n")
            report.append(f"- 社区数: {results['network']['num_communities']}\n")
            report.append(f"- 模块度: {results['network']['modularity']:.4f}\n")
            report.append(f"- 创新点: 构建悖论词汇共现网络，识别关键节点和社区结构\n\n")
        
        # PII结果
        if 'pii' in results and 'error' not in results['pii']:
            report.append(f"### 悖论强度指数（PII）\n")
            report.append(f"- 分析词对数: {results['pii']['num_pairs']}\n")
            report.append(f"- 平均PII: {results['pii']['avg_pii']:.4f}\n")
            if results['pii']['top_paradox']:
                report.append(f"- 最强悖论: {' vs '.join(results['pii']['top_paradox']['word_pair'])}\n")
            report.append(f"- 创新点: 整合语义、矛盾、主题、时序四个维度\n\n")
        
        # DTM结果
        if 'dtm' in results and 'error' not in results['dtm']:
            report.append(f"### 动态主题模型（DTM）\n")
            report.append(f"- 时间片数: {results['dtm']['num_time_slices']}\n")
            report.append(f"- 新兴悖论主题: {results['dtm']['emerging_paradoxes']} 个\n")
            report.append(f"- 创新点: 追踪主题演变，识别转折点和新兴悖论\n\n")
        
        # 方法论对比
        report.append("## 新旧方法对比\n\n")
        report.append("| 维度 | 原有方法 | 新方法 | 改进 |\n")
        report.append("|------|---------|--------|------|\n")
        report.append("| 主题建模 | LDA（词袋） | BERTopic（Transformer） | 语义理解能力大幅提升 |\n")
        report.append("| 悖论识别 | 关键词匹配 | 语义网络+社区发现 | 发现隐性悖论关系 |\n")
        report.append("| 强度测量 | 简单距离 | PII四维度整合 | 更全面的悖论量化 |\n")
        report.append("| 时序分析 | 相关分析 | DTM动态追踪 | 捕捉演变轨迹 |\n\n")
        
        # 学术价值
        report.append("## 学术价值\n\n")
        report.append("### 方法创新\n")
        report.append("1. **BERTopic应用**: 首次将BERTopic应用于中文管理文本分析\n")
        report.append("2. **语义网络**: 构建悖论词汇网络，引入社会网络分析方法\n")
        report.append("3. **PII指数**: 提出悖论强度综合测量工具，可跨案例比较\n")
        report.append("4. **DTM追踪**: 实现主题动态演化分析，识别关键转折点\n\n")
        
        report.append("### 理论贡献\n")
        report.append("- 为悖论管理研究提供新的计算分析方法\n")
        report.append("- 建立悖论强度的量化测量框架\n")
        report.append("- 揭示悖论表达的动态演化规律\n\n")
        
        return "\n".join(report)


def run_integrated_analysis(use_new_methods: bool = True) -> dict[str, Any]:
    """
    运行整合分析（便捷函数）
    
    Args:
        use_new_methods: 是否使用新方法
        
    Returns:
        分析结果和报告
    """
    analyzer = IntegratedAnalyzer(use_new_methods)
    
    # 运行分析
    results = analyzer.run_full_analysis()
    
    # 生成报告
    report = analyzer.generate_comparison_report(results)
    
    # 保存报告
    config = get_config()
    output_dir = Path(config.get('output.topic_dir', 'results/topics'))
    report_file = output_dir / 'methodology_innovation_report.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"方法论创新报告已保存: {report_file}")
    
    return {
        'results': results,
        'report': report,
        'report_file': str(report_file)
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 运行整合分析
    output = run_integrated_analysis()
    print("\n" + "=" * 60)
    print("整合分析完成!")
    print("=" * 60)
    print(f"报告文件: {output['report_file']}")
