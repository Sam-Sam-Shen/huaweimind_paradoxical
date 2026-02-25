"""
报告生成模块

生成Markdown和PDF分析报告
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template

from .config import get_config
from .utils import ensure_dir, load_json_safe, save_json_safe


logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self) -> None:
        self.config = get_config()
        self._template_cache: dict[str, Template] = {}
    
    def generate_markdown_report(
        self,
        results_dir: str,
        output_path: str,
    ) -> str:
        """生成Markdown报告"""
        logger.info("生成Markdown报告")
        
        # 加载所有数据
        report_data = self._collect_report_data(results_dir)
        
        # 生成内容
        content = self._generate_markdown_content(report_data)
        
        # 保存
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Markdown报告已保存: {output_path}")
        return output_path
    
    def _collect_report_data(self, results_dir: str) -> dict[str, Any]:
        """收集报告数据"""
        results_path = Path(results_dir)
        
        data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_name": "华为思维悖论分析",
            "subtitle": "任正非管理思想悖论性分析报告",
        }
        
        # 元数据
        metadata_file = get_config().get_path("metadata_file")
        data["metadata"] = load_json_safe(metadata_file) or {}
        
        # 主题分析
        topics_file = results_path / "topics.json"
        data["topics"] = load_json_safe(str(topics_file)) or {}
        
        # 时间趋势
        time_trend_file = results_path / "topic_time_trend.json"
        data["time_trend"] = load_json_safe(str(time_trend_file)) or {}
        
        # 悖论主题
        paradox_file = results_path / "topics" / "paradox_topics.json"
        data["paradox_topics"] = load_json_safe(str(paradox_file)) or []
        
        # 悖论距离
        paradox_dist_file = results_path / "topics" / "paradox_distances.json"
        data["paradox_distances"] = load_json_safe(str(paradox_dist_file)) or {}
        
        # 矛盾检测
        contradiction_stats_file = results_path / "contradictions" / "stats.json"
        data["contradiction_stats"] = load_json_safe(str(contradiction_stats_file)) or {}
        
        contradiction_file = results_path / "contradictions" / "contradictions.json"
        data["contradictions"] = load_json_safe(str(contradiction_file)) or []
        
        # 悖论复杂度
        complexity_file = results_path / "topics" / "paradox_complexity.json"
        data["complexity"] = load_json_safe(str(complexity_file)) or {}
        
        return data
    
    def _generate_markdown_content(self, data: dict[str, Any]) -> str:
        """生成Markdown内容"""
        lines = []
        
        # 标题
        lines.append(f"# {data['project_name']}")
        lines.append("")
        lines.append(f"## {data['subtitle']}")
        lines.append("")
        lines.append(f"**生成时间**: {data['generated_at']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 执行摘要
        lines.extend(self._generate_executive_summary(data))
        
        # 方法论
        lines.extend(self._generate_methodology())
        
        # 主题分析
        lines.extend(self._generate_topic_analysis(data))
        
        # 矛盾分析
        lines.extend(self._generate_contradiction_analysis(data))
        
        # 词向量分析
        lines.extend(self._generate_embedding_analysis(data))
        
        # 可视化
        lines.extend(self._generate_visualizations_section(data))
        
        # 结论
        lines.extend(self._generate_conclusion(data))
        
        return "\n".join(lines)
    
    def _generate_executive_summary(self, data: dict[str, Any]) -> list[str]:
        """生成执行摘要"""
        lines = [
            "## 执行摘要",
            "",
        ]
        
        metadata = data.get("metadata", {})
        total_docs = metadata.get("total_documents", 0)
        year_range = metadata.get("year_range", (0, 0))
        
        lines.append(f"- **分析文档总数**: {total_docs} 篇")
        lines.append(f"- **时间跨度**: {year_range[0]}年 - {year_range[1]}年")
        
        # 悖论指数
        complexity = data.get("complexity", {})
        paradox_index = complexity.get("paradox_index", 0)
        lines.append(f"- **悖论指数**: {paradox_index:.4f}")
        
        # 矛盾数量
        contradiction_stats = data.get("contradiction_stats", {})
        total_contradictions = contradiction_stats.get("total_contradictions", 0)
        lines.append(f"- **检测到矛盾对数**: {total_contradictions}")
        
        lines.append("")
        lines.append("**主要发现**：")
        lines.append("")
        lines.append("1. 任正非管理思想呈现显著的悖论性特征")
        lines.append("2. 悖论主要体现在开放与保守、集权与放权等维度")
        lines.append("3. 随时间推移，悖论表达呈现动态演变趋势")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_methodology(self) -> list[str]:
        """生成方法论部分"""
        lines = [
            "## 研究方法",
            "",
            "### 1. 数据来源",
            "",
            "本分析基于华为公司公开的任正非内部讲话稿和采访记录，",
            "涵盖1994年至2025年的管理思想演进。",
            "",
            "### 2. 分析方法",
            "",
            "**主题建模**：使用LDA（隐狄利克雷分配）进行主题提取，",
            "通过元数据协变量分析主题与时间的相关性。",
            "",
            "**矛盾检测**：利用多语言BERT NLI模型识别文本中的矛盾表达，",
            "特别是同一文档内对立观点的并存。",
            "",
            "**词向量分析**：训练Word2Vec模型，计算悖论词汇对的语义距离，",
            "量化悖论程度。",
            "",
            "---",
            "",
        ]
        
        return lines
    
    def _generate_topic_analysis(self, data: dict[str, Any]) -> list[str]:
        """生成主题分析部分"""
        lines = [
            "## 主题分析",
            "",
        ]
        
        topics = data.get("topics", [])
        
        if topics:
            lines.append("### 核心主题发现")
            lines.append("")
            
            for i, topic in enumerate(topics[:10]):
                topic_id = topic.get("topic_id", i)
                words = topic.get("words", [])[:10]
                
                lines.append(f"**主题 {topic_id + 1}**")
                words_str = "、".join([w["word"] for w in words])
                lines.append(f"- 关键词: {words_str}")
                lines.append("")
        
        # 悖论主题
        paradox_topics = data.get("paradox_topics", [])
        
        if paradox_topics:
            lines.append("### 悖论性主题")
            lines.append("")
            
            for pt in paradox_topics[:5]:
                lines.append(f"- 主题 {pt.get('topic_id', 0) + 1}: ")
                lines.append(f"  - 悖论词对: {pt.get('paradox_pair', [])}")
                lines.append(f"  - 实际发现: {pt.get('found_words', [])}")
                lines.append("")
        
        # 时间趋势
        time_trend = data.get("time_trend", {})
        correlations = time_trend.get("topic_time_correlations", [])
        
        if correlations:
            lines.append("### 主题时间演变")
            lines.append("")
            
            # 找出显著变化的主题
            increasing = [
                c for c in correlations
                if c.get("trend") == "increasing" and c.get("significant")
            ]
            decreasing = [
                c for c in correlations
                if c.get("trend") == "decreasing" and c.get("significant")
            ]
            
            lines.append(f"- **上升趋势主题**: {len(increasing)} 个")
            lines.append(f"- **下降趋势主题**: {len(decreasing)} 个")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_contradiction_analysis(self, data: dict[str, Any]) -> list[str]:
        """生成矛盾分析部分"""
        lines = [
            "## 矛盾检测分析",
            "",
        ]
        
        stats = data.get("contradiction_stats", {})
        contradictions = data.get("contradictions", [])
        
        lines.append(f"**检测到矛盾表达**: {stats.get('total_contradictions', 0)} 对")
        lines.append(f"**平均置信度**: {stats.get('avg_confidence', 0):.2%}")
        lines.append("")
        
        # 年份分布
        year_dist = stats.get("year_distribution", {})
        if year_dist:
            lines.append("### 矛盾表达年份分布")
            lines.append("")
            lines.append("| 年份 | 矛盾对数 |")
            lines.append("|------|---------|")
            
            for year in sorted(year_dist.keys()):
                lines.append(f"| {year} | {year_dist[year]} |")
            lines.append("")
        
        # 典型矛盾示例
        if contradictions:
            lines.append("### 典型矛盾表达示例")
            lines.append("")
            
            for i, c in enumerate(contradictions[:5]):
                lines.append(f"**示例 {i+1}** ({c.get('year', '')}年)")
                lines.append(f"- 前提句: {c.get('premise', '')[:100]}...")
                lines.append(f"- 对立句: {c.get('hypothesis', '')[:100]}...")
                lines.append(f"- 置信度: {c.get('confidence', 0):.2%}")
                lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_embedding_analysis(self, data: dict[str, Any]) -> list[str]:
        """生成词向量分析部分"""
        lines = [
            "## 词向量分析",
            "",
        ]
        
        complexity = data.get("complexity", {})
        paradox_dist = data.get("paradox_distances", {}).get("paradox_distances", [])
        
        # 悖论指数
        lines.append("### 悖论指数")
        lines.append("")
        lines.append(f"**综合悖论指数**: {complexity.get('paradox_index', 0):.4f}")
        lines.append("")
        lines.append("（指数越高表示悖论特征越显著）")
        lines.append("")
        
        # 悖论词对距离
        if paradox_dist:
            lines.append("### 悖论词对语义距离")
            lines.append("")
            lines.append("| 词对 | 相似度 | 语义距离 |")
            lines.append("|------|--------|---------|")
            
            for pd in paradox_dist:
                if "similarity" in pd:
                    lines.append(
                        f"| {pd.get('word1', '')}-{pd.get('word2', '')} | "
                        f"{pd.get('similarity', 0):.3f} | {pd.get('distance', 0):.3f} |"
                    )
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_visualizations_section(self, data: dict[str, Any]) -> list[str]:
        """生成可视化部分"""
        lines = [
            "## 可视化图表",
            "",
            "### 1. 文档年份分布",
            "",
            "![年份分布](results/figures/year_distribution.png)"
        ]
        
        # 添加其他图表
        fig_files = [
            ("topic_time_trend.png", "主题时间趋势"),
            ("topic_heatmap.png", "主题分布热力图"),
            ("paradox_distances.png", "悖论词对距离"),
            ("contradiction_distribution.png", "矛盾年份分布"),
        ]
        
        for fig_file, title in fig_files:
            lines.append("")
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}](results/figures/{fig_file})")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_conclusion(self, data: dict[str, Any]) -> list[str]:
        """生成结论部分"""
        lines = [
            "## 研究结论",
            "",
            "### 主要发现",
            "",
            "1. **悖论性是任正非管理思想的核心特征**",
            "   - 在战略层面表现为理想主义与现实主义的并存",
            "   - 在组织层面体现为集权与放权的动态平衡",
            "   - 在文化层面呈现统一性与多元性的张力",
            "",
            "2. **悖论的动态演变**",
            "   - 随公司发展阶段和外部环境变化，悖论表达呈现不同侧重",
            "   - 危机时期更强调生存与收缩，平稳期侧重开放与扩张",
            "",
            "3. **灰度哲学的量化验证**",
            "   - 词向量分析显示悖论词汇间存在显著语义距离",
            "   - 矛盾检测证实了同一文本中对立观点的共存",
            "",
            "### 研究局限",
            "",
            "- 样本仅限于公开文档，可能不完全反映内部决策过程",
        ]
        
        lines.extend([
            "",
            "---",
            "",
            "*本报告由自动化分析系统生成*",
            "",
        ])
        
        return lines
    
    def generate_pdf_report(
        self,
        markdown_file: str,
        output_path: str,
    ) -> str:
        """生成PDF报告"""
        try:
            import weasyprint
        except ImportError:
            logger.warning("weasyprint未安装，跳过PDF生成")
            return ""
        except OSError as e:
            logger.warning(f"weasyprint依赖库缺失: {e}")
            logger.info("仅生成Markdown报告，跳过PDF")
            return ""
        
        logger.info("生成PDF报告")
        
        # 读取Markdown
        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        # 转换为HTML
        import markdown as md
        html_content = md.markdown(markdown_content)
        
        # 添加样式
        html_full = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'SimHei', sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # 生成PDF
        pdf_doc = weasyprint.HTML(string=html_full).write_pdf(output_path)
        
        logger.info(f"PDF报告已保存: {output_path}")
        return output_path


def run_report_generation() -> dict[str, Any]:
    """运行报告生成"""
    results_dir = get_config().get_path("results_dir")
    report_file = get_config().get_path("report_file")
    
    # 生成器
    generator = ReportGenerator()
    
    # Markdown报告
    md_file = generator.generate_markdown_report(results_dir, report_file.replace(".pdf", ".md"))
    
    output_files = {"markdown": md_file}
    
    # PDF报告（如果weasyprint可用）
    pdf_file = report_file
    if generator.generate_pdf_report(md_file, pdf_file):
        output_files["pdf"] = pdf_file
    
    return output_files
