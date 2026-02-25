"""
主程序入口

任正非管理思想悖论分析 - 主程序
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import get_config
from src.utils import Timer, setup_logging


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="任正非管理思想悖论性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --all                    # 运行完整分析流程
  python main.py --preprocess             # 仅预处理
  python main.py --topics                 # 仅主题建模
  python main.py --contradict             # 仅矛盾检测
  python main.py --visualize              # 仅可视化
  python main.py --report                 # 仅生成报告
        """,
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行完整分析流程",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="仅运行预处理",
    )
    parser.add_argument(
        "--topics",
        action="store_true",
        help="仅运行主题建模",
    )
    parser.add_argument(
        "--contradict",
        action="store_true",
        help="仅运行矛盾检测",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="仅运行词向量分析",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="仅运行可视化",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="仅生成报告",
    )
    parser.add_argument(
        "--innovation",
        action="store_true",
        help="运行方法论创新分析（BERTopic, 语义网络, PII, DTM）",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=None,
        help="主题建模的主题数量",
    )
    parser.add_argument(
        "--method",
        choices=["lda", "nmf"],
        default="lda",
        help="主题建模方法",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    return parser.parse_args()


def run_preprocessing() -> dict:
    """运行预处理"""
    from src.preprocess import run_preprocessing
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("开始预处理")
    logger.info("=" * 50)
    
    with Timer() as timer:
        result = run_preprocessing()
    
    logger.info(f"预处理完成，耗时: {timer.elapsed():.2f}秒")
    return result


def run_topic_modeling(method: str = "lda", n_topics: int | None = None) -> dict:
    """运行主题建模"""
    from src.stm_analysis import run_topic_modeling
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info(f"开始主题建模 (方法: {method})")
    logger.info("=" * 50)
    
    # 获取配置中的默认值
    config = get_config()
    if n_topics is None:
        n_topics = config.get("topic_modeling.num_topics", 25)
    
    with Timer() as timer:
        result = run_topic_modeling(method=method, n_topics=n_topics)
    
    logger.info(f"主题建模完成，耗时: {timer.elapsed():.2f}秒")
    return result


def run_contradiction_detection() -> dict:
    """运行矛盾检测"""
    from src.contradiction_detect import run_contradiction_detection
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("开始矛盾检测")
    logger.info("=" * 50)
    
    with Timer() as timer:
        result = run_contradiction_detection()
    
    logger.info(f"矛盾检测完成，耗时: {timer.elapsed():.2f}秒")
    return result


def run_word_embedding() -> dict:
    """运行词向量分析"""
    from src.word_embedding import run_word_embedding
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("开始词向量分析")
    logger.info("=" * 50)
    
    with Timer() as timer:
        result = run_word_embedding()
    
    logger.info(f"词向量分析完成，耗时: {timer.elapsed():.2f}秒")
    return result


def run_visualization() -> dict:
    """运行可视化"""
    from src.visualize import run_visualization
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("开始可视化")
    logger.info("=" * 50)
    
    with Timer() as timer:
        result = run_visualization()
    
    logger.info(f"可视化完成，耗时: {timer.elapsed():.2f}秒")
    return result


def run_report() -> dict:
    """运行报告生成"""
    from src.report_generator import run_report_generation
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("开始生成报告")
    logger.info("=" * 50)
    
    with Timer() as timer:
        result = run_report_generation()
    
    logger.info(f"报告生成完成，耗时: {timer.elapsed():.2f}秒")
    return result


def main() -> int:
    """主函数"""
    args = parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging()
    logging.getLogger().setLevel(log_level)
    
    logger = logging.getLogger(__name__)
    
    # 检查是否指定了任何选项
    run_all = args.all or not any([
        args.preprocess,
        args.topics,
        args.contradict,
        args.embedding,
        args.visualize,
        args.report,
        args.innovation,
    ])
    
    if run_all:
        logger.info("运行完整分析流程")
    
    results = {}
    
    try:
        # 预处理
        if args.preprocess or run_all:
            results["preprocess"] = run_preprocessing()
        
        # 主题建模
        if args.topics or run_all:
            results["topics"] = run_topic_modeling(
                method=args.method,
                n_topics=args.n_topics,
            )
        
        # 矛盾检测
        if args.contradict or run_all:
            results["contradiction"] = run_contradiction_detection()
        
        # 词向量分析
        if args.embedding or run_all:
            results["embedding"] = run_word_embedding()
        
        # 可视化
        if args.visualize or run_all:
            results["visualize"] = run_visualization()
        
        # 报告生成
        if args.report or run_all:
            results["report"] = run_report()
        
        # 方法论创新分析
        if args.innovation:
            from src.integrated_analysis import run_integrated_analysis
            logger.info("=" * 50)
            logger.info("运行方法论创新分析")
            logger.info("=" * 50)
            results["innovation"] = run_integrated_analysis()
        
        logger.info("=" * 50)
        logger.info("所有任务完成!")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
