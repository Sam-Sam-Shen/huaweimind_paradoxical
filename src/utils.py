"""
工具函数模块
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

from .config import get_config


def setup_logging(name: str = __name__) -> logging.Logger:
    """设置日志记录器"""
    config = get_config()
    log_config = config.logging
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_config.get("format"))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # 添加文件处理器
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "analysis.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def extract_year_from_filename(filename: str) -> int | None:
    """从文件名提取年份"""
    # 匹配YYYYMMDD格式
    match = re.match(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        return int(match.group(1))
    
    # 匹配年份目录
    year_match = re.search(r"/?(\d{4})/", filename)
    if year_match:
        return int(year_match.group(1))
    
    return None


def extract_date_from_filename(filename: str) -> str | None:
    """从文件名提取日期"""
    match = re.match(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def clean_text(text: str) -> str:
    """基础文本清理"""
    # 移除多余空白
    text = re.sub(r"\s+", " ", text)
    # 移除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s,，。！？、；：""''（）【】《》]", "", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """简单的中文分句"""
    # 按常见句末标点分句
    sentences = re.split(r"[。！？；\n]+", text)
    # 过滤空句子和过短的句子
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def load_json_safe(filepath: str | Path) -> dict | list | None:
    """安全加载JSON文件"""
    import json
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_json_safe(data: dict | list, filepath: str | Path, indent: int = 2) -> bool:
    """安全保存JSON文件"""
    import json
    try:
        ensure_dir(Path(filepath).parent)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception:
        return False


def save_json(data: dict | list, filepath: str | Path, indent: int = 2) -> None:
    """保存JSON文件（不安全版本，出错时抛出异常）"""
    import json
    ensure_dir(Path(filepath).parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json_safe(filepath: str | Path, default: Any = None) -> Any:
    """安全加载JSON文件，失败时返回默认值"""
    import json
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def timer(func):
    """装饰器：计算函数执行时间"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} 耗时: {elapsed:.2f}秒")
        return result
    return wrapper


def format_size(size_bytes: float) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


class Timer:
    """简单计时器"""
    
    def __init__(self) -> None:
        import time
        self._start = time.time()
    
    def elapsed(self) -> float:
        import time
        return time.time() - self._start
    
    def __enter__(self) -> "Timer":
        return self
    
    def __exit__(self, *args: Any) -> None:
        import time
        elapsed = time.time() - self._start
        logger = logging.getLogger(__name__)
        logger.info(f"耗时: {elapsed:.2f}秒")
