"""
数据加载模块

加载和处理任正非讲话文档
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_config
from .utils import (
    ensure_dir,
    extract_date_from_filename,
    extract_year_from_filename,
    get_project_root,
    load_json_safe,
    save_json_safe,
)


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据类"""
    id: str
    title: str
    content: str
    year: int
    date: str | None = None
    filepath: str = ""
    doc_type: str = "讲话"  # 讲话、采访、文章等
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "year": self.year,
            "date": self.date,
            "filepath": self.filepath,
            "doc_type": self.doc_type,
        }


@dataclass
class Corpus:
    """语料库数据类"""
    documents: list[Document] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "total_documents": len(self.documents),
                "year_range": self.get_year_range(),
            },
            "documents": [doc.to_dict() for doc in self.documents],
        }
    
    def get_year_range(self) -> tuple[int, int]:
        """获取年份范围"""
        if not self.documents:
            return (0, 0)
        years = [doc.year for doc in self.documents]
        return (min(years), max(years))
    
    def get_by_year(self, year: int) -> list[Document]:
        """按年份获取文档"""
        return [doc for doc in self.documents if doc.year == year]
    
    def filter_by_year_range(self, start: int, end: int) -> list[Document]:
        """按年份范围筛选"""
        return [doc for doc in self.documents if start <= doc.year <= end]


class DataLoader:
    """数据加载器"""
    
    def __init__(self) -> None:
        self.config = get_config()
        self._source_dir: Path | None = None
    
    @property
    def source_dir(self) -> Path:
        """获取源数据目录"""
        if self._source_dir is None:
            source_path = self.config.get("paths.source_dir", "huawei")
            self._source_dir = get_project_root() / source_path
        assert self._source_dir is not None
        return self._source_dir
    
    def load_all_documents(self, force_reload: bool = False) -> Corpus:
        """加载所有文档"""
        corpus_file = self.config.get_path("corpus_file")
        
        # 尝试从缓存加载
        if not force_reload:
            cached = load_json_safe(corpus_file)
            if cached and isinstance(cached, dict):
                logger.info(f"从缓存加载语料库: {len(cached.get('documents', []))} 篇文档")
                return self._load_from_cache(cached)
        
        # 扫描并加载文档
        logger.info(f"扫描文档目录: {self.source_dir}")
        documents = []
        
        for md_file in self.source_dir.rglob("*.md"):
            # 跳过README等非讲话文件
            if md_file.name.lower() in ["readme.md", "readme.txt"]:
                continue
            
            doc = self._load_document(md_file)
            if doc:
                documents.append(doc)
        
        # 按年份排序
        documents.sort(key=lambda d: (d.year, d.title))
        
        # 分配ID
        for i, doc in enumerate(documents):
            doc.id = f"doc_{i+1:04d}"
        
        corpus = Corpus(documents=documents)
        
        # 保存到缓存
        save_json_safe(corpus.to_dict(), corpus_file)
        logger.info(f"加载完成: {len(documents)} 篇文档")
        
        return corpus
    
    def _load_document(self, filepath: Path) -> Document | None:
        """加载单个文档"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content or len(content) < 100:
                logger.warning(f"文档内容过短: {filepath}")
                return None
            
            # 提取标题（从文件名或第一行）
            title = filepath.stem
            
            # 从文件名提取年份
            year = extract_year_from_filename(filepath.name)
            if year is None:
                # 从文件内容第一行提取
                year = self._extract_year_from_content(content)
            
            if year is None:
                # 从目录名提取
                year = self._extract_year_from_path(filepath)
            
            if year is None:
                logger.warning(f"无法确定年份: {filepath}")
                year = 2000  # 默认值
            
            # 提取日期
            date = extract_date_from_filename(filepath.name)
            
            # 确定文档类型
            doc_type = self._infer_doc_type(filepath, content)
            
            return Document(
                id="",
                title=title,
                content=content,
                year=year,
                date=date,
                filepath=str(filepath),
                doc_type=doc_type,
            )
            
        except Exception as e:
            logger.error(f"加载文档失败 {filepath}: {e}")
            return None
    
    def _extract_year_from_content(self, content: str) -> int | None:
        """从内容中提取年份"""
        # 匹配4位数年份
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", content)
        if matches:
            # 返回最小的年份（通常更准确）
            years = [int(m) for m in matches]
            return min(years)
        return None
    
    def _extract_year_from_path(self, filepath: Path) -> int | None:
        """从文件路径提取年份"""
        for part in filepath.parts:
            if part.isdigit() and len(part) == 4:
                year = int(part)
                if 1990 <= year <= 2030:
                    return year
        return None
    
    def _infer_doc_type(self, filepath: Path, content: str) -> str:
        """推断文档类型"""
        path_str = str(filepath).lower()
        content_lower = content[:500].lower()
        
        # 根据关键词判断
        if any(kw in path_str for kw in ["采访", "采访纪要", "答记者问", "记者问"]):
            return "采访"
        elif any(kw in path_str for kw in ["咖啡对话", "座谈", "讲话", "发言"]):
            return "讲话"
        elif any(kw in content_lower for kw in ["答记者问", "记者问", "采访时"]):
            return "采访"
        elif "决议" in path_str:
            return "决议"
        elif "通知" in path_str:
            return "通知"
        else:
            return "讲话"
    
    def _load_from_cache(self, data: dict[str, Any]) -> Corpus:
        """从缓存加载"""
        docs = []
        for doc_data in data.get("documents", []):
            docs.append(Document(**doc_data))
        return Corpus(documents=docs)
    
    def get_metadata(self) -> dict[str, Any]:
        """获取语料库元数据"""
        corpus = self.load_all_documents()
        
        # 统计年份分布
        year_counts: dict[int, int] = {}
        type_counts: dict[str, int] = {}
        
        for doc in corpus.documents:
            year_counts[doc.year] = year_counts.get(doc.year, 0) + 1
            type_counts[doc.doc_type] = type_counts.get(doc.doc_type, 0) + 1
        
        return {
            "total_documents": len(corpus.documents),
            "year_range": corpus.get_year_range(),
            "year_distribution": dict(sorted(year_counts.items())),
            "type_distribution": type_counts,
        }
