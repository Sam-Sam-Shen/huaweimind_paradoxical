"""
矛盾检测模块

使用BERT NLI模型检测文本中的矛盾表达
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from .config import get_config
from .data_loader import DataLoader
from .utils import ensure_dir, save_json_safe, split_sentences


logger = logging.getLogger(__name__)


@dataclass
class SentencePair:
    """句子对"""
    premise: str
    hypothesis: str
    premise_idx: int
    hypothesis_idx: int
    doc_id: str
    doc_title: str
    year: int


@dataclass
class Contradiction:
    """矛盾检测结果"""
    sentence_pair: SentencePair
    label: str  # entailment, contradiction, neutral
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "premise": self.sentence_pair.premise,
            "hypothesis": self.sentence_pair.hypothesis,
            "premise_idx": self.sentence_pair.premise_idx,
            "hypothesis_idx": self.sentence_pair.hypothesis_idx,
            "doc_id": self.sentence_pair.doc_id,
            "doc_title": self.sentence_pair.doc_title,
            "year": self.sentence_pair.year,
            "label": self.label,
            "confidence": self.confidence,
        }


class ContradictionDetector:
    """矛盾检测器"""
    
    def __init__(self) -> None:
        self.config = get_config()
        self._nli_pipeline = None
        self._device = "cpu"  # 无GPU时使用CPU
        # 获取项目根目录并创建models目录
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._model_cache_dir = os.path.join(project_root, "models")
        os.makedirs(self._model_cache_dir, exist_ok=True)
        logger.info(f"模型缓存目录: {self._model_cache_dir}")
    
    @property
    def nli_pipeline(self):
        """获取NLI管道"""
        if self._nli_pipeline is None:
            model_name = self.config.get(
                "contradiction_detection.model_name",
                "Fengshenbang/Erlangshen-RoBERTa-330M-NLI",
            )
            
            logger.info(f"=" * 50)
            logger.info(f"开始加载NLI模型")
            logger.info(f"模型名称: {model_name}")
            logger.info(f"缓存目录: {self._model_cache_dir}")
            logger.info(f"=" * 50)
            
            # 使用ModelScope加载模型
            try:
                import modelscope
                logger.info("使用ModelScope加载模型...")
                
                from modelscope import AutoModelForSequenceClassification, AutoTokenizer
                
                # 使用snapshot_download下载模型到指定目录
                from modelscope.hub.snapshot_download import snapshot_download
                
                logger.info("正在从ModelScope下载模型（首次下载可能需要几分钟）...")
                
                model_dir = snapshot_download(
                    model_name,
                    cache_dir=self._model_cache_dir,
                    revision='v1.0'
                )
                
                logger.info(f"模型下载完成，路径: {model_dir}")
                
                # 列出下载的文件
                import os
                logger.info("下载的文件:")
                for f in os.listdir(model_dir):
                    fpath = os.path.join(model_dir, f)
                    size = os.path.getsize(fpath) / (1024*1024)  # MB
                    logger.info(f"  - {f} ({size:.1f} MB)")
                
                # 从本地加载
                logger.info("加载模型到内存...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )
                
                # 构建pipeline
                from transformers import pipeline
                self._nli_pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU
                    truncation=True,
                    max_length=self.config.get(
                        "contradiction_detection.max_length", 256
                    ),
                )
                
                logger.info("NLI模型加载完成!")
                logger.info(f"模型目录: {model_dir}")
                
            except Exception as e:
                logger.error(f"ModelScope加载失败: {e}")
                logger.info("尝试使用transformers...")
                
                # 回退到transformers
                try:
                    from transformers import pipeline
                    
                    self._nli_pipeline = pipeline(
                        "text-classification",
                        model=model_name,
                        cache_dir=self._model_cache_dir,
                        device=-1,
                        truncation=True,
                        max_length=self.config.get(
                            "contradiction_detection.max_length", 256
                        ),
                    )
                    logger.info("transformers模型加载完成")
                    
                except Exception as e2:
                    logger.error(f"transformers也加载失败: {e2}")
                    raise
        
        return self._nli_pipeline
    
    def extract_sentences(self, text: str) -> list[str]:
        """提取句子"""
        sentences = split_sentences(text)
        # 过滤过短或过长的句子
        sentences = [s for s in sentences if 10 <= len(s) <= 200]
        return sentences
    
    def generate_candidate_pairs(
        self,
        documents: list[dict[str, Any]],
    ) -> list[SentencePair]:
        """生成候选句子对"""
        paradox_pairs = self.config.get(
            "contradiction_detection.paradox_pairs",
            [
                ["开放", "保守"],
                ["激进", "稳健"],
                ["集权", "放权"],
            ],
        )
        
        # 构建正则模式
        patterns = []
        for pair in paradox_pairs:
            if len(pair) >= 2:
                pattern = f".*{pair[0]}.*{pair[1]}.*|.*{pair[1]}.*{pair[0]}.*"
                patterns.append((re.compile(pattern), pair))
        
        candidate_pairs = []
        
        for doc in tqdm(documents, desc="生成候选句对"):
            sentences = self.extract_sentences(doc["content"])
            
            # 找出包含悖论关键词的句子
            paradox_sentences = []
            for idx, sent in enumerate(sentences):
                for pattern, pair in patterns:
                    if pattern.search(sent):
                        paradox_sentences.append((idx, sent, pair))
                        break
            
            # 生成句子对
            for i, (idx1, sent1, _) in enumerate(paradox_sentences):
                for idx2, sent2, _ in paradox_sentences[i+1:]:
                    # 避免重复对
                    if abs(idx1 - idx2) < 3:  # 至少隔3句
                        continue
                    
                    # 随机选择顺序
                    if np.random.random() > 0.5:
                        premise, hypothesis = sent1, sent2
                        premise_idx, hypothesis_idx = idx1, idx2
                    else:
                        premise, hypothesis = sent2, sent1
                        premise_idx, hypothesis_idx = idx2, idx1
                    
                    candidate_pairs.append(SentencePair(
                        premise=premise,
                        hypothesis=hypothesis,
                        premise_idx=premise_idx,
                        hypothesis_idx=hypothesis_idx,
                        doc_id=doc["id"],
                        doc_title=doc["title"],
                        year=doc["year"],
                    ))
        
        logger.info(f"生成 {len(candidate_pairs)} 个候选句子对")
        return candidate_pairs
    
    def detect_contradictions(
        self,
        candidate_pairs: list[SentencePair],
        batch_size: int = 8,
        threshold: float = 0.7,
    ) -> list[Contradiction]:
        """检测矛盾"""
        logger.info(f"开始矛盾检测，阈值: {threshold}")
        
        results = []
        
        # 批量处理
        batch_texts = []
        batch_indices = []
        
        for idx, pair in enumerate(tqdm(candidate_pairs, desc="检测矛盾")):
            # 格式化输入
            text = f"{pair.premise} [SEP] {pair.hypothesis}"
            batch_texts.append(text)
            batch_indices.append(idx)
            
            # 达到批次大小时处理
            if len(batch_texts) >= batch_size:
                batch_results = self._process_batch(batch_texts)
                
                for i, result in enumerate(batch_results):
                    pair_idx = batch_indices[i]
                    if result["label"] == "contradiction" and result["score"] >= threshold:
                        results.append(Contradiction(
                            sentence_pair=candidate_pairs[pair_idx],
                            label=result["label"],
                            confidence=result["score"],
                        ))
                
                batch_texts = []
                batch_indices = []
        
        # 处理剩余的
        if batch_texts:
            batch_results = self._process_batch(batch_texts)
            
            for i, result in enumerate(batch_results):
                pair_idx = batch_indices[i]
                if result["label"] == "contradiction" and result["score"] >= threshold:
                    results.append(Contradiction(
                        sentence_pair=candidate_pairs[pair_idx],
                        label=result["label"],
                        confidence=result["score"],
                    ))
        
        logger.info(f"发现 {len(results)} 对矛盾表达")
        return results
    
    def _process_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """处理一批文本"""
        # 模型输出格式可能是 "contradiction" 或 "CONTRADICTION"
        raw_results = self.nli_pipeline(texts)
        
        results = []
        for res in raw_results:
            # 处理标签格式
            label = res["label"].lower()
            if "contradict" in label:
                label = "contradiction"
            elif "entail" in label:
                label = "entailment"
            else:
                label = "neutral"
            
            results.append({
                "label": label,
                "score": res["score"],
            })
        
        return results
    
    def find_self_contradictions(
        self,
        documents: list[dict[str, Any]],
    ) -> list[Contradiction]:
        """在单个文档内寻找自相矛盾"""
        all_contradictions = []
        
        for doc in tqdm(documents, desc="分析文档矛盾"):
            sentences = self.extract_sentences(doc["content"])
            
            # 提取所有包含悖论词的句子
            paradox_keywords = []
            for pairs in self.config.get("contradiction_detection.paradox_pairs", []):
                paradox_keywords.extend(pairs)
            
            paradox_sents = [
                (i, s)
                for i, s in enumerate(sentences)
                if any(kw in s for kw in paradox_keywords)
            ]
            
            # 成对检测
            for i, (idx1, sent1) in enumerate(paradox_sents):
                for idx2, sent2 in paradox_sents[i+1:]:
                    if abs(idx1 - idx2) < 2:
                        continue
                    
                    try:
                        result = self.nli_pipeline([
                            f"{sent1} [SEP] {sent2}"
                        ])[0]
                        
                        label = result["label"].lower()
                        if "contradict" in label:
                            label = "contradiction"
                        elif "entail" in label:
                            label = "entailment"
                        else:
                            label = "neutral"
                        
                        threshold = self.config.get(
                            "contradiction_detection.confidence_threshold", 0.7
                        )
                        
                        if label == "contradiction" and result["score"] >= threshold:
                            all_contradictions.append(Contradiction(
                                sentence_pair=SentencePair(
                                    premise=sent1,
                                    hypothesis=sent2,
                                    premise_idx=idx1,
                                    hypothesis_idx=idx2,
                                    doc_id=doc["id"],
                                    doc_title=doc["title"],
                                    year=doc["year"],
                                ),
                                label=label,
                                confidence=result["score"],
                            ))
                    except Exception as e:
                        logger.warning(f"处理句子对失败: {e}")
                        continue
        
        return all_contradictions
    
    def save_results(
        self,
        contradictions: list[Contradiction],
        output_dir: str,
    ) -> None:
        """保存结果"""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        # 保存所有矛盾
        results = [c.to_dict() for c in contradictions]
        save_json_safe(results, output_path / "contradictions.json")
        
        # 按年份统计
        year_stats: dict[int, int] = {}
        for c in contradictions:
            year_stats[c.sentence_pair.year] = year_stats.get(
                c.sentence_pair.year, 0
            ) + 1
        
        stats = {
            "total_contradictions": len(contradictions),
            "year_distribution": year_stats,
            "avg_confidence": np.mean([c.confidence for c in contradictions])
            if contradictions else 0,
        }
        
        save_json_safe(stats, output_path / "stats.json")
        
        logger.info(f"矛盾检测结果已保存到: {output_path}")


def run_contradiction_detection() -> dict[str, Any]:
    """运行矛盾检测"""
    # 加载数据
    loader = DataLoader()
    corpus = loader.load_all_documents()
    
    # 准备文档数据
    documents = [
        {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "year": doc.year,
        }
        for doc in corpus.documents
    ]
    
    # 初始化检测器
    detector = ContradictionDetector()
    
    # 生成候选对并检测
    # 注意：由于无GPU，这里简化处理
    # 实际上可以先用规则筛选候选对，再小批量检测
    
    logger.info("开始矛盾检测流程...")
    
    # 为演示，这里生成少量候选对
    sample_docs = documents[:10]  # 限制样本
    
    candidate_pairs = detector.generate_candidate_pairs(sample_docs)
    
    # 如果候选对太多，随机采样
    max_pairs = 100
    if len(candidate_pairs) > max_pairs:
        import random
        random.seed(42)
        candidate_pairs = random.sample(candidate_pairs, max_pairs)
    
    # 检测矛盾
    contradictions = detector.detect_contradictions(candidate_pairs)
    
    # 保存结果
    output_dir = get_config().get_path("contradictions_dir")
    detector.save_results(contradictions, output_dir)
    
    return {
        "total_candidates": len(candidate_pairs),
        "contradictions_found": len(contradictions),
        "output_dir": output_dir,
    }
