"""
ArXiv 文献检索工具
==================
对接 ArXiv API，实现关键词检索与元数据提取。
"""

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    """论文元数据"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    categories: list[str]
    url: str
    pdf_url: str
    citation_count: int = 0
    relevance_score: float = 0.0
    structured_summary: Optional[dict] = None  # {method, conclusion, limitation}

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract[:500],
            "published": self.published,
            "categories": self.categories,
            "url": self.url,
            "relevance_score": self.relevance_score,
            "structured_summary": self.structured_summary,
        }

    def short_repr(self) -> str:
        authors_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            authors_str += f" et al."
        return f"[{self.arxiv_id}] {self.title} — {authors_str} ({self.published[:4]})"


# ── 中文 → 英文关键词映射（ArXiv API 仅支持英文检索）─────────
_CN_TO_EN: dict[str, str] = {
    # 模型架构
    "transformer": "transformer",
    "transformer架构": "transformer architecture",
    "注意力机制": "attention mechanism",
    "自注意力": "self-attention",
    "多头注意力": "multi-head attention",
    "位置编码": "positional encoding",
    "前馈网络": "feed-forward network",
    # 大语言模型
    "大语言模型": "large language model",
    "大模型": "large language model",
    "预训练": "pre-training",
    "微调": "fine-tuning",
    "提示学习": "prompt learning",
    "上下文学习": "in-context learning",
    "思维链": "chain of thought",
    "强化学习": "reinforcement learning",
    "人类反馈": "reinforcement learning from human feedback",
    # 视觉
    "视觉": "vision",
    "图像分类": "image classification",
    "目标检测": "object detection",
    "图像分割": "image segmentation",
    "卷积神经网络": "convolutional neural network",
    "视觉transformer": "vision transformer",
    # 训练方法
    "对比学习": "contrastive learning",
    "自监督学习": "self-supervised learning",
    "半监督学习": "semi-supervised learning",
    "迁移学习": "transfer learning",
    "元学习": "meta-learning",
    "联邦学习": "federated learning",
    "知识蒸馏": "knowledge distillation",
    # 优化
    "梯度下降": "gradient descent",
    "反向传播": "backpropagation",
    "批归一化": "batch normalization",
    "dropout": "dropout",
    "过拟合": "overfitting",
    # 图神经网络
    "图神经网络": "graph neural network",
    "图卷积网络": "graph convolutional network",
    "知识图谱": "knowledge graph",
    # 生成模型
    "生成对抗网络": "generative adversarial network",
    "变分自编码器": "variational autoencoder",
    "扩散模型": "diffusion model",
    # NLP
    "自然语言处理": "natural language processing",
    "文本分类": "text classification",
    "情感分析": "sentiment analysis",
    "机器翻译": "machine translation",
    "命名实体识别": "named entity recognition",
    "问答系统": "question answering",
    "信息检索": "information retrieval",
    # 推荐系统
    "推荐系统": "recommendation system",
    "协同过滤": "collaborative filtering",
    # Agent
    "智能体": "agent",
    "多智能体": "multi-agent",
    "强化学习": "reinforcement learning",
    # 数据
    "数据增强": "data augmentation",
    "数据集": "dataset",
    "基准测试": "benchmark",
    # 通用
    "深度学习": "deep learning",
    "机器学习": "machine learning",
    "人工智能": "artificial intelligence",
    "神经网络": "neural network",
    "循环神经网络": "recurrent neural network",
    "长短时记忆": "long short-term memory",
    "架构": "architecture",
    "模型": "model",
    "训练": "training",
    "推理": "inference",
    "效率": "efficiency",
    "性能": "performance",
    "评估": "evaluation",
    "实验": "experiment",
    "分类": "classification",
    "回归": "regression",
    "聚类": "clustering",
}


def _contains_chinese(text: str) -> bool:
    """判断字符串中是否包含中文字符"""
    return any('一' <= c <= '鿿' for c in text)


def _translate_query(query: str) -> tuple[str, bool]:
    """
    将查询词转换为英文（ArXiv 仅支持英文检索）。
    返回 (英文查询词, 是否发生了翻译)
    """
    if not _contains_chinese(query):
        return query, False

    q = query.strip()
    # 先尝试完整短语匹配
    if q in _CN_TO_EN:
        return _CN_TO_EN[q], True

    # 逐词/短语替换（按词长降序，优先替换长短语）
    result = q
    sorted_terms = sorted(_CN_TO_EN.keys(), key=len, reverse=True)
    for cn_term in sorted_terms:
        if cn_term in result:
            result = result.replace(cn_term, ' ' + _CN_TO_EN[cn_term] + ' ')

    # 去除剩余无法翻译的中文字符
    result = re.sub(r'[一-鿿]+', ' ', result)
    # 清理多余空格
    result = re.sub(r'\s+', ' ', result).strip()

    if not result:
        english_parts = re.sub(r'[一-鿿]+', ' ', q).strip()
        result = english_parts if english_parts else "deep learning"

    return result, True


class ArXivSearchTool:
    """
    ArXiv 文献检索工具
    使用官方 Atom API（无需 Key）
    自动将中文查询词翻译为英文
    """

    BASE_URL = "http://export.arxiv.org/api/query"
    NS = {"atom": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, max_results: int = 10, logger=None):
        self.max_results = max_results
        self.logger = logger

    def search(self, query: str, max_results: Optional[int] = None,
               categories: Optional[list[str]] = None) -> list[Paper]:
        """
        搜索 ArXiv 论文（自动将中文查询翻译为英文）
        Args:
            query: 搜索关键词，支持中文或英文
            max_results: 最大返回数量
            categories: 限定分类，如 ["cs.AI", "cs.LG"]
        Returns:
            论文列表，按相关性排序
        """
        n = max_results or self.max_results

        # 中文 → 英文转换（ArXiv API 仅支持英文检索）
        en_query, translated = _translate_query(query)
        if translated and self.logger:
            self.logger.info("ArXivTool", f"中文查询已转换为英文: 「{query}」→「{en_query}」")

        search_q = self._build_query(en_query, categories)

        params = urllib.parse.urlencode({
            "search_query": search_q,
            "start": 0,
            "max_results": n,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        url = f"{self.BASE_URL}?{params}"

        if self.logger:
            self.logger.info("ArXivTool", f"检索: {en_query[:60]}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MindPilot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read().decode("utf-8")
        except Exception as e:
            if self.logger:
                self.logger.warning("ArXivTool", f"网络请求失败，返回 Mock 数据: {e}")
            return self._mock_papers(query, n)

        papers = self._parse_xml(xml_data, query)
        if self.logger:
            self.logger.success("ArXivTool", f"找到 {len(papers)} 篇论文")
        return papers

    def _build_query(self, query: str, categories: Optional[list[str]]) -> str:
        q = f"all:{urllib.parse.quote(query)}"
        if categories:
            cat_q = " OR ".join(f"cat:{c}" for c in categories)
            q = f"({q}) AND ({cat_q})"
        return q

    def _parse_xml(self, xml_data: str, query: str) -> list[Paper]:
        root = ET.fromstring(xml_data)
        papers = []
        query_words = set(query.lower().split())

        for entry in root.findall("atom:entry", self.NS):
            try:
                arxiv_id = entry.find("atom:id", self.NS).text.split("/abs/")[-1]
                title = entry.find("atom:title", self.NS).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", self.NS).text.strip().replace("\n", " ")
                published = entry.find("atom:published", self.NS).text[:10]
                authors = [
                    a.find("atom:name", self.NS).text
                    for a in entry.findall("atom:author", self.NS)
                ]
                categories = [
                    c.attrib.get("term", "")
                    for c in entry.findall("arxiv:category", self.NS)
                ]
                url = f"https://arxiv.org/abs/{arxiv_id}"
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

                # 计算简单相关性分数
                text = (title + " " + abstract).lower()
                score = sum(1 for w in query_words if w in text) / max(len(query_words), 1)

                papers.append(Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    categories=categories,
                    url=url,
                    pdf_url=pdf_url,
                    relevance_score=round(score, 3),
                ))
            except Exception:
                continue

        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    def _mock_papers(self, query: str, n: int) -> list[Paper]:
        """网络不可用时的 Mock 论文数据"""
        topics = query.split()[:3]
        mock = []
        templates = [
            ("A Comprehensive Survey on {}", ["John Smith", "Alice Wang"], "cs.AI"),
            ("Efficient {} with Transformer Architecture", ["Bob Chen", "Carol Lee"], "cs.LG"),
            ("{}: A Novel Approach via Reinforcement Learning", ["David Kim", "Eve Zhang"], "cs.CL"),
            ("Scaling {} to Large Language Models", ["Frank Liu", "Grace Zhao"], "cs.CV"),
            ("Benchmark Study of {} Methods", ["Henry Wu", "Iris Ma"], "stat.ML"),
        ]
        for i, (tmpl, authors, cat) in enumerate(templates[:n]):
            topic = " ".join(topics)
            mock.append(Paper(
                arxiv_id=f"2024.{1000+i:05d}",
                title=tmpl.format(topic.title()),
                authors=authors,
                abstract=f"In this paper, we study {topic} and propose a novel framework. "
                         f"Experiments on standard benchmarks demonstrate significant improvements.",
                published=f"2024-0{i+1}-15",
                categories=[cat],
                url=f"https://arxiv.org/abs/2024.{1000+i:05d}",
                pdf_url=f"https://arxiv.org/pdf/2024.{1000+i:05d}",
                relevance_score=round(0.9 - i * 0.1, 2),
            ))
        return mock

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """按 ID 获取单篇论文"""
        results = self.search(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None
