"""
模块② — 文献检索与知识图谱 Agent
===================================
混合检索（关键词 + 语义向量）+ 知识图谱构建 + 结构化摘要生成。
"""

import json
import re
import math
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    node_id: str
    node_type: str          # paper | author | method | dataset | concept
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """知识图谱边"""
    source: str
    target: str
    relation: str           # cites | uses_method | applied_to | authored_by | related_to
    weight: float = 1.0


class LightKnowledgeGraph:
    """
    轻量知识图谱（基于 NetworkX，无需 Neo4j）
    生产环境可替换为 Neo4j 后端。
    """

    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[KnowledgeEdge] = []
        self._adj: dict[str, list[str]] = {}   # 邻接表（用于多跳推理）

    def add_node(self, node: KnowledgeNode):
        self.nodes[node.node_id] = node
        self._adj.setdefault(node.node_id, [])

    def add_edge(self, edge: KnowledgeEdge):
        self.edges.append(edge)
        self._adj.setdefault(edge.source, []).append(edge.target)
        self._adj.setdefault(edge.target, []).append(edge.source)

    def add_paper(self, paper) -> str:
        """从 Paper 对象构建图谱节点与关系"""
        pid = f"paper:{paper.arxiv_id}"
        self.add_node(KnowledgeNode(
            node_id=pid, node_type="paper",
            label=paper.title[:60],
            properties={"year": paper.published[:4], "url": paper.url,
                        "relevance": paper.relevance_score}
        ))
        # 作者节点
        for author in paper.authors[:3]:
            aid = f"author:{author.replace(' ', '_')}"
            self.add_node(KnowledgeNode(node_id=aid, node_type="author", label=author))
            self.add_edge(KnowledgeEdge(pid, aid, "authored_by"))
        # 类别节点
        for cat in paper.categories[:2]:
            cid = f"cat:{cat}"
            self.add_node(KnowledgeNode(node_id=cid, node_type="category", label=cat))
            self.add_edge(KnowledgeEdge(pid, cid, "belongs_to"))
        return pid

    def multi_hop_query(self, start_label: str, hops: int = 2) -> list[KnowledgeNode]:
        """从起点出发进行 N 跳检索"""
        # 找起点节点
        start_nodes = [
            nid for nid, n in self.nodes.items()
            if start_label.lower() in n.label.lower()
        ]
        if not start_nodes:
            return []
        visited = set(start_nodes)
        frontier = set(start_nodes)
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                for neighbor in self._adj.get(nid, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
        return [self.nodes[nid] for nid in visited if nid in self.nodes]

    def stats(self) -> dict:
        type_counts = {}
        for n in self.nodes.values():
            type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "types": type_counts}


class StructuredSummarizer:
    """
    文献结构化摘要生成器
    将原始摘要压缩为：方法 / 结论 / 局限性 三段式
    """

    def __init__(self, llm_client, max_len: int = 300, logger=None):
        self.llm = llm_client
        self.max_len = max_len
        self.logger = logger

    def summarize(self, paper) -> dict:
        """生成结构化摘要"""
        system = (
            "你是学术论文分析专家。请将以下论文摘要压缩为结构化摘要。"
            "以 JSON 格式输出（字段：method, conclusion, limitation），每项不超过60字。"
        )
        text = f"标题：{paper.title}\n摘要：{paper.abstract[:800]}"
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ])
        try:
            m = re.search(r"\{[\s\S]+\}", resp)
            summary = json.loads(m.group(0) if m else resp)
            return {
                "method": summary.get("method", "未提取到"),
                "conclusion": summary.get("conclusion", "未提取到"),
                "limitation": summary.get("limitation", "未提取到"),
            }
        except Exception:
            # 简单 fallback：截取摘要前三句
            sents = paper.abstract.split(". ")
            return {
                "method": sents[0][:100] if len(sents) > 0 else "",
                "conclusion": sents[1][:100] if len(sents) > 1 else "",
                "limitation": sents[-1][:100] if len(sents) > 2 else "",
            }


class LiteratureAgent:
    """
    模块② — 文献检索与知识图谱 Agent
    混合检索 + 图谱构建 + 结构化摘要
    """

    AGENT_NAME = "LiteratureAgent"

    def __init__(self, config, llm_client, arxiv_tool, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.arxiv = arxiv_tool
        self.memory = memory_store
        self.logger = logger
        self.summarizer = StructuredSummarizer(llm_client, config.literature.summary_max_len, logger)
        self.kg = LightKnowledgeGraph()

    def run(self, task_description: str, query: str = "") -> dict:
        """
        主入口
        Returns:
            {papers, knowledge_graph_stats, literature_review, recall_at_k}
        """
        search_query = query or task_description
        call = self.logger.start_call(self.AGENT_NAME, "literature_search", search_query)

        try:
            # Step 1: ArXiv 检索
            self.logger.info(self.AGENT_NAME, "开始 ArXiv 混合检索...")
            papers = self.arxiv.search(
                search_query,
                max_results=self.config.literature.arxiv_max_results
            )

            # Step 2: 语义重排序（TF-IDF 近似）
            papers = self._rerank(papers, search_query)

            # Step 3: 生成结构化摘要 + 构建知识图谱
            self.logger.info(self.AGENT_NAME, f"为 {len(papers)} 篇论文生成摘要...")
            for paper in papers:
                paper.structured_summary = self.summarizer.summarize(paper)
                self.kg.add_paper(paper)

            # Step 4: 计算 Recall@K 近似值（基于相关性分数）
            recall_5 = self._compute_recall_at_k(papers, k=5)
            recall_10 = self._compute_recall_at_k(papers, k=10)

            # Step 5: 生成文献综述段落
            review = self._generate_review(search_query, papers[:5])

            # Step 6: 存入记忆
            self.memory.add(
                content=f"文献检索: {search_query}，找到 {len(papers)} 篇",
                agent=self.AGENT_NAME,
                payload={"papers": [p.to_dict() for p in papers[:5]]},
                tags=["literature"],
            )

            result = {
                "papers": [p.to_dict() for p in papers],
                "top_papers": [p.to_dict() for p in papers[:self.config.literature.retrieval_top_k]],
                "knowledge_graph": self.kg.stats(),
                "literature_review": review,
                "metrics": {"recall@5": recall_5, "recall@10": recall_10},
                "total_found": len(papers),
            }
            self.logger.finish_call(call, result)
            self._print_results(papers[:5])
            return result

        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    def _rerank(self, papers, query: str):
        """TF-IDF 近似重排序"""
        query_words = set(query.lower().split())
        for p in papers:
            text = (p.title + " " + p.abstract).lower()
            words = text.split()
            total = len(words)
            if total == 0:
                continue
            tf_score = sum(text.count(w) / total for w in query_words)
            p.relevance_score = round(
                0.6 * p.relevance_score + 0.4 * min(tf_score * 10, 1.0), 3
            )
        return sorted(papers, key=lambda p: p.relevance_score, reverse=True)

    def _compute_recall_at_k(self, papers, k: int) -> float:
        """基于相关性分数近似计算 Recall@K"""
        top_k = papers[:k]
        relevant = sum(1 for p in top_k if p.relevance_score > 0.3)
        total_relevant = sum(1 for p in papers if p.relevance_score > 0.3)
        if total_relevant == 0:
            return 0.0
        return round(relevant / total_relevant, 3)

    def _generate_review(self, query: str, papers: list) -> str:
        """用 LLM 生成文献综述段落"""
        if not papers:
            return "未找到相关文献。"
        paper_summaries = "\n".join([
            f"[{i+1}] {p.title}\n   方法：{p.structured_summary.get('method','') if p.structured_summary else ''}"
            for i, p in enumerate(papers)
        ])
        system = "你是学术写作专家。请根据以下论文列表，写一段200字以内的中文文献综述。"
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"研究主题：{query}\n\n相关论文：\n{paper_summaries}"}
        ])
        return resp[:600]

    def _print_results(self, papers: list):
        print(f"\n{'━'*58}")
        print(f"  📚 文献检索结果 (Top {len(papers)})")
        print(f"{'━'*58}")
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
            print(f"  [{i}] {p.title[:52]}")
            print(f"       {authors} ({p.published[:4]}) | 相关性: {p.relevance_score:.2f}")
        print(f"  知识图谱: {self.kg.stats()}")
        print(f"{'━'*58}\n")
