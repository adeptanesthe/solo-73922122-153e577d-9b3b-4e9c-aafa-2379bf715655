"""搜索核心：BM25 全文检索、两套排序、交错（interleaving）合并。

两套排序（候选集合完全相同，仅顺序不同）：
  * full  —— 标准全文 BM25，标题列与正文列字段权重均为 1
  * title —— 标题加权 BM25，标题列字段权重为正文列的 10 倍

为什么不用 FTS5 内置 bm25()：演示语料只有十几篇文档，常见词会命中半数以上
文档，FTS5 把负 IDF 钳制为 0，叠加长度归一化后分数几乎全为 0，无法区分排序。
这里按经典 BM25 公式自行实现（Robertson-Spärck Jones IDF + 平滑），
字段命中通过字段权重进入打分，行为确定、可解释。

中文处理：FTS5 自带 unicode61 分词器会把连续中文切成一个整词，无法按单字
检索。因此在建索引与构造查询前，先把中文按字切开，英文/数字整体保留为词条。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# 排序方案标识。前端与数据库统一使用英文标识，页面文案才显示中文名。
SIDE_FULL = "full"
SIDE_TITLE = "title"
SIDES = (SIDE_FULL, SIDE_TITLE)
SIDE_LABELS = {
    SIDE_FULL: "全文 BM25",
    SIDE_TITLE: "标题加权 BM25",
}

# 标题列相对正文列的权重倍数。
TITLE_WEIGHT = 10.0
BODY_WEIGHT = 1.0

# BM25 经典参数。
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """把文本切成词条：中文逐字成词，英文/数字连续成词，统一小写。"""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum() and ord(ch) < 128:
            buf.append(ch.lower())
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
            if not ch.isspace():
                # 中文字符（以及其它非 ASCII 字符）单独成词。
                tokens.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def fts_index_text(text: str) -> str:
    """写入 FTS 索引前的规范化文本（词条以空格连接）。"""
    return " ".join(tokenize(text))


def build_match_query(query: str) -> str:
    """把用户查询转为词条列表的 OR 表达式（FTS MATCH 用）。"""
    tokens = tokenize(query)
    if not tokens:
        return ""
    safe = [t.replace('"', '""') for t in tokens]
    return " OR ".join(f'"{t}"' for t in safe)


# --------------------------------------------------------------------------- #
# 纯 Python BM25
# --------------------------------------------------------------------------- #
@dataclass
class RankedDoc:
    doc_id: int
    score: float


class BM25Index:
    """内存中的 BM25 索引：按字段分别统计词频与文档频率。

    文档规模为帮助中心级别（数百至数千篇），全量驻留内存成本很低，
    且让打分逻辑完全可控。
    """

    def __init__(self) -> None:
        self.doc_ids: list[int] = []
        # field -> doc_id -> term -> tf
        self._tf: dict[str, dict[int, dict[str, int]]] = {"title": {}, "body": {}}
        # field -> term -> 出现该词的文档数
        self._df: dict[str, dict[str, int]] = {"title": {}, "body": {}}
        self._field_len: dict[str, dict[int, int]] = {"title": {}, "body": {}}
        self._avg_len: dict[str, float] = {"title": 0.0, "body": 0.0}

    def add_document(self, doc_id: int, title: str, body: str) -> None:
        self.doc_ids.append(doc_id)
        for field, text in (("title", title), ("body", body)):
            tokens = tokenize(text)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf[field][doc_id] = tf
            self._field_len[field][doc_id] = len(tokens)
            for term in tf:
                self._df[field][term] = self._df[field].get(term, 0) + 1

    def freeze(self) -> None:
        """文档灌入完成后计算平均字段长度。"""
        for field in ("title", "body"):
            lens = self._field_len[field].values()
            self._avg_len[field] = (sum(lens) / len(lens)) if lens else 0.0

    def _idf(self, field: str, term: str) -> float:
        n = len(self.doc_ids)
        df = self._df[field].get(term, 0)
        # 平滑 IDF，保证常见词也是正值但权重低，避免小语料下分数塌缩。
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _field_score(self, field: str, doc_id: int, terms: Sequence[str],
                     field_weight: float) -> float:
        tf_map = self._tf[field].get(doc_id, {})
        if not tf_map:
            return 0.0
        dl = self._field_len[field][doc_id]
        avgdl = self._avg_len[field] or 1.0
        denom_norm = 1.0 - BM25_B + BM25_B * dl / avgdl
        total = 0.0
        for term in terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(field, term)
            total += idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * denom_norm)
        return field_weight * total

    def search(self, query: str, title_weight: float,
               body_weight: float) -> list[RankedDoc]:
        terms = tokenize(query)
        if not terms:
            return []
        term_set = set(terms)
        results: list[RankedDoc] = []
        for doc_id in self.doc_ids:
            # 召回条件：任一字段命中任一查询词条（OR 语义）。
            hit = any(
                term in self._tf[field].get(doc_id, {})
                for field in ("title", "body") for term in term_set
            )
            if not hit:
                continue
            score = (
                self._field_score("title", doc_id, terms, title_weight)
                + self._field_score("body", doc_id, terms, body_weight)
            )
            results.append(RankedDoc(doc_id=doc_id, score=score))
        # 分数越高越相关；同分时按 doc_id 保证两套排序结果稳定可复现。
        results.sort(key=lambda d: (-d.score, d.doc_id))
        return results


def rank_documents(index: BM25Index, query: str) -> dict[str, list[RankedDoc]]:
    """对同一批命中文档分别用两套 BM25 打分，各返回一张榜单。

    候选集合天然一致：两套排序使用同一个索引、同一份召回条件，
    只有字段权重不同。
    """
    if not tokenize(query):
        return {SIDE_FULL: [], SIDE_TITLE: []}
    full_rank = index.search(query, title_weight=BODY_WEIGHT, body_weight=BODY_WEIGHT)
    title_rank = index.search(query, title_weight=TITLE_WEIGHT, body_weight=BODY_WEIGHT)
    return {SIDE_FULL: full_rank, SIDE_TITLE: title_rank}


@dataclass
class MergedItem:
    """交错合并后的一个展示位。"""

    position: int
    doc_id: int
    # 由哪一方选出：full / title；平手时为 None（不计票、不换手）。
    picked_by: str | None
    # 是否为两边首选相同的平手文档。
    tie: bool


def interleave(
    full_rank: Sequence[RankedDoc],
    title_rank: Sequence[RankedDoc],
    first_side: str,
) -> list[MergedItem]:
    """按盲测规则交错两张榜单。

    规则：
      1. 随机先手 first_side，之后两边轮流行动；
      2. 每一方选择自己榜单中最靠前且尚未展示的文档；
      3. 若两边当前首选相同，则该文档只展示一次，标记为平手：
         不计票、不换手（下一轮仍由当前方先手）；
      4. 否则由当前行动方展示其首选，并换手；
      5. 一方榜单耗尽后，另一方依次展示剩余文档（归属仍可计票）。
    """
    if first_side not in SIDES:
        raise ValueError(f"invalid first_side: {first_side}")

    full_queue = [d.doc_id for d in full_rank]
    title_queue = [d.doc_id for d in title_rank]
    queues = {SIDE_FULL: full_queue, SIDE_TITLE: title_queue}
    shown: set[int] = set()
    merged: list[MergedItem] = []

    side = first_side
    while full_queue or title_queue:
        other = SIDE_TITLE if side == SIDE_FULL else SIDE_FULL

        # 跳过已经展示过的文档，取各方当前首选。
        while queues[side] and queues[side][0] in shown:
            queues[side].pop(0)
        while queues[other] and queues[other][0] in shown:
            queues[other].pop(0)

        pick_side = queues[side][0] if queues[side] else None
        pick_other = queues[other][0] if queues[other] else None

        if pick_side is None:
            # 当前方耗尽：另一方接管，继续用另一方的归属展示。
            side = other
            continue
        if pick_other is not None and pick_side == pick_other:
            # 两边首选相同：只展示一次，平手，不计票，不换手。
            merged.append(
                MergedItem(position=len(merged) + 1, doc_id=pick_side,
                           picked_by=None, tie=True)
            )
            shown.add(pick_side)
            continue

        # 当前方展示自己的首选，归属当前方，然后换手。
        merged.append(
            MergedItem(position=len(merged) + 1, doc_id=pick_side,
                       picked_by=side, tie=False)
        )
        shown.add(pick_side)
        side = other

    return merged
