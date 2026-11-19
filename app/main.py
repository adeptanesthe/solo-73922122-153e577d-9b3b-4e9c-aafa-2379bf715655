"""FastAPI 应用：帮助中心搜索盲测。

页面（HTML）：
  GET /                首页：搜索框 + 规则说明 + 历史轮次
  GET /rounds/{rid}    盲测轮次页：交错文档列表（隐藏归属）+ 本轮计票
  GET /docs            帮助文档目录（内置文档可直接点开阅读）
  GET /docs/{doc_id}   单篇帮助文档阅读页
  GET /summary         汇总页：两套排序的得票对比

接口（JSON）：
  POST /api/search                 创建盲测轮次
  GET  /api/rounds/{rid}           读取轮次快照（列表 + 投票状态，可重开复原）
  POST /api/rounds/{rid}/click     点击文档（并发/重复点击最多一票）
  GET  /api/summary                全局汇总数据
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database as db
from .search import (
    SIDE_FULL,
    SIDE_LABELS,
    SIDE_TITLE,
    SIDES,
    BM25Index,
    MergedItem,
    interleave,
    rank_documents,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DB_PATH = os.environ.get("BLIND_DB_PATH", os.path.join(BASE_DIR, "data", "blind.db"))

# 进程级 BM25 索引（只读，线程安全），lifespan 中构建。
BM25: BM25Index | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BM25
    db.init_db(DB_PATH)
    conn = db.connect(DB_PATH)
    try:
        BM25 = db.build_bm25_index(conn)
    finally:
        conn.close()
    yield


app = FastAPI(title="帮助中心搜索盲测", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _round_payload(row) -> dict[str, Any]:
    """把一轮记录组装成前端所需 JSON。

    盲测安全：响应中不包含任何可反推排序归属的信息——
    不返回两套榜单（ranking_json 仅服务端内部使用）、不返回先手方、
    平手标记在对应文档被点开之前也不暴露（否则结合交错规则可推归属）。
    归属 picked_by 只存在服务端，用于点击计票。
    """
    merged = json.loads(row["merged_json"])
    conn = db.connect(DB_PATH)
    try:
        docs = {
            d["id"]: {"id": d["id"], "title": d["title"], "body": d["body"]}
            for d in db.list_documents(conn)
        }
        vote_rows = db.get_votes(conn, row["id"])
        click_count = db.get_click_count(conn, row["id"])
        clicked = db.get_clicked_doc_ids(conn, row["id"])
    finally:
        conn.close()

    votes = {v["doc_id"]: v["side"] for v in vote_rows}
    items = []
    for m in merged:
        doc = docs[m["doc_id"]]
        item = {
            "position": m["position"],
            "doc_id": m["doc_id"],
            "title": doc["title"],
            "snippet": _snippet(doc["body"]),
            # 仅在该文档已被点开后揭示平手标记；未交互前不暴露结构信息。
            "tie": bool(m["tie"]) and m["doc_id"] in clicked,
            "voted": m["doc_id"] in votes,
        }
        items.append(item)

    # 本轮计票状态（只给聚合票数，不给文档与归属的映射），重开链接时复原。
    tally = {SIDE_FULL: 0, SIDE_TITLE: 0}
    for side in votes.values():
        tally[side] = tally.get(side, 0) + 1

    return {
        "round_id": row["id"],
        "query": row["query"],
        "side_labels": SIDE_LABELS,
        "created_at": row["created_at"],
        "items": items,
        "tally": tally,
        "click_count": click_count,
    }


def _snippet(body: str, length: int = 90) -> str:
    body = body.strip()
    return body if len(body) <= length else body[:length] + "…"


# --------------------------------------------------------------------------- #
# JSON 接口
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)


@app.post("/api/search")
def create_search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="搜索词不能为空")

    conn = db.connect(DB_PATH)
    try:
        ranks = rank_documents(BM25, query)
        if not ranks[SIDE_FULL] and not ranks[SIDE_TITLE]:
            return JSONResponse(
                status_code=404,
                content={"detail": "没有找到与搜索词匹配的帮助文档", "query": query},
            )

        # 随机先手。
        import random
        first_side = random.choice(list(SIDES))
        merged: list[MergedItem] = interleave(
            ranks[SIDE_FULL], ranks[SIDE_TITLE], first_side
        )

        round_id = uuid.uuid4().hex
        ranking_snapshot = {
            side: [
                {"doc_id": d.doc_id, "score": round(d.score, 6)}
                for d in ranks[side]
            ]
            for side in SIDES
        }
        merged_snapshot = [
            {
                "position": m.position,
                "doc_id": m.doc_id,
                "picked_by": m.picked_by,
                "tie": m.tie,
            }
            for m in merged
        ]
        db.create_round(
            conn, round_id, query, first_side,
            ranking_snapshot, merged_snapshot,
        )
        row = db.get_round(conn, round_id)
        return _round_payload(row)
    finally:
        conn.close()


@app.get("/api/rounds/{round_id}")
def read_round(round_id: str):
    conn = db.connect(DB_PATH)
    try:
        row = db.get_round(conn, round_id)
        if row is None:
            raise HTTPException(status_code=404, detail="轮次不存在或已过期")
        return _round_payload(row)
    finally:
        conn.close()


class ClickRequest(BaseModel):
    doc_id: int


@app.post("/api/rounds/{round_id}/click")
def click_document(round_id: str, req: ClickRequest):
    conn = db.connect(DB_PATH)
    try:
        row = db.get_round(conn, round_id)
        if row is None:
            raise HTTPException(status_code=404, detail="轮次不存在或已过期")
        merged = json.loads(row["merged_json"])
        target = next((m for m in merged if m["doc_id"] == req.doc_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="该文档不在本轮列表中")
        if db.get_document(conn, req.doc_id) is None:
            raise HTTPException(status_code=404, detail="文档不存在")

        result = db.record_click_and_vote(
            conn, round_id, req.doc_id,
            picked_by=target["picked_by"], position=target["position"],
        )
        # 不向客户端返回归属方 side：归属全程只在服务端可见。
        return {
            "round_id": round_id,
            "doc_id": req.doc_id,
            "tie": target["tie"],
            "voted": result["voted"],
            "vote_recorded_now": result["vote_recorded_now"],
        }
    finally:
        conn.close()


@app.get("/api/summary")
def summary():
    conn = db.connect(DB_PATH)
    try:
        data = db.get_summary(conn)
        data["side_labels"] = SIDE_LABELS
        # 汇总页只展示聚合计票，不暴露每轮先手，避免与交错列表结合反推归属。
        for row in data["rounds"]:
            row.pop("first_side", None)
        return data
    finally:
        conn.close()


@app.get("/api/docs")
def api_list_docs():
    conn = db.connect(DB_PATH)
    try:
        return [dict(d) for d in db.list_documents(conn)]
    finally:
        conn.close()


@app.get("/api/docs/{doc_id}")
def api_get_doc(doc_id: int):
    conn = db.connect(DB_PATH)
    try:
        row = db.get_document(conn, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        return dict(row)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# HTML 页面（静态文件，由前端 JS 调接口渲染）
# --------------------------------------------------------------------------- #
@app.get("/")
def index_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/rounds/{round_id}")
def round_page(round_id: str):
    return FileResponse(os.path.join(STATIC_DIR, "round.html"))


@app.get("/docs")
def docs_index_page():
    return FileResponse(os.path.join(STATIC_DIR, "docs.html"))


@app.get("/docs/{doc_id}")
def doc_page(doc_id: int):
    return FileResponse(os.path.join(STATIC_DIR, "doc.html"))


@app.get("/summary")
def summary_page():
    return FileResponse(os.path.join(STATIC_DIR, "summary.html"))
