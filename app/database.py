"""SQLite 持久化层：文档、FTS 索引、盲测轮次、点击与投票。"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .corpus import DOCUMENTS
from .search import BM25Index, fts_index_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
    title,
    body,
    content='',
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS rounds (
    id            TEXT PRIMARY KEY,
    query         TEXT NOT NULL,
    first_side    TEXT NOT NULL,
    ranking_json  TEXT NOT NULL,
    merged_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS clicks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id    TEXT NOT NULL REFERENCES rounds(id),
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_clicks_round ON clicks(round_id, doc_id);

-- 一票约束：同一轮同一篇有归属文档最多一票。
CREATE TABLE IF NOT EXISTS votes (
    round_id    TEXT NOT NULL REFERENCES rounds(id),
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    side        TEXT NOT NULL,
    position    INTEGER NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (round_id, doc_id)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """打开一个数据库连接（短连接，每请求一个），开启 WAL 与外键。"""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: str) -> None:
    """建表并写入内置帮助文档（幂等：重复调用不会重复灌数据）。"""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count == 0:
            for doc in DOCUMENTS:
                cur = conn.execute(
                    "INSERT INTO documents(title, body) VALUES (?, ?)",
                    (doc["title"], doc["body"]),
                )
                doc_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO doc_fts(rowid, title, body) VALUES (?, ?, ?)",
                    (doc_id, fts_index_text(doc["title"]),
                     fts_index_text(doc["body"])),
                )
        conn.commit()
    finally:
        conn.close()


def create_round(
    conn: sqlite3.Connection,
    round_id: str,
    query: str,
    first_side: str,
    ranking: dict[str, list[dict[str, Any]]],
    merged: list[dict[str, Any]],
) -> None:
    conn.execute(
        "INSERT INTO rounds(id, query, first_side, ranking_json, merged_json)"
        " VALUES (?, ?, ?, ?, ?)",
        (round_id, query, first_side,
         json.dumps(ranking, ensure_ascii=False),
         json.dumps(merged, ensure_ascii=False)),
    )
    conn.commit()


def get_round(conn: sqlite3.Connection, round_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM rounds WHERE id = ?", (round_id,)
    ).fetchone()


def list_rounds(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM rounds ORDER BY created_at DESC, id DESC"
    ).fetchall()


def get_document(conn: sqlite3.Connection, doc_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, title, body FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, title, body FROM documents ORDER BY id"
    ).fetchall()


def build_bm25_index(conn: sqlite3.Connection) -> BM25Index:
    """从 documents 表构建内存 BM25 索引（应用启动时调用一次）。"""
    index = BM25Index()
    for row in list_documents(conn):
        index.add_document(row["id"], row["title"], row["body"])
    index.freeze()
    return index


def record_click_and_vote(
    conn: sqlite3.Connection,
    round_id: str,
    doc_id: int,
    picked_by: str | None,
    position: int,
) -> dict[str, Any]:
    """记录一次点击，并在满足条件时投出票。

    幂等与并发安全：
      * 每次请求都写 clicks（用于排查重复/并发点击）；
      * votes 对 (round_id, doc_id) 为主键，INSERT OR IGNORE 保证
        并发或重复点击合计最多一票；
      * 平手文档 picked_by 为 None，只记点击、不写投票。
    返回本次请求后的状态（是否新投票、归属方）。
    """
    conn.execute(
        "INSERT INTO clicks(round_id, doc_id) VALUES (?, ?)",
        (round_id, doc_id),
    )
    vote_recorded = False
    if picked_by is not None:
        cur = conn.execute(
            "INSERT OR IGNORE INTO votes(round_id, doc_id, side, position)"
            " VALUES (?, ?, ?, ?)",
            (round_id, doc_id, picked_by, position),
        )
        vote_recorded = cur.rowcount > 0
    conn.commit()

    existing = conn.execute(
        "SELECT side FROM votes WHERE round_id = ? AND doc_id = ?",
        (round_id, doc_id),
    ).fetchone()
    return {
        "voted": existing is not None,
        "vote_recorded_now": vote_recorded,
        "side": existing["side"] if existing else None,
    }


def get_votes(conn: sqlite3.Connection, round_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT doc_id, side, position, created_at FROM votes"
        " WHERE round_id = ? ORDER BY position",
        (round_id,),
    ).fetchall()


def get_click_count(conn: sqlite3.Connection, round_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM clicks WHERE round_id = ?", (round_id,)
    ).fetchone()
    return int(row["n"])


def get_clicked_doc_ids(conn: sqlite3.Connection, round_id: str) -> set[int]:
    """本轮已被点开的文档 id 集合（用于决定何时可以揭示平手标记）。"""
    rows = conn.execute(
        "SELECT DISTINCT doc_id FROM clicks WHERE round_id = ?", (round_id,)
    ).fetchall()
    return {int(r["doc_id"]) for r in rows}


def get_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """汇总两套排序在全部轮次中的得票与点击情况。"""
    rows = conn.execute(
        "SELECT side, COUNT(*) AS n FROM votes GROUP BY side"
    ).fetchall()
    votes_by_side = {r["side"]: int(r["n"]) for r in rows}

    per_round = conn.execute(
        """
        SELECT r.id AS round_id, r.query AS query, r.first_side AS first_side,
               r.created_at AS created_at,
               SUM(CASE WHEN v.side = 'full' THEN 1 ELSE 0 END) AS full_votes,
               SUM(CASE WHEN v.side = 'title' THEN 1 ELSE 0 END) AS title_votes,
               COUNT(v.doc_id) AS total_votes,
               (SELECT COUNT(*) FROM clicks c WHERE c.round_id = r.id) AS clicks
        FROM rounds r
        LEFT JOIN votes v ON v.round_id = r.id
        GROUP BY r.id
        ORDER BY r.created_at DESC, r.id DESC
        """
    ).fetchall()
    return {
        "votes_by_side": votes_by_side,
        "rounds": [dict(r) for r in per_round],
    }
