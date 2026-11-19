"""关键操作自测。

直接用解释器运行：
    python3 -I tests/test_app.py
覆盖：分词/FTS、两套 BM25 差异、交错规则、平手规则、投票幂等与并发、
重开链接复原、归属隐藏、汇总统计、HTTP 端到端流程。
"""

import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/workspace/.pylibs")

from app import database as db
from app.search import (
    SIDE_FULL,
    SIDE_TITLE,
    SIDES,
    build_match_query,
    interleave,
    rank_documents,
    tokenize,
)


class TokenizerTest(unittest.TestCase):
    def test_cjk_split_per_char(self):
        self.assertEqual(tokenize("重置密码"), ["重", "置", "密", "码"])

    def test_latin_kept_whole_and_lower(self):
        self.assertEqual(tokenize("Hello CSV"), ["hello", "csv"])

    def test_mixed(self):
        self.assertEqual(tokenize("App 闪退x1"), ["app", "闪", "退", "x1"])

    def test_match_query_quoted_or(self):
        self.assertEqual(build_match_query("密码"), '"密" OR "码"')


class SearchRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmp.name, "t.db")
        db.init_db(cls.db_path)
        cls.conn = db.connect(cls.db_path)
        cls.index = db.build_bm25_index(cls.conn)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        cls.tmp.cleanup()

    def test_corpus_loaded_with_fts(self):
        n = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        self.assertGreaterEqual(n, 16)
        nfts = self.conn.execute("SELECT COUNT(*) FROM doc_fts").fetchone()[0]
        self.assertEqual(n, nfts)

    def test_chinese_query_returns_results(self):
        ranks = rank_documents(self.index, "密码")
        self.assertTrue(ranks[SIDE_FULL])
        self.assertTrue(ranks[SIDE_TITLE])

    def test_both_rankings_same_candidate_set(self):
        for q in ["密码", "退款", "登录", "验证码", "搜索 文档"]:
            ranks = rank_documents(self.index, q)
            a = [d.doc_id for d in ranks[SIDE_FULL]]
            b = [d.doc_id for d in ranks[SIDE_TITLE]]
            self.assertEqual(set(a), set(b), f"候选集合不一致: {q}")

    def test_title_weight_changes_ordering_or_scores(self):
        # 标题加权必须真正产生不同的分数（标题命中被放大）。
        ranks = rank_documents(self.index, "密码")
        full_scores = {d.doc_id: d.score for d in ranks[SIDE_FULL]}
        title_scores = {d.doc_id: d.score for d in ranks[SIDE_TITLE]}
        changed = [i for i in full_scores if full_scores[i] != title_scores[i]]
        self.assertTrue(changed, "标题加权未改变任何文档分数")

    def test_no_match(self):
        ranks = rank_documents(self.index, "龘齉爩")
        self.assertEqual(ranks[SIDE_FULL], [])
        self.assertEqual(ranks[SIDE_TITLE], [])


class InterleaveTest(unittest.TestCase):
    @staticmethod
    def _rank(ids):
        from app.search import RankedDoc
        return [RankedDoc(doc_id=i, score=0.0) for i in ids]

    def test_alternation_and_attribution(self):
        # full: [1,2,3], title: [4,5,6]，full 先手
        merged = interleave(self._rank([1, 2, 3]), self._rank([4, 5, 6]), SIDE_FULL)
        self.assertEqual([m.doc_id for m in merged], [1, 4, 2, 5, 3, 6])
        self.assertEqual(
            [m.picked_by for m in merged],
            [SIDE_FULL, SIDE_TITLE, SIDE_FULL, SIDE_TITLE, SIDE_FULL, SIDE_TITLE],
        )
        self.assertTrue(all(not m.tie for m in merged))
        # 位置序号连续
        self.assertEqual([m.position for m in merged], [1, 2, 3, 4, 5, 6])

    def test_tie_shown_once_no_vote_no_turn_switch(self):
        # full: [1,2], title: [1,3]，full 先手
        # 第 1 轮：两边首选都是 1 -> 平手，不换手（仍 full 行动）
        # 第 2 轮：full 首选 2，title 首选 3 -> full 拿 2，换手
        # 第 3 轮：title 拿 3
        merged = interleave(self._rank([1, 2]), self._rank([1, 3]), SIDE_FULL)
        self.assertEqual([m.doc_id for m in merged], [1, 2, 3])
        self.assertTrue(merged[0].tie and merged[0].picked_by is None)
        self.assertFalse(merged[1].tie and merged[2].tie)
        self.assertEqual(merged[1].picked_by, SIDE_FULL)
        self.assertEqual(merged[2].picked_by, SIDE_TITLE)

    def test_tie_title_first(self):
        # title 先手时平手后同样不换手
        merged = interleave(self._rank([1, 2]), self._rank([1, 3]), SIDE_TITLE)
        self.assertEqual([m.doc_id for m in merged], [1, 3, 2])
        self.assertTrue(merged[0].tie)
        self.assertEqual(merged[1].picked_by, SIDE_TITLE)
        self.assertEqual(merged[2].picked_by, SIDE_FULL)

    def test_consecutive_ties(self):
        # full=[1,2,5], title=[1,2,6]：1、2 连续平手，之后 full 拿 5、换手 title 拿 6
        merged = interleave(self._rank([1, 2, 5]), self._rank([1, 2, 6]), SIDE_FULL)
        self.assertEqual([m.doc_id for m in merged], [1, 2, 5, 6])
        self.assertTrue(all(m.tie for m in merged[:2]))
        self.assertEqual(merged[2].picked_by, SIDE_FULL)
        self.assertEqual(merged[3].picked_by, SIDE_TITLE)

    def test_one_side_exhausted(self):
        merged = interleave(self._rank([1, 2, 3]), self._rank([1]), SIDE_FULL)
        # 1 平手 -> full 拿 2 -> title 已耗尽 -> full 继续拿 3
        self.assertEqual([m.doc_id for m in merged], [1, 2, 3])
        self.assertTrue(merged[0].tie)
        self.assertEqual([m.picked_by for m in merged[1:]], [SIDE_FULL, SIDE_FULL])

    def test_no_duplicate_and_union_coverage(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            a = rng.sample(range(1, 11), 8)
            b = rng.sample(range(1, 11), 9)
            merged = interleave(self._rank(a), self._rank(b), rng.choice(SIDES))
            ids = [m.doc_id for m in merged]
            self.assertEqual(len(ids), len(set(ids)), "出现重复展示")
            self.assertEqual(set(ids), set(a) | set(b), "未覆盖全部候选")

    def test_invalid_first_side(self):
        with self.assertRaises(ValueError):
            interleave(self._rank([1]), self._rank([1]), "bogus")


class VotePersistenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "v.db")
        db.init_db(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_round(self, picked_by, doc_id=1, position=1, tie=False):
        conn = db.connect(self.db_path)
        try:
            rid = "r1"
            db.create_round(
                conn, rid, "密码", SIDE_FULL,
                {s: [] for s in SIDES},
                [{"position": position, "doc_id": doc_id,
                  "picked_by": picked_by, "tie": tie}],
            )
            return rid
        finally:
            conn.close()

    def test_first_click_votes_repeat_clicks_one_vote(self):
        rid = self._make_round(SIDE_FULL, doc_id=1)
        conn = db.connect(self.db_path)
        try:
            r1 = db.record_click_and_vote(conn, rid, 1, SIDE_FULL, 1)
            self.assertTrue(r1["vote_recorded_now"] and r1["voted"])
            r2 = db.record_click_and_vote(conn, rid, 1, SIDE_FULL, 1)
            self.assertFalse(r2["vote_recorded_now"])  # 重复点击不再计票
            votes = db.get_votes(conn, rid)
            self.assertEqual(len(votes), 1)
            self.assertEqual(db.get_click_count(conn, rid), 2)  # 点击都留痕
        finally:
            conn.close()

    def test_tie_doc_never_votes(self):
        rid = self._make_round(None, doc_id=2, tie=True)
        conn = db.connect(self.db_path)
        try:
            for _ in range(3):
                r = db.record_click_and_vote(conn, rid, 2, None, 1)
                self.assertFalse(r["voted"])
            self.assertEqual(db.get_votes(conn, rid), [])
            self.assertEqual(db.get_click_count(conn, rid), 3)
        finally:
            conn.close()

    def test_concurrent_clicks_at_most_one_vote(self):
        rid = self._make_round(SIDE_TITLE, doc_id=3)
        errors = []

        def worker():
            try:
                conn = db.connect(self.db_path)
                try:
                    db.record_click_and_vote(conn, rid, 3, SIDE_TITLE, 1)
                finally:
                    conn.close()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors, f"并发点击出错: {errors}")
        conn = db.connect(self.db_path)
        try:
            votes = db.get_votes(conn, rid)
            self.assertEqual(len(votes), 1, "并发下票数必须恰好为 1")
            self.assertEqual(votes[0]["side"], SIDE_TITLE)
            self.assertEqual(db.get_click_count(conn, rid), 16)
        finally:
            conn.close()

    def test_round_reopen_restores_state(self):
        rid = self._make_round(SIDE_FULL, doc_id=1)
        conn = db.connect(self.db_path)
        try:
            db.record_click_and_vote(conn, rid, 1, SIDE_FULL, 1)
        finally:
            conn.close()
        # 全新连接模拟重开链接
        conn2 = db.connect(self.db_path)
        try:
            row = db.get_round(conn2, rid)
            merged = json.loads(row["merged_json"])
            self.assertEqual(merged[0]["doc_id"], 1)  # 列表快照不变
            self.assertEqual(len(db.get_votes(conn2, rid)), 1)  # 计票状态复原
        finally:
            conn2.close()


class HttpApiTest(unittest.TestCase):
    """端到端：FastAPI + TestClient（httpx）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["BLIND_DB_PATH"] = os.path.join(cls.tmp.name, "api.db")
        # 显式建表并构建 BM25 索引（TestClient 不进入 with 时不会触发 lifespan）
        db.init_db(os.environ["BLIND_DB_PATH"])
        # 延迟导入，确保环境变量生效
        from fastapi.testclient import TestClient

        import app.main as mainmod
        conn = db.connect(os.environ["BLIND_DB_PATH"])
        try:
            mainmod.BM25 = db.build_bm25_index(conn)
        finally:
            conn.close()
        cls.client = TestClient(mainmod.app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_pages_render(self):
        for path in ["/", "/docs", "/summary"]:
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            self.assertIn("text/html", r.headers["content-type"])

    def test_02_docs_api(self):
        r = self.client.get("/api/docs")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.json()), 16)
        doc_id = r.json()[0]["id"]
        r2 = self.client.get(f"/api/docs/{doc_id}")
        self.assertEqual(r2.status_code, 200)
        self.assertIn("title", r2.json())
        self.assertEqual(self.client.get(f"/docs/{doc_id}").status_code, 200)

    def test_03_search_creates_round_and_hides_attribution(self):
        # 「验证码」在两套排序下顺序不同，保证同时存在归属文档与平手文档
        r = self.client.post("/api/search", json={"query": "验证码"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("round_id", data)
        self.assertGreater(len(data["items"]), 0)

        # 归属隐藏：轮次响应不允许出现任何可反推归属的字段
        forbidden_top = {"rankings", "first_side", "picked_by", "side"}
        self.assertTrue(forbidden_top.isdisjoint(data.keys()),
                        f"顶层泄露字段: {forbidden_top & set(data.keys())}")
        for item in data["items"]:
            self.assertNotIn("picked_by", item)
            self.assertNotIn("side", item)
            # 未发生任何点击前，平手标记也不得暴露
            self.assertFalse(item["tie"], "未交互前不应返回平手标记")

        self.__class__.round_id = data["round_id"]
        # 合并列表覆盖全部候选且无重复（候选集合从服务端快照核对）
        import app.main as mainmod
        conn = db.connect(mainmod.DB_PATH)
        try:
            snap = json.loads(db.get_round(conn, data["round_id"])["ranking_json"])
        finally:
            conn.close()
        full_ids = [x["doc_id"] for x in snap[SIDE_FULL]]
        title_ids = [x["doc_id"] for x in snap[SIDE_TITLE]]
        self.assertEqual(set(full_ids), set(title_ids))
        merged_ids = [i["doc_id"] for i in data["items"]]
        self.assertEqual(len(merged_ids), len(set(merged_ids)))
        self.assertEqual(set(merged_ids), set(full_ids))

    def test_04_click_voting_and_idempotency(self):
        rid = self.__class__.round_id
        # 直接查库拿到一篇归属文档与一篇平手文档（接口本身隐藏归属）
        import app.main as mainmod
        conn = db.connect(mainmod.DB_PATH)
        try:
            row = db.get_round(conn, rid)
            merged = json.loads(row["merged_json"])
        finally:
            conn.close()
        owned = next(m for m in merged if not m["tie"])
        tied = next((m for m in merged if m["tie"]), None)

        r1 = self.client.post(
            f"/api/rounds/{rid}/click", json={"doc_id": owned["doc_id"]})
        self.assertEqual(r1.status_code, 200)
        body1 = r1.json()
        self.assertTrue(body1["vote_recorded_now"])
        # 点击响应也不得暴露归属方
        self.assertNotIn("side", body1)
        self.assertNotIn("picked_by", body1)
        # 归属是否正确记录由服务端数据库核对
        conn = db.connect(mainmod.DB_PATH)
        try:
            votes = {v["doc_id"]: v["side"] for v in db.get_votes(conn, rid)}
        finally:
            conn.close()
        self.assertEqual(votes[owned["doc_id"]], owned["picked_by"])

        r2 = self.client.post(
            f"/api/rounds/{rid}/click", json={"doc_id": owned["doc_id"]})
        self.assertFalse(r2.json()["vote_recorded_now"])  # 重复点击一票

        if tied:
            rt = self.client.post(
                f"/api/rounds/{rid}/click", json={"doc_id": tied["doc_id"]})
            self.assertTrue(rt.json()["tie"])
            self.assertFalse(rt.json()["voted"])
            self.assertNotIn("side", rt.json())

        # 计票面板与复原
        again = self.client.get(f"/api/rounds/{rid}").json()
        total = again["tally"][SIDE_FULL] + again["tally"][SIDE_TITLE]
        self.assertEqual(total, 1)
        voted_item = next(i for i in again["items"] if i["doc_id"] == owned["doc_id"])
        self.assertTrue(voted_item["voted"])
        # 平手文档被点开后，平手标记才在响应中揭示
        if tied:
            tie_item = next(i for i in again["items"] if i["doc_id"] == tied["doc_id"])
            self.assertTrue(tie_item["tie"])
            self.assertFalse(tie_item["voted"])
        # 未点击的文档仍不暴露平手标记
        untouched = [i for i in again["items"]
                     if i["doc_id"] not in {owned["doc_id"], *( [tied["doc_id"]] if tied else [])}]
        self.assertTrue(all(not i["tie"] for i in untouched))

    def test_05_concurrent_clicks_http(self):
        r = self.client.post("/api/search", json={"query": "退款 流程"})
        rid = r.json()["round_id"]
        import app.main as mainmod
        conn = db.connect(mainmod.DB_PATH)
        try:
            merged = json.loads(db.get_round(conn, rid)["merged_json"])
        finally:
            conn.close()
        owned = next(m for m in merged if not m["tie"])
        results = []

        def fire():
            resp = self.client.post(
                f"/api/rounds/{rid}/click", json={"doc_id": owned["doc_id"]})
            results.append(resp.json())

        threads = [threading.Thread(target=fire) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        recorded = sum(1 for x in results if x["vote_recorded_now"])
        self.assertEqual(recorded, 1, "HTTP 并发下只能有一次真正计票")
        payload = self.client.get(f"/api/rounds/{rid}").json()
        self.assertEqual(
            payload["tally"][SIDE_FULL] + payload["tally"][SIDE_TITLE], 1)

    def test_06_summary_aggregates_both_rankings(self):
        r = self.client.get("/api/summary")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # 键只能来自两套排序；具体哪一方得票取决于随机先手
        self.assertTrue(set(data["votes_by_side"]).issubset(set(SIDES)))
        total_votes = sum(data["votes_by_side"].values())
        self.assertGreaterEqual(total_votes, 2)  # 前面用例已产生票数
        # 分轮次明细字段齐全，且不泄露先手方
        self.assertTrue(all("full_votes" in x and "title_votes" in x
                            for x in data["rounds"]))
        self.assertTrue(all("first_side" not in x for x in data["rounds"]))
        # 明细票数之和与全局一致
        detail_sum = sum((r["full_votes"] or 0) + (r["title_votes"] or 0)
                         for r in data["rounds"])
        self.assertEqual(detail_sum, total_votes)

    def test_07_unknown_round_and_doc_404(self):
        self.assertEqual(self.client.get("/api/rounds/nope").status_code, 404)
        self.assertEqual(
            self.client.post("/api/rounds/nope/click", json={"doc_id": 1}).status_code,
            404)
        self.assertEqual(self.client.get("/api/docs/9999").status_code, 404)

    def test_08_empty_query_and_no_match(self):
        # 空白查询：应用层显式拒绝（400）；Pydantic 字段约束拒绝空串（422）
        self.assertIn(
            self.client.post("/api/search", json={"query": "   "}).status_code,
            (400, 422))
        self.assertEqual(
            self.client.post("/api/search", json={"query": ""}).status_code, 422)
        r = self.client.post("/api/search", json={"query": "龘齉爩"})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
