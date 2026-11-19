# 帮助中心搜索盲测（Search Blind Test）

一个用于对比两套 BM25 排序效果的盲测演示应用。同一次搜索会生成两张榜单，
按交错规则合并成一份**隐藏归属**的文档列表；评审者点开文档即完成投票，
页面汇总两套排序各自的得票。

- 后端：FastAPI + SQLite（FTS5 辅助索引 + 自实现 BM25 打分）
- 前端：原生 HTML/CSS/JS，无构建步骤
- 部署：Docker Compose 一键启动，数据持久化到命名卷

## 一、盲测规则

1. 每次搜索同时产生两套排序：
   - **全文 BM25（full）**：标题、正文字段权重相同；
   - **标题加权 BM25（title）**：标题字段权重为正文的 10 倍。
2. 系统**随机决定先手**，之后两边轮流选择自己榜单中「最靠前且尚未展示」的文档。
3. 当两边在剩余文档中的首选是同一篇时，该文档**只展示一次**，标记为平手：
   **不计票、也不换手**（下一轮仍由当前方先手）。
4. 页面**隐藏归属**：接口不返回每篇文档由哪套排序选出、不返回两套榜单顺序、
   不返回先手方；平手标记在该文档被点开之前也不暴露，避免反推归属。
5. 每轮中一篇**有归属文档的首次点开**记给选出它的一方；
   并发或重复点击合计**最多一票**（数据库主键约束兜底）。平手文档只留点击记录、不投票。
6. 每轮有固定链接 `/rounds/{id}`，重开可看到**原始列表与计票状态**（持久化在 SQLite）。
7. `/summary` 汇总两套排序在全部轮次中的得票。

## 二、目录结构

```
.
├── app/
│   ├── main.py          # FastAPI 路由与接口
│   ├── search.py        # 分词、BM25 索引/打分、交错合并算法
│   ├── database.py      # SQLite schema、轮次/点击/投票持久化
│   ├── corpus.py        # 内置 16 篇帮助文档语料
│   └── static/          # 前端页面（首页/轮次页/文档页/汇总页）
├── tests/test_app.py    # 28 项自测（含并发投票、归属隐藏、复原）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 三、用 Docker Compose 启动（演示方式）

```bash
docker compose up --build
```

启动后访问 <http://localhost:8000>：

- `/` 发起盲测搜索；
- `/docs` 浏览内置帮助文档，可点开阅读全文；
- `/rounds/{id}` 盲测轮次页（固定链接，可重开复原）；
- `/summary` 两套排序得票汇总。

SQLite 数据库保存在命名卷 `blind_data`（容器内 `/data/blind.db`），
容器重建后数据不丢失。停止并清理：

```bash
docker compose down          # 停止，保留数据
docker compose down -v       # 停止并删除数据卷
```

## 四、本地直接运行（不用 Docker）

需要 Python 3.11+：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 可选：指定数据库路径
BLIND_DB_PATH=/tmp/blind.db uvicorn app.main:app --port 8000
```

## 五、自测

```bash
python3 -I tests/test_app.py
```

覆盖内容：中文分词、两套 BM25 候选集一致性与标题加权差异、交错/平手/换手规则、
投票幂等、16 与 32 并发点击只计一票、重开链接复原、接口归属隐藏、404/空查询等。

## 六、设计说明

- **为什么自实现 BM25 而不是直接用 FTS5 的 `bm25()`**：演示语料只有十几篇文档，
  常见词会命中半数以上文档，FTS5 会把负 IDF 钳制为 0，叠加长度归一化后分数几乎全为 0，
  两套排序无法区分。本项目用经典 Robertson-Spärck Jones IDF（带平滑）+ 字段权重
  自行计算 BM25，打分确定、可解释；FTS5 表仍保留为全文索引底座。
- **中文分词**：FTS5 的 unicode61 会把连续中文切成一个整词，因此索引/查询前先按字切分，
  英文与数字整体保留为词条。
- **并发安全**：SQLite 开启 WAL；`votes` 表以 `(round_id, doc_id)` 为主键，
  `INSERT OR IGNORE` 保证并发/重复点击最多一票；`clicks` 表保留全部点击留痕。
