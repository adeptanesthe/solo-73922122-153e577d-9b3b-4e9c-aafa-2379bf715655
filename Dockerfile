FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BLIND_DB_PATH=/data/blind.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# SQLite 数据文件持久化到挂载卷
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 多 worker 也安全：SQLite 开启 WAL，投票由主键唯一约束兜底
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
