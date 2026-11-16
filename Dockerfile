# 운영 노드(EC2 Graviton)와 같은 ARM64로 CI에서 빌드
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CACHE_DIR=/app/cache

# 의존성만 먼저 설치해 코드만 바뀌면 이 레이어를 재사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY data data

# 임베딩 캐시는 PVC를 /app/cache에 붙여 재시작에도 유지
RUN useradd --system --uid 10001 app && mkdir cache && chown app cache
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
