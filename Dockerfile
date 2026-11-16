# 운영 노드(EC2 Graviton)와 같은 ARM64로 CI에서 빌드
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/hf

# 의존성만 먼저 설치해 코드만 바뀌면 이 레이어를 재사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 임베딩 모델을 이미지에 미리 받아 둠. 운영 파드는 인터넷에서 모델을 받지 않음
ARG EMBED_MODEL=dragonkue/multilingual-e5-small-ko
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"
ENV HF_HUB_OFFLINE=1

COPY app app
# 임베딩 파일(data/embeddings-*.npz)도 함께 들어감. 운영은 읽기만 함
COPY data data

RUN useradd --system --uid 10001 app
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
