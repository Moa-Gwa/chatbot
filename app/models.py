"""임베딩은 로컬 Hugging Face 모델, 답변은 Gemini. 테스트는 같은 인터페이스의 가짜 구현을 넣어 돌림."""
from langchain_google_genai import ChatGoogleGenerativeAI

from app import config

# 키가 없어도 서버는 뜨게 하고, 실제 호출에서 실패하면 오류 이벤트로 알림
KEY = config.GEMINI_API_KEY or "missing"


class LocalEmbedder:
    """e5 계열은 질문에 "query: ", 문서에 "passage: "를 붙여야 학습 때와 같은 방식으로 비교됨"""

    def __init__(self, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # torch가 무거워 실제로 쓸 때만 불러옴

        self.name = config.EMBED_MODEL.split("/")[-1]
        self.model = SentenceTransformer(config.EMBED_MODEL, device=device)

    def embed_documents(self, texts: list[str]):
        return self.model.encode([f"passage: {t}" for t in texts], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 100)

    def embed_query(self, text: str):
        return self.model.encode(f"query: {text}", normalize_embeddings=True)


def analyzer() -> ChatGoogleGenerativeAI:
    """질문 분석용. JSON으로만 답하게 함"""
    # gemini-3.5-flash-lite는 샘플링 값이 고정이라 temperature를 지정해도 무시됨(docs/decisions.md 6장)
    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=KEY,
                                  response_mime_type="application/json", max_retries=2, timeout=20)


def writer() -> ChatGoogleGenerativeAI:
    """답변 작성용. 글자 단위로 스트리밍"""
    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=KEY, max_retries=2, timeout=60)
