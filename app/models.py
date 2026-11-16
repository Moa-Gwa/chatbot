"""Gemini 임베딩, 채팅 모델. 테스트는 같은 인터페이스의 가짜 구현을 넣어 API 키 없이 돌림."""
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app import config

# 키가 없어도 서버는 뜨게 하고, 실제 호출에서 실패하면 /readyz에 원인을 보임
KEY = config.GEMINI_API_KEY or "missing"


class GeminiEmbedder:
    def __init__(self) -> None:
        self.name = f"{config.EMBED_MODEL}-{config.EMBED_DIM}"
        self.client = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL, google_api_key=KEY)

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts, task_type="RETRIEVAL_DOCUMENT", titles=titles,
                                           output_dimensionality=config.EMBED_DIM)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text, task_type="RETRIEVAL_QUERY", output_dimensionality=config.EMBED_DIM)


def analyzer() -> ChatGoogleGenerativeAI:
    """질문 분석용. JSON으로만 답하게 함"""
    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=KEY, temperature=0,
                                  response_mime_type="application/json", max_retries=2, timeout=20)


def writer() -> ChatGoogleGenerativeAI:
    """답변 작성용. 글자 단위로 스트리밍"""
    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, google_api_key=KEY, temperature=0.2,
                                  max_retries=2, timeout=60)
