import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
# 한국어 검색용으로 추가 학습한 경량 모델(118M, 384차원). 고른 근거는 docs/decisions.md
EMBED_MODEL = os.getenv("EMBED_MODEL", "dragonkue/multilingual-e5-small-ko")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))

# BE만 호출하도록 공유 토큰 확인. 비어 있으면 설정 실수로 보고 모든 채팅 요청을 거부함
INTERNAL_TOKEN = os.getenv("CHATBOT_INTERNAL_TOKEN", "")
