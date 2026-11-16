import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# 768차원: 3072 대비 저장량과 계산량이 1/4이고 검색 품질 차이는 작음(Gemini 권장 축소 크기)
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT / "cache"))

# BE만 호출하도록 공유 토큰 확인. 비어 있으면 설정 실수로 보고 모든 채팅 요청을 거부함
INTERNAL_TOKEN = os.getenv("CHATBOT_INTERNAL_TOKEN", "")
