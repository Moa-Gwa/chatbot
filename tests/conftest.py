import zlib

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.corpus import Alias, Doc
from app.index import Index, bigrams

DIM = 512


class FakeEmbedder:
    """2글자 해시 벡터. 글자가 많이 겹칠수록 가까워져 API 없이 검색 순서를 검사할 수 있음"""
    name = "fake"

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(DIM, dtype=np.float32)
        for g in bigrams(text):
            v[zlib.crc32(g.encode()) % DIM] += 1
        return v.tolist()

    def embed_documents(self, texts, titles):
        self.calls += 1
        return [self._vec(f"{title} {text}") for text, title in zip(texts, titles)]

    def embed_query(self, text):
        return self._vec(text)


DOCS = [
    Doc("site0-0", "site", "학사지원 > 학적변동 > 휴학", "[학교 홈페이지 휴학]\n문의: 학사행정팀 02-2164-4170\n휴학 신청은 Trinity 학사정보 학적/졸업에서 신청",
        "https://www.catholic.ac.kr/ko/support/leave_of_absence.do", "학사행정팀", "02-2164-4170", "성심"),
    Doc("site1-0", "site", "학사지원 > 증명발급 > 증명서 발급", "[학교 홈페이지 증명서 발급]\n문의: 학생지원팀 02-2164-4732\n성적증명서 재학증명서는 무인발급기와 인터넷 증명발급으로 발급",
        "https://www.catholic.ac.kr/ko/support/certificate.do", "학생지원팀", "02-2164-4732", "성심"),
    Doc("site2-0", "site", "학사지원 > 군휴학", "[학교 홈페이지 군휴학]\n문의: 학사지원팀\n군휴학은 입영통지서를 지참해 방문 제출",
        "https://www.catholic.ac.kr/ko/support/military.do", "학사지원팀", None, "성심"),
    Doc("reg33-a19", "article", "가톨릭대학교 학칙 제21조(휴학)", "[가톨릭대학교 학칙 제21조(휴학)]\n휴학 기간은 통산 3년을 넘을 수 없다",
        "http://rule.catholic.ac.kr/33"),
    Doc("reg33-b1", "addenda", "가톨릭대학교 학칙 부칙", "[가톨릭대학교 학칙 부칙]\n이 학칙은 휴학 관련 개정 후 시행한다", "http://rule.catholic.ac.kr/33"),
    Doc("contact-0", "contact", "성심 학사행정팀", "[연락처] 성심 학사행정팀, 담당업무: 학적/졸업, 전화 02-2164-4170",
        department="학사행정팀", phone="02-2164-4170", campus="성심"),
    Doc("contact-1", "contact", "성심 학생지원팀", "[연락처] 성심 학생지원팀, 담당업무: 증명서 발급, 전화 02-2164-4732",
        department="학생지원팀", phone="02-2164-4732", campus="성심"),
    Doc("contact-2", "contact", "성의 교학팀", "[연락처] 성의 교학팀, 담당업무: 학적, 전화 02-3147-8000",
        department="교학팀", phone="02-3147-8000", campus="성의"),
]
ALIASES = [Alias("학사지원팀", "학사행정팀", "번호가 다른 부서로 옮겨감", "6522")]


@pytest.fixture
def index(tmp_path):
    idx = Index(DOCS, ALIASES, FakeEmbedder(), tmp_path)
    idx.build()
    return idx


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))
