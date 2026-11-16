"""임베딩 캐시와 혼합 검색(의미 + 키워드). 문서 1만 개 규모라 벡터 DB 없이 메모리에서 계산함."""
import hashlib
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.corpus import Alias, Doc

log = logging.getLogger(__name__)

# 값마다 정한 근거와 측정 결과는 docs/decisions.md
RRF_K = 60  # 순위 합산 상수(RRF 기본값)
KEYWORD_WEIGHT = 0.2  # 키워드 순위의 비중. 평가 세트 31개에서 가장 좋았던 값
WINDOW_CHARS = 550  # 검색용 창 크기. 모델 입력 한도 512토큰 안에 들도록 정함
QUOTA = {"site": 4, "regulation": 3, "contact": 3}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class Hit:
    doc: Doc
    similarity: float  # 질문과의 코사인 유사도(근거 판정용)


def windows(text: str, limit: int = WINDOW_CHARS) -> list[str]:
    """답변용 조각은 그대로 두고, 임베딩만 짧은 창으로 나눔. 창마다 첫 줄(메뉴 경로, 조문 제목)을 다시 붙임"""
    head, _, body = text.partition("\n")
    if len(text) <= limit or not body:
        return [text[:limit]]
    room = max(limit - len(head) - 1, 100)
    out, current = [], ""
    for line in body.split("\n"):
        while len(line) > room:  # 한 줄이 너무 길면(긴 표 행) 글자 수로 자름
            if current:
                out.append(current)
                current = ""
            out.append(line[:room])
            line = line[room:]
        if current and len(current) + 1 + len(line) > room:
            out.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        out.append(current)
    return [f"{head}\n{w}" for w in out]


def bigrams(text: str) -> list[str]:
    """띄어쓰기를 빼고 2글자씩 자름. 형태소 분석기 없이 부서명, 업무명 같은 한국어 낱말을 잡기 위함"""
    s = "".join(text.lower().split())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class Keyword:
    """2글자 단위 BM25"""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(len(texts), dtype=np.float32)
        for i, text in enumerate(texts):
            grams = Counter(bigrams(text))
            self.lengths[i] = sum(grams.values())
            for g, tf in grams.items():
                self.postings[g].append((i, tf))
        self.avg = float(self.lengths.mean()) if len(texts) else 1.0
        n = len(texts)
        self.idf = {g: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for g, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.lengths), dtype=np.float32)
        for g in set(bigrams(query)):
            for i, tf in self.postings.get(g, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                out[i] += self.idf[g] * tf * (self.k1 + 1) / norm
        return out


class Index:
    def __init__(self, docs: list[Doc], aliases: list[Alias], embedder: Embedder, data_dir: Path):
        self.docs, self.aliases, self.embedder = docs, aliases, embedder
        # 로컬에서 미리 계산해 커밋한 파일. 운영은 읽기만 함(scripts/build_embeddings.py)
        self.cache = data_dir / f"embeddings-{embedder.name}.npz"
        self.embedded, self.ready, self.error = 0, False, None
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        # 제목은 키워드에 한 번 더 넣어 메뉴 경로, 조문 제목을 무겁게 봄
        self.keyword = Keyword([f"{d.title} {d.title} {d.text}" for d in docs])
        self.groups = np.array([d.group for d in docs])
        self.addenda = np.array([d.kind == "addenda" for d in docs])

    def build(self) -> None:
        """임베딩 파일을 읽고 빠진 문서만 계산. 실패하면 /readyz로 알리고 요청은 503으로 거절"""
        try:
            self._build()
        except Exception as e:
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            log.exception("색인 실패")

    def _build(self) -> None:
        # 조각마다 검색용 창을 만들고, 창이 어느 조각 것인지 기억함
        texts = [(i, w) for i, d in enumerate(self.docs) for w in windows(d.text)]
        self.owner = np.array([i for i, _ in texts], dtype=int)
        digests = [hashlib.sha1(w.encode()).hexdigest() for _, w in texts]
        known: dict[str, np.ndarray] = {}
        if self.cache.exists():
            data = np.load(self.cache)
            current = set(digests)
            # 지금 쓰지 않는 옛 벡터는 버려 파일이 다시 수집할 때마다 커지지 않게 함
            known = {k: v for k, v in zip(data["digests"].tolist(), data["vectors"]) if k in current}
        missing = [(h, w) for h, (_, w) in dict(zip(digests, texts)).items() if h not in known]
        self.embedded = len(self.docs) if not missing else 0
        if missing:
            log.warning("임베딩 파일에 없는 창 %d개를 계산함", len(missing))
            known.update({h: np.asarray(v, dtype=np.float32) for (h, _), v in
                          zip(missing, self.embedder.embed_documents([w for _, w in missing]))})
            self.embedded = len(self.docs)
            self._save(known)
        matrix = np.stack([known[h] for h in digests]).astype(np.float32) if digests else np.zeros((0, 1), np.float32)
        self.vectors = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        self.ready = True

    def _save(self, known: dict[str, np.ndarray]) -> None:
        try:
            tmp = self.cache.with_suffix(".tmp.npz")
            # float16으로 저장해 파일 크기를 반으로 줄임. 평가 수치(docs/decisions.md)는 이 파일로 잰 값
            np.savez(tmp, digests=np.array(list(known)), vectors=np.stack(list(known.values())).astype(np.float16))
            tmp.replace(self.cache)  # 저장 도중 꺼져도 이전 파일이 남게 바꿔치기
        except OSError:  # 운영 이미지의 data/는 읽기 전용. 메모리에서만 쓰고 계속 진행
            log.warning("임베딩 파일을 저장하지 못함: %s", self.cache)

    def search(self, query: str, campus: str = "성심") -> list[Hit]:
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        # 조각 점수 = 그 조각에 속한 창 중 가장 가까운 창의 점수
        similarity = np.full(len(self.docs), -1.0, dtype=np.float32)
        np.maximum.at(similarity, self.owner, self.vectors @ (q / max(float(np.linalg.norm(q)), 1e-12)))
        keyword = self.keyword.scores(query)
        include_addenda = any(w in query for w in ("부칙", "시행일", "경과조치"))
        hits: list[Hit] = []
        for group, quota in QUOTA.items():
            idx = np.flatnonzero((self.groups == group) & (include_addenda | ~self.addenda))
            if group == "contact":
                idx = np.array([i for i in idx if self.docs[i].campus == campus], dtype=int)
            if not len(idx):
                continue
            hits += self._fuse(idx, similarity, keyword)[:quota]
        return hits

    def _fuse(self, idx: np.ndarray, similarity: np.ndarray, keyword: np.ndarray) -> list[Hit]:
        """의미 순위와 키워드 순위를 순위 역수로 합침(RRF). 점수 단위가 달라도 섞을 수 있음"""
        dense_rank = idx[np.argsort(-similarity[idx])]
        score: dict[int, float] = defaultdict(float)
        for rank, i in enumerate(dense_rank[:50]):
            score[int(i)] += 1 / (RRF_K + rank)
        sparse = idx[keyword[idx] > 0]
        for rank, i in enumerate(sparse[np.argsort(-keyword[sparse])][:50]):
            score[int(i)] += KEYWORD_WEIGHT / (RRF_K + rank)
        ranked = sorted(score, key=score.get, reverse=True)
        return [Hit(self.docs[i], float(similarity[i])) for i in ranked]

    def current_department(self, name: str) -> str:
        """옛 부서명이면 지금 부서명으로 바꿈(대조표에 지금 부서가 있을 때만)"""
        for a in self.aliases:
            if a.old == name and a.new:
                return a.new
        return name

    def contacts_for(self, hits: list[Hit], campus: str = "성심", limit: int = 2) -> list[Doc]:
        """고른 홈페이지 안내의 담당부서로 전화번호부를 다시 찾아 문의처를 보강"""
        # "학생지원팀 교내장학 / 국가장학"처럼 업무 설명이 붙은 부서명은 첫 단어로 맞춤
        wanted = list(dict.fromkeys(self.current_department(h.doc.department.removeprefix("성심교정 ").split()[0]) for h in hits
                                    if h.doc.kind == "site" and h.doc.department))
        found = [d for d in self.docs if d.kind == "contact" and d.campus == campus and d.department in wanted]
        found.sort(key=lambda d: wanted.index(d.department))
        return found[:limit]

    def aliases_in(self, texts: list[str]) -> list[Alias]:
        joined = "\n".join(texts)
        return [a for a in self.aliases if a.old in joined]
