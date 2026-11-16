"""임베딩 캐시와 혼합 검색(의미 + 키워드). 문서 1만 개 규모라 벡터 DB 없이 메모리에서 계산함."""
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.corpus import Alias, Doc

log = logging.getLogger(__name__)

BATCH = 100  # Gemini batchEmbedContents 한 번에 보낼 수 있는 최대 개수
TRANSIENT = re.compile(r"429|500|502|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE|timed? ?out", re.I)
RRF_K = 60  # 순위 합산 상수(일반적으로 쓰는 값)
QUOTA = {"site": 4, "regulation": 3, "contact": 3}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class Hit:
    doc: Doc
    similarity: float  # 질문과의 코사인 유사도(근거 판정용)


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
    def __init__(self, docs: list[Doc], aliases: list[Alias], embedder: Embedder, cache_dir: Path):
        self.docs, self.aliases, self.embedder = docs, aliases, embedder
        self.cache = cache_dir / f"embeddings-{embedder.name}.npz"
        self.embedded, self.ready, self.error = 0, False, None
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        # 제목은 키워드에 한 번 더 넣어 메뉴 경로, 조문 제목을 무겁게 봄
        self.keyword = Keyword([f"{d.title} {d.title} {d.text}" for d in docs])
        self.groups = np.array([d.group for d in docs])
        self.addenda = np.array([d.kind == "addenda" for d in docs])

    def build(self) -> None:
        """캐시에 없는 문서만 임베딩. 요청 한도에 걸리면 기다렸다가 이어서 계산하고, 묶음마다 캐시에 저장"""
        try:
            self._build()
        except Exception as e:  # 키 오류 등은 /readyz로 알리고 요청은 503으로 거절
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            log.exception("색인 실패")

    def _build(self) -> None:
        known: dict[str, np.ndarray] = {}
        if self.cache.exists():
            data = np.load(self.cache)
            current = {d.digest for d in self.docs}
            # 지금 문서에 없는 옛 벡터는 버려 캐시가 수집할 때마다 커지지 않게 함
            known = {k: v for k, v in zip(data["digests"].tolist(), data["vectors"]) if k in current}
        missing = [d for d in self.docs if d.digest not in known]
        self.embedded = len(self.docs) - len(missing)
        for start in range(0, len(missing), BATCH):
            batch = missing[start:start + BATCH]
            for attempt in range(6):
                try:
                    vectors = self.embedder.embed_documents([d.text for d in batch], [d.title for d in batch])
                    break
                except Exception as e:  # 요청 한도(429), 일시 오류만 점점 길게 기다렸다 다시 시도
                    if attempt == 5 or not TRANSIENT.search(str(e)):
                        raise
                    log.warning("임베딩 재시도 %d: %s", attempt + 1, e)
                    time.sleep(10 * 2 ** attempt)
            known.update({d.digest: np.asarray(v, dtype=np.float32) for d, v in zip(batch, vectors)})
            self.embedded += len(batch)
            self._save(known)
        matrix = np.stack([known[d.digest] for d in self.docs]) if self.docs else np.zeros((0, 1), np.float32)
        self.vectors = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        self.ready = True

    def _save(self, known: dict[str, np.ndarray]) -> None:
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache.with_suffix(".tmp.npz")
        np.savez(tmp, digests=np.array(list(known)), vectors=np.stack(list(known.values())))
        tmp.replace(self.cache)  # 저장 도중 꺼져도 이전 캐시가 남게 바꿔치기

    def search(self, query: str, campus: str = "성심") -> list[Hit]:
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        similarity = self.vectors @ (q / max(float(np.linalg.norm(q)), 1e-12))
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
            score[int(i)] += 1 / (RRF_K + rank)
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
        wanted = list(dict.fromkeys(self.current_department(h.doc.department) for h in hits
                                    if h.doc.kind == "site" and h.doc.department))
        found = [d for d in self.docs if d.kind == "contact" and d.campus == campus and d.department in wanted]
        found.sort(key=lambda d: wanted.index(d.department))
        return found[:limit]

    def aliases_in(self, texts: list[str]) -> list[Alias]:
        joined = "\n".join(texts)
        return [a for a in self.aliases if a.old in joined]
