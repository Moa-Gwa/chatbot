"""docs/decisions.md의 수치를 다시 만듦.
1. 검색 순위: eval/questions.jsonl의 정답 홈페이지가 의미, 키워드, 혼합 검색에서 몇 위에 오는지
2. 근거 판정 기준값: 업무 질문과 무관 질문(eval/off_topic.jsonl)의 최고 유사도 분포
3. 창 크기: 검색용 창 중 모델 입력 한도(512토큰)를 넘는 창 수

    .venv/bin/python scripts/eval_retrieval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.corpus import load_aliases, load_docs  # noqa: E402
from app import graph  # noqa: E402
from app.index import Index, windows  # noqa: E402
from app.models import LocalEmbedder  # noqa: E402

index = Index(load_docs(config.DATA_DIR), load_aliases(config.DATA_DIR), LocalEmbedder(), config.DATA_DIR)
index.build()
questions = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
site = np.flatnonzero(index.groups == "site")


def rank(order: list[int], page: str) -> int | None:
    for r, i in enumerate(order, 1):
        if (index.docs[i].url or "").endswith(page):
            return r
    return None


def ranks(query: str, page: str) -> dict[str, int | None]:
    q = np.asarray(index.embedder.embed_query(query), dtype=np.float32)
    sim = np.full(len(index.docs), -1.0, dtype=np.float32)
    np.maximum.at(sim, index.owner, index.vectors @ q)
    kw = index.keyword.scores(query)
    hybrid = [index.docs.index(h.doc) for h in index._fuse(site, sim, kw)]
    return {"의미": rank(list(site[np.argsort(-sim[site])]), page), "키워드": rank(list(site[np.argsort(-kw[site])]), page),
            "혼합": rank(hybrid, page)}


results = [(q, ranks(q["question"], q["page"])) for q in questions]
for method in ("의미", "키워드", "혼합"):
    rs = [r[method] for _, r in results]
    hit1 = sum(r == 1 for r in rs) / len(rs)
    hit4 = sum(r is not None and r <= 4 for r in rs) / len(rs)
    mrr = sum(1 / r for r in rs if r) / len(rs)
    print(f"{method}: hit@1 {hit1:.2f}, hit@4 {hit4:.2f}, MRR {mrr:.2f}")
print("\n혼합 검색 4위 밖:")
for q, r in results:
    if not r["혼합"] or r["혼합"] > 4:
        print(f"  {q['question']} → 정답 {q['page']} {r}")


def best(query: str) -> float:
    return max(h.similarity for h in index.search(query) if h.doc.group != "contact")


off_topic = [json.loads(line)["question"] for line in (ROOT / "eval" / "off_topic.jsonl").open(encoding="utf-8")]
on = sorted(best(q["question"]) for q in questions)
off = sorted(best(q) for q in off_topic)
print(f"\n근거 판정(현재 기준값 {graph.GROUND_MIN_SIMILARITY}):")
print(f"  업무 질문 최고 유사도: 최저 {on[0]:.3f}, 하위 10% {np.percentile(on, 10):.3f}, 중앙값 {np.median(on):.3f}")
print(f"  무관 질문 최고 유사도: 중앙값 {np.median(off):.3f}, 최고 {off[-1]:.3f}")
print(f"  기준값 미만으로 빠지는 업무 질문 {sum(v < graph.GROUND_MIN_SIMILARITY for v in on)}개, 기준값 이상인 무관 질문 {sum(v >= graph.GROUND_MIN_SIMILARITY for v in off)}개")

tokenizer = index.embedder.model.tokenizer
lengths = [len(tokenizer("passage: " + w)["input_ids"]) for d in index.docs for w in windows(d.text)]
print(f"\n검색용 창: {len(lengths)}개, 최대 {max(lengths)}토큰, 512토큰 초과 {sum(n > 512 for n in lengths)}개")
