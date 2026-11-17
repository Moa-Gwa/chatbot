"""질문 분석 → (헷갈리면 학생 확인) → 검색 → 답변 또는 문의처 안내."""
import json
import logging
import re
from datetime import datetime
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app import prompts
from app.corpus import KST, Doc
from app.index import Index

log = logging.getLogger(__name__)

# 가장 가까운 홈페이지 안내나 규정이 이보다 멀면 근거 부족으로 보고 문의처만 안내(근거: docs/decisions.md)
GROUND_MIN_SIMILARITY = 0.42


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class State(TypedDict, total=False):
    question: str
    history: list[Turn]
    interpretation: str
    intent: str
    campus: str
    alternatives: list[str]
    needs_confirm: bool  # 학생에게 해석을 확인받을지. 그래프 분기와 API 이벤트가 함께 씀
    retry: bool
    sources: list[Doc]  # 번호 순서 = 답변의 [n]
    grounded: bool
    answer: str


def _json(text: str) -> dict:
    """모델이 코드 블록으로 감싸도 JSON만 꺼냄"""
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {}


def _content(message) -> str:
    content = message.content
    if isinstance(content, list):  # Gemini는 여러 조각(part)으로 줄 때가 있음
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content


def source_block(sources: list[Doc], annotate=lambda text: text) -> str:
    return "\n\n".join(f"[{i}] {d.title}\n{annotate(d.text)}" + (f"\n원문: {d.url}" if d.url else "")
                       for i, d in enumerate(sources, 1))


def cited(answer: str, count: int) -> list[int]:
    """답변이 실제로 인용한 근거 번호. 화면에는 이것만 보여 줌"""
    # "[1]"뿐 아니라 "[1, 9]"처럼 한 괄호에 묶어 쓴 번호도 읽음
    numbers = (int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in group.split(","))
    return sorted({n for n in numbers if 1 <= n <= count})


def build(index: Index, analyzer: BaseChatModel, writer: BaseChatModel, checkpointer: BaseCheckpointSaver):
    def analyze(state: State) -> State:
        history = "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", [])) or "(없음)"
        try:
            result = _json(_content(analyzer.invoke(prompts.ANALYZE.format(history=history, question=state["question"]))))
        except Exception:  # 분석이 실패해도 질문 그대로 검색해 답함
            log.exception("질문 분석 실패")
            result = {}
        intent = str(result.get("intent") or "procedure")
        alts = result.get("alternatives")
        # 모델이 목록 대신 문자열을 주면 글자 단위로 쪼개지므로 목록일 때만 씀
        alternatives = [str(a) for a in alts if str(a).strip()][:2] if isinstance(alts, list) else []
        return {
            "interpretation": str(result.get("interpretation") or state["question"]),
            "intent": intent,
            "campus": result.get("campus") if result.get("campus") in ("성심", "성의", "성신") else "성심",
            "alternatives": alternatives,
            # 고를 대안이 있을 때만 확인함
            # 모델이 false를 문자열로 줘도 분명하지 않음으로 읽음
            "needs_confirm": intent != "out_of_scope" and str(result.get("clear", True)).lower() == "false" and bool(alternatives),
        }

    def after_analyze(state: State) -> str:
        if state["intent"] == "out_of_scope":
            return "refuse"
        return "confirm" if state["needs_confirm"] else "retrieve"

    def confirm(state: State) -> State:
        # 여기서 그래프가 멈추고, 학생 응답이 Command(resume=...)로 들어오면 이 줄부터 이어서 실행됨
        reply = interrupt({"interpretation": state["interpretation"], "alternatives": state["alternatives"]})
        choice = reply.get("choice") if isinstance(reply, dict) else None
        if choice == "alternative":
            i = reply.get("index")
            if isinstance(i, int) and 0 <= i < len(state["alternatives"]):
                return {"interpretation": state["alternatives"][i], "retry": False}
        return {"retry": choice not in ("confirm", "alternative")}

    def retrieve(state: State) -> State:
        hits = index.search(state["interpretation"], state["campus"])
        evidence = [h for h in hits if h.doc.group != "contact"]
        sources = [h.doc for h in evidence]
        for contact in index.contacts_for(evidence, state["campus"]) + [h.doc for h in hits if h.doc.group == "contact"]:
            if contact not in sources:
                sources.append(contact)
        grounded = any(h.similarity >= GROUND_MIN_SIMILARITY for h in evidence)
        return {"sources": sources, "grounded": grounded}

    def write(template: str, state: State) -> State:
        prompt = template.format(sources=source_block(state["sources"], index.annotate), question=state["interpretation"],
                                 today=datetime.now(KST).strftime("%Y-%m-%d"))
        return {"answer": _content(writer.invoke(prompt))}

    def answer(state: State) -> State:
        return write(prompts.ANSWER, state)

    def fallback(state: State) -> State:
        return write(prompts.FALLBACK, state)

    def refuse(state: State) -> State:
        return {"answer": prompts.OUT_OF_SCOPE, "sources": []}

    graph = StateGraph(State)
    for name, node in [("analyze", analyze), ("confirm", confirm), ("retrieve", retrieve),
                       ("answer", answer), ("fallback", fallback), ("refuse", refuse)]:
        graph.add_node(name, node)
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", after_analyze, ["refuse", "retrieve", "confirm"])
    graph.add_conditional_edges("confirm", lambda s: END if s.get("retry") else "retrieve", [END, "retrieve"])
    graph.add_conditional_edges("retrieve", lambda s: "answer" if s["grounded"] else "fallback", ["answer", "fallback"])
    for name in ("answer", "fallback", "refuse"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=checkpointer)


CONTACT = re.compile(r"0\d{1,2}-\d{3,4}-\d{4}|https?://[A-Za-z0-9\-._~:/?#@!$&'*+,;=%]+")


def contacts(text: str) -> list[str]:
    """전화번호, 링크를 뽑아 끝의 문장부호와 / 를 뗌. "…/에서", "…do." 같은 한국어 문장 대비"""
    return [x.rstrip(".,;:!?'/") for x in CONTACT.findall(text)]


def unverified(answer: str, sources: list[Doc]) -> list[str]:
    """답변에 나온 전화번호, 링크 중 자료에 없는 것. 모델이 지어낸 값을 잡아냄"""
    # 부분 문자열로 비교하면 실제 링크의 앞부분만 잘라 쓴 링크를 놓치므로 값 단위로 정확히 비교
    known = set(contacts("\n".join(f"{d.text} {d.url or ''} {d.phone or ''}" for d in sources) + " 02-2164-4114"))
    return [x for x in dict.fromkeys(contacts(answer)) if x not in known]

