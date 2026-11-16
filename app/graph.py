"""질문 분석 → (헷갈리면 학생 확인) → 검색 → 답변 또는 문의처 안내."""
import json
import logging
import re
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app import prompts
from app.corpus import Doc
from app.index import Index

log = logging.getLogger(__name__)

# 가장 가까운 홈페이지 안내나 규정이 이보다 멀면 근거 부족으로 보고 문의처만 안내
# shortcut: 기준값을 고정함, 평가 세트로 다시 정할 것
GROUND_MIN_SIMILARITY = 0.55


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
    aliases: list[str]
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


def source_block(sources: list[Doc]) -> str:
    return "\n\n".join(f"[{i}] {d.title}\n{d.text}" + (f"\n원문: {d.url}" if d.url else "")
                       for i, d in enumerate(sources, 1))


def build(index: Index, analyzer: BaseChatModel, writer: BaseChatModel, checkpointer: BaseCheckpointSaver):
    def analyze(state: State) -> State:
        history = "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", [])) or "(없음)"
        try:
            result = _json(_content(analyzer.invoke(prompts.ANALYZE.format(history=history, question=state["question"]))))
        except Exception:  # 분석이 실패해도 질문 그대로 검색해 답함
            log.exception("질문 분석 실패")
            result = {}
        intent = str(result.get("intent") or "procedure")
        alternatives = [str(a) for a in result.get("alternatives") or []][:2]
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
        aliases = [f"'{a.old}' → 지금 '{a.new}'" for a in index.aliases_in([d.text for d in sources]) if a.new]
        grounded = any(h.similarity >= GROUND_MIN_SIMILARITY for h in evidence)
        return {"sources": sources, "aliases": aliases, "grounded": grounded}

    def write(template: str, state: State) -> State:
        aliases = "[부서명 안내]\n" + "\n".join(state["aliases"]) + "\n\n" if state.get("aliases") else ""
        prompt = template.format(aliases=aliases, sources=source_block(state["sources"]), question=state["interpretation"])
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


def unverified(answer: str, sources: list[Doc]) -> list[str]:
    """답변에 나온 전화번호, 링크 중 자료에 없는 것. 모델이 지어낸 값을 잡아냄"""
    known = "\n".join(f"{d.text} {d.url or ''} {d.phone or ''}" for d in sources) + " 02-2164-4114"
    # 링크는 URL에 쓰는 ASCII 글자까지만 보고 끝의 문장부호는 뗌("…/에서", "…do." 같은 한국어 문장 대비)
    found = [x.rstrip(".,;:!?'") for x in re.findall(r"0\d{1,2}-\d{3,4}-\d{4}|https?://[A-Za-z0-9\-._~:/?#@!$&'*+,;=%]+", answer)]
    return [x for x in dict.fromkeys(found) if x not in known]

