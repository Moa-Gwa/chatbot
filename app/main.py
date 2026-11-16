"""챗봇 REST API. 클러스터 안에서 BE만 호출하며 답변은 SSE로 스트리밍함."""
import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import config, prompts
from app.corpus import Doc, load_aliases, load_docs
from app.graph import build, unverified
from app.index import Index

log = logging.getLogger(__name__)
PENDING_TTL = 600  # 의도 확인을 기다리는 대화는 10분 뒤 지움


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=6)


class ResumeRequest(BaseModel):
    choice: Literal["confirm", "alternative", "retry"]
    index: int | None = None


class Service:
    """색인, 그래프, 확인 대기 중인 대화. 테스트는 가짜 모델로 만든 Service를 넣음"""

    def __init__(self, index: Index, analyzer, writer):
        self.index = index
        # 상태에 담는 Doc을 체크포인트에서 되살릴 수 있게 허용(미등록 타입은 이후 LangGraph에서 차단됨)
        self.checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[("app.corpus", "Doc")]))
        self.graph = build(index, analyzer, writer, self.checkpointer)
        self.pending: dict[str, float] = {}  # thread_id → 멈춘 시각

    def expire(self) -> None:
        now = time.monotonic()
        # 다른 요청 스레드가 동시에 추가하므로 한 번에 복사한 뒤 훑음
        for thread_id in [t for t, at in list(self.pending.items()) if now - at > PENDING_TTL]:
            self.pending.pop(thread_id, None)
            self.checkpointer.delete_thread(thread_id)


def default_service() -> Service:
    from app.models import LocalEmbedder, analyzer, writer
    index = Index(load_docs(config.DATA_DIR), load_aliases(config.DATA_DIR), LocalEmbedder(), config.DATA_DIR)
    return Service(index, analyzer(), writer())


def create_app(service: Service | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or default_service()
        if not app.state.service.index.ready:
            # 색인은 처음 한 번 수 분 걸릴 수 있어 서버를 먼저 띄우고 뒤에서 계산함
            app.state.indexing = asyncio.create_task(asyncio.to_thread(app.state.service.index.build))
        yield

    app = FastAPI(title="Moa-Gwa Chatbot", lifespan=lifespan)

    def internal(x_internal_token: str = Header(default="")) -> None:
        if not config.INTERNAL_TOKEN or not hmac.compare_digest(x_internal_token, config.INTERNAL_TOKEN):
            raise HTTPException(401, "invalid internal token")

    def ready_service() -> Service:
        svc: Service = app.state.service
        if not svc.index.ready:
            raise HTTPException(503, "색인을 준비하고 있어요. 잠시 후 다시 시도해 주세요.")
        svc.expire()
        return svc

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        index: Index = app.state.service.index
        body = {"ready": index.ready, "embedded": index.embedded, "total": len(index.docs), "error": index.error}
        return JSONResponse(body, status_code=200 if index.ready else 503)

    @app.post("/v1/chat", dependencies=[Depends(internal)])
    def chat(req: ChatRequest, svc: Service = Depends(ready_service)) -> StreamingResponse:
        thread_id = f"t_{uuid.uuid4().hex}"
        state = {"question": req.question, "history": [t.model_dump() for t in req.history]}
        return sse(svc, thread_id, state)

    @app.post("/v1/chat/{thread_id}/resume", dependencies=[Depends(internal)])
    def resume(thread_id: str, req: ResumeRequest, svc: Service = Depends(ready_service)) -> StreamingResponse:
        if svc.pending.pop(thread_id, None) is None:
            raise HTTPException(404, "확인 시간이 지났어요. 다시 질문해 주세요.")
        return sse(svc, thread_id, Command(resume=req.model_dump()))

    return app


def event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def source_view(n: int, d: Doc) -> dict:
    kind = "regulation" if d.group == "regulation" else d.kind
    return {"n": n, "kind": kind, "title": d.title, "url": d.url, "department": d.department, "phone": d.phone}


def sse(svc: Service, thread_id: str, payload) -> StreamingResponse:
    config_ = {"configurable": {"thread_id": thread_id}}

    def stream():
        started, paused = time.monotonic(), False
        try:
            for mode, chunk in svc.graph.stream(payload, config_, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    message, meta = chunk
                    if meta.get("langgraph_node") in ("answer", "fallback") and message.content:
                        text = message.content if isinstance(message.content, str) else "".join(
                            p.get("text", "") for p in message.content if isinstance(p, dict))
                        if text:
                            yield event("token", {"text": text})
                    continue
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        paused = True
                        svc.pending[thread_id] = time.monotonic()
                        yield event("clarify", {"thread_id": thread_id, **update[0].value})
                    elif node == "analyze" and update["intent"] != "out_of_scope" and not update["needs_confirm"]:
                        yield event("understood", {"interpretation": update["interpretation"]})
                    elif node == "confirm" and update.get("retry"):
                        yield event("retry", {"message": prompts.RETRY})
                    elif node == "retrieve":
                        yield event("sources", [source_view(i, d) for i, d in enumerate(update["sources"], 1)])
                    elif node == "refuse":
                        yield event("token", {"text": update["answer"]})
            if paused:
                return
            state = svc.graph.get_state(config_).values
            checks = unverified(state.get("answer", ""), state.get("sources", []))
            if checks:
                log.warning("자료에 없는 연락처나 링크 %d건", len(checks))
            # 질문 원문은 남기지 않고 처리 결과만 기록
            log.info("chat done grounded=%s ms=%d", state.get("grounded"), (time.monotonic() - started) * 1000)
            yield event("done", {"grounded": bool(state.get("grounded")), "unverified": checks})
        except Exception:
            log.exception("chat failed")
            yield event("error", {"code": "UPSTREAM", "message": "잠시 후 다시 시도해 주세요."})
        finally:
            if not paused:
                svc.checkpointer.delete_thread(thread_id)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app = create_app()
