"""챗봇 REST API. 클러스터 안에서 BE만 호출하며 답변은 SSE로 스트리밍함."""
import asyncio
import hmac
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app import config, prompts
from app.corpus import Doc, load_aliases, load_docs, notice_doc
from app.graph import build, cited, unverified
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


class Notice(BaseModel):
    """BE가 보내는 공지 한 건(필드 이름은 BE JSON 그대로 camelCase)"""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    key: str = Field(max_length=40)  # 예: CRAWLED:123
    title: str = Field(max_length=300)
    organization: str = Field(max_length=100)
    body: str = Field(default="", max_length=8000)
    url: str | None = Field(default=None, max_length=500)
    published_at: str | None = None
    deadline_at: str | None = None
    event_start_at: str | None = None
    event_end_at: str | None = None
    location: str | None = Field(default=None, max_length=200)


class NoticeSync(BaseModel):
    notices: list[Notice] = Field(max_length=5000)


class Service:
    """색인, 그래프, 확인 대기 중인 대화. 테스트는 가짜 모델로 만든 Service를 넣음"""

    def __init__(self, index: Index, analyzer, writer):
        self.index, self.analyzer, self.writer = index, analyzer, writer
        self.base_docs = index.docs  # 공지를 뺀 홈페이지, 규정, 연락처
        self.notices: list[Doc] = []
        self._pending: list[Doc] | None = None
        self._lock = threading.Lock()
        self.notice_worker: threading.Thread | None = None
        # shortcut: 확인 대기 대화가 파드 메모리에만 있어 재시작이나 다른 파드로 간 /resume은 404, 파드를 늘릴 때 Redis/Postgres 체크포인터로 교체
        # 상태에 담는 Doc을 체크포인트에서 되살릴 수 있게 허용(미등록 타입은 이후 LangGraph에서 차단됨)
        self.checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[("app.corpus", "Doc")]))
        self.graph = build(index, analyzer, writer, self.checkpointer)
        self.pending: dict[str, float] = {}  # thread_id → 멈춘 시각

    def sync_notices(self, notices: list[Doc]) -> bool:
        """공지 목록을 통째로 바꿈. 색인은 뒤에서 만들고, 그사이 들어온 요청은 마지막 목록만 반영함"""
        with self._lock:
            latest = self._pending if self._pending is not None else self.notices
            if [(d.id, d.digest) for d in latest] == [(d.id, d.digest) for d in notices]:
                return False
            self._pending = notices
            if not (self.notice_worker and self.notice_worker.is_alive()):
                self.notice_worker = threading.Thread(target=self._rebuild, daemon=True)
                self.notice_worker.start()
        return True

    def _rebuild(self) -> None:
        while True:
            with self._lock:
                notices, self._pending = self._pending, None
            if notices is None:
                return
            index = Index(self.base_docs + notices, self.index.aliases, self.index.embedder, self.index.data_dir,
                          known=self.index.known)
            index.build()
            if not index.ready:  # 실패하면 기존 색인을 그대로 쓰고 다음 동기화를 기다림
                continue
            # 그래프는 색인을 붙잡고 있어 함께 바꿈. 확인 대기 대화는 체크포인터에 있어 이어짐
            self.graph, self.index, self.notices = build(index, self.analyzer, self.writer, self.checkpointer), index, notices
            log.info("공지 %d건으로 색인을 바꿈", len(notices))

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
        body = {"ready": index.ready, "embedded": index.embedded, "total": len(index.docs),
                "notices": len(app.state.service.notices), "error": index.error}
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

    @app.put("/v1/notices", dependencies=[Depends(internal)], status_code=202)
    def notices(req: NoticeSync, svc: Service = Depends(ready_service)) -> dict:
        changed = svc.sync_notices([notice_doc(n.model_dump()) for n in req.notices])
        return {"received": len(req.notices), "changed": changed}

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
            answer = state.get("answer", "")
            yield event("done", {"grounded": bool(state.get("grounded")), "unverified": checks,
                                 "cited": cited(answer, len(state.get("sources", [])))})
        except Exception:
            log.exception("chat failed")
            yield event("error", {"code": "UPSTREAM", "message": "잠시 후 다시 시도해 주세요."})
        finally:
            if not paused:
                svc.checkpointer.delete_thread(thread_id)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app = create_app()
