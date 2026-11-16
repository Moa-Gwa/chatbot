# Moa-Gwa Chatbot

가톨릭대학교 학생이 학교 일을 처리할 때 **어디에 문의하고 어떤 절차로 진행하는지** 안내하는 RAG 챗봇 서비스.
학교 홈페이지 안내, 부서 연락처, 규정을 근거로만 답하고 근거 출처를 함께 보여준다.

학생 요청은 BE(로그인 확인, 질문 한도)를 거쳐 이 서비스의 REST API로 전달된다. 클러스터 안에서만 열린다.

## 구조

```
analyze ─┬─ 분명함 ─────────────▶ retrieve ─┬─ 근거 있음 ─▶ answer   (문의처, 절차, 참고)
         ├─ 헷갈림 ─▶ confirm ⏸ ─┘          └─ 근거 부족 ─▶ fallback (확인 안 됨 + 문의처)
         │            └ 아니에요 ─▶ 다시 질문 받기
         └─ 범위 밖 ─▶ refuse
```

| 파일 | 역할 |
|---|---|
| `app/corpus.py` | `data/`의 홈페이지 안내, 규정, 연락처, 옛 부서명 대조표 읽기 |
| `app/index.py` | Gemini 임베딩 캐시, 의미 검색과 2글자 키워드 검색(BM25)을 순위로 합치는 혼합 검색, 담당부서 연락처 보강 |
| `app/graph.py` | LangGraph 흐름. 헷갈리는 질문은 `interrupt()`로 멈추고 학생 확인 후 이어서 실행 |
| `app/prompts.py` | 질문 분석, 답변, 근거 부족 안내 프롬프트 |
| `app/main.py` | FastAPI, SSE 스트리밍, 내부 토큰 확인, 확인 대기 대화 10분 보관 |
| `scripts/crawl_site.py` | 학교 홈페이지 안내 수집(`data/README.md` 참고) |

문서 1만 개 규모라 벡터 DB 없이 메모리에서 계산한다. 임베딩은 `cache/`에 저장하고 바뀐 문서만 다시 계산한다.

## API

| 요청 | 설명 |
|---|---|
| `POST /v1/chat` | `{question, history[≤6]}`. 응답은 `text/event-stream` |
| `POST /v1/chat/{thread_id}/resume` | `{choice: "confirm" \| "alternative" \| "retry", index?}`. 멈춘 대화를 이어서 실행 |
| `GET /healthz` | 프로세스 생존 |
| `GET /readyz` | 색인 준비 여부와 진행률, 오류 |

SSE 이벤트

| 이벤트 | 데이터 |
|---|---|
| `understood` | 분명한 질문의 해석. 답변 위 "이렇게 이해했어요" |
| `clarify` | 헷갈리는 질문. `thread_id`, 해석, 대안 해석. 여기서 스트림이 끝나고 `resume`을 기다림 |
| `retry` | 학생이 "아니에요"를 고름. 다시 질문 받기 안내 |
| `sources` | 근거 목록(번호, 종류, 제목, 링크, 담당부서, 전화) |
| `token` | 답변 글자 |
| `done` | `grounded`(근거 여부), `unverified`(자료에 없는 전화번호나 링크) |
| `error` | 모델 호출 실패 등 |

채팅 요청은 `X-Internal-Token` 헤더가 `CHATBOT_INTERNAL_TOKEN`과 같아야 한다. 토큰이 설정되지 않으면 설정 실수로 보고 모든 채팅 요청을 거부한다.

## 실행

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pytest -q                      # API 키 없이 가짜 임베딩, 가짜 모델로 실행

printf "GEMINI_API_KEY=...\nCHATBOT_INTERNAL_TOKEN=local-dev\n" > .env   # 실제 모델로 실행할 때(.env는 커밋하지 않음)
set -a; . ./.env; set +a
.venv/bin/uvicorn app.main:app --port 8000
curl localhost:8000/readyz               # 처음에는 임베딩 계산에 몇 분 걸림
```

| 환경변수 | 기본값 |
|---|---|
| `GEMINI_API_KEY` | 없음(없으면 `/readyz`에 오류, 채팅은 503) |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` |
| `EMBED_MODEL`, `EMBED_DIM` | `gemini-embedding-001`, `768` |
| `CACHE_DIR` | `cache/` |
| `CHATBOT_INTERNAL_TOKEN` | 없음(없으면 채팅 요청 401. 로컬 개발도 아무 값이나 지정) |
