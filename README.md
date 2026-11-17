# Moa-Gwa Chatbot

가톨릭대학교 규정, 부서 연락처를 근거로 답하는 RAG 챗봇 서비스.
학생 요청은 BE(로그인 확인, 요청 한도)를 거쳐 이 서비스의 REST API로 전달된다. 클러스터 내부에서만 열린다.

- Python, FastAPI, LangGraph
- 임베딩, 답변 생성: Gemini

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
| `app/index.py` | 임베딩 파일 읽기, 의미 검색과 2글자 키워드 검색(BM25)을 순위로 합치는 혼합 검색, 담당부서 연락처 보강 |
| `app/models.py` | 임베딩은 로컬 Hugging Face 모델(`dragonkue/multilingual-e5-small-ko`), 답변은 Gemini |
| `app/graph.py` | LangGraph 흐름. 헷갈리는 질문은 `interrupt()`로 멈추고 학생 확인 후 이어서 실행 |
| `app/prompts.py` | 질문 분석, 답변, 근거 부족 안내 프롬프트 |
| `app/main.py` | FastAPI, SSE 스트리밍, 내부 토큰 확인, 확인 대기 대화 10분 보관 |
| `scripts/crawl_site.py` | 학교 홈페이지 안내 수집(`data/README.md` 참고) |
| `scripts/build_embeddings.py` | 임베딩 파일 `data/embeddings-<모델>.npz`를 로컬에서 만듦 |
| `scripts/eval_retrieval.py` | 평가 세트(`eval/`)로 검색 품질, 근거 판정 기준값, 창 크기를 잼 |
| `docs/decisions.md` | 모델과 조정한 값의 근거, 측정 결과 |

문서 1만 개 규모라 벡터 DB 없이 메모리에서 계산한다. 조각은 답변용(최대 1,500자)으로 두고, 임베딩은 모델 입력 한도(512토큰)에 맞춰 550자 창으로 나눠 계산한다.
임베딩은 로컬에서 미리 계산해 커밋하고 운영은 읽기만 한다. `data/`를 바꾸면 임베딩 파일도 다시 만들어야 테스트가 통과한다.

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
.venv/bin/pytest -q                              # 가짜 임베딩, 가짜 모델로 실행(API 키 불필요)

.venv/bin/python scripts/build_embeddings.py      # data/를 바꿨을 때(Apple Silicon은 GPU로 약 3분)
.venv/bin/python scripts/eval_retrieval.py        # 검색 품질 확인

printf "GEMINI_API_KEY=...\nCHATBOT_INTERNAL_TOKEN=local-dev\n" > .env   # .env는 커밋하지 않음
set -a; . ./.env; set +a
.venv/bin/uvicorn app.main:app --port 8000
```

| 환경변수 | 기본값 |
|---|---|
| `GEMINI_API_KEY` | 없음(없으면 답변 생성이 실패해 `error` 이벤트) |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` |
| `EMBED_MODEL` | `dragonkue/multilingual-e5-small-ko`(바꾸면 임베딩 파일을 다시 만들어야 함) |
| `CHATBOT_INTERNAL_TOKEN` | 없음(없으면 채팅 요청 401. 로컬 개발도 아무 값이나 지정) |
