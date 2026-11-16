import json

import pytest
from fastapi.testclient import TestClient

from app import config, graph, prompts
from app.main import Service, create_app
from tests.conftest import TOKEN, fake_model

CLEAR = json.dumps({"interpretation": "성적증명서 발급 방법과 문의처", "intent": "procedure", "campus": "성심", "clear": True, "alternatives": []})
AMBIGUOUS = json.dumps({"interpretation": "일반휴학 신청 절차", "intent": "procedure", "campus": "성심", "clear": False,
                        "alternatives": ["군휴학 입영통지서 제출", "질병휴학 신청"]})


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def client(index, analyzer, writer) -> TestClient:
    return TestClient(create_app(Service(index, analyzer, writer)), headers={"X-Internal-Token": TOKEN})


def test_clear_question_streams_understood_sources_answer_and_done(index):
    with client(index, fake_model(CLEAR), fake_model("문의처 학생지원팀 02-2164-4732 [1]")) as c:
        result = events(c.post("/v1/chat", json={"question": "성적증명서 어디서 떼요?"}))
    names = [n for n, _ in result]
    assert names[0] == "understood" and names[1] == "sources" and names[-1] == "done"
    assert result[1][1][0]["url"].endswith("certificate.do")
    assert "".join(d["text"] for n, d in result if n == "token") == "문의처 학생지원팀 02-2164-4732 [1]"
    assert result[-1][1] == {"grounded": True, "unverified": [], "cited": [1]}


def test_ambiguous_question_pauses_then_answers_the_chosen_interpretation(index):
    with client(index, fake_model(AMBIGUOUS), fake_model("군휴학 안내 [1]")) as c:
        first = events(c.post("/v1/chat", json={"question": "휴학 신청 어떻게 해요?"}))
        assert [n for n, _ in first] == ["clarify"]
        clarify = first[0][1]
        assert clarify["interpretation"] == "일반휴학 신청 절차" and len(clarify["alternatives"]) == 2
        second = events(c.post(f"/v1/chat/{clarify['thread_id']}/resume", json={"choice": "alternative", "index": 0}))
        assert second[0][0] == "sources" and second[0][1][0]["title"] == "학사지원 > 군휴학"
        assert second[-1][0] == "done"
        # 한 번 이어서 실행한 대화는 다시 이어갈 수 없음
        assert c.post(f"/v1/chat/{clarify['thread_id']}/resume", json={"choice": "confirm"}).status_code == 404


def test_rejected_interpretation_asks_again_without_answering(index):
    with client(index, fake_model(AMBIGUOUS), fake_model("쓰이면 안 됨")) as c:
        thread = events(c.post("/v1/chat", json={"question": "신청 어떻게 해요?"}))[0][1]["thread_id"]
        result = events(c.post(f"/v1/chat/{thread}/resume", json={"choice": "retry"}))
    assert result[0] == ("retry", {"message": prompts.RETRY})
    assert "sources" not in [n for n, _ in result]


def test_pending_confirmation_expires(index, monkeypatch):
    with client(index, fake_model(AMBIGUOUS), fake_model("x")) as c:
        thread = events(c.post("/v1/chat", json={"question": "신청?"}))[0][1]["thread_id"]
        c.app.state.service.pending[thread] -= 601
        assert c.post(f"/v1/chat/{thread}/resume", json={"choice": "confirm"}).status_code == 404


def test_out_of_scope_is_refused_without_search(index):
    reply = json.dumps({"interpretation": "코드 작성", "intent": "out_of_scope", "campus": "성심", "clear": True, "alternatives": []})
    with client(index, fake_model(reply), fake_model("쓰이면 안 됨")) as c:
        result = events(c.post("/v1/chat", json={"question": "파이썬 코드 짜줘"}))
    assert ("token", {"text": prompts.OUT_OF_SCOPE}) in result and "sources" not in [n for n, _ in result]


def test_weak_evidence_goes_to_fallback_and_invented_numbers_are_flagged(index, monkeypatch):
    monkeypatch.setattr(graph, "GROUND_MIN_SIMILARITY", 1.1)
    with client(index, fake_model(CLEAR), fake_model("확인되지 않아요. 02-9999-0000으로 문의하세요")) as c:
        done = events(c.post("/v1/chat", json={"question": "증명서"}))[-1]
    assert done == ("done", {"grounded": False, "unverified": ["02-9999-0000"], "cited": []})


def test_requests_need_the_internal_token_and_a_ready_index(index, monkeypatch):
    with client(index, fake_model(CLEAR), fake_model("x")) as c:
        assert c.post("/v1/chat", json={"question": "q"}, headers={"X-Internal-Token": "wrong"}).status_code == 401
        assert c.post("/v1/chat", json={"question": ""}).status_code == 422
        index.ready = False
        assert c.post("/v1/chat", json={"question": "q"}).status_code == 503
        assert c.get("/readyz").status_code == 503


def test_missing_token_setting_rejects_every_chat_request(index, monkeypatch):
    # 배포에서 토큰 설정이 빠지면 확인을 건너뛰지 않고 모두 거부
    monkeypatch.setattr(config, "INTERNAL_TOKEN", "")
    with client(index, fake_model(CLEAR), fake_model("x")) as c:
        assert c.post("/v1/chat", json={"question": "q"}, headers={"X-Internal-Token": ""}).status_code == 401
        assert c.post("/v1/chat/t_x/resume", json={"choice": "confirm"}, headers={"X-Internal-Token": ""}).status_code == 401


def test_string_false_from_the_model_still_asks_for_confirmation(index):
    reply = AMBIGUOUS.replace('"clear": false', '"clear": "false"')
    with client(index, fake_model(reply), fake_model("x")) as c:
        assert [n for n, _ in events(c.post("/v1/chat", json={"question": "휴학 신청?"}))] == ["clarify"]
