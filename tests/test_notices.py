import json

from fastapi.testclient import TestClient

from app.corpus import notice_doc
from app.index import Index
from app.main import Service, create_app
from tests.conftest import ALIASES, DOCS, TOKEN, FakeEmbedder, fake_model

SCHOLARSHIP = {"key": "CRAWLED:7", "title": "2026학년도 2학기 성적우수 장학금 신청 안내", "organization": "학생지원팀",
               "body": "성적우수 장학금 신청은 Trinity에서 접수", "url": "https://moagwa.example/notices/crawled/7",
               "publishedAt": "2026-11-10T00:00:00Z", "deadlineAt": "2026-11-23T09:00:00Z", "location": None}
QUESTION = json.dumps({"interpretation": "성적우수 장학금 신청 마감", "intent": "criteria", "campus": "성심", "clear": True,
                       "alternatives": []})


def test_notice_doc_puts_kst_dates_before_the_body():
    d = notice_doc({"key": "MANUAL:1", "title": "학술제", "organization": "총학생회", "body": "본문",
                    "published_at": "2026-11-10T00:00:00Z", "event_start_at": "2026-12-03T09:00:00Z",
                    "event_end_at": "2026-12-03T11:00:00Z", "location": "대강당"})
    assert d.kind == "notice" and d.group == "notice" and d.title == "공지 > 총학생회 > 학술제"
    assert d.text.splitlines()[:5] == ["[공지] 학술제", "작성: 총학생회", "게시일: 2026-11-10 09:00",
                                       "행사 일시: 2026-12-03 18:00 ~ 2026-12-03 20:00", "장소: 대강당"]


def test_rebuilding_with_notices_embeds_only_new_windows_and_keeps_the_file(index, tmp_path):
    before = (tmp_path / "embeddings-fake.npz").read_bytes()
    embedder = FakeEmbedder()
    notice = notice_doc({**SCHOLARSHIP, "published_at": SCHOLARSHIP["publishedAt"], "deadline_at": SCHOLARSHIP["deadlineAt"]})
    rebuilt = Index(DOCS + [notice], ALIASES, embedder, tmp_path, known=index.known)
    rebuilt.build()
    assert rebuilt.ready and embedder.calls == 1  # 공지 창만 계산
    assert (tmp_path / "embeddings-fake.npz").read_bytes() == before
    hits = rebuilt.search("성적우수 장학금 신청")
    assert [h.doc.id for h in hits if h.doc.group == "notice"] == ["notice-CRAWLED:7"]


def test_notice_sync_swaps_the_index_and_answers_from_the_notice(index):
    svc = Service(index, fake_model(QUESTION), fake_model("학생지원팀 공지 기준 11월 23일 18시까지예요 [1]"))
    with TestClient(create_app(svc), headers={"X-Internal-Token": TOKEN}) as c:
        assert c.put("/v1/notices", json={"notices": [SCHOLARSHIP]}, headers={"X-Internal-Token": "x"}).status_code == 401
        r = c.put("/v1/notices", json={"notices": [SCHOLARSHIP]})
        assert r.status_code == 202 and r.json() == {"received": 1, "changed": True}
        svc.notice_worker.join(10)
        assert c.get("/readyz").json()["notices"] == 1
        # 같은 목록을 다시 보내면 색인을 다시 만들지 않음(BE가 10분마다 전체를 보냄)
        assert c.put("/v1/notices", json={"notices": [SCHOLARSHIP]}).json()["changed"] is False
        body = c.post("/v1/chat", json={"question": "성적우수 장학금 언제까지 신청해요?"}).text
        sources = json.loads(body.split("event: sources\ndata: ", 1)[1].split("\n", 1)[0])
        assert any(s["kind"] == "notice" and s["url"] == SCHOLARSHIP["url"] for s in sources)
        # 빈 목록이 오면 공지가 빠짐(마감이 지나 BE 목록에서 빠진 경우)
        assert c.put("/v1/notices", json={"notices": []}).json()["changed"] is True
        svc.notice_worker.join(10)
        assert c.get("/readyz").json()["notices"] == 0
