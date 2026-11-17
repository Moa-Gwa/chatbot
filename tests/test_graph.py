from langgraph.checkpoint.memory import InMemorySaver

from app.corpus import Doc
from app.graph import build, cited, unverified
from tests.conftest import fake_model

SOURCES = [Doc("s", "site", "증명서 발급", "인터넷 증명발급 신청하기(https://catholic.certpia.com/) 문의: 학생지원팀 02-2164-4732",
               url="https://www.catholic.ac.kr/ko/support/certificate.do")]


def test_links_followed_by_korean_or_punctuation_are_matched_against_sources():
    answer = ("인터넷 발급은 https://catholic.certpia.com/에서 신청해요 [1]. 학생지원팀(02-2164-4732)에 문의하세요. "
              "자세한 내용은 https://www.catholic.ac.kr/ko/support/certificate.do.")
    assert unverified(answer, SOURCES) == []


def test_invented_numbers_and_links_are_reported():
    answer = "02-9999-0000으로 전화하거나 https://fake.example.com/apply[1]에서 신청하세요."
    assert unverified(answer, SOURCES) == ["02-9999-0000", "https://fake.example.com/apply"]


def test_truncated_link_is_reported_but_trailing_slash_is_ignored():
    answer = "https://www.catholic.ac.kr/ko/support 또는 https://catholic.certpia.com 에서 신청하세요."
    assert unverified(answer, SOURCES) == ["https://www.catholic.ac.kr/ko/support"]


def test_only_cited_source_numbers_within_range_are_returned():
    assert cited("학생지원팀에 문의하세요 [1]. 발급 방법 [3][1], 없는 번호 [9]", count=4) == [1, 3]
    assert cited("구매관재팀 [1, 9]. 발급 [2,3]", count=12) == [1, 2, 3, 9]


def test_alternatives_that_are_not_a_list_are_ignored(index):
    index.build()
    analyzer = fake_model('{"interpretation": "휴학 신청", "intent": "procedure", "campus": "성심", '
                          '"clear": false, "alternatives": "군 휴학 신청"}')
    graph = build(index, analyzer, fake_model("답변 [1]"), InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    list(graph.stream({"question": "휴학", "history": []}, config))
    state = graph.get_state(config).values
    # 고를 대안이 없으므로 되묻지 않고 바로 답함
    assert state["alternatives"] == []
    assert state["needs_confirm"] is False
    assert state["answer"] == "답변 [1]"
