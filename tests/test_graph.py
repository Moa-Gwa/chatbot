from app.corpus import Doc
from app.graph import cited, unverified

SOURCES = [Doc("s", "site", "증명서 발급", "인터넷 증명발급 신청하기(https://catholic.certpia.com/) 문의: 학생지원팀 02-2164-4732",
               url="https://www.catholic.ac.kr/ko/support/certificate.do")]


def test_links_followed_by_korean_or_punctuation_are_matched_against_sources():
    answer = ("인터넷 발급은 https://catholic.certpia.com/에서 신청해요 [1]. 학생지원팀(02-2164-4732)에 문의하세요. "
              "자세한 내용은 https://www.catholic.ac.kr/ko/support/certificate.do.")
    assert unverified(answer, SOURCES) == []


def test_invented_numbers_and_links_are_reported():
    answer = "02-9999-0000으로 전화하거나 https://fake.example.com/apply[1]에서 신청하세요."
    assert unverified(answer, SOURCES) == ["02-9999-0000", "https://fake.example.com/apply"]


def test_only_cited_source_numbers_within_range_are_returned():
    assert cited("학생지원팀에 문의하세요 [1]. 발급 방법 [3][1], 없는 번호 [9]", count=4) == [1, 3]
    assert cited("구매관재팀 [1, 9]. 발급 [2,3]", count=12) == [1, 2, 3, 9]
