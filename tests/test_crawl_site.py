import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_site  # noqa: E402


def test_merged_cells_repeat_on_every_row():
    table = BeautifulSoup(
        '<table><tr><th>계열</th><th>학과</th><th>학점</th></tr>'
        '<tr><td rowspan="2">인문사회계열</td><td>국어국문학과</td><td rowspan="2">6</td></tr>'
        '<tr><td>철학과</td></tr><tr><td colspan="2">합계</td><td>12</td></tr></table>', "html.parser").table
    assert crawl_site.table_rows(table) == [
        crawl_site.TABLE_HEAD + "계열 | 학과 | 학점", "인문사회계열 | 국어국문학과 | 6", "인문사회계열 | 철학과 | 6", "합계 | 합계 | 12"]


def test_contact_header_omits_missing_values():
    page = {"path": "대학생활 > 학생군사교육단", "tab": None, "title": "", "department": "학생군사교육단", "phone": None,
            "email": None, "url": "u", "updated_on": None, "lines": ["안내"]}
    assert crawl_site.chunks(page, 0)[0]["text"] == "[학교 홈페이지 대학생활 > 학생군사교육단]\n문의: 학생군사교육단\n안내"


def test_relative_links_resolve_against_the_page():
    node = BeautifulSoup('<div><a href="?mode=view&amp;articleNo=1">서식</a></div>', "html.parser").div
    assert crawl_site.to_lines(node, "https://www.catholic.ac.kr/ko/about/various-forms.do") == [
        "서식(https://www.catholic.ac.kr/ko/about/various-forms.do?mode=view&articleNo=1)"]


def test_table_header_repeats_in_continuation_chunks(monkeypatch):
    monkeypatch.setattr(crawl_site, "MAX_CHARS", 80)
    rows = crawl_site.table_rows(BeautifulSoup(
        "<table><tr><th>과목</th><th>학점</th></tr>" + "".join(f"<tr><td>교과목{i:02d}</td><td>3</td></tr>" for i in range(8)) + "</table>",
        "html.parser").table)
    page = {"path": "학사지원 > 교과과정", "tab": None, "title": "", "department": None, "phone": None, "email": None,
            "url": "u", "updated_on": None, "lines": ["안내", *rows]}
    parts = crawl_site.chunks(page, 0)
    assert len(parts) > 1 and all("과목 | 학점" in p["text"] for p in parts[1:])
    assert all(crawl_site.TABLE_HEAD not in p["text"] for p in parts)
