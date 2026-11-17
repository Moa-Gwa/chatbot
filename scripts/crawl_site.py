"""가톨릭대학교 홈페이지의 학생 안내 페이지를 모아 data/site_pages.jsonl로 저장.

학사지원, 대학생활, 캠퍼스 안내 메뉴에서 시작해 같은 메뉴의 탭 페이지까지 따라감.
게시판 글(mode=)과 홍보, 연혁 등 안내와 무관한 메뉴는 제외. 1초에 한 페이지씩 요청함.

    .venv/bin/python scripts/crawl_site.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString

BASE = "https://www.catholic.ac.kr"
OUT = Path(__file__).resolve().parent.parent / "data" / "site_pages.jsonl"
HEADERS = {"User-Agent": "MoaGwaChatbotCrawler/0.1 (+https://moagwa.zruor.cc)"}
MAX_CHARS = 1500  # 규정 청크와 같은 기준으로 자름
TABLE_HEAD = "\u0001"  # 표 머리줄 표시. 청크를 나눌 때 이어지는 조각에 머리줄을 다시 붙이는 데 씀

# 학생이 묻는 업무 안내가 있는 메뉴만 수집
ALLOW = re.compile(r"^/ko/(support/|campuslife/|certificate_domestic\.do$|service/(safety|webmail_notice)\.do$|about/("
                   r"phone_book|campus-map|campus-map-info|location_\w+|various-forms[\w-]*|rule|univ_bulletin|brochure_rule|organization)\.do$)")
DENY = re.compile(r"/notice[\w]*\.do$|/campuslife/(media|club|institutional_club\d*|CUBIG|catchup|supporters|cat-cert)\.do$")


# 학번별 교과과정, 졸업학점 페이지는 재학 가능성이 있는 학번만 수집
# shortcut: 기준 학번을 고정함, 매년 초 MIN_YEAR를 올리거나 현재 연도 기준으로 바꿀 것
MIN_YEAR = 2019
YEAR_PAGE = re.compile(r"_(\d{4})(?:-(\d{4}))?\.do$")


ROBOTS = urllib.robotparser.RobotFileParser(BASE + "/robots.txt")


def fetch(url: str) -> str:
    """요청 한도 초과, 서버 오류는 세 번까지 다시 시도. 그 밖의 오류는 바로 올림"""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def allowed(href: str) -> str | None:
    """사이트 내부 안내 페이지면 경로를 돌려줌."""
    u = urlparse(urljoin(BASE, href))
    if u.netloc not in ("www.catholic.ac.kr", "") or u.query or not u.path.endswith(".do"):
        return None
    year = YEAR_PAGE.search(u.path)
    if year and int(year.group(2) or year.group(1)) < MIN_YEAR:
        return None
    if not ALLOW.match(u.path) or DENY.search(u.path) or not ROBOTS.can_fetch(HEADERS["User-Agent"], BASE + u.path):
        return None
    return u.path


def cell_text(node) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def span(cell, name: str) -> int:
    value = re.match(r"\d+", str(cell.get(name, "1")))
    return max(1, int(value.group(0))) if value else 1


def table_rows(table) -> list[str]:
    """합친 칸(rowspan, colspan)의 값을 행과 열마다 되풀이해 한 행만 읽어도 뜻이 통하게 함"""
    carry: dict[int, tuple[int, str]] = {}  # 열 번호 → (아래로 남은 행 수, 값)
    rows = []
    for tr in table.find_all("tr"):
        cells: list[str] = []

        def fill_carried() -> None:
            while len(cells) in carry:
                left, value = carry.pop(len(cells))
                if left > 1:
                    carry[len(cells)] = (left - 1, value)
                cells.append(value)

        for cell in tr.find_all(["th", "td"]):
            fill_carried()
            value = cell_text(cell)
            for _ in range(span(cell, "colspan")):
                if span(cell, "rowspan") > 1:
                    carry[len(cells)] = (span(cell, "rowspan") - 1, value)
                cells.append(value)
        fill_carried()
        rows.append(" | ".join(cells))
    return [TABLE_HEAD + rows[0], *rows[1:]] if rows else rows


def to_lines(node, page_url: str) -> list[str]:
    """본문을 줄 목록으로. 표는 한 행을 '칸 | 칸' 한 줄로, 링크는 '글자(주소)'로."""
    for a in node.find_all("a", href=True):
        href = urljoin(page_url, a["href"])  # ?mode= 같은 상대 주소는 현재 페이지 기준
        label = cell_text(a)
        if label and not href.startswith("javascript"):
            a.replace_with(NavigableString(f"{label}({href})"))
    for table in node.find_all("table"):
        rows = table_rows(table)
        table.replace_with(NavigableString("\n" + "\n".join(r for r in rows if r.strip(" |")) + "\n"))
    for br in node.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for block in node.find_all(["p", "li", "h2", "h3", "h4", "h5", "dt", "dd", "div"]):
        block.insert_after(NavigableString("\n"))
    return [" ".join(line.split()) for line in node.get_text().split("\n") if line.strip()]


def parse(path: str, html: str) -> tuple[dict | None, set[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = {p for a in soup.select(".tab-box a[href], .content a[href]") if (p := allowed(a["href"]))}
    content = soup.select_one(".content-wrap .content")
    if content is None:
        return None, links
    for junk in content.select("script, style, .fxd-btn-wrap, .gtx-trans-icon"):
        junk.decompose()
    # 게시판(유실물, 서식 목록 등)은 글이 계속 바뀌고 작성자명이 섞여 있어 목록은 빼고 위치만 안내
    boards = content.select(".co-board")
    for board in boards:
        board.decompose()
    crumbs = [cell_text(a) for a in soup.select(".path-box a.path-selected")]
    title = cell_text(soup.select_one(".page-title h3") or soup.title)
    tab = soup.select_one(".tab-box li.active a")
    info = soup.select_one(".sub-info-box")
    dept = info.select_one(".info-tit .name") if info else None
    phone = info.select_one("a.tel") if info else None
    email = info.select_one("a.mail") if info else None
    updated = re.search(r"([\d]{4})\.\s*(\d{2})\.\s*(\d{2})", cell_text(info.select_one(".info-date-box"))) if info and info.select_one(".info-date-box") else None
    page = {
        "url": BASE + path,
        "path": " > ".join(dict.fromkeys(c for c in crumbs if c and c != "홈")),
        "title": title,
        "tab": cell_text(tab) if tab else None,
        "department": cell_text(dept) if dept else None,
        "phone": cell_text(phone) if phone else None,
        "email": cell_text(email) if email else None,
        "updated_on": "-".join(updated.groups()) if updated else None,
        "lines": to_lines(content, BASE + path) + ([f"게시판 목록은 홈페이지에서 확인({BASE + path})"] if boards else []),
    }
    return page, links


def chunks(page: dict, index: int) -> list[dict]:
    path, tab = page["path"], page["tab"]
    name = (path if tab and path.endswith(tab) else " > ".join(x for x in [path, tab] if x)) or page["title"]
    contact = " ".join(x for x in ["문의:", page["department"], page["phone"], page["email"]] if x) if page["department"] else ""
    head = f"[학교 홈페이지 {name}]" + (f"\n{contact}" if contact else "")
    parts, current, header = [], [], None
    for raw in page["lines"]:
        line = raw.lstrip(TABLE_HEAD)
        if raw.startswith(TABLE_HEAD):
            header = line
        elif " | " not in line:
            header = None
        if current and len("\n".join(current + [line])) > MAX_CHARS - len(head) - 10:
            parts.append(current)
            # 표 중간에서 나뉘면 열 이름을 알 수 있게 머리줄을 다시 붙임
            current = [header] if header and line != header else []
        current.append(line)
    if current:
        parts.append(current)
    meta = {k: page[k] for k in ("url", "department", "phone", "email", "updated_on")}
    return [{"id": f"site{index}-{i}", "type": "site", "title": name, "part": f"{i + 1}/{len(parts)}",
             "text": head + (f" ({i + 1}/{len(parts)})" if len(parts) > 1 else "") + "\n" + "\n".join(p), **meta}
            for i, p in enumerate(parts)]


def main() -> None:
    ROBOTS.read()
    seeds = {p for a in BeautifulSoup(fetch(BASE + "/ko/index.do"), "html.parser").select("a[href]") if (p := allowed(a["href"]))}
    queue, seen, pages, failed = deque(sorted(seeds)), set(seeds), [], []
    while queue:
        path = queue.popleft()
        time.sleep(1)
        try:
            page, links = parse(path, fetch(BASE + path))
        except urllib.error.HTTPError as e:
            if e.code not in (404, 410):  # 404, 410만 메뉴에 남은 죽은 링크로 보고 건너뜀
                failed.append(path)
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            failed.append(path)
            print(f"fail {path}: {e}", file=sys.stderr)
            continue
        for link in sorted(links - seen):
            seen.add(link)
            queue.append(link)
        if page and page["lines"]:
            pages.append(page)
            print(f"{len(pages):3d} {path} {page['department'] or '-'} {len(page['lines'])} lines", file=sys.stderr)
    if failed:
        # 일부만 모인 결과로 기존 데이터를 덮어쓰지 않음
        sys.exit(f"{len(failed)}개 페이지 실패, {OUT.name}을 바꾸지 않음: {failed}")
    rows = [c for i, p in enumerate(pages) for c in chunks(p, i)]
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(pages)} pages, {len(rows)} chunks → {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
