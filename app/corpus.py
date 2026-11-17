"""data/의 홈페이지 안내, 규정, 연락처를 검색 단위 문서로 읽음."""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REGULATION_KINDS = ("article", "attachment", "addenda")


@dataclass(frozen=True)
class Doc:
    id: str
    kind: str  # site | article | attachment | addenda | contact
    title: str  # 출처 목록에 보일 이름(예: 학사지원 > 학적변동 > 휴학, 가톨릭대학교 학칙 제21조(휴학))
    text: str  # 임베딩과 프롬프트에 쓰는 본문
    url: str | None = None
    department: str | None = None
    phone: str | None = None
    campus: str | None = None

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()

    @property
    def group(self) -> str:
        """검색 몫을 나누는 묶음: site, regulation, contact"""
        return "regulation" if self.kind in REGULATION_KINDS else self.kind


@dataclass(frozen=True)
class Alias:
    old: str
    new: str  # 비어 있으면 지금 부서를 확인할 수 없음
    result: str
    phone: str


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:  # 엑셀용 BOM 포함
        return list(csv.DictReader(f))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_docs(data_dir: Path) -> list[Doc]:
    docs = [Doc(r["id"], "site", r["title"], r["text"], r["url"], r["department"], r["phone"], "성심")
            for r in _jsonl(data_dir / "site_pages.jsonl")]
    for r in _jsonl(data_dir / "regulations_chunks.jsonl"):
        # 본문 첫 줄 "[규정명 제N조(제목)]"을 출처 이름으로 씀
        docs.append(Doc(r["id"], r["type"], r["text"].split("\n", 1)[0].strip("[]"), r["text"], r["source_url"]))
    for i, r in enumerate(_csv(data_dir / "contacts.csv")):
        work = r["담당업무"].strip() or "담당업무 표기 없음"
        docs.append(Doc(f"contact-{i}", "contact", f"{r['교정']} {r['부서']}",
                        f"[연락처] {r['교정']} {r['부서']}, 담당업무: {work}, 전화 {r['전화번호']}",
                        department=r["부서"], phone=r["전화번호"], campus=r["교정"]))
    return docs


def load_aliases(data_dir: Path) -> list[Alias]:
    return [Alias(r["옛 부서명(주요연락처 표)"], r["현재 부서"], r["대조 결과"], r["번호"])
            for r in _csv(data_dir / "department_aliases.csv")]
