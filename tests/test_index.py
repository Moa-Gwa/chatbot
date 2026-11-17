import numpy as np

from app import config
from app.corpus import load_aliases, load_docs
from app.corpus import Doc
from app.index import WINDOW_CHARS, Index, windows
from tests.conftest import ALIASES, DOCS, FakeEmbedder


def test_real_data_loads_every_source():
    docs = load_docs(config.DATA_DIR)
    groups = {g: sum(d.group == g for d in docs) for g in ("site", "regulation", "contact")}
    assert groups == {"site": 308, "regulation": 9131, "contact": 620}
    assert len({d.id for d in docs}) == len(docs)
    assert len(load_aliases(config.DATA_DIR)) == 37


def test_search_prefers_matching_guide_and_keeps_group_quotas(index):
    hits = index.search("휴학 신청 방법")
    assert hits[0].doc.id == "site0-0"
    assert sum(h.doc.group == "site" for h in hits) <= 4
    assert {h.doc.group for h in hits} == {"site", "regulation", "contact"}


def test_addenda_only_when_asked_and_contacts_filtered_by_campus(index):
    assert all(h.doc.kind != "addenda" for h in index.search("휴학 기간"))
    assert any(h.doc.kind == "addenda" for h in index.search("휴학 부칙 시행일"))
    assert {h.doc.campus for h in index.search("학적 문의", campus="성의") if h.doc.group == "contact"} == {"성의"}


def test_old_department_names_map_to_current_contacts(index):
    hits = [h for h in index.search("군휴학 입영통지서") if h.doc.id == "site2-0"]
    assert [c.id for c in index.contacts_for(hits)] == ["contact-0"]
    assert index.annotate("학사지원팀에 방문 제출") == "학사지원팀(지금 문의처: 학사행정팀)에 방문 제출"
    assert index.annotate("학사행정팀에 문의") == "학사행정팀에 문의"


def test_cache_reuses_vectors_and_only_embeds_changed_docs(tmp_path):
    first = FakeEmbedder()
    Index(DOCS, ALIASES, first, tmp_path).build()
    again = FakeEmbedder()
    Index(DOCS, ALIASES, again, tmp_path).build()
    assert first.calls == 1 and again.calls == 0
    changed = FakeEmbedder()
    edited = [DOCS[0].__class__(**{**DOCS[0].__dict__, "text": DOCS[0].text + " 추가"}), *DOCS[1:]]
    Index(edited, ALIASES, changed, tmp_path).build()
    assert changed.calls == 1
    # 바뀌기 전 문서의 벡터는 캐시에서 빠짐
    assert len(np.load(tmp_path / "embeddings-fake.npz")["digests"]) == len(DOCS)


def test_permanent_errors_stop_indexing_without_retrying(tmp_path):
    class BadKey(FakeEmbedder):
        def embed_documents(self, texts):
            self.calls += 1
            raise ValueError("API key not valid")

    embedder = BadKey()
    index = Index(DOCS, ALIASES, embedder, tmp_path)
    index.build()
    assert not index.ready and embedder.calls == 1 and "API key not valid" in index.error


def test_long_chunks_are_embedded_in_windows_with_the_header_repeated():
    text = "[학교 홈페이지 학사지원 > 장학]\n" + "\n".join(f"{i}번째 안내 문장 " * 6 for i in range(40)) + "\n국가장학금 신청은 한국장학재단에서"
    parts = windows(text)
    assert len(parts) > 1
    assert all(len(p) <= WINDOW_CHARS and p.startswith("[학교 홈페이지 학사지원 > 장학]\n") for p in parts)
    assert "국가장학금" in parts[-1]


def test_content_at_the_end_of_a_long_chunk_is_still_found(tmp_path):
    long = Doc("site9-0", "site", "학사지원 > 장학", "[학교 홈페이지 학사지원 > 장학]\n" + "\n".join(
        f"교내 장학 안내 {i}" * 8 for i in range(30)) + "\n국가장학금 신청은 한국장학재단 홈페이지에서", None, "학생지원팀", None, "성심")
    index = Index([long, *DOCS], ALIASES, FakeEmbedder(), tmp_path)
    index.build()
    assert index.search("국가장학금 한국장학재단")[0].doc.id == "site9-0"


def test_department_with_work_description_still_finds_its_contact(tmp_path):
    page = Doc("site8-0", "site", "학사지원 > 장학제도", "[학교 홈페이지 장학]\n문의: 학생지원팀 교내장학 / 국가장학\n장학금 안내",
               None, "학생지원팀 교내장학 / 국가장학", None, "성심")
    index = Index([page, *DOCS], ALIASES, FakeEmbedder(), tmp_path)
    index.build()
    hits = [h for h in index.search("장학금 안내") if h.doc.id == "site8-0"]
    assert [c.id for c in index.contacts_for(hits)] == ["contact-1"]


def test_annotation_keeps_the_old_name_when_a_unit_moved_into_another_team(tmp_path):
    from app.corpus import Alias
    index = Index(DOCS, [Alias("보건실", "학생지원팀", "번호가 다른 부서로 옮겨감", "4126")], FakeEmbedder(), tmp_path)
    assert index.annotate("보건실 위치는 니콜스관") == "보건실(지금 문의처: 학생지원팀) 위치는 니콜스관"
