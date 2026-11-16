from app import config
from app.corpus import load_aliases, load_docs
from app.index import Index
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
    assert [a.old for a in index.aliases_in(["문의: 학사지원팀"])] == ["학사지원팀"]


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


def test_permanent_errors_stop_indexing_without_retrying(tmp_path):
    class BadKey(FakeEmbedder):
        def embed_documents(self, texts, titles):
            self.calls += 1
            raise ValueError("API key not valid")

    embedder = BadKey()
    index = Index(DOCS, ALIASES, embedder, tmp_path)
    index.build()
    assert not index.ready and embedder.calls == 1 and "API key not valid" in index.error
