import hashlib

import numpy as np

from app import config
from app.corpus import load_docs
from app.index import windows


def test_committed_embeddings_cover_every_search_window():
    """data/를 바꾸고 scripts/build_embeddings.py를 다시 돌리지 않으면 실패함. 운영에서 임베딩을 새로 계산하지 않게 하기 위함"""
    name = config.EMBED_MODEL.split("/")[-1]
    stored = set(np.load(config.DATA_DIR / f"embeddings-{name}.npz")["digests"].tolist())
    needed = {hashlib.sha1(w.encode()).hexdigest() for d in load_docs(config.DATA_DIR) for w in windows(d.text)}
    assert needed - stored == set(), "data/가 바뀌었음. scripts/build_embeddings.py로 임베딩 파일을 다시 만들 것"
