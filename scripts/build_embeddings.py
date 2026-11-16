"""data/를 바꾼 뒤 로컬에서 임베딩 파일(data/embeddings-<모델>.npz)을 다시 만듦. 바뀐 창만 계산함.

    .venv/bin/python scripts/build_embeddings.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.corpus import load_aliases, load_docs  # noqa: E402
from app.index import Index  # noqa: E402
from app.models import LocalEmbedder  # noqa: E402

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
started = time.time()
index = Index(load_docs(config.DATA_DIR), load_aliases(config.DATA_DIR), LocalEmbedder(device), config.DATA_DIR)
index.build()
if index.error:
    sys.exit(index.error)
size = index.cache.stat().st_size / 1e6
print(f"{device}: 조각 {len(index.docs)}개, 창 {len(index.vectors)}개, {time.time() - started:.0f}초, {index.cache.name} {size:.1f}MB")
