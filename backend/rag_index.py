import json
import os
import shutil
import tempfile
from typing import List, Tuple, Any

import faiss
import numpy as np


def _has_non_ascii(path: str) -> bool:
    return any(ord(c) > 127 for c in path)


def write_faiss_index(index: faiss.Index, path: str) -> None:
    """Save FAISS index; on Windows, non-ASCII paths need a temp-file workaround."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.name == "nt" and _has_non_ascii(path):
        fd, tmp_path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            faiss.write_index(index, tmp_path)
            shutil.copy2(tmp_path, path)
        finally:
            os.remove(tmp_path)
    else:
        faiss.write_index(index, path)


def read_faiss_index(path: str) -> faiss.Index:
    """Load FAISS index; on Windows, non-ASCII paths need a temp-file workaround."""
    path = os.path.abspath(path)
    if os.name == "nt" and _has_non_ascii(path):
        fd, tmp_path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            shutil.copy2(path, tmp_path)
            return faiss.read_index(tmp_path)
        finally:
            os.remove(tmp_path)
    return faiss.read_index(path)


def load_index(index_path: str, meta_path: str) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise RuntimeError(
            "FAISS index or metadata not found. "
            "Run `python backend/build_index.py` first to build the RAG index."
        )

    index = read_faiss_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return index, metadata


def search_similar(
    index: faiss.IndexFlatL2,
    metadata: np.ndarray,
    query_vec: np.ndarray,
    k: int = 3,
) -> List[Any]:
    distances, indices = index.search(query_vec, k)
    idxs = indices[0]
    results = []
    for i in idxs:
        if 0 <= i < len(metadata):
            results.append(metadata[i])
    return results


def load_faq_data(path: str):
    """Загружает FAQ данные из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


