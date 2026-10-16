from __future__ import annotations

from pathlib import Path

import chromadb

from .config import ROOT_DIR


class VectorDB:
    def __init__(self, persistence_path: str | Path = ROOT_DIR / "data" / "chroma"):
        path = Path(persistence_path)
        if not path.is_absolute():
            path = ROOT_DIR / path

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection("astra_knowledge")

    def add_devices(self, devices_list):
        ids = [device["id"] for device in devices_list]
        docs = [device["desc"] for device in devices_list]
        self.collection.upsert(ids=ids, documents=docs)

    def search(self, query, n_results: int = 5):
        return self.collection.query(query_texts=[query], n_results=n_results)
