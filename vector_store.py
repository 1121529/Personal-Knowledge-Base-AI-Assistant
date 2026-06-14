"""
vector_store.py

功能：
1. 建立 FAISS Index
2. 儲存向量
3. 載入向量
4. 儲存 Metadata

Author: Team RAG
"""

import os
import pickle
import re
import numpy as np
import faiss


class VectorStore:

    def __init__(
        self,
        index_path="vector_db/faiss.index",
        metadata_path="vector_db/metadata.pkl"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []

    def build_index(self, embedded_chunks):
        """
        建立 FAISS Index

        embedded_chunks:
        [
            {
                "chunk_id": 0,
                "source": "...",
                "page": 1,
                "content": "...",
                "embedding": [...]
            }
        ]
        """

        if not embedded_chunks:
            raise ValueError("embedded_chunks 為空")

        vectors = np.array(
            [item["embedding"] for item in embedded_chunks],
            dtype=np.float32
        )

        dimension = vectors.shape[1]

        # Cosine Similarity
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(vectors)

        self.metadata = []

        for item in embedded_chunks:

            self.metadata.append({
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "page": item["page"],
                "content": item["content"]
            })

        print(
            f"建立完成，共 {len(self.metadata)} 個 Chunk"
        )

    def add_documents(
        self,
        embedded_chunks,
        replace_existing=True
    ):
        """
        將新文件增量加入既有 FAISS Index
        """

        if not embedded_chunks:
            raise ValueError("embedded_chunks 為空")

        if self.index is None:
            self.build_index(embedded_chunks)
            return len(embedded_chunks)

        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS Index 與 Metadata 數量不一致")

        sources = {
            item["source"]
            for item in embedded_chunks
        }

        if replace_existing:

            for source in sources:
                self.delete_source(
                    source,
                    save_after_delete=False
                )

        next_chunk_id = (
            max(
                [
                    item["chunk_id"]
                    for item in self.metadata
                ],
                default=-1
            )
            + 1
        )

        vectors = np.array(
            [
                item["embedding"]
                for item in embedded_chunks
            ],
            dtype=np.float32
        )

        faiss.normalize_L2(vectors)

        if self.index.ntotal == 0:
            self.index = faiss.IndexFlatIP(
                vectors.shape[1]
            )
        elif self.index.d != vectors.shape[1]:
            raise ValueError("新文件向量維度與現有資料庫不一致")

        self.index.add(vectors)

        added_count = 0

        for item in embedded_chunks:

            self.metadata.append({
                "chunk_id": next_chunk_id,
                "source": item["source"],
                "page": item["page"],
                "content": item["content"]
            })

            next_chunk_id += 1
            added_count += 1

        return added_count

    def save(self):
        """
        儲存 Index 與 Metadata
        """

        os.makedirs(
            os.path.dirname(self.index_path),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(
            self.metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

        print("Vector DB 已儲存")

    def load(self):
        """
        載入 Index 與 Metadata
        """

        if not os.path.exists(
            self.index_path
        ):
            raise FileNotFoundError(
                "找不到 FAISS Index"
            )

        if not os.path.exists(
            self.metadata_path
        ):
            raise FileNotFoundError(
                "找不到 Metadata"
            )

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.metadata_path,
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        print(
            f"成功載入 {len(self.metadata)} 個 Chunk"
        )

    def get_metadata(self):
        """
        回傳 Metadata
        """

        return self.metadata

    def get_pages(
        self,
        pages,
        source=None,
        page_window=0
    ):
        """
        依頁碼回傳資料庫中的 Chunk
        """

        target_pages = set()

        for page in pages:

            for nearby_page in range(
                page - page_window,
                page + page_window + 1
            ):

                if nearby_page > 0:
                    target_pages.add(nearby_page)

        results = []

        for item in self.metadata:

            if source and item["source"] != source:
                continue

            if item["page"] not in target_pages:
                continue

            results.append({
                "score": 1.0,
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "page": item["page"],
                "content": item["content"],
                "match_type": "page"
            })

        return sorted(
            results,
            key=lambda item: (
                item["source"],
                item["page"],
                item["chunk_id"]
            )
        )

    def get_image_chunks(
        self,
        source=None,
        pages=None,
        limit=12
    ):
        """
        回傳含圖片/圖表分析的 Chunk
        """

        target_pages = set(pages) if pages else None
        results = []

        for item in self.metadata:

            if source and item["source"] != source:
                continue

            if target_pages and item["page"] not in target_pages:
                continue

            if "[圖片與圖表分析]" not in item["content"]:
                continue

            results.append({
                "score": 1.0,
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "page": item["page"],
                "content": item["content"],
                "match_type": "image"
            })

        return sorted(
            results,
            key=lambda item: (
                item["source"],
                item["page"],
                item["chunk_id"]
            )
        )[:limit]

    def infer_source_from_query(self, query):
        """
        若問題中提到檔名，回傳最可能的資料庫來源
        """

        sources = self.list_sources()

        for source in sources:

            if source in query:
                return source

            source_stem = re.sub(
                r"\.[^.]+$",
                "",
                source
            )

            if source_stem and source_stem in query:
                return source

        return None

    def list_sources(self):
        """
        回傳目前資料庫內的檔案名稱
        """

        return sorted(
            {
                item["source"]
                for item in self.metadata
            }
        )

    def get_source_text(self, source, max_chars=18000):
        """
        回傳指定檔案在資料庫中的文字內容
        """

        chunks = [
            item
            for item in self.metadata
            if item["source"] == source
        ]

        chunks = sorted(
            chunks,
            key=lambda item: (
                item["page"],
                item["chunk_id"]
            )
        )

        contents = [
            (
                f"[第 {item['page']} 頁]\n"
                f"{item['content']}"
            )
            for item in chunks
        ]

        if not contents:
            return ""

        if len("\n\n".join(contents)) <= max_chars:
            return "\n\n".join(contents)

        target_parts = min(
            12,
            len(contents)
        )

        selected_indices = []

        for index in range(target_parts):

            selected_index = round(
                index *
                (len(contents) - 1) /
                max(target_parts - 1, 1)
            )

            if selected_index not in selected_indices:
                selected_indices.append(selected_index)

        per_part_chars = max_chars // len(selected_indices)
        text_parts = []

        for index in selected_indices:

            content = contents[index]

            text_parts.append(
                content[:per_part_chars]
            )

        text_parts.append(
            "[提示]\n以上內容為全文前中後段抽樣，請整理整份文件的核心主題。"
        )

        return "\n\n".join(text_parts)

    def delete_source(
        self,
        source,
        save_after_delete=True
    ):
        """
        從向量資料庫刪除指定來源檔案的所有 Chunk
        """

        if self.index is None:
            raise ValueError("請先載入 Vector Store")

        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS Index 與 Metadata 數量不一致")

        remaining_metadata = []
        remaining_vectors = []
        deleted_count = 0
        dimension = self.index.d

        for idx, item in enumerate(self.metadata):

            if item["source"] == source:
                deleted_count += 1
                continue

            remaining_metadata.append(item)
            remaining_vectors.append(
                self.index.reconstruct(idx)
            )

        if deleted_count == 0:
            return 0

        self.index = faiss.IndexFlatIP(
            dimension
        )

        if remaining_vectors:
            vectors = np.array(
                remaining_vectors,
                dtype=np.float32
            )

            faiss.normalize_L2(vectors)

            self.index.add(vectors)

        self.metadata = remaining_metadata

        if save_after_delete:
            self.save()

        return deleted_count


if __name__ == "__main__":

    sample_data = [

        {
            "chunk_id": 0,
            "source": "numpy.pdf",
            "page": 1,
            "content": "NumPy 是數值運算函式庫",
            "embedding": [0.1] * 384
        },

        {
            "chunk_id": 1,
            "source": "numpy.pdf",
            "page": 2,
            "content": "reshape 可以改變形狀",
            "embedding": [0.2] * 384
        }

    ]

    store = VectorStore()

    store.build_index(
        sample_data
    )

    store.save()

    store.load()

    print(
        store.get_metadata()[0]
    )
