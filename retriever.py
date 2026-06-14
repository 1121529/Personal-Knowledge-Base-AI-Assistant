"""
retriever.py

功能：
1. 問題向量化
2. FAISS 相似度搜尋
3. 回傳 Top-K 結果

Author: Team RAG
"""

import re
import faiss
import numpy as np

from embedding import EmbeddingService
from vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def search(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        搜尋最相關 Chunk

        Parameters
        ----------
        query : str
            使用者問題

        top_k : int
            回傳筆數

        Returns
        -------
        list
        """

        if self.vector_store.index is None:
            raise ValueError(
                "請先載入 Vector Store"
            )

        page_results = self.search_pages_from_query(
            query
        )

        image_results = self.search_images_from_query(
            query
        )

        # Query → Embedding
        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        query_vector = np.array(
            [query_vector],
            dtype=np.float32
        )

        # 與建立 Index 時一致
        faiss.normalize_L2(
            query_vector
        )

        scores, indices = (
            self.vector_store.index.search(
                query_vector,
                top_k
            )
        )

        semantic_results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            metadata = (
                self.vector_store
                .metadata[idx]
            )

            semantic_results.append({

                "score":
                float(score),

                "chunk_id":
                metadata["chunk_id"],

                "source":
                metadata["source"],

                "page":
                metadata["page"],

                "content":
                metadata["content"],

                "match_type":
                "semantic"

            })

        return self.merge_results(
            page_results,
            image_results,
            semantic_results,
            top_k
        )

    def extract_page_numbers(self, query):
        """
        從問題中抓出頁碼
        """

        patterns = [
            r"第\s*(\d+)\s*頁",
            r"第\s*(\d+)\s*页",
            r"頁\s*(\d+)",
            r"页\s*(\d+)",
            r"p\.?\s*(\d+)",
            r"page\s*(\d+)"
        ]

        pages = []

        for pattern in patterns:

            for match in re.findall(
                pattern,
                query,
                flags=re.IGNORECASE
            ):

                page = int(match)

                if page not in pages:
                    pages.append(page)

        return pages

    def search_pages_from_query(self, query):
        """
        若問題指定頁碼，直接取該頁附近內容
        """

        pages = self.extract_page_numbers(query)

        if not pages:
            return []

        source = (
            self.vector_store
            .infer_source_from_query(query)
        )

        return self.vector_store.get_pages(
            pages,
            source=source,
            page_window=1
        )

    def is_image_query(self, query):
        """
        判斷問題是否在詢問圖片或圖表
        """

        keywords = [
            "圖片",
            "圖表",
            "圖像",
            "照片",
            "截圖",
            "流程圖",
            "架構圖",
            "示意圖",
            "表格",
            "視覺",
            "image",
            "chart",
            "figure",
            "diagram"
        ]

        query_lower = query.lower()

        return any(
            keyword.lower() in query_lower
            for keyword in keywords
        )

    def search_images_from_query(self, query):
        """
        若問題詢問圖片/圖表，優先回傳圖片分析 Chunk
        """

        if not self.is_image_query(query):
            return []

        pages = self.extract_page_numbers(query)

        source = (
            self.vector_store
            .infer_source_from_query(query)
        )

        return self.vector_store.get_image_chunks(
            source=source,
            pages=pages or None,
            limit=12
        )

    def merge_results(
        self,
        page_results,
        image_results,
        semantic_results,
        top_k
    ):
        """
        合併頁碼檢索與語意檢索，頁碼結果優先
        """

        merged = []
        seen = set()

        priority_results = (
            page_results +
            image_results +
            semantic_results
        )

        for result in priority_results:

            key = (
                result["source"],
                result["page"],
                result["chunk_id"]
            )

            if key in seen:
                continue

            merged.append(result)
            seen.add(key)

        minimum_results = max(
            top_k,
            len(page_results) + len(image_results)
        )

        return merged[:minimum_results]


if __name__ == "__main__":

    # 載入向量庫

    store = VectorStore()

    store.load()

    # 載入 Embedding Model

    embedder = EmbeddingService()

    # 建立 Retriever

    retriever = Retriever(
        vector_store=store,
        embedding_service=embedder
    )

    # 查詢

    query = "NumPy有哪些功能？"

    results = retriever.search(
        query=query,
        top_k=3
    )

    print("\n搜尋結果\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print("=" * 50)

        print(f"Top {i}")

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            result["content"][:200]
        )
