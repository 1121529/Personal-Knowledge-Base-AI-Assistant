"""
embedding.py

功能：
1. 載入 Embedding Model
2. 將 Chunk 轉換成向量
3. 支援 Query Embedding
4. 保留原始 Metadata

Author: Team RAG
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np


class EmbeddingService:

    def __init__(
        self,
        model_name="BAAI/bge-m3"
    ):
        """
        初始化 Embedding Model
        """

        print("Loading Embedding Model...")

        self.model = SentenceTransformer(
            model_name
        )

        print("Model Loaded.")

    def embed_text(
        self,
        text: str
    ) -> np.ndarray:
        """
        單一文字轉向量
        """

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:
        """
        問題轉向量
        """

        return self.embed_text(query)

    def embed_documents(
        self,
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        將 Chunk 清單轉向量
        """

        results = []

        for chunk in chunks:

            vector = self.embed_text(
                chunk["content"]
            )

            results.append({

                "chunk_id":
                chunk["chunk_id"],

                "source":
                chunk["source"],

                "page":
                chunk["page"],

                "content":
                chunk["content"],

                "embedding":
                vector.tolist()
            })

        return results


if __name__ == "__main__":

    sample_chunks = [

        {
            "chunk_id": 0,
            "source": "numpy.pdf",
            "page": 1,
            "content":
            "NumPy 是 Python 的數值運算函式庫"
        },

        {
            "chunk_id": 1,
            "source": "numpy.pdf",
            "page": 2,
            "content":
            "reshape() 可以改變陣列形狀"
        }

    ]

    embedder = EmbeddingService()

    embedded_chunks = (
        embedder.embed_documents(
            sample_chunks
        )
    )

    print()

    print(
        embedded_chunks[0]["source"]
    )

    print(
        embedded_chunks[0]["page"]
    )

    print(
        len(
            embedded_chunks[0]["embedding"]
        )
    )
