"""
chunker.py

功能：
1. 將文件內容切成 Chunk
2. 保留來源與頁碼
3. 支援 Chunk Overlap
"""

from typing import List, Dict


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        將單一文字切塊
        """

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunks.append(text[start:end])

            start += (
                self.chunk_size -
                self.chunk_overlap
            )

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        documents:
        [
            {
                "source": "...",
                "page": 1,
                "content": "..."
            }
        ]
        """

        results = []

        chunk_id = 0

        for doc in documents:

            source = doc["source"]
            page = doc["page"]
            content = doc["content"]

            chunks = self.split_text(content)

            for chunk in chunks:

                results.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                    "content": chunk
                })

                chunk_id += 1

        return results


if __name__ == "__main__":

    sample_docs = [
        {
            "source": "numpy.pdf",
            "page": 1,
            "content":
            "NumPy 是 Python 的數值運算函式庫。"
            * 100
        }
    ]

    chunker = TextChunker(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = chunker.chunk_documents(
        sample_docs
    )

    print(f"Chunk數量: {len(chunks)}")

    print("\n第一個Chunk\n")

    print(chunks[0])
