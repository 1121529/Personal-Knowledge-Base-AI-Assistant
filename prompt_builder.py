"""
prompt_builder.py

功能：
1. 將檢索結果組成 Prompt
2. 提供給 LLM 使用
3. 保留來源資訊

Author: Team RAG
"""

from typing import List, Dict


class PromptBuilder:

    def __init__(self):
        self.system_prompt = """
你是一個專業的知識庫AI助手。

請根據提供的文件內容回答問題。

規則：
1. 只能根據文件內容回答。
2. 若文件中沒有答案，請明確回答：
   「文件中沒有足夠資訊回答此問題。」
3. 不得自行編造資訊。
4. 回答請使用繁體中文。
5. 盡量條列式整理重點。
6. 若使用者指定頁碼，請優先根據該頁碼及相鄰頁面的內容回答。
7. 只要提供的參考資料中有相關頁碼內容，就不要回答「文件中沒有足夠資訊」。
8. 若參考資料中含有「圖片與圖表分析」，請把它視為文件圖片內容的描述。
"""

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict]
    ) -> str:
        """
        建立完整 Prompt

        Parameters
        ----------
        query : str
            使用者問題

        retrieved_docs : List[Dict]
            retriever.py 回傳結果

        Returns
        -------
        str
        """

        context_parts = []

        for i, doc in enumerate(retrieved_docs, start=1):

            source = doc["source"]
            page = doc["page"]
            content = doc["content"]
            match_type = doc.get(
                "match_type",
                "semantic"
            )

            if match_type == "page":
                match_note = "頁碼指定匹配"
            elif match_type == "image":
                match_note = "圖片/圖表分析匹配"
            else:
                match_note = "語意檢索匹配"

            context_parts.append(
                f"""
文件 {i}
來源：{source}
頁碼：{page}
匹配方式：{match_note}

內容：
{content}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
{self.system_prompt}

====================
參考資料
====================

{context}

====================
使用者問題
====================

{query}

====================
請根據上述資料回答
====================
"""

        return prompt

    def build_citation(
        self,
        retrieved_docs: List[Dict]
    ) -> str:
        """
        產生引用來源
        """

        citations = []

        seen = set()

        for doc in retrieved_docs:

            key = (
                doc["source"],
                doc["page"]
            )

            if key not in seen:

                citations.append(
                    f"{doc['source']} 第 {doc['page']} 頁"
                )

                seen.add(key)

        return "\n".join(citations)


if __name__ == "__main__":

    sample_results = [

        {
            "score": 0.92,
            "source": "numpy.pdf",
            "page": 3,
            "content":
            "NumPy 是 Python 的數值運算函式庫，支援多維陣列。"
        },

        {
            "score": 0.88,
            "source": "numpy.pdf",
            "page": 5,
            "content":
            "reshape() 可用於改變陣列形狀。"
        }

    ]

    builder = PromptBuilder()

    prompt = builder.build_prompt(
        query="NumPy有哪些功能？",
        retrieved_docs=sample_results
    )

    print(prompt)

    print("\n來源資訊：")
    print(
        builder.build_citation(
            sample_results
        )
    )
