import requests


class LLMService:

    def __init__(
        self,
        model_name="gemma3",
        base_url="http://localhost:11434/api/chat"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str) -> str:

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=600
            )

            response.raise_for_status()

            result = response.json()

            # chat API 回傳格式
            return result["message"]["content"]

        except requests.exceptions.ConnectionError:
            return "無法連線到 Ollama，請確認 ollama serve 是否已啟動"

        except Exception as e:
            return f"發生錯誤：{str(e)}"
