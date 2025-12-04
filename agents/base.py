from typing import Any
from dataclasses import dataclass
from langchain_openai import ChatOpenAI 
from config.settings import OPENAI_API_KEY, LLM_MODEL_NAME

@dataclass
class BaseAgent:
    name: str

    def __post_init__(self): 
        #runs after dataclass initialization
        # gets auto generated __init__(name: str)
        # allows additional setup after init
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your environment."
            )

        self.llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY,
        )

    def call_llm(self, messages: list[dict]) -> str:
        resp = self.llm.invoke(messages)
        return resp.content
