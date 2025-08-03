# src/qa_bot/groq_llm.py

import os
import groq
from typing import List, Dict
from langchain.schema import BaseMessage, HumanMessage

class GroqLLM:
    """
    Minimal chat‑style wrapper around the Groq Python SDK.
    Accepts a list of HumanMessage and returns a dict mimicking LangChain’s ChatResult.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        self.client = groq.Client(api_key=api_key)
        self.model = model_name
        self.temperature = temperature

    def __call__(self, messages: List[BaseMessage], **kwargs) -> Dict:
        # Convert LangChain HumanMessage → Groq format
        groq_msgs = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            else:
                # For simplicity, treat all other as assistant
                role = "assistant"
            groq_msgs.append({"role": role, "content": m.content})

        # Call Groq’s chat API
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=groq_msgs,
            temperature=self.temperature,
            **kwargs
        )
        # groq returns a dict with 'choices' list
        text = resp.choices[0].message.content

        # Build a minimal “LangChain‑style” response
        return {
            "generations": [[{"message": HumanMessage(content=text)}]]
        }
