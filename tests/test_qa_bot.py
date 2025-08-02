# tests/test_qa_bot.py

import unittest
from src.qa_bot.bot import QABot
from src.utils.vector_db import VectorDB

class DummyVDB:
    def __init__(self):
        # Return two fake hits
        self.hits = [
            {"metadata": {"section_id":"root.3.1","change_type":"modified"}, "text":"Old vs new..."},
            {"metadata": {"section_id":"root.4","change_type":"added"}, "text":"New section added..."}
        ]
    def query_changes(self, question, top_k):
        return self.hits

class TestQABot(unittest.TestCase):
    def test_answer(self):
        dummy = DummyVDB()
        bot = QABot(vector_db=dummy, llm_model="gpt-3.5-turbo", temperature=0.0)
        ans = bot.answer_question("What changed in section 3.1?")
        self.assertIsInstance(ans, str)
        self.assertTrue(len(ans) > 0)
