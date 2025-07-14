# tests/test_docx_parser.py
import unittest
from src.parsers.docx_parser import DOCXParser

class TestDOCXParser(unittest.TestCase):
    def setUp(self):
        self.parser = DOCXParser()
        self.sample = r"data\raw\24301-af0.docx"

    def test_parse_structure(self):
        root = self.parser.parse(self.sample)
        # at least one top-level section
        self.assertGreater(len(root.subsections), 0)
        # ensure tables is a list
        self.assertIsInstance(root.tables, list)
        # ensure content is a string
        self.assertIsInstance(root.content, str)
