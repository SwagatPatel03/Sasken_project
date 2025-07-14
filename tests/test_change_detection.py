import unittest
from src.change_detection.detector import ChangeDetector, ChangeType
from src.parsers.base_parser import DocumentSection

class DummySection(DocumentSection):
    """Quick constructor for tests"""
    def __init__(self, sid, text):
        super().__init__(sid, sid, text, [], [], {})

class TestChangeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ChangeDetector(similarity_threshold=0.8)

    def test_added_removed(self):
        old = DummySection("root", "")
        new = DummySection("root", "")
        old.subsections = [DummySection("root.1", "A")] 
        new.subsections = [DummySection("root.2", "B")]
        changes = self.detector.detect_changes(old, new)
        types = {c.change_type for c in changes}
        self.assertIn(ChangeType.ADDED, types)
        self.assertIn(ChangeType.REMOVED, types)

    def test_modified(self):
        old = DummySection("root.1", "hello world")
        new = DummySection("root.1", "hello brave new world")
        changes = self.detector.detect_changes(old, new)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, ChangeType.MODIFIED)
