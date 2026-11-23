import unittest

from local_model_dock.selection import Candidate, select


def candidate(**changes):
    values = dict(name="model", family="qwen", moe=True, compatible=True,
                  accessible=True, licensed=True, memory_bytes=10, disk_bytes=10,
                  model_revision="sha", backend_version="1", tested_revision="sha",
                  tested_backend="1", tool_test_passed=True, coding_score=1)
    values.update(changes)
    return Candidate(**values)


class SelectionTests(unittest.TestCase):
    def test_unknown_resources_and_moe_never_recommend(self):
        for change in ({"moe": None}, {"memory_bytes": None}, {"disk_bytes": None},
                       {"licensed": None}, {"accessible": None}, {"compatible": None}):
            with self.subTest(change=change):
                self.assertEqual(select([candidate(**change)], 100, 100, ("qwen",)), [])

    def test_exact_model_and_backend_validation_required(self):
        for change in ({"tested_revision": "old"}, {"tested_backend": "old"},
                       {"tool_test_passed": False}):
            with self.subTest(change=change):
                self.assertEqual(select([candidate(**change)], 100, 100, ("qwen",)), [])

    def test_family_priority_and_fallback(self):
        models = [candidate(name="fallback", family="other", coding_score=10),
                  candidate(name="q", coding_score=2),
                  candidate(name="d", family="deepseek", coding_score=0)]
        self.assertEqual([m.name for m in select(models, 100, 100, ("deepseek", "qwen"))],
                         ["d", "q", "fallback"])

    def test_insufficient_or_unknown_machine_resources(self):
        for memory, disk in ((None, 100), (100, None), (9, 100), (100, 9)):
            self.assertEqual(select([candidate()], memory, disk, ("qwen",)), [])
