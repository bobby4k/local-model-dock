import unittest

from local_model_dock.selection import Candidate, select
from local_model_dock.discovery import DiscoveryCandidate


def candidate(**changes):
    values = dict(name="model", family="qwen", moe=True, compatible=True,
                  accessible=True, licensed=True, memory_bytes=10, disk_bytes=10,
                  model_revision="a" * 40, backend_version="1", tested_revision="a" * 40,
                  tested_backend="1", tool_test_passed=True, coding_score=1,
                  backend="ollama", tested_backend_name="ollama",
                  version=(3, 5), total_parameters=9_000_000_000)
    values.update(changes)
    return Candidate(**values)


class SelectionTests(unittest.TestCase):
    def test_fallback_quantization_exactly_matches_real_filename(self):
        from local_model_dock.discovery import DiscoveryCandidate
        from local_model_dock.selection import resource_estimate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None, gguf_files=(
            ("DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf", 400),
            ("model.i1-Q4_K_S.gguf", 100), ("model.Q4_K_M.gguf", 200),
            ("model.Q8_0.gguf", 800)))
        name, ram, disk = resource_estimate(model, "ollama", quantization="i1-Q4_K_M")
        self.assertTrue(name.endswith(".i1-Q4_K_M.gguf"))
        self.assertEqual(ram, 600 + 2 * 1024**3)
        self.assertEqual(disk, 800 + 256 * 1024**2)
        self.assertEqual(resource_estimate(model, "ollama", quantization="Q8_0")[0], "model.Q8_0.gguf")
        self.assertEqual(resource_estimate(model, "ollama", quantization="Q4_K_M")[0], "model.Q4_K_M.gguf")
        for quantization in ("i1-Q5_K_M", "unknown"):
            with self.subTest(quantization=quantization), self.assertRaisesRegex(ValueError, "quantization"):
                resource_estimate(model, "ollama", quantization=quantization)
    def test_regular_verified_candidate_beats_fallback(self):
        fallback = candidate(name="owner/fallback", family="deepseek", version=(4,), moe=False)
        regular = candidate(name="owner/regular", family="qwen", version=(3, 5))
        result = select([fallback, regular], 100, 100, ("deepseek", "qwen"), fallback_repo=fallback.name)
        self.assertEqual([item.name for item in result], [regular.name, fallback.name])
    def test_qwen_minimum_version_and_unknowns(self):
        from local_model_dock.selection import meets_model_limits
        self.assertTrue(meets_model_limits("qwen", (3, 5), 9_000_000_000))
        self.assertFalse(meets_model_limits("qwen", (3, 4), 9_000_000_000))
        self.assertFalse(meets_model_limits("qwen", None, 9_000_000_000))
        self.assertFalse(meets_model_limits("qwen", (3, 5), None))

    def test_explicit_fallback_still_requires_verified_identity(self):
        model = candidate(name="owner/fallback", moe=False, version=(3, 5),
                          total_parameters=9_000_000_000)
        self.assertEqual(select([model], 100, 100, ("qwen",)), [])
        self.assertEqual(select([model], 100, 100, ("qwen",), fallback_repo="owner/fallback"), [model])
        from dataclasses import replace
        self.assertEqual(select([replace(model, tool_test_passed=False)], 100, 100,
                                ("qwen",), fallback_repo="owner/fallback"), [])
    def test_verified_discovery_requires_exact_checked_identity_and_license(self):
        from local_model_dock.selection import assess_verified

        gib = 1024 ** 3
        model = DiscoveryCandidate("owner/Qwen", "a" * 40, "apache-2.0", 1,
                                   moe=True, accessible=True, formats=("mlx",),
                                   disk_bytes=gib, family="qwen", version=(3, 5),
                                   total_parameters=9_000_000_000)
        def check(repo, revision, backend, version):
            return (repo, revision, backend, version) == ("owner/Qwen", "a" * 40, "omlx", "1.0")

        result = assess_verified([model], 10*gib, 10*gib, ("qwen",),
                                 "omlx", "1.0", {"apache-2.0"}, check)
        self.assertEqual([c.name for c in result], ["owner/Qwen"])
        self.assertEqual(assess_verified([model], 10*gib, 10*gib, ("qwen",),
                                         "omlx", "2.0", {"apache-2.0"}, check), [])
        self.assertEqual(assess_verified([model], 10*gib, 10*gib, ("qwen",),
                                         "omlx", "1.0", set(), check), [])

    def test_verified_discovery_fails_closed_for_mismatches_and_untrusted_values(self):
        from dataclasses import replace
        from local_model_dock.selection import assess_verified

        gib = 1024 ** 3
        model = DiscoveryCandidate("owner/Qwen", "a" * 40, "apache-2.0", 1,
                                   moe=True, accessible=True, formats=("mlx", "gguf"),
                                   disk_bytes=gib, gguf_files=(("model-Q4_K_M.gguf", gib),))
        def check(repo, revision, backend, version):
            return (repo, revision, backend, version) == ("owner/Qwen", "a" * 40, "omlx", "1")

        for changed, backend, checker in (
            (replace(model, name="other/Qwen"), "omlx", check),
            (replace(model, revision="b" * 40), "omlx", check),
            (model, "ollama", check),
            (model, "omlx", lambda *args: {"verified": True}),
            (replace(model, license=None), "omlx", check),
            (replace(model, moe=None), "omlx", check),
        ):
            with self.subTest(model=changed, backend=backend, checker=checker):
                self.assertEqual(assess_verified([changed], 10*gib, 10*gib, ("qwen",),
                                                 backend, "1", {"apache-2.0"}, checker), [])

    def test_ranks_platform_artifacts_without_recommending_them(self):
        from local_model_dock.selection import assess_discovered
        gib = 1024 ** 3
        qwen = DiscoveryCandidate("mlx-community/Qwen-MoE", "a" * 40, "apache-2.0", 2,
                                   moe=True, formats=("mlx",), accessible=True, disk_bytes=gib,
                                   family="qwen")
        deepseek = DiscoveryCandidate("owner/DeepSeek-MoE", "b" * 40, "apache-2.0", 1,
                                       moe=True, formats=("mlx",), accessible=True, disk_bytes=gib,
                                       family="deepseek")
        recommendations, pending = assess_discovered([qwen, deepseek], 10*gib, 10*gib,
                                                     ("deepseek", "qwen"), backend="omlx")
        self.assertEqual(recommendations, [])
        self.assertEqual([m.name for m, _ in pending], [deepseek.name, qwen.name])
        self.assertIn("tool-call verification", pending[0][1])
        self.assertNotIn("memory estimate", pending[0][1])

    def test_incompatible_and_unknown_resources_are_not_matched(self):
        from local_model_dock.selection import assess_discovered
        model = DiscoveryCandidate("owner/Qwen", "a" * 40, "apache-2.0", 1, moe=True,
                                   accessible=True, formats=("gguf",),
                                   gguf_files=(("q4.gguf", 1024),))
        _, pending = assess_discovered([model], None, 10000, ("qwen",), backend="ollama")
        self.assertIn("available memory", pending[0][1])
        _, pending = assess_discovered([model], 10**12, 10**12, ("qwen",), backend="omlx")
        self.assertIn("backend format", pending[0][1])

    def test_split_gguf_is_not_counted_as_complete_artifact(self):
        from local_model_dock.selection import resource_estimate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None,
                                   gguf_files=(("model-00001-of-00002.gguf", 100),))
        self.assertEqual(resource_estimate(model, "ollama"), (None, None, None))

    def test_gguf_projector_is_not_selected_as_model(self):
        from local_model_dock.selection import resource_estimate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None,
                                   gguf_files=(("mmproj-Q4_K_M.gguf", 1),
                                               ("model-Q4_K_M.gguf", 1000)))
        self.assertEqual(resource_estimate(model, "ollama")[0], "model-Q4_K_M.gguf")

    def test_unknown_resources_and_moe_never_recommend(self):
        for change in ({"moe": None}, {"memory_bytes": None}, {"disk_bytes": None},
                       {"licensed": None}, {"accessible": None}, {"compatible": None}):
            with self.subTest(change=change):
                self.assertEqual(select([candidate(**change)], 100, 100, ("qwen",)), [])

    def test_exact_model_and_backend_validation_required(self):
        for change in ({"tested_revision": "old"}, {"tested_backend": "old"},
                       {"tool_test_passed": False}, {"tested_backend_name": "omlx"},
                       {"backend": ""}, {"model_revision": "latest",
                                         "tested_revision": "latest"}):
            with self.subTest(change=change):
                self.assertEqual(select([candidate(**change)], 100, 100, ("qwen",)), [])

    def test_family_priority_and_fallback(self):
        models = [candidate(name="fallback", family="other", coding_score=10),
                  candidate(name="q", coding_score=2),
                  candidate(name="d", family="deepseek", version=(4,), coding_score=0)]
        self.assertEqual([m.name for m in select(models, 100, 100, ("deepseek", "qwen"))],
                         ["d", "q", "fallback"])

    def test_insufficient_or_unknown_machine_resources(self):
        for memory, disk in ((None, 100), (100, None), (9, 100), (100, 9)):
            self.assertEqual(select([candidate()], memory, disk, ("qwen",)), [])

    def test_discovery_metadata_is_not_promoted_to_a_recommendation(self):
        from local_model_dock.selection import assess_discovered

        model = DiscoveryCandidate("Qwen/example", "abc", "apache-2.0", 1000)
        recommendations, pending = assess_discovered([model], 1000, 1000, ("qwen",))
        self.assertEqual(recommendations, [])
        self.assertEqual(pending[0][0], model)
        self.assertIn("MoE", pending[0][1])
        self.assertIn("tool-call verification", pending[0][1])
        self.assertIn("backend compatibility", pending[0][1])
        self.assertIn("memory estimate", pending[0][1])
        self.assertIn("disk estimate", pending[0][1])
        self.assertIn("access conditions", pending[0][1])
        self.assertIn("license eligibility", pending[0][1])

    def test_known_discovery_evidence_is_not_reported_as_unknown(self):
        from local_model_dock.selection import assess_discovered

        model = DiscoveryCandidate("Qwen/example", "a" * 40, "apache-2.0", 10,
                                   moe=True, accessible=True)
        recommendations, pending = assess_discovered([model], 1000, 1000, ("qwen",))
        self.assertEqual(recommendations, [])
        self.assertNotIn("MoE", pending[0][1])
        self.assertIn("access conditions", pending[0][1])
        self.assertIn("tool-call verification", pending[0][1])
