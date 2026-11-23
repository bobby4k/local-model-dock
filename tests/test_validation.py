import unittest

from local_model_dock.validation import detect_moe, validate_tool_call


class ValidationTests(unittest.TestCase):
    def test_qwen2_moe_requires_valid_expert_structure(self):
        config = {"model_type": "qwen2_moe", "num_experts": 60, "num_experts_per_tok": 4}
        self.assertTrue(detect_moe(config))
        for changes in ({"num_experts": None}, {"num_experts_per_tok": None},
                        {"num_experts": True}, {"num_experts_per_tok": True},
                        {"num_experts": 1}, {"num_experts_per_tok": 0},
                        {"num_experts_per_tok": 61}, {"num_experts": "60"}):
            with self.subTest(changes=changes):
                self.assertIsNone(detect_moe({**config, **changes}))

    def test_qwen3_moe_requires_expert_counts(self):
        self.assertTrue(detect_moe({"model_type": "qwen3_moe", "num_experts": 128,
                                    "num_experts_per_tok": 8}))
        self.assertIsNone(detect_moe({"model_type": "qwen3_moe", "num_experts": 128}))

    def test_moe_requires_known_structure_and_consistent_expert_counts(self):
        self.assertTrue(detect_moe({"model_type": "mixtral", "num_local_experts": 8,
                                    "num_experts_per_tok": 2}))
        for config in ({}, {"model_type": "mixtral", "num_local_experts": 8},
                       {"model_type": "llama", "num_local_experts": 8, "num_experts_per_tok": 2},
                       {"model_type": "mixtral", "num_local_experts": True, "num_experts_per_tok": 1},
                       {"model_type": "mixtral", "num_local_experts": 8, "num_experts_per_tok": 9}):
            with self.subTest(config=config):
                self.assertIsNone(detect_moe(config))

    def test_tool_call_must_match_probe_and_versions(self):
        def respond(request):
            self.assertEqual(request["tools"][0]["function"]["name"], "lookup")
            return {"choices": [{"message": {"tool_calls": [
                {"type": "function", "function": {"name": "lookup",
                                                    "arguments": '{"query":"dock"}'}}]}}]}

        record = validate_tool_call(respond, "a" * 40, "ollama", "0.1.0")
        self.assertTrue(record.passed)
        self.assertTrue(record.matches("a" * 40, "ollama", "0.1.0"))
        self.assertFalse(record.matches("b" * 40, "ollama", "0.1.0"))
        self.assertFalse(record.matches("a" * 40, "omlx", "0.1.0"))
        self.assertFalse(record.matches("a" * 40, "ollama", "0.2.0"))

    def test_invalid_identity_and_responses_fail_closed(self):
        valid = {"choices": [{"message": {"tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": '{"query":"dock"}'}}]}}]}
        bad = [None, {}, {"choices": []},
               {"choices": [{"message": {"tool_calls": [
                   {"type": "function", "function": {"name": "wrong", "arguments": '{"query":"dock"}'}}]}}]},
               {"choices": [{"message": {"tool_calls": [
                   {"type": "function", "function": {"name": "lookup", "arguments": '{"query": NaN}'}}]}}]},
               {"choices": [{"message": {"tool_calls": [
                   {"type": "function", "function": {"name": "lookup", "arguments": '{"query":"other"}'}}]}}]}]
        for response in bad:
            with self.subTest(response=response):
                self.assertFalse(validate_tool_call(lambda _: response, "a" * 40, "ollama", "1").passed)
        for revision, backend, version in (("main", "ollama", "1"), ("a" * 40, "", "1"),
                                           ("a" * 40, "ollama", "")):
            with self.subTest(revision=revision, backend=backend, version=version):
                self.assertFalse(validate_tool_call(lambda _: valid, revision, backend, version).passed)

    def test_request_error_fails_closed(self):
        def broken(_):
            raise OSError("offline")
        self.assertFalse(validate_tool_call(broken, "a" * 40, "ollama", "1").passed)
