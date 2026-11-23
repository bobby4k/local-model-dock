import json
import unittest
import urllib.error
import urllib.request
from email.message import Message
from unittest.mock import patch

from local_model_dock.discovery import discover


class Response:
    def __init__(self, data, link=None):
        self.data = json.dumps(data).encode()
        self.headers = {"Link": link} if link else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size=-1):
        return self.data[:size] if size >= 0 else self.data


class DiscoveryTests(unittest.TestCase):
    def test_fallback_resolves_exact_repository_to_immutable_revision(self):
        from local_model_dock.discovery import discover_fallback
        with patch("local_model_dock.discovery._open", return_value=Response({
            "id": "owner/model", "sha": "a" * 40, "private": False,
            "gated": False, "cardData": {"license": "mit"}
        })) as request:
            model = discover_fallback("owner/model")
        self.assertEqual((model.name, model.revision, model.accessible), ("owner/model", "a" * 40, True))
        self.assertIn("/api/models/owner/model", request.call_args.args[0])
        with patch("local_model_dock.discovery._open", return_value=Response({"id": "other/model", "sha": "a" * 40})):
            with self.assertRaises(ValueError):
                discover_fallback("owner/model")
    def test_architecture_and_parameter_metadata_never_come_from_repo_name(self):
        from local_model_dock.discovery import DiscoveryCandidate, inspect_candidate
        model = DiscoveryCandidate("owner/DeepSeek-V4-Qwen3.5-9B", "a" * 40,
                                   None, None, total_parameters=9_000_000_000)
        with patch("local_model_dock.discovery._open", side_effect=[
            Response({"model_type": "qwen3", "quantization": {"bits": 4}}),
            Response([{"path": "model-Q4.gguf", "size": 100}])
        ]):
            found = inspect_candidate(model)
        self.assertEqual((found.family, found.version), ("qwen", (3,)))
        self.assertIsNone(found.total_parameters)

    def test_targeted_search_includes_requested_families(self):
        from local_model_dock.discovery import discover_targeted
        with patch("local_model_dock.discovery._open", return_value=Response([])) as request:
            discover_targeted(max_pages=1)
        urls = [call.args[0].lower() for call in request.call_args_list]
        self.assertTrue(any("search=qwen3.5" in url for url in urls))
        self.assertTrue(any("search=deepseek" in url for url in urls))

    def test_qwen35_moe_config_is_recognized(self):
        from local_model_dock.discovery import DiscoveryCandidate, inspect_candidate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None)
        with patch("local_model_dock.discovery._open", side_effect=[
            Response({"model_type": "qwen3_5_moe", "num_experts": 64,
                      "num_experts_per_tok": 8}),
            Response([{"path": "model-Q4.gguf", "size": 100}])
        ]):
            found = inspect_candidate(model)
        self.assertEqual((found.family, found.version, found.moe), ("qwen", (3, 5), True))
    def test_targeted_search_queries_both_backends_and_deduplicates(self):
        from local_model_dock.discovery import discover_targeted
        item = {"id": "owner/model", "sha": "a" * 40}
        with patch("local_model_dock.discovery._open", side_effect=lambda url: Response([item])) as request:
            found = discover_targeted(max_pages=1)
        self.assertEqual([m.name for m in found], ["owner/model"])
        urls = [call.args[0] for call in request.call_args_list]
        self.assertTrue(any("search=mlx" in url.lower() for url in urls))
        self.assertTrue(any("search=gguf" in url.lower() for url in urls))

    def test_targeted_search_includes_compact_moe_not_in_general_results(self):
        from local_model_dock.discovery import discover_targeted
        name = "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"
        def response(url):
            return Response([{"id": name, "sha": "a" * 40}]
                            if "search=moe-a2" in url.lower() else [])
        with patch("local_model_dock.discovery._open", side_effect=response):
            models = discover_targeted(max_pages=1)
        self.assertEqual([model.name for model in models], [name])
        self.assertFalse(models[0].verified)

    def test_non_mlx_quantization_is_not_mlx_evidence(self):
        from local_model_dock.discovery import DiscoveryCandidate, inspect_candidate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None)
        config = {"quantization": {"bits": 4, "quant_method": "gptq"}}
        with patch("local_model_dock.discovery._open", side_effect=[
            Response(config), Response([{"path": "model.safetensors", "size": 100}])
        ]):
            self.assertNotIn("mlx", inspect_candidate(model).formats)

    def test_gguf_files_keep_separate_sizes(self):
        from local_model_dock.discovery import DiscoveryCandidate, inspect_candidate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None)
        with patch("local_model_dock.discovery._open", side_effect=[
            Response({"model_type": "mixtral", "num_local_experts": 8, "num_experts_per_tok": 2}),
            Response([{"path": "model-Q4_K_M.gguf", "size": 100},
                      {"path": "model-Q8_0.gguf", "size": 200}])
        ]):
            result = inspect_candidate(model)
        self.assertTrue(result.moe)
        self.assertEqual(result.gguf_files, (("model-Q4_K_M.gguf", 100), ("model-Q8_0.gguf", 200)))
        self.assertIsNone(result.disk_bytes)

    def test_missing_config_still_reports_gguf_without_claiming_moe(self):
        from local_model_dock.discovery import DiscoveryCandidate, inspect_candidate
        model = DiscoveryCandidate("owner/model", "a" * 40, None, None)
        missing = urllib.error.HTTPError("https://huggingface.co/", 404, "missing", {}, None)
        with patch("local_model_dock.discovery._open", side_effect=[
            missing, Response([{"path": "model-q4.gguf", "size": 100}])
        ]):
            result = inspect_candidate(model)
        self.assertIsNone(result.moe)
        self.assertEqual(result.gguf_files, (("model-q4.gguf", 100),))

    def test_pinned_metadata_uses_config_and_file_sizes(self):
        from local_model_dock.discovery import inspect_candidate

        model = __import__("local_model_dock.discovery", fromlist=["DiscoveryCandidate"]).DiscoveryCandidate(
            "mlx-community/example", "a" * 40, "apache-2.0", 10, accessible=True)
        config = {"model_type": "qwen3_moe", "num_experts": 128,
                  "num_experts_per_tok": 8, "quantization": {"bits": 4}}
        files = [{"path": "model.safetensors", "size": 100},
                 {"path": "config.json", "size": 20}]
        with patch("local_model_dock.discovery._open", side_effect=[Response(config), Response(files)]) as request:
            result = inspect_candidate(model)
        self.assertTrue(result.moe)
        self.assertEqual(result.formats, ("mlx",))
        self.assertEqual(result.disk_bytes, 120)
        self.assertIn("/" + "a" * 40 + "/config.json", request.call_args_list[0].args[0])
        self.assertIn("/tree/" + "a" * 40, request.call_args_list[1].args[0])
        self.assertFalse(result.verified)

    def test_pinned_metadata_refuses_moving_revision(self):
        from local_model_dock.discovery import DiscoveryCandidate, inspect_candidate
        model = DiscoveryCandidate("owner/model", "main", None, None)
        with patch("local_model_dock.discovery._open") as request:
            self.assertEqual(inspect_candidate(model), model)
        request.assert_not_called()

    def test_redirects_are_not_followed(self):
        from local_model_dock.discovery import NoRedirect
        handler = NoRedirect()
        request = urllib.request.Request("https://huggingface.co/api/models")
        headers = Message()
        headers["Location"] = "http://127.0.0.1/private"
        self.assertIsNone(handler.http_error_302(request, None, 302, "redirect", headers))

    def test_oversized_response_fails_closed(self):
        with patch("local_model_dock.discovery._open", return_value=Response(["x" * 2_000_001])):
            with self.assertRaises(ValueError):
                discover(max_pages=1)

    def test_online_config_and_format_are_evidence_not_verification(self):
        item = {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "sha": "a" * 40,
                "cardData": {"license": "apache-2.0"},
                "config": {"model_type": "qwen3_moe", "num_experts": 128,
                           "num_experts_per_tok": 8},
                "siblings": [{"rfilename": "model-00001-of-00002.safetensors"},
                             {"rfilename": "model-00002-of-00002.safetensors"}]}
        with patch("local_model_dock.discovery._open", return_value=Response([item])):
            model = discover(max_pages=1)[0]
        self.assertTrue(model.moe)
        self.assertEqual(model.formats, ("safetensors",))
        self.assertFalse(model.verified)

    def test_access_is_known_only_when_both_flags_are_public(self):
        models = [
            {"id": "a", "private": False, "gated": False},
            {"id": "b", "private": False, "gated": "manual"},
            {"id": "c", "private": False},
        ]
        with patch("local_model_dock.discovery._open", return_value=Response(models)):
            found = discover(1)
        self.assertEqual([m.accessible for m in found], [True, False, None])

    def test_models_remain_unverified_and_unknown(self):
        with patch("local_model_dock.discovery._open", return_value=Response([
            {"id": "Qwen/example", "sha": "abc", "downloads": 12,
             "cardData": {"license": "apache-2.0"}}
        ])):
            models = discover()
        self.assertEqual(models[0].name, "Qwen/example")
        self.assertEqual(models[0].revision, "abc")
        self.assertEqual(models[0].license, "apache-2.0")
        self.assertIsNone(models[0].moe)
        self.assertFalse(models[0].verified)

    def test_follows_huggingface_pagination(self):
        responses = [Response([{"id": "first"}],
                              '<https://huggingface.co/api/models?cursor=next>; rel="next"'),
                     Response([{"id": "second"}])]
        with patch("local_model_dock.discovery._open", side_effect=responses):
            self.assertEqual([m.name for m in discover()], ["first", "second"])

    def test_rejects_external_pagination_target(self):
        with patch("local_model_dock.discovery._open", return_value=Response(
            [{"id": "first"}], '<https://evil.example/path>; rel="next"')) as request:
            with self.assertRaises(ValueError):
                discover()
        self.assertEqual(request.call_count, 1)

    def test_bad_response_is_reported(self):
        with patch("local_model_dock.discovery._open", return_value=Response({"error": "bad"})):
            with self.assertRaises(ValueError):
                discover()
