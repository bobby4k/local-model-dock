import os
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from local_model_dock.discovery import DiscoveryCandidate


class CliTests(unittest.TestCase):
    def test_fallback_ollama_uses_configured_quantization_and_rejects_missing(self):
        from local_model_dock.cli import main
        from local_model_dock.config import load_config
        fallback = DiscoveryCandidate(load_config(environ={}).fallback_repo, "a" * 40, "mit", 1,
            formats=("gguf",), gguf_files=(("model.i1-Q4_K_M.gguf", 400), ("model.Q8_0.gguf", 100)))
        for quant, expected, status in (("i1-Q4_K_M", "artifact: model.i1-Q4_K_M.gguf", 0),
                                        ("Q8_0", "artifact: model.Q8_0.gguf", 0),
                                        ("Q5_K_M", "quantization", 1)):
            output, errors = StringIO(), StringIO()
            with self.subTest(quant=quant), patch.object(sys, "argv", ["dock", "update"]), \
                 patch.dict(os.environ, {"FALLBACK_QUANTIZATION": quant}), \
                 patch("local_model_dock.discovery.discover_targeted", return_value=[]), \
                 patch("local_model_dock.discovery.discover_fallback", return_value=fallback), \
                 patch("local_model_dock.discovery.inspect_candidate", side_effect=lambda model: model), \
                 patch("local_model_dock.hardware.detect_backend", return_value="ollama"), \
                 redirect_stdout(output), __import__("contextlib").redirect_stderr(errors):
                self.assertEqual(main(), status)
            self.assertIn(expected, output.getvalue() + errors.getvalue())

    def setUp(self):
        fallback = patch("local_model_dock.discovery.discover_fallback", return_value=None)
        fallback.start()
        self.addCleanup(fallback.stop)

    def test_fallback_is_inspected_and_gguf_variants_are_displayed_unverified(self):
        from local_model_dock.cli import main
        from local_model_dock.config import load_config
        fallback = DiscoveryCandidate(load_config(environ={}).fallback_repo, "a" * 40, "mit", 1,
                                      accessible=True, formats=("gguf",),
                                      gguf_files=(("model.i1-Q4_K_M.gguf", 100), ("model-Q8_0.gguf", 200)))
        output = StringIO()
        with patch.object(sys, "argv", ["dock", "update"]), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[]), \
             patch("local_model_dock.discovery.discover_fallback", return_value=fallback), \
             patch("local_model_dock.discovery.inspect_candidate", side_effect=lambda model: model), \
             patch("local_model_dock.hardware.detect_backend", return_value="ollama"), \
             patch("local_model_dock.model_download.download") as download, redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn(fallback.name, output.getvalue())
        self.assertIn("model.i1-Q4_K_M.gguf", output.getvalue())
        self.assertIn("model-Q8_0.gguf", output.getvalue())
        self.assertIn("tool-call verification", output.getvalue())
        self.assertNotIn("Recommended:", output.getvalue())
        download.assert_not_called()

    def test_fallback_is_still_discovered_when_general_search_fails(self):
        from local_model_dock.cli import main
        from local_model_dock.config import load_config
        fallback = DiscoveryCandidate(load_config(environ={}).fallback_repo, "a" * 40, None, None,
                                      gguf_files=(("model.i1-Q4_K_M.gguf", 100),))
        output = StringIO()
        with patch.object(sys, "argv", ["dock", "update"]), \
             patch("local_model_dock.discovery.discover_targeted", side_effect=OSError("offline")), \
             patch("local_model_dock.discovery.discover_fallback", return_value=fallback), \
             patch("local_model_dock.discovery.inspect_candidate", side_effect=lambda model: model), \
             redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn(fallback.name, output.getvalue())

    def test_fallback_recommendation_requires_budget_and_checked_identity(self):
        from local_model_dock.cli import main
        from local_model_dock.config import load_config
        from local_model_dock.storage import StorageStatus
        fallback = DiscoveryCandidate(load_config(environ={}).fallback_repo, "a" * 40, "mit", 1,
                                      moe=False, accessible=True, formats=("mlx",), disk_bytes=1024,
                                      family="qwen", version=(3, 5), total_parameters=9_000_000_000)
        identity = {"repo": fallback.name, "revision": fallback.revision,
                    "backend": "omlx", "backend_version": "1"}
        for over_budget, verified in ((False, True), (True, True), (False, False)):
            output = StringIO()
            with self.subTest(over_budget=over_budget, verified=verified), \
                 patch.object(sys, "argv", ["dock", "update"]), \
                 patch("local_model_dock.discovery.discover_targeted", return_value=[]), \
                 patch("local_model_dock.discovery.discover_fallback", return_value=fallback), \
                 patch("local_model_dock.discovery.inspect_candidate", side_effect=lambda model: model), \
                 patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
                 patch("local_model_dock.hardware.available_memory", return_value=10 * 1024**3), \
                 patch("local_model_dock.hardware.available_disk", return_value=10 * 1024**3), \
                 patch("local_model_dock.storage.inspect_storage", return_value=StorageStatus(0, 1024, 10**10, over_budget)), \
                 patch("local_model_dock.cli._control", return_value={"verified": verified, "identity": identity}), \
                 redirect_stdout(output):
                self.assertEqual(main(), 0)
            self.assertEqual("Recommended: " + fallback.name in output.getvalue(), not over_budget and verified)

    def test_gguf_fallback_is_not_recommended_from_repository_config_alone(self):
        from local_model_dock.cli import main
        from local_model_dock.config import load_config
        from local_model_dock.storage import StorageStatus
        fallback = DiscoveryCandidate(load_config(environ={}).fallback_repo, "a" * 40, "mit", 1,
                                      accessible=True, formats=("gguf",), gguf_files=(("model.i1-Q4_K_M.gguf", 1024),),
                                      family="qwen", version=(3, 5), total_parameters=9_000_000_000)
        output = StringIO()
        with patch.object(sys, "argv", ["dock", "update"]), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[]), \
             patch("local_model_dock.discovery.discover_fallback", return_value=fallback), \
             patch("local_model_dock.discovery.inspect_candidate", side_effect=lambda model: model), \
             patch("local_model_dock.hardware.detect_backend", return_value="ollama"), \
             patch("local_model_dock.hardware.available_memory", return_value=10 * 1024**3), \
             patch("local_model_dock.hardware.available_disk", return_value=10 * 1024**3), \
             patch("local_model_dock.storage.inspect_storage", return_value=StorageStatus(0, 1024, 10**10, False)), \
             patch("local_model_dock.cli._control") as control, redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertNotIn("Recommended:", output.getvalue())
        self.assertIn("GGUF metadata require verification", output.getvalue())
        control.assert_not_called()
    def test_existing_model_without_trusted_policy_metadata_is_rejected(self):
        from local_model_dock.cli import main
        name = "owner--Qwen3.8-9B--" + "a" * 40
        for option in ("--verify", "--enable"):
            with self.subTest(option=option), \
                 patch.object(sys, "argv", ["dock", "update", option, name]), \
                 patch("local_model_dock.cli._control") as control, \
                 redirect_stdout(StringIO()):
                self.assertEqual(main(), 1)
                control.assert_not_called()

    def test_install_rejects_missing_or_disallowed_model_evidence_before_download(self):
        from local_model_dock.cli import main
        candidate = DiscoveryCandidate("owner/model", "a" * 40, "apache-2.0", 10,
                                       moe=True, accessible=True, formats=("mlx",),
                                       disk_bytes=1024**3, family="qwen",
                                       version=(3, 8), total_parameters=9_000_000_000)
        terminal = StringIO()
        terminal.isatty = lambda: True
        for model in (replace(candidate, version=None), replace(candidate, version=(3, 4)),
                      replace(candidate, family="deepseek", version=(3,)),
                      replace(candidate, total_parameters=None),
                      replace(candidate, total_parameters=8_999_999_999),
                      replace(candidate, family=None)):
            with self.subTest(model=model), \
                 patch.object(sys, "argv", ["dock", "update", "--install", candidate.name]), \
                 patch.object(sys, "stdin", terminal), patch.object(sys, "stdout", terminal), \
                 patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
                 patch("local_model_dock.discovery.discover_targeted", return_value=[model]), \
                 patch("local_model_dock.discovery.inspect_candidate", return_value=model), \
                 patch("local_model_dock.install_candidate.install_candidate") as install:
                self.assertEqual(main(), 1)
                install.assert_not_called()

    def test_read_only_update_recommends_only_matching_checked_identity(self):
        from local_model_dock.cli import main
        candidate = DiscoveryCandidate("owner/Qwen", "a" * 40, "apache-2.0", 10,
                                       moe=True, accessible=True, formats=("mlx",),
                                       disk_bytes=1024**3, family="qwen",
                                       version=(3, 8), total_parameters=9_000_000_000)
        identity = {"repo": candidate.name, "revision": candidate.revision,
                    "backend": "omlx", "backend_version": "1.2.3"}
        output = StringIO()
        with patch.object(sys, "argv", ["dock", "update"]), \
             patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
             patch("local_model_dock.hardware.available_memory", return_value=20 * 1024**3), \
             patch("local_model_dock.hardware.available_disk", return_value=20 * 1024**3), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate), \
             patch("local_model_dock.cli._control", return_value={"verified": True, "identity": identity}) as control, \
             patch("local_model_dock.model_download.download") as download, redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn("Recommended: owner/Qwen", output.getvalue())
        self.assertTrue(control.called)
        self.assertTrue(all(call.args[0] == "check" for call in control.call_args_list))
        download.assert_not_called()

    def test_read_only_update_rejects_untrusted_identity_and_license(self):
        from local_model_dock.cli import main
        candidate = DiscoveryCandidate("owner/Qwen", "a" * 40, "apache-2.0", 10,
                                       moe=True, accessible=True, formats=("mlx",),
                                       disk_bytes=1024**3, family="qwen",
                                       version=(3, 8), total_parameters=9_000_000_000)
        identity = {"repo": candidate.name, "revision": candidate.revision,
                    "backend": "omlx", "backend_version": "1.2.3"}
        good = {"verified": True, "identity": identity}
        cases = [
            (candidate, [good, {"verified": True, "identity": {**identity, key: value}}])
            for key, value in (("repo", "other/Qwen"), ("revision", "b" * 40),
                               ("backend", "ollama"), ("backend_version", "2.0"))
        ]
        cases += [(candidate, [{"verified": True}]),
                  (candidate, [{"verified": True, "identity": {**identity, "backend_version": ""}}]),
                  (candidate, [OSError("supervisor unavailable")]),
                  (replace(candidate, license="other"), [good])]
        for model, responses in cases:
            output = StringIO()
            with self.subTest(model=model, responses=responses), \
                 patch.object(sys, "argv", ["dock", "update"]), \
                 patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
                 patch("local_model_dock.hardware.available_memory", return_value=20 * 1024**3), \
                 patch("local_model_dock.hardware.available_disk", return_value=20 * 1024**3), \
                 patch("local_model_dock.discovery.discover_targeted", return_value=[model]), \
                 patch("local_model_dock.discovery.inspect_candidate", return_value=model), \
                 patch("local_model_dock.cli._control", side_effect=responses), \
                 patch("local_model_dock.model_download.download") as download, redirect_stdout(output):
                self.assertEqual(main(), 0)
                self.assertNotIn("Recommended:", output.getvalue())
                download.assert_not_called()

    def test_verify_existing_without_discovery_or_download(self):
        from local_model_dock.cli import main
        name = "owner--model--" + "a" * 40
        with patch.object(sys, "argv", ["local-model-dock", "update", "--verify", name]), \
             patch("local_model_dock.discovery.discover_targeted") as discover, \
             patch("local_model_dock.model_download.download") as download, \
             patch("local_model_dock.cli._control") as control, redirect_stdout(StringIO()):
            self.assertEqual(main(), 1)
        control.assert_not_called()
        discover.assert_not_called()
        download.assert_not_called()

    def test_enable_existing_checks_record_before_enabling(self):
        from local_model_dock.cli import main
        name = "owner--model--" + "a" * 40
        with patch.object(sys, "argv", ["local-model-dock", "update", "--enable", name]), \
             patch("local_model_dock.cli._control") as control, redirect_stdout(StringIO()):
            self.assertEqual(main(), 1)
        control.assert_not_called()

    def test_failed_record_check_never_enables_existing(self):
        from local_model_dock.cli import main
        name = "owner--model--" + "a" * 40
        with patch.object(sys, "argv", ["local-model-dock", "update", "--enable", name]), \
             patch("local_model_dock.cli._control", side_effect=RuntimeError("check failed")) as control, \
             redirect_stdout(StringIO()):
            self.assertEqual(main(), 1)
        control.assert_not_called()

    def test_retry_with_activate_only_enables_after_verify_and_check(self):
        from local_model_dock.cli import main
        name = "owner--model--" + "a" * 40
        for failure, expected in ((None, ["verify", "check", "enable"]),
                                  ("verify", ["verify"]),
                                  ("check", ["verify", "check"])):
            calls = []
            def control(action, model):
                calls.append(action)
                if action == failure:
                    raise RuntimeError(f"{action} failed")
            with self.subTest(failure=failure), \
                 patch.object(sys, "argv", ["local-model-dock", "update", "--verify", name, "--activate"]), \
                 patch("local_model_dock.cli._control", side_effect=control), redirect_stdout(StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual(calls, [])

    def test_existing_operations_reject_conflicting_options(self):
        from local_model_dock.cli import main
        for args in (["update", "--verify", "x", "--enable", "y"],
                     ["update", "--enable", "x", "--activate"],
                     ["start", "--verify", "x"]):
            with self.subTest(args=args), patch.object(sys, "argv", ["dock", *args]), \
                 patch("local_model_dock.cli._control") as control, \
                 self.assertRaises(SystemExit):
                main()
            control.assert_not_called()

    def test_control_exchanges_private_json(self):
        from local_model_dock.cli import _control
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.sock"
            requests = []
            with socket.socket(socket.AF_UNIX) as server:
                server.bind(str(path))
                server.listen()
                def respond():
                    with server.accept()[0] as client:
                        requests.append(json.loads(client.recv(4096)))
                        client.sendall(b'{"verified":true}\n')
                thread = threading.Thread(target=respond)
                thread.start()
                with patch("local_model_dock.supervisor.private_root", return_value=Path(directory)):
                    _control("verify", "model-id")
                thread.join()
            self.assertEqual(requests, [{"action": "verify", "model": "model-id"}])

    def test_check_returns_supervisor_identity(self):
        from local_model_dock.cli import _control
        identity = {"repo": "owner/model", "revision": "a" * 40,
                    "backend": "omlx", "backend_version": "1.2"}
        with tempfile.TemporaryDirectory() as directory, socket.socket(socket.AF_UNIX) as server:
            server.bind(str(Path(directory) / "control.sock"))
            server.listen()
            requests = []
            def respond():
                with server.accept()[0] as client:
                    requests.append(json.loads(client.recv(4096)))
                    client.sendall(json.dumps({"verified": True, "identity": identity}).encode() + b"\n")
            thread = threading.Thread(target=respond)
            thread.start()
            with patch("local_model_dock.supervisor.private_root", return_value=Path(directory)):
                result = _control("check", "model-id")
            thread.join()
        self.assertEqual(result["identity"], identity)
        self.assertEqual(requests, [{"action": "check", "model": "model-id"}])

    def test_existing_verify_activate_uses_literal_supervisor_protocol(self):
        from local_model_dock.cli import main
        name = "owner--model--" + "a" * 40
        with patch.object(sys, "argv", ["dock", "update", "--verify", name, "--activate"]), \
             patch("local_model_dock.cli._control") as control, redirect_stdout(StringIO()):
            self.assertEqual(main(), 1)
        control.assert_not_called()

    def test_install_verifies_checks_then_explicitly_enables(self):
        from local_model_dock.cli import main
        candidate = DiscoveryCandidate("owner/model", "a" * 40, "apache-2.0", 10,
                                       moe=True, accessible=True, formats=("mlx",),
                                       disk_bytes=1024**3, family="qwen",
                                       version=(3, 8), total_parameters=9_000_000_000)
        terminal = StringIO()
        terminal.isatty = lambda: True
        with patch.object(sys, "argv", ["local-model-dock", "update", "--install", candidate.name, "--activate"]), \
             patch.object(sys, "stdin", terminal), patch.object(sys, "stdout", terminal), \
             patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate), \
             patch("local_model_dock.supervisor.private_root", return_value=Path("/private/dock")), \
             patch("local_model_dock.install_candidate.install_candidate", return_value=Path("/private/dock/models/model-id")) as install, \
             patch("local_model_dock.cli._control") as control:
            self.assertEqual(main(), 0)
        self.assertEqual(install.call_args.args[0], candidate)
        self.assertEqual([call.args for call in control.call_args_list],
                         [("verify", "model-id"), ("check", "model-id"), ("enable", "model-id")])

    def test_failed_verification_never_enables(self):
        from local_model_dock.cli import main
        candidate = DiscoveryCandidate("owner/model", "a" * 40, "apache-2.0", 10,
                                       family="qwen", version=(3, 8), total_parameters=9_000_000_000)
        terminal = StringIO()
        terminal.isatty = lambda: True
        with patch.object(sys, "argv", ["local-model-dock", "update", "--install", candidate.name, "--activate"]), \
             patch.object(sys, "stdin", terminal), patch.object(sys, "stdout", terminal), \
             patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate), \
             patch("local_model_dock.supervisor.private_root", return_value=Path("/private/dock")), \
             patch("local_model_dock.install_candidate.install_candidate", return_value=Path("/private/dock/models/model-id")), \
             patch("local_model_dock.cli._control", side_effect=RuntimeError("verify failed")) as control:
            self.assertEqual(main(), 1)
        self.assertEqual(control.call_count, 1)

    def test_install_refuses_noninteractive_before_download(self):
        from local_model_dock.cli import main
        candidate = DiscoveryCandidate("owner/model", "a" * 40, "apache-2.0", 10,
                                       moe=True, accessible=True, formats=("mlx",),
                                       disk_bytes=1024**3)
        with patch.object(sys, "argv", ["local-model-dock", "update", "--install", "owner/model"]), \
             patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate), \
             patch("local_model_dock.model_download.download") as download, \
             patch.object(sys, "stdin", StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(main(), 1)
        download.assert_not_called()

    def test_update_inspects_targeted_candidates_and_never_downloads(self):
        from local_model_dock.cli import main
        output = StringIO()
        candidate = DiscoveryCandidate("mlx-community/Qwen", "a" * 40, "apache-2.0", 10,
                                       moe=True, accessible=True, formats=("mlx",),
                                       disk_bytes=1024 ** 3)
        with patch.object(sys, "argv", ["local-model-dock", "update"]), \
             patch("local_model_dock.hardware.detect_backend", return_value="omlx"), \
             patch("local_model_dock.hardware.available_memory", return_value=20 * 1024**3), \
             patch("local_model_dock.hardware.available_disk", return_value=20 * 1024**3), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate) as inspect, \
             redirect_stdout(output):
            self.assertEqual(main(), 0)
        inspect.assert_called_once_with(candidate)
        self.assertIn("Qwen", output.getvalue())
        self.assertIn("tool-call verification", output.getvalue())
        self.assertIn("estimate", output.getvalue())

    def test_update_prints_hardware_even_if_discovery_fails(self):
        from local_model_dock.cli import main
        output = StringIO()
        with patch.object(sys, "argv", ["local-model-dock", "update"]), \
             patch("local_model_dock.hardware.platform.system", return_value="Darwin"), \
             patch("local_model_dock.hardware.platform.machine", return_value="arm64"), \
             patch("local_model_dock.hardware.cpu_name", return_value="Apple M4", create=True), \
             patch("local_model_dock.hardware.total_memory", return_value=16 * 1024**3, create=True), \
             patch("local_model_dock.hardware.available_memory", return_value=None), \
             patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(100, 50, 20 * 1024**3)), \
             patch("local_model_dock.discovery.discover_targeted", side_effect=OSError("offline")), \
             redirect_stdout(output):
            self.assertEqual(main(), 1)
        text = output.getvalue()
        for expected in ("Darwin", "arm64", "Apple M4", "16.0 GiB", "unknown", "20.0 GiB", "ollama"):
            self.assertIn(expected, text)

    def test_update_reports_unverified_candidates_without_downloading(self):
        from local_model_dock.cli import main
        output = StringIO()
        candidate = DiscoveryCandidate("owner/model", "abc", "apache-2.0", 10)
        with patch.object(sys, "argv", ["local-model-dock", "update"]), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate), \
             redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn("owner/model", output.getvalue())
        self.assertIn("unverified", output.getvalue())
        self.assertIn("MoE: unknown", output.getvalue())
        self.assertIn("No verified recommendation", output.getvalue())
        self.assertIn("tool-call verification", output.getvalue())

    def test_update_shows_known_moe_without_claiming_verification(self):
        from local_model_dock.cli import main
        output = StringIO()
        candidate = DiscoveryCandidate("owner/model", "a" * 40, "apache-2.0", 10, moe=True)
        with patch.object(sys, "argv", ["local-model-dock", "update"]), \
             patch("local_model_dock.discovery.discover_targeted", return_value=[candidate]), \
             patch("local_model_dock.discovery.inspect_candidate", return_value=candidate), \
             redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn("MoE: yes", output.getvalue())
        self.assertIn("unverified", output.getvalue())
