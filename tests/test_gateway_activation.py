import threading
import unittest
from unittest.mock import patch

from local_model_dock.supervisor import ModelState
from local_model_dock.gateway import create_gateway
from test_gateway import Backend, request, running
from http.server import ThreadingHTTPServer


class ModelStateTests(unittest.TestCase):
    @patch("local_model_dock.supervisor.check_record")
    def test_enable_and_revalidate_each_request(self, check):
        state = ModelState(object())
        check.return_value = True
        self.assertTrue(state.enable("first"))
        with state.request() as model:
            self.assertEqual(model, "first")
        check.return_value = False
        with state.request() as model:
            self.assertIsNone(model)

    @patch("local_model_dock.supervisor.check_record", return_value=True)
    def test_restart_requires_explicit_enable(self, check):
        first = ModelState(object())
        self.assertTrue(first.enable("first"))
        restarted = ModelState(object())
        with restarted.request() as model:
            self.assertIsNone(model)
        self.assertTrue(restarted.enable("first"))
        with restarted.request() as model:
            self.assertEqual(model, "first")

    @patch("local_model_dock.supervisor.verify_model")
    @patch("local_model_dock.supervisor.check_record", return_value=True)
    def test_inflight_request_refuses_verification_restart(self, check, verify):
        state = ModelState(object())
        self.assertTrue(state.enable("first"))
        with state.request() as model:
            self.assertEqual(model, "first")
            self.assertFalse(state.verify("second"))
            self.assertFalse(state.enable("second"))
            verify.assert_not_called()

    @patch("local_model_dock.supervisor.verify_model")
    @patch("local_model_dock.supervisor.check_record", return_value=True)
    def test_probe_busy_does_not_block_other_control_operations(self, check, verify):
        entered, release = threading.Event(), threading.Event()
        def probe(*args):
            entered.set()
            release.wait(3)
            return True
        verify.side_effect = probe
        state = ModelState(object())
        worker = threading.Thread(target=lambda: state.verify("first"))
        worker.start()
        self.addCleanup(worker.join, 4)
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))
        self.assertFalse(state.enable("second"))
        with state.request() as model:
            self.assertIsNone(model)

    @patch("local_model_dock.supervisor.check_record")
    def test_gateway_exposes_only_currently_valid_model(self, check):
        state = ModelState(object())
        check.return_value = True
        self.assertTrue(state.enable("first"))
        with running(ThreadingHTTPServer(("127.0.0.1", 0), Backend)) as backend:
            url = f"http://127.0.0.1:{backend.server_port}"
            gateway = create_gateway(("127.0.0.1", 0), url, "secret", "local-model",
                                     lambda: None, model_session=state.request)
            with running(gateway):
                status = request(gateway, "POST", "/v1/chat/completions",
                                 "Bearer secret", b'{"model":"local-model"}')[0]
                self.assertEqual(status, 200)
                self.assertEqual(Backend.requests[-1][2]["model"], "first")
                check.return_value = False
                status = request(gateway, "POST", "/v1/chat/completions",
                                 "Bearer secret", b'{"model":"local-model"}')[0]
                self.assertEqual(status, 503)

    @patch("local_model_dock.supervisor.verify_model")
    @patch("local_model_dock.supervisor.check_record", return_value=True)
    def test_inflight_gateway_call_survives_rejected_switch_and_probe(self, check, verify):
        arrived, release = threading.Event(), threading.Event()

        class SlowBackend(Backend):
            def do_POST(self):
                arrived.set()
                release.wait(3)
                super().do_POST()

        state = ModelState(object())
        self.assertTrue(state.enable("first"))
        with running(ThreadingHTTPServer(("127.0.0.1", 0), SlowBackend)) as backend:
            gateway = create_gateway(("127.0.0.1", 0), f"http://127.0.0.1:{backend.server_port}",
                                     "secret", "local-model", lambda: None,
                                     model_session=state.request)
            with running(gateway):
                result = []
                worker = threading.Thread(target=lambda: result.append(
                    request(gateway, "POST", "/v1/chat/completions", "Bearer secret",
                            b'{"model":"local-model"}')[0]))
                worker.start()
                try:
                    self.assertTrue(arrived.wait(2))
                    self.assertFalse(state.enable("second"))
                    self.assertFalse(state.verify("second"))
                finally:
                    release.set()
                    worker.join(4)
                self.assertEqual(result, [200])
