import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from local_model_dock.process_identity import process_owns_loopback_port


class ProcessIdentityTests(unittest.TestCase):
    def test_real_child_owns_its_loopback_listener(self):
        code = (
            "import socket,sys; s=socket.socket(); "
            "s.bind(('127.0.0.1',0)); s.listen(); "
            "print(s.getsockname()[1],flush=True); sys.stdin.read()"
        )
        with subprocess.Popen(
            [sys.executable, "-c", code], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True,
        ) as child:
            port = int(child.stdout.readline())
            self.assertTrue(process_owns_loopback_port(child, port))
            self.assertFalse(process_owns_loopback_port(child, 0))
            child.stdin.close()
            child.wait()
            self.assertFalse(process_owns_loopback_port(child, port))

    def test_other_process_listener_is_not_owned(self):
        with socket.socket() as listener, subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
        ) as child:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            self.assertFalse(process_owns_loopback_port(child, listener.getsockname()[1]))
            child.stdin.close()
            child.wait()

    def test_lsof_results_require_exact_live_owner_and_loopback(self):
        class Child:
            pid = 123

            def poll(self):
                return None

        with patch("local_model_dock.process_identity.sys.platform", "darwin"), \
                patch("local_model_dock.process_identity.subprocess.run") as run:
            run.return_value.returncode = 0
            for output, host, expected in [
                ("p123\nn127.0.0.1:4321\n", "127.0.0.1", True),
                ("p123\nn[::1]:4321\n", "::1", True),
                ("p123\nn*:4321\n", "127.0.0.1", False),
                ("p124\nn127.0.0.1:4321\n", "127.0.0.1", False),
                ("p123\nn192.0.2.1:4321\n", "127.0.0.1", False),
                ("p123\nn127.0.0.1:4321\np124\nn127.0.0.1:4321\n", "127.0.0.1", False),
                ("p123\nn127.0.0.1:43210\n", "127.0.0.1", False),
                ("garbage\n", "127.0.0.1", False),
            ]:
                with self.subTest(output=output):
                    run.return_value.stdout = output
                    self.assertEqual(process_owns_loopback_port(Child(), 4321, host), expected)
            run.side_effect = OSError("unavailable")
            self.assertFalse(process_owns_loopback_port(Child(), 4321))

    def test_invalid_input_and_unsupported_platform_fail_closed(self):
        class Child:
            pid = 123
            def poll(self):
                return None
        for port in (-1, 65536, "123", True):
            self.assertFalse(process_owns_loopback_port(Child(), port))
        with patch("local_model_dock.process_identity.sys.platform", "win32"):
            self.assertFalse(process_owns_loopback_port(Child(), 123))

    def test_linux_inode_matching_ipv4_ipv6_and_failures(self):
        class Child:
            pid = 123
            def poll(self):
                return None

        header = "header\n"
        row4 = "0: 0100007F:10E1 00000000:0000 0A 0:0 0:0 0 0 0 42\n"
        row6 = "0: 00000000000000000000000001000000:10E1 0:0 0A 0:0 0:0 0 0 0 43\n"
        tables = {"tcp": header + row4, "tcp6": header + row6}
        def read_table(path):
            return tables[path.name]

        with patch("local_model_dock.process_identity.sys.platform", "linux"), \
                patch.object(Path, "read_text", read_table), \
                patch.object(Path, "iterdir", return_value=iter([Path("4"), Path("5")])), \
                patch("local_model_dock.process_identity.os.readlink",
                      side_effect=["socket:[42]", "socket:[43]"]):
            self.assertTrue(process_owns_loopback_port(Child(), 4321))
        with patch("local_model_dock.process_identity.sys.platform", "linux"), \
                patch.object(Path, "read_text", read_table), \
                patch.object(Path, "iterdir", return_value=iter([Path("4"), Path("5")])), \
                patch("local_model_dock.process_identity.os.readlink",
                      side_effect=["socket:[42]", "socket:[43]"]):
            self.assertTrue(process_owns_loopback_port(Child(), 4321, "::1"))
        with patch("local_model_dock.process_identity.sys.platform", "linux"), \
                patch.object(Path, "read_text", read_table), \
                patch.object(Path, "iterdir", return_value=iter([Path("4")])), \
                patch("local_model_dock.process_identity.os.readlink", return_value="socket:[42]"):
            self.assertFalse(process_owns_loopback_port(Child(), 4321))
        with patch("local_model_dock.process_identity.sys.platform", "linux"), \
                patch.object(Path, "read_text", side_effect=PermissionError):
            self.assertFalse(process_owns_loopback_port(Child(), 4321))
