import http.client
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import unittest

from local_model_dock.gateway import create_gateway


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Backend(BaseHTTPRequestHandler):
    requests = []
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.requests.append((self.path, dict(self.headers), json.loads(body)))
        data = b'data: {"choices":[{"delta":{"tool_calls":[]}}]}\n\ndata: [DONE]\n\n' if json.loads(body).get('stream') else b'{"choices":[{"message":{"tool_calls":[]}}]}'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if json.loads(body).get('stream') else 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass


def request(server, method, path, key=None, body=None, headers=None):
    conn = http.client.HTTPConnection(*server.server_address, timeout=3)
    headers = dict(headers or {})
    if key is not None:
        headers['Authorization'] = key
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    conn.close()
    return result


class GatewayTests(unittest.TestCase):
    def test_rejects_empty_key_and_non_loopback_backend(self):
        for key, backend in [('', 'http://127.0.0.1:1'), ('   ', 'http://127.0.0.1:1'), ('key', 'http://example.com:1'), ('key', 'http://127.0.0.1:1/evil')]:
            with self.assertRaises(ValueError):
                create_gateway(('127.0.0.1', 0), backend, key, 'stable', lambda: 'actual')


    def test_authentication_paths_and_forwarding(self):
        Backend.requests = []
        with running(ThreadingHTTPServer(('127.0.0.1', 0), Backend)) as backend:
            url = f'http://127.0.0.1:{backend.server_port}'
            model = ['actual']
            with running(create_gateway(('127.0.0.1', 0), url, 'secret', 'stable', lambda: model[0])) as gateway:
                for auth in [None, 'Bearer', 'Bearer wrong', 'Bearer secretx']:
                    assert request(gateway, 'GET', '/v1/models', auth)[0] == 401
                assert request(gateway, 'GET', '/v1/models', 'Bearer secret')[0] == 200
                assert json.loads(request(gateway, 'GET', '/v1/models', 'Bearer secret')[2])['data'][0]['id'] == 'stable'
                assert request(gateway, 'GET', '/v1/models?x=1', 'Bearer secret')[0] == 404
                assert request(gateway, 'POST', '/v1/../api/generate', 'Bearer secret', b'{}')[0] == 404
                assert request(gateway, 'POST', '/v1/chat/completions', 'Bearer secret', b'bad')[0] == 400
                assert request(gateway, 'POST', '/v1/chat/completions', 'Bearer secret', b'{"model":"other"}')[0] == 400
                assert not Backend.requests
                payload = {'model': 'stable', 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': [{'type': 'function', 'function': {'name': 'test'}}]}
                status, _, body = request(gateway, 'POST', '/v1/chat/completions', 'Bearer secret', json.dumps(payload))
                assert status == 200 and json.loads(body)['choices'][0]['message']['tool_calls'] == []
                assert Backend.requests[-1][2] == {**payload, 'model': 'actual'}
                assert 'Authorization' not in Backend.requests[-1][1]
                model[0] = 'next'
                payload['stream'] = True
                status, headers, body = request(gateway, 'POST', '/v1/chat/completions', 'Bearer secret', json.dumps(payload))
                assert status == 200 and headers['Content-Type'] == 'text/event-stream'
                assert body.endswith(b'data: [DONE]\n\n')
                assert Backend.requests[-1][2]['model'] == 'next'

    def test_backend_failure_returns_gateway_error(self):
        backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        port = backend.server_port
        backend.server_close()
        with running(create_gateway(('127.0.0.1', 0), f'http://127.0.0.1:{port}', 'secret', 'stable', lambda: 'actual')) as gateway:
            status, _, body = request(gateway, 'POST', '/v1/chat/completions', 'Bearer secret', b'{"model":"stable"}')
            assert status == 502
            assert b'secret' not in body
