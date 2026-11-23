"""Small authenticated gateway for the OpenAI chat-completions subset."""

import hmac
import http.client
import ipaddress
import json
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


_MAX_BODY = 16 * 1024 * 1024


def create_gateway(address, backend_url, api_key, stable_model, get_backend_model, *, model_session=None):
    """Create a server; caller runs serve_forever() and isolates the backend listener.

    get_backend_model is called once per request so new requests see model changes.
    """
    if not isinstance(api_key, str) or not api_key.strip() or '\r' in api_key or '\n' in api_key:
        raise ValueError('A nonempty API key is required')
    if not isinstance(stable_model, str) or not stable_model.strip():
        raise ValueError('A stable model name is required')
    try:
        url = urlsplit(backend_url)
        host = url.hostname
        port = url.port
        if (url.scheme != 'http' or not host or not ipaddress.ip_address(host).is_loopback
                or url.username is not None or url.password is not None
                or url.path not in ('', '/') or url.query or url.fragment):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError('Backend must be a loopback HTTP origin') from None

    if model_session is None:
        model_session = lambda: nullcontext(get_backend_model())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def error(self, status, message):
            body = json.dumps({'error': {'message': message}}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def authorized(self):
            values = self.headers.get_all('Authorization', [])
            expected = ('Bearer ' + api_key).encode('utf-8')
            if len(values) != 1 or not hmac.compare_digest(values[0].encode('utf-8'), expected):
                self.error(401, 'Unauthorized')
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != '/v1/models':
                self.error(404, 'Not found')
                return
            with model_session() as model:
                models = ([{'id': stable_model, 'object': 'model', 'owned_by': 'local'}]
                          if model else [])
            body = json.dumps({'object': 'list', 'data': models}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/chat/completions':
                self.error(404, 'Not found')
                return
            lengths = self.headers.get_all('Content-Length', [])
            if self.headers.get('Transfer-Encoding') is not None or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                self.error(400, 'Invalid request length')
                return
            if len(lengths[0]) > 10:
                self.error(413, 'Request too large')
                return
            length = int(lengths[0])
            if length > _MAX_BODY:
                self.error(413, 'Request too large')
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or payload.get('model') != stable_model:
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self.error(400, 'Invalid request or model')
                return
            with model_session() as model:
                if not isinstance(model, str) or not model.strip():
                    self.error(503, 'No verified model available')
                    return
                self.forward(payload, model)

        def forward(self, payload, model):
            started = False
            try:
                payload['model'] = model
                body = json.dumps(payload).encode()
                conn = http.client.HTTPConnection(host, port or 80, timeout=30)
                try:
                    conn.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json', 'Accept': self.headers.get('Accept', '*/*')})
                    response = conn.getresponse()
                    self.send_response(response.status)
                    content_type = response.getheader('Content-Type')
                    if content_type:
                        self.send_header('Content-Type', content_type)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    started = True
                    self.close_connection = True
                    while chunk := response.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                finally:
                    conn.close()
            except (OSError, http.client.HTTPException, ValueError):
                if not started:
                    self.error(502, 'Backend unavailable')

    return ThreadingHTTPServer(address, Handler)
