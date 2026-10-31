import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client import ApiError, Client, SignInRequired  # noqa: E402


class Resp:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code = status
        self._body = body
        self.content = content
        self.ok = status < 400

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeHttp:
    """Routes by (method, path); a route may be a response or a list consumed in order."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _answer(self, method, url, kwargs):
        path = url.split("imagetolayers.test", 1)[1]
        self.calls.append((method, path, kwargs))
        route = self.routes[(method, path)]
        return route.pop(0) if isinstance(route, list) else route

    def post(self, url, **kwargs):
        return self._answer("POST", url, kwargs)

    def request(self, method, url, **kwargs):
        return self._answer(method, url, kwargs)


def make(routes, session=None, now=1000.0):
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "session.json"
    if session:
        path.write_text(json.dumps({"baseUrl": "https://imagetolayers.test", **session}))
    http = FakeHttp(routes)
    client = Client(base_url="https://imagetolayers.test", session_path=path, http=http, now=lambda: now, sleep=lambda s: None)
    return client, http, path


GOOD = {"accessToken": "at", "refreshToken": "rt", "expiresAt": 10_000}


class Session(unittest.TestCase):
    def test_not_signed_in(self):
        client, _, _ = make({})
        self.assertFalse(client.has_session())
        with self.assertRaises(SignInRequired):
            client.request("GET", "/api/credits")

    def test_session_for_another_server_is_ignored(self):
        client, _, path = make({}, session=GOOD)
        path.write_text(json.dumps({"baseUrl": "https://other.example", **GOOD}))
        self.assertFalse(client.has_session())

    def test_bearer_header_is_sent(self):
        client, http, _ = make({("GET", "/api/credits"): Resp(200, {"credits": {"remaining_credits": 42}})}, session=GOOD)
        self.assertEqual(client.credits(), 42)
        self.assertEqual(http.calls[0][2]["headers"]["Authorization"], "Bearer at")

    def test_refreshes_before_expiry_and_saves(self):
        routes = {
            ("POST", "/api/plugin/auth/refresh"): Resp(200, {"accessToken": "at2", "refreshToken": "rt2", "expiresAt": 20_000}),
            ("GET", "/api/credits"): Resp(200, {"credits": {"remaining_credits": 5}}),
        }
        client, http, path = make(routes, session={**GOOD, "expiresAt": 1030})
        client.credits()
        self.assertEqual(http.calls[-1][2]["headers"]["Authorization"], "Bearer at2")
        self.assertEqual(json.loads(path.read_text())["refreshToken"], "rt2")

    def test_rejected_refresh_signs_out(self):
        routes = {("POST", "/api/plugin/auth/refresh"): Resp(401, {})}
        client, _, path = make(routes, session={**GOOD, "expiresAt": 1000})
        with self.assertRaises(SignInRequired):
            client.credits()
        self.assertFalse(path.exists())

    def test_retries_once_after_a_401(self):
        routes = {
            ("GET", "/api/credits"): [Resp(401, {}), Resp(200, {"credits": {"remaining_credits": 7}})],
            ("POST", "/api/plugin/auth/refresh"): Resp(200, {"accessToken": "at2", "refreshToken": "rt2", "expiresAt": 20_000}),
        }
        client, _, _ = make(routes, session=GOOD)
        self.assertEqual(client.credits(), 7)


class SignIn(unittest.TestCase):
    def test_start_returns_code_and_approval_is_saved(self):
        routes = {
            ("POST", "/api/plugin/auth/start"): Resp(200, {"deviceCode": "d" * 30, "userCode": "ABCD-1234", "verificationUrl": "https://imagetolayers.test/plugin-auth?code=ABCD-1234", "interval": 1, "expiresIn": 600}),
            ("POST", "/api/plugin/auth/poll"): [Resp(200, {"status": "pending"}), Resp(200, {"status": "approved", **GOOD, "email": "a@b.c"})],
        }
        client, _, path = make(routes)
        pending = client.start_sign_in()
        self.assertEqual(pending["userCode"], "ABCD-1234")
        pending["thread"].join(timeout=5)
        self.assertTrue(client.has_session())
        self.assertEqual(client.email(), "a@b.c")
        self.assertTrue(path.exists())

    def test_second_start_reuses_the_pending_code(self):
        routes = {("POST", "/api/plugin/auth/start"): Resp(200, {"deviceCode": "d" * 30, "userCode": "X", "verificationUrl": "u", "interval": 1, "expiresIn": 600}),
                  ("POST", "/api/plugin/auth/poll"): [Resp(200, {"status": "expired"})]}
        client, http, _ = make(routes)
        first = client.start_sign_in()
        first["thread"].join(timeout=5)
        self.assertFalse(client.has_session())


class Jobs(unittest.TestCase):
    def test_full_job(self):
        routes = {
            ("POST", "/api/plugin/upload"): Resp(200, {"fileKey": "k"}),
            ("POST", "/api/detailed-layers/jobs"): Resp(200, {"jobId": "j1"}),
            ("GET", "/api/layer-jobs/j1"): [Resp(200, {"job": {"status": "processing"}}), Resp(200, {"job": {"status": "succeeded"}, "layers": [{"id": "l1", "name": "A", "bbox": None}]})],
            ("GET", "/api/plugin/jobs/j1/layers/l1/image"): Resp(200, content=b"PNG"),
        }
        client, http, _ = make(routes, session=GOOD)
        key = client.upload_image(b"x")
        job = client.start_job(key, "bottle")
        self.assertEqual(http.calls[-1][2]["json"], {"inputPath": "k", "prompt": "bottle"})
        layers = client.wait_for_layers(job)
        self.assertEqual(layers[0]["id"], "l1")
        self.assertEqual(client.download_layer(job, "l1"), b"PNG")

    def test_failed_job_carries_the_server_message(self):
        routes = {("GET", "/api/layer-jobs/j"): Resp(200, {"job": {"status": "failed", "error_message": "boom"}})}
        client, _, _ = make(routes, session=GOOD)
        with self.assertRaises(ApiError) as caught:
            client.wait_for_layers("j")
        self.assertIn("boom", str(caught.exception))

    def test_server_error_message_is_surfaced(self):
        routes = {("POST", "/api/detailed-layers/jobs"): Resp(402, {"error": "Not enough credits", "code": "INSUFFICIENT_CREDITS"})}
        client, _, _ = make(routes, session=GOOD)
        with self.assertRaises(ApiError) as caught:
            client.start_job("k")
        self.assertEqual(caught.exception.status, 402)
        self.assertEqual(caught.exception.code, "INSUFFICIENT_CREDITS")
        self.assertIn("credits", str(caught.exception))

    def test_prompt_is_trimmed_to_the_server_limit(self):
        routes = {("POST", "/api/detailed-layers/jobs"): Resp(200, {"job": {"id": "j"}})}
        client, http, _ = make(routes, session=GOOD)
        client.start_job("k", "x" * 2000)
        self.assertEqual(len(http.calls[-1][2]["json"]["prompt"]), 1000)

    def test_oversize_upload_is_refused_before_sending(self):
        client, http, _ = make({}, session=GOOD)
        with self.assertRaises(ApiError):
            client.upload_image(b"0" * (25 * 1024 * 1024 + 1))
        self.assertEqual(http.calls, [])


if __name__ == "__main__":
    unittest.main()
