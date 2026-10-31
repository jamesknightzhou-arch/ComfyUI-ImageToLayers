"""Client for the Image to Layers website (https://www.imagetolayers.com).

The node signs in the same way the Photoshop and Adobe Express plugins do: a short code is approved in the
browser, then the API is called with the resulting session as ``Authorization: Bearer <access token>``.
There is no API key to copy around, and the node never sees the user's password.

Pure Python (``requests`` only) so it can be tested without ComfyUI or torch.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

# www is the real host: the bare domain 301-redirects, which turns POSTs into GETs and drops Authorization.
DEFAULT_BASE_URL = "https://www.imagetolayers.com"
CLIENT_LABEL = "ComfyUI"
SESSION_REFRESH_MARGIN_S = 60
JOB_POLL_INTERVAL_S = 2.5
JOB_TIMEOUT_S = 10 * 60
UPLOAD_MAX_BYTES = 25 * 1024 * 1024
PROMPT_MAX_LENGTH = 1000
HTTP_TIMEOUT_S = 60


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int = 0, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


class SignInRequired(RuntimeError):
    """The user has to approve a sign-in in the browser; the message says how."""


def default_session_path() -> Path:
    override = os.environ.get("IMAGETOLAYERS_SESSION_FILE")
    if override:
        return Path(override)
    return Path.home() / ".imagetolayers" / "comfyui_session.json"


class Client:
    def __init__(
        self,
        base_url: Optional[str] = None,
        session_path: Optional[Path] = None,
        http: Any = None,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = (base_url or os.environ.get("IMAGETOLAYERS_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.session_path = Path(session_path) if session_path else default_session_path()
        self.http = http or requests
        self.now = now
        self.sleep = sleep
        self._lock = threading.Lock()
        self._session: Optional[Dict[str, Any]] = None
        self._pending: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ session storage
    def _load(self) -> Optional[Dict[str, Any]]:
        if self._session:
            return self._session
        try:
            data = json.loads(self.session_path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("baseUrl") not in (None, self.base_url):
            return None  # a session for another server (for example staging) is not valid here
        if data.get("accessToken") and data.get("refreshToken"):
            self._session = data
            return data
        return None

    def _save(self, tokens: Dict[str, Any]) -> None:
        current = self._session or {}
        self._session = {
            "baseUrl": self.base_url,
            "accessToken": tokens["accessToken"],
            "refreshToken": tokens["refreshToken"],
            "expiresAt": tokens.get("expiresAt", 0),
            "email": tokens.get("email") or current.get("email") or "",
        }
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._session), "utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.session_path)

    def _clear(self) -> None:
        self._session = None
        try:
            self.session_path.unlink()
        except OSError:
            pass

    def has_session(self) -> bool:
        return self._load() is not None

    def email(self) -> str:
        session = self._load()
        return (session or {}).get("email", "")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _failure(response: Any, fallback: str) -> ApiError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        message = payload.get("error") if isinstance(payload, dict) else None
        code = payload.get("code") if isinstance(payload, dict) else None
        return ApiError(message if isinstance(message, str) else f"{fallback} ({response.status_code}).", response.status_code, code if isinstance(code, str) else None)

    def _post_json(self, path: str, body: Dict[str, Any]) -> Any:
        return self.http.post(f"{self.base_url}{path}", json=body, timeout=HTTP_TIMEOUT_S)

    def _refresh(self) -> bool:
        session = self._load()
        if not session:
            return False
        response = self._post_json("/api/plugin/auth/refresh", {"refreshToken": session["refreshToken"]})
        if response.status_code in (400, 401):
            self._clear()
            return False
        if not response.ok:
            raise ApiError(f"Could not refresh your session ({response.status_code}).", response.status_code)
        self._save(response.json())
        return True

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Every authenticated call: attaches the session, refreshes it before expiry, retries once after a 401."""
        with self._lock:
            session = self._load()
            if not session:
                raise SignInRequired("Not signed in.")
            if session.get("expiresAt", 0) - self.now() < SESSION_REFRESH_MARGIN_S and not self._refresh():
                raise SignInRequired("Your session has ended.")

        def send() -> Any:
            headers = dict(kwargs.get("headers") or {})
            headers["Authorization"] = f"Bearer {(self._session or {}).get('accessToken', '')}"
            call = {**kwargs, "headers": headers}
            return self.http.request(method, f"{self.base_url}{path}", timeout=HTTP_TIMEOUT_S, **call)

        response = send()
        if response.status_code == 401:
            with self._lock:
                if not self._refresh():
                    raise SignInRequired("Your session has ended.")
            response = send()
            if response.status_code == 401:
                with self._lock:
                    self._clear()
                raise SignInRequired("Your session has ended.")
        return response

    # ------------------------------------------------------------------ sign-in (device code)
    def start_sign_in(self) -> Dict[str, Any]:
        """Starts a sign-in and approves it in the background; returns the code and URL to show the user.

        ComfyUI nodes cannot show a dialog, so the node raises an error that carries the code, and this thread keeps
        polling. Once the user approves in the browser the session is saved and the next run just works.
        """
        with self._lock:
            pending = self._pending
            if pending and pending["deadline"] > self.now() and pending["thread"].is_alive():
                return pending
            response = self._post_json("/api/plugin/auth/start", {"clientLabel": CLIENT_LABEL})
            if not response.ok:
                raise self._failure(response, "Could not start sign-in")
            start = response.json()
            pending = {
                "userCode": start["userCode"],
                "url": start["verificationUrl"],
                "deadline": self.now() + float(start.get("expiresIn", 600)),
            }
            thread = threading.Thread(target=self._poll, args=(start, pending["deadline"]), daemon=True, name="imagetolayers-signin")
            pending["thread"] = thread
            self._pending = pending
            thread.start()
            return pending

    def _poll(self, start: Dict[str, Any], deadline: float) -> None:
        failures = 0
        interval = float(start.get("interval", 3))
        while self.now() < deadline:
            self.sleep(interval)
            try:
                response = self._post_json("/api/plugin/auth/poll", {"deviceCode": start["deviceCode"]})
                if response.ok:
                    failures = 0
                    data = response.json()
                    status = data.get("status")
                    if status == "approved":
                        with self._lock:
                            self._save(data)
                        return
                    if status in ("denied", "expired", "consumed"):
                        return
                else:
                    failures += 1
            except (requests.RequestException, ValueError):
                failures += 1
            if failures >= 5:
                return

    def sign_out(self) -> None:
        session = self._load()
        self._clear()
        if session:
            try:
                self.http.post(f"{self.base_url}/api/plugin/auth/logout", headers={"Authorization": f"Bearer {session['accessToken']}"}, timeout=10)
            except requests.RequestException:
                pass

    # ------------------------------------------------------------------ the API
    def credits(self) -> Optional[int]:
        response = self.request("GET", "/api/credits")
        if not response.ok:
            return None
        return ((response.json() or {}).get("credits") or {}).get("remaining_credits")

    def upload_image(self, data: bytes, content_type: str = "image/png") -> str:
        if len(data) > UPLOAD_MAX_BYTES:
            raise ApiError("This image is too large to upload (25 MB limit).", 413)
        response = self.request("POST", "/api/plugin/upload", data=data, headers={"Content-Type": content_type})
        if not response.ok:
            raise self._failure(response, "Failed to upload the image")
        return response.json()["fileKey"]

    def start_job(self, input_path: str, prompt: Optional[str] = None) -> str:
        body: Dict[str, Any] = {"inputPath": input_path}
        if prompt:
            body["prompt"] = prompt[:PROMPT_MAX_LENGTH]
        response = self.request("POST", "/api/detailed-layers/jobs", json=body)
        if not response.ok:
            raise self._failure(response, "Server error")
        data = response.json()
        job_id = data.get("jobId") or (data.get("job") or {}).get("id")
        if not job_id:
            raise ApiError("Could not start the layer separation job.")
        return job_id

    def wait_for_layers(self, job_id: str, on_poll: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
        started = self.now()
        while True:
            if self.now() - started > JOB_TIMEOUT_S:
                raise ApiError("This is taking longer than expected. Check the job on imagetolayers.com.")
            self.sleep(JOB_POLL_INTERVAL_S)
            if on_poll:
                on_poll()
            response = self.request("GET", f"/api/layer-jobs/{job_id}")
            if not response.ok:
                raise self._failure(response, "Could not check the job")
            result = response.json()
            status = str((result.get("job") or {}).get("status", "")).lower()
            if status == "succeeded":
                return result.get("layers") or []
            if status in ("failed", "canceled", "cancelled"):
                raise ApiError((result.get("job") or {}).get("error_message") or "Layer separation failed.")

    def download_layer(self, job_id: str, layer_id: str) -> bytes:
        response = self.request("GET", f"/api/plugin/jobs/{job_id}/layers/{layer_id}/image")
        if not response.ok:
            raise self._failure(response, "Could not download a layer")
        return response.content
