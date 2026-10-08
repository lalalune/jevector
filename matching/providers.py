"""Provider-neutral, evidence-aware profile vector extraction (Python 3.11+)."""

import datetime
import hashlib
import http.client
import json
import math
import os
import tempfile
import threading
import weakref
from pathlib import Path
import time
import urllib.error
import urllib.request
import tomllib

VERSION = "jevector-2.1"  # Retain existing request-cache identities.


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Refusing authenticated HTTP redirect")


def digest(value):
    # Order matters to Clef; do not sort question maps when fingerprinting requests.
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_answers(answers, qs):
    if set(answers) != set(qs):
        raise ValueError("Response question IDs do not match request")
    for key, q in qs.items():
        a = answers[key]
        p = a.get("probabilities", {})
        if a.get("type") != "choice" or set(p) != set(q["criteria"]):
            raise ValueError(f"Invalid answer shape: {key}")
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in p.values()
        ):
            raise ValueError(f"Invalid probability: {key}")
        if abs(sum(p.values()) - 1) > len(p) * 0.005 + 0.00001:
            raise ValueError(f"Probabilities do not sum to one: {key}")
        if a.get("choice") not in p:
            raise ValueError(f"Invalid selected choice: {key}")
        c = a.get("confidence")
        if not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
            raise ValueError(f"Invalid confidence: {key}")


class Client:
    _cache_locks = weakref.WeakValueDictionary()
    _cache_guard = threading.Lock()

    def __init__(
        self,
        provider,
        account=None,
        model=None,
        key=None,
        cache="runs/cache",
        batch_size=None,
    ):
        if provider not in ("clef", "jev"):
            raise ValueError("Unknown provider")
        self.provider = provider
        self.model = model or ("clef-flash" if provider == "clef" else "jev-1.13.0")
        self.account = account or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.key = key
        self.cache = Path(cache)
        self.batch_size = batch_size or (64 if provider == "clef" else 256)
        if not 1 <= self.batch_size <= (64 if provider == "clef" else 256):
            raise ValueError("Invalid batch size")
        if provider == "clef" and (
            not self.account or self.model not in ("clef", "clef-flash")
        ):
            raise ValueError("Clef needs an account and model clef or clef-flash")

    def call(self, state, qs, nonce=None):
        lock_key = (
            str(self.cache.resolve()),
            self.provider,
            self.model,
            self.account,
            digest({"state": state, "questions": qs, "nonce": nonce}),
        )
        with self._cache_guard:
            lock = self._cache_locks.setdefault(lock_key, threading.RLock())
        with lock:
            return self._call(state, qs, nonce)

    def _call(self, state, qs, nonce=None):
        body = {"model": self.model, "state": state, "questions": qs}
        identity = {
            "provider": self.provider,
            "account": self.account if self.provider == "clef" else None,
            "version": VERSION,
            "request": body,
            "nonce": nonce,
        }
        request_hash = digest(identity)
        path = self.cache / self.provider / (request_hash + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record["request_hash"] != request_hash:
                raise ValueError("Cache fingerprint mismatch")
            validate_answers(record["response"]["answers"], qs)
            return record
        secret = (
            (
                self.key
                or os.environ.get("TYPESAFE_API_KEY")
                or os.environ.get("JEV_API_KEY")
            )
            if self.provider == "jev"
            else cloudflare_token()
        )
        if not secret:
            raise ValueError("Set TYPESAFE_API_KEY or use --prompt-key")
        url = (
            "https://api.typesafe.ai/v1/systemone"
            if self.provider == "jev"
            else f"https://api.cloudflare.com/client/v4/accounts/{self.account}/ai/run/@cf/cloudflare/{self.model}"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": "Bearer " + secret,
                "Content-Type": "application/json",
            },
        )
        start = time.monotonic()
        for attempt in range(4):
            try:
                with urllib.request.build_opener(NoRedirect).open(
                    req, timeout=120
                ) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if (
                    e.code in (429, 500, 502, 503, 504, 520, 521, 522, 524, 529)
                    and attempt < 3
                ):
                    retry = e.headers.get("Retry-After", "")
                    time.sleep(min(float(retry), 30) if retry.isdigit() else 2**attempt)
                    continue
                # Do not expose response bodies, which may contain input or authentication details.
                raise RuntimeError(
                    f"{self.provider} HTTP {e.code}; request {request_hash[:12]}"
                ) from None
            except (
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                TimeoutError,
                ConnectionError,
            ):
                if attempt == 3:
                    raise RuntimeError(
                        f"{self.provider} connection failed; request {request_hash[:12]}"
                    ) from None
                time.sleep(2**attempt)
        if self.provider == "clef":
            if not data.get("success"):
                raise RuntimeError("Cloudflare reported an unsuccessful request")
            data = data["result"]
        record = {
            "request_hash": request_hash,
            "provider": self.provider,
            "requested_model": self.model,
            "schema_version": VERSION,
            "question_ids": list(qs),
            "question_count": len(qs),
            "state_hash": digest(state),
            "elapsed_seconds": time.monotonic() - start,
            "attempts": attempt + 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response": data,
        }
        validate_answers(data["answers"], qs)
        atomic_json(path, record)
        return record


def cloudflare_token():
    for key in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_AUTH_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    for p in [
        Path.home() / "Library/Preferences/.wrangler/config/default.toml",
        Path.home() / ".wrangler/config/default.toml",
    ]:
        if p.exists():
            return tomllib.loads(p.read_text())["oauth_token"]
    raise RuntimeError("No Cloudflare API token or Wrangler login found")


def api(path, body=None):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/" + path,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": "Bearer " + cloudflare_token(),
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=180) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Cloudflare HTTP {e.code}") from None
    if not data.get("success", True):
        raise RuntimeError("Cloudflare reported an unsuccessful request")
    return data.get("result", data)
