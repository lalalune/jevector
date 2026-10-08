import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from matching.providers import Client, validate_answers, digest


def answer(q, choice):
    return {
        "type": "choice",
        "choice": choice,
        "confidence": 1.0,
        "probabilities": {k: float(k == choice) for k in q["criteria"]},
    }


class SchemaTests(unittest.TestCase):
    def test_question_order_changes_hash(self):
        self.assertNotEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))


class VectorTests(unittest.TestCase):
    def test_probability_validation_rejects_nan_and_missing(self):
        q = {"x": {"criteria": {"yes": "", "no": ""}}}
        a = {"x": answer(q["x"], "yes")}
        validate_answers(a, q)
        a["x"]["probabilities"]["yes"] = float("nan")
        with self.assertRaises(ValueError):
            validate_answers(a, q)
        with self.assertRaises(ValueError):
            validate_answers({}, q)

    def test_rounded_probabilities_and_provider_choice_preserved(self):
        q = {"x": {"criteria": {"yes": "", "no": "", "unknown": ""}}}
        a = {
            "x": {
                "type": "choice",
                "choice": "yes",
                "confidence": 0.2,
                "probabilities": {"yes": 0.33, "no": 0.34, "unknown": 0.34},
            }
        }
        validate_answers(a, q)
        self.assertEqual(a["x"]["choice"], "yes")

    def test_corrupt_probability_sum_rejected(self):
        q = {"x": {"criteria": {"yes": "", "no": ""}}}
        a = {"x": answer(q["x"], "yes")}
        a["x"]["probabilities"]["no"] = 0.5
        with self.assertRaises(ValueError):
            validate_answers(a, q)


class ClientTests(unittest.TestCase):
    def test_transient_520_retries(self):
        import io, urllib.error
        from unittest.mock import MagicMock

        qs = {"x": {"type": "choice", "criteria": {"yes": "", "no": ""}}}
        payload = {"model": "test", "answers": {"x": answer(qs["x"], "yes")}}
        opener = MagicMock()
        opener.open.side_effect = [
            urllib.error.HTTPError(
                "https://example.invalid", 520, "temporary", {}, None
            ),
            io.BytesIO(json.dumps(payload).encode()),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "urllib.request.build_opener", return_value=opener
        ), patch("matching.providers.time.sleep") as sleep:
            result = Client("jev", key="test-only", cache=tmp).call("state", qs)
            self.assertEqual(result["attempts"], 2)
            sleep.assert_called_once()

    def test_disconnect_retry_and_concurrent_request_deduplication(self):
        import io, http.client, threading
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import MagicMock

        qs = {"x": {"type": "choice", "criteria": {"yes": "", "no": ""}}}
        payload = {"model": "test", "answers": {"x": answer(qs["x"], "yes")}}
        opener = MagicMock()
        opener.open.side_effect = [
            http.client.RemoteDisconnected(),
            io.BytesIO(json.dumps(payload).encode()),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "urllib.request.build_opener", return_value=opener
        ), patch("matching.providers.time.sleep"):
            clients = [Client("jev", key="test-only", cache=tmp) for _ in range(8)]
            barrier = threading.Barrier(8)

            def work(c):
                barrier.wait()
                return c.call("state", qs)

            with ThreadPoolExecutor(max_workers=8) as pool:
                records = list(pool.map(work, clients))
            self.assertEqual(opener.open.call_count, 2)
            self.assertEqual(len({r["request_hash"] for r in records}), 1)

    def test_invalid_network_answer_is_not_cached(self):
        import io
        from unittest.mock import MagicMock

        qs = {"x": {"type": "choice", "criteria": {"yes": "", "no": ""}}}
        opener = MagicMock()
        opener.open.return_value = io.BytesIO(json.dumps({"answers": {}}).encode())
        with tempfile.TemporaryDirectory() as tmp, patch(
            "urllib.request.build_opener", return_value=opener
        ):
            with self.assertRaises(ValueError):
                Client("jev", key="test-only", cache=tmp).call("state", qs)
            self.assertEqual(list(Path(tmp).rglob("*.json")), [])

    def test_validated_cache_without_credentials(self):
        from matching.providers import VERSION, atomic_json

        with tempfile.TemporaryDirectory() as tmp:
            c = Client("jev", cache=tmp)
            qs = {"x": {"type": "choice", "criteria": {"yes": "", "unknown": ""}}}
            body = {"model": c.model, "state": "s", "questions": qs}
            key = digest(
                {
                    "provider": "jev",
                    "account": None,
                    "version": VERSION,
                    "request": body,
                    "nonce": None,
                }
            )
            atomic_json(
                Path(tmp) / "jev" / (key + ".json"),
                {
                    "request_hash": key,
                    "response": {"answers": {"x": answer(qs["x"], "yes")}},
                },
            )
            with patch(
                "urllib.request.build_opener",
                side_effect=AssertionError("Network should not run"),
            ):
                self.assertEqual(
                    c.call("s", qs)["response"]["answers"]["x"]["choice"], "yes"
                )
