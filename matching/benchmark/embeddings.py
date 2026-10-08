"""Shared token-safe embeddings, cached API calls, and retrieval metrics."""

import json, math, time, urllib.request, urllib.error, concurrent.futures
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from matching.providers import api
from matching.providers import atomic_json, digest

BGE = "@cf/baai/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
CACHE = Path("runs/evaluation/cache")


def cached_call(account, model, body):
    key = digest({"model": model, "body": body})
    path = CACHE / (key + ".json")
    if path.exists():
        r = json.loads(path.read_text())
        return r, True
    start = time.perf_counter()
    for attempt in range(4):
        try:
            response = api(f"accounts/{account}/ai/run/{model}", body)
            break
        except (RuntimeError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    r = {
        "model": model,
        "request_hash": key,
        "elapsed_seconds": time.perf_counter() - start,
        "response": response,
    }
    atomic_json(path, r)
    return r, False


def tokenizer():
    path = Path("runs/evaluation/tokenizer.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Pin model revision for reproducible token counts and chunk boundaries.
        url = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json"
        path.write_bytes(urllib.request.urlopen(url).read())
    t = Tokenizer.from_file(str(path))
    t.no_truncation()
    t.no_padding()
    return t


def chunks(text, t, budget=460):
    enc = t.encode(text, add_special_tokens=False)
    if len(enc.ids) <= budget:
        return [text]
    out = []
    for start in range(0, len(enc.ids), budget):
        offsets = enc.offsets[start : start + budget]
        part = text[offsets[0][0] : offsets[-1][1]]
        if len(t.encode(part).ids) > 512:
            raise ValueError("Chunk exceeds model limit")
        out.append(part)
    return out


def embed(account, texts, prefix="", chunk=True):
    t = tokenizer()
    inputs = []
    owners = []
    lengths = []
    for i, text in enumerate(texts):
        lengths.append(len(t.encode(prefix + text).ids))
        for part in chunks(text, t) if chunk else [text]:
            value = prefix + part
            if chunk and len(t.encode(value).ids) > 512:
                raise ValueError("Embedding input exceeds 512 tokens")
            inputs.append(value)
            owners.append(i)
    jobs = [inputs[i : i + 16] for i in range(0, len(inputs), 16)]

    def work(batch):
        return cached_call(account, BGE, {"text": batch, "pooling": "cls"})

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(work, jobs))
    a = np.asarray(
        [v for (r, _) in records for v in r["response"]["data"]], dtype=np.float32
    )
    if (
        a.shape != (len(inputs), 384)
        or not np.isfinite(a).all()
        or np.any(np.linalg.norm(a, axis=1) == 0)
    ):
        raise ValueError("Invalid embeddings")
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, v in zip(owners, a):
        out[i] += v
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out, {
        "texts": len(texts),
        "chunks": len(inputs),
        "over_512_before_chunking": sum(n > 512 for n in lengths),
        "max_tokens_before_chunking": max(lengths),
        "input_tokens": sum(len(t.encode(x).ids) for x in inputs),
        "requests": len(records),
        "cache_hits": sum(hit for _, hit in records),
        "recorded_request_seconds": sum(r["elapsed_seconds"] for r, _ in records),
        "wall_seconds_this_run": time.perf_counter() - started,
    }


def plain(state):
    if isinstance(state, dict):
        return "\n".join(
            k.replace("_", " ") + ": " + plain(v) for k, v in state.items()
        )
    if isinstance(state, list):
        return "\n".join(plain(v) for v in state)
    return str(state)


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for i, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0) + 1 / (k + i + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))
