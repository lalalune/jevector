# Jevector

Profile matching needs specific criteria. Similar text alone does not show whether two people meet each other's preferences.

Jevector uses Clef or Jev to encode these criteria as 64 or 256 numeric values. HNSW retrieves candidate profiles. The matcher checks mandatory conditions in both directions, then ranks the remaining profiles by preference coverage.

## Benchmarks

All methods search the same 52 synthetic profiles. The test uses eight queries from four profile-pair families excluded from development. Each query has one intended partner.

The table measures retrieval **before rule filtering**.

<!-- benchmarks:start -->

| Method | Top-1 | Top-5 | Top-10 | MRR@10 | nDCG@10 | Tie-adjusted top-1 |
|---|---|---|---|---|---|---|
| Clef 64 | 50.0% | 100.0% | 100.0% | 0.750 | 0.815 | 50.0% |
| Clef 256 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 100.0% |
| Jev 64 | 87.5% | 100.0% | 100.0% | 0.938 | 0.954 | 44.8% |
| Jev 256 | 87.5% | 100.0% | 100.0% | 0.938 | 0.954 | 81.2% |
| BM25 | 12.5% | 12.5% | 12.5% | 0.125 | 0.125 | 12.5% |
| BGE-small 384 | 0.0% | 12.5% | 12.5% | 0.042 | 0.062 | 0.0% |
| BM25 + BGE-small | 0.0% | 0.0% | 12.5% | 0.018 | 0.042 | 0.0% |


<!-- benchmarks:end -->

Top-k measures how often the intended partner appears in the first k results. MRR@10 and nDCG@10 measure its rank. Higher scores are better. Tie-adjusted top-1 divides credit between results with equal scores.

Clef and Jev use HNSW with 32 candidates. BM25 and BGE-small search all profiles. BM25 + BGE-small selects ten BM25 results, then ranks them by BGE cosine similarity.

Decision vectors perform better on this test. The profiles were built around the selected criteria, and the test is small. These results do not establish better performance on real profiles or general search.

See the [full benchmark report](docs/benchmarks.md) for methods, rule-filtered results, and HNSW recall.

## Run

**1. Install.** Use Python 3.11 or later.

```sh
git clone https://github.com/lalalune/jevector.git
cd jevector
python3 -m venv .venv-search312
.venv-search312/bin/python -m pip install -r requirements.txt
```

**2. Search the included profiles.** No API key is needed.

```sh
scripts/python -m matching.cli rank \
  --query runs/matching-v2/jev-256/family0-persona.json \
  --gallery runs/matching-v2/jev-256/*.json \
  --method hnsw --candidates 32 --k 5
```

HNSW can miss candidates. Use `--method exact` to check all profiles.

**3. Encode a new profile.** Use either provider.

For Clef, set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`, or use an existing Wrangler login.

```sh
scripts/python -m matching.cli extract --provider clef --dimensions 256 \
  --account "$CLOUDFLARE_ACCOUNT_ID" \
  --input matching/example-v2.json --output output/profile-clef.json
```

For Jev, enter the API key at the hidden prompt.

```sh
scripts/python -m matching.cli extract --provider jev --dimensions 256 --prompt-key \
  --input matching/example-v2.json --output output/profile-jev.json
```

Use `--dimensions 64` for the smaller vector. The example input has a profile ID, a description, and partner conditions. See [input and rule instructions](docs/usage.md).

**4. Run the benchmark.** This step needs both provider accounts and can incur API charges.

```sh
scripts/python -m matching.qualification --account "$CLOUDFLARE_ACCOUNT_ID" --prompt-key
scripts/python -m matching.comparison
scripts/python -m matching.render
```

The first command obtains model outputs and fills the local cache. The next commands calculate metrics and update the reports.
