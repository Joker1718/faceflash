# FaceFlash

Face search that keeps 1M faces in 61 MB of RAM at the same recall@1 as exact brute-force cosine. ArcFace embeddings are stored as 512-bit binary codes instead of full float vectors, scanned by Hamming distance, and the top ~100 candidates are reranked with exact cosine.

One thing to understand up front: the 61 MB is the in-RAM index. The original float vectors (~2 GB at 1M) stay on disk — since v0.2 in a single SQLite file — and the rerank reads only the ~100 candidate rows per query. So it's 61 MB of RAM plus ~2 GB of disk, not 61 MB total. RAM is usually the scarce resource, which is why the number is framed that way.

This is a compression + packaging project, not a new search algorithm. The search is a brute-force Hamming scan (same as FAISS `IndexBinaryFlat`) plus a cosine rerank. You could build the same pipeline from `faiss.PCAMatrix` + `IndexBinaryFlat` + manual reranking; FaceFlash packages it behind one API and tunes the SIMD kernel. The reason it works without losing accuracy is that ArcFace embeddings are low-rank, so PCA+ITQ binary codes preserve nearest-neighbor ordering. On general or random vectors this does not hold (see [Limitations](#limitations)).

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/faceflash.svg)](https://pypi.org/project/faceflash/)
[![CI](https://github.com/raghavenderreddygrudhanti/faceflash/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavenderreddygrudhanti/faceflash/actions)

</div>

```python
from faceflash import FaceFlash

ff = FaceFlash("faces.db")             # single SQLite file; created if missing.
                                       # (downloads ArcFace model ~166 MB on first run)
ff.register("Alice", "alice.jpg")      # persisted immediately — no save() needed
ff.register("Bob", "bob.jpg")
ff.register_folder("employees/")       # bulk enroll (folder/person_name/photo.jpg)

result = ff.search("query.jpg")        # detect → embed → binary search → cosine rerank
# {"matches": [{"name": "Alice", "confidence": 0.92}], ...}

ff.verify("photo1.jpg", "photo2.jpg")  # 1:1 — same person?
# {"match": True, "confidence": 0.87}

ff.remove("Bob")                       # deletes from the .db file too (GDPR)
```

Reopen `faces.db` tomorrow and keep adding faces — streaming insertion needs no PCA refit and no file rewrite. Binary codes load into RAM at open (~92ms at 100K); float embeddings stay on disk and the rerank fetches only the ~100 candidate rows per query, which costs about +0.2ms over RAM-resident floats. If you want the old explicit-snapshot workflow, `ff.save("my_index/")` still writes the .npy directory format, and either format converts to the other via `save()`/`load()`.

There's also a batch path (`ff.index.search_batch(embeddings, k=1)`) for bulk dedup, watchlists, and video frames — roughly 11× the throughput of one-at-a-time at 100K.

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/demo.gif" width="720" alt="FaceFlash live demo"/>

</div>

---

## The numbers

All FaceFlash numbers use the default 512-bit config, measured on MS1MV2 embeddings with an AMD EPYC 9355 (AVX-512). Recall@1 = returns the same nearest neighbor as exact brute-force cosine (FAISS-Flat ground truth). That's a statement about the compression being lossless for retrieval — not about ArcFace being perfect; embedding-model limits (pose, occlusion, age) are upstream.

| | FaceFlash | HNSWLIB (ef=128) | USearch (exp=256) | FAISS-Flat |
|---|---|---|---|---|
| **Recall@1** | **100%** | 100% | 99.1–99.9% | 100% (ground truth) |
| **Index RAM @ 100K** | **6.1 MB** | 222 MB | 112 MB | 195 MB |
| **Index RAM @ 1M** | **61 MB** | 2,217 MB | 1,118 MB | 1,953 MB |
| **Single-query @ 100K** | **0.40ms** | 0.60ms | 0.60ms | 4.99ms |
| **Single-query @ 1M** | 3.40ms | 0.65ms | 0.94ms | 51.6ms |

Notes on this table:

- All memory is *measured* (saved-index file size — HNSWLIB `save_index`, USearch `save`, FAISS `write_index`), not estimated. Earlier published versions used estimates that ran ~30% high for competitors.
- Competitors are tuned, not defaults: HNSWLIB gets an ef sweep and USearch an expansion sweep; the table shows each at its best swept recall.
- The 1M scale is tiled from 644K real embeddings (the smaller scales are not). Recall scoring is tie-aware — an equidistant duplicate counts as correct for every method — so tiling doesn't favor anyone.
- FaceFlash's latency here reranks from *disk* just fine: the mmap-rerank row measured 3.26ms at 1M on NVMe, indistinguishable from RAM-resident floats.

| Scale | Recall@1 | Single-query | Batched QPS | Index RAM |
|---|---|---|---|---|
| 100K | 100% | 0.40ms | 28,792 | 6.1 MB |
| 200K | 100% | 0.70ms | 18,422 | 12.2 MB |
| 300K | 100% | 1.20ms | 13,858 | 18.3 MB |
| 500K | 100% | 1.70ms | 9,091 | 30.5 MB |
| 1M | 100% | 3.40ms | 5,057 | 61 MB |

At 100K the scan is faster per query than HNSW (0.40ms vs 0.60ms) because the codes fit in cache; at 200K they're roughly tied; from ~300K up HNSW's O(log N) graph wins on single-query latency while FaceFlash keeps the ~36× smaller footprint. A 256-bit config halves the index at the same recall for about 2× the single-query latency.

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_memory_bar.png" width="760" alt="Measured index memory at 100K faces"/>

<sub>Measured index memory at 100K (256-bit config): the gap is float-vs-binary storage, not a smarter index.</sub>

</div>

<details>
<summary>Per-scale benchmark tables (all methods, measured 2026-07-02)</summary>

### 100K Faces (6,939 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (512b/100c, batched) | 100.0% | 0.035ms | 28,792 | 6.1 MB |
| FaceFlash (512b/100c) | 100.0% | 0.398ms | 2,512 | 6.1 MB |
| FaceFlash (512b/100c, mmap rerank) | 100.0% | 0.327ms | 3,059 | 6.1 MB |
| FAISS-BinaryFlat + rerank | 100.0% | 0.195ms | 5,135 | 6.1 MB |
| FAISS-BinaryHNSW + rerank | 95.8% | 0.066ms | 15,202 | 32.1 MB |
| HNSWLIB (ef=128) | 100.0% | 0.604ms | 1,656 | 221.7 MB |
| HNSWLIB (ef=128, batched) | 100.0% | 0.175ms | 5,721 | 221.7 MB |
| USearch (exp=256) | 99.1% | 0.599ms | 1,670 | 111.8 MB |
| USearch (exp=128, batched) | 99.5% | 0.011ms | 88,115 | 111.8 MB |
| FAISS-Flat (exact) | 100.0% | 4.987ms | 201 | 195.3 MB |
| ScaNN (leaves=316) | 98.4% | 0.104ms | 9,620 | 12.2 MB |

### 200K Faces (13,749 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (512b/100c, batched) | 100.0% | 0.054ms | 18,422 | 12.2 MB |
| FaceFlash (512b/100c) | 100.0% | 0.704ms | 1,420 | 12.2 MB |
| FaceFlash (512b/100c, mmap rerank) | 100.0% | 0.636ms | 1,573 | 12.2 MB |
| FAISS-BinaryFlat + rerank | 100.0% | 0.343ms | 2,915 | 12.2 MB |
| FAISS-BinaryHNSW + rerank | 91.6% | 0.075ms | 13,345 | 64.1 MB |
| HNSWLIB (ef=128) | 99.6% | 0.657ms | 1,523 | 443.3 MB |
| HNSWLIB (ef=128, batched) | 99.7% | 0.178ms | 5,608 | 443.3 MB |
| USearch (exp=256) | 99.6% | 0.681ms | 1,468 | 223.7 MB |
| USearch (exp=128, batched) | 99.2% | 0.044ms | 22,714 | 223.7 MB |
| FAISS-Flat (exact) | 100.0% | 9.921ms | 101 | 390.6 MB |
| ScaNN (leaves=447) | 97.2% | 0.189ms | 5,298 | 24.4 MB |

### 300K Faces (20,615 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (512b/100c, batched) | 100.0% | 0.072ms | 13,858 | 18.3 MB |
| FaceFlash (512b/100c) | 100.0% | 1.200ms | 833 | 18.3 MB |
| FaceFlash (512b/100c, mmap rerank) | 100.0% | 0.943ms | 1,060 | 18.3 MB |
| FAISS-BinaryFlat + rerank | 100.0% | 0.490ms | 2,041 | 18.3 MB |
| FAISS-BinaryHNSW + rerank | 86.3% | 0.087ms | 11,547 | 96.2 MB |
| HNSWLIB (ef=128) | 99.8% | 0.794ms | 1,260 | 665.0 MB |
| HNSWLIB (ef=128, batched) | 99.7% | 0.177ms | 5,664 | 665.0 MB |
| USearch (exp=256) | 99.6% | 0.806ms | 1,241 | 335.4 MB |
| USearch (exp=128, batched) | 99.4% | 0.015ms | 64,733 | 335.4 MB |
| FAISS-Flat (exact) | 100.0% | 14.406ms | 69 | 585.9 MB |
| ScaNN (leaves=547) | 97.8% | 0.271ms | 3,691 | 36.6 MB |

### 500K Faces (34,328 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (512b/100c, batched) | 100.0% | 0.110ms | 9,091 | 30.5 MB |
| FaceFlash (512b/100c) | 100.0% | 1.698ms | 589 | 30.5 MB |
| FaceFlash (512b/100c, mmap rerank) | 100.0% | 1.657ms | 603 | 30.5 MB |
| FAISS-BinaryFlat + rerank | 100.0% | 0.865ms | 1,157 | 30.5 MB |
| FAISS-BinaryHNSW + rerank | 78.0% | 0.146ms | 6,843 | 160.3 MB |
| HNSWLIB (ef=128) | 99.8% | 0.723ms | 1,383 | 1,108.3 MB |
| HNSWLIB (ef=128, batched) | 99.8% | 0.165ms | 6,047 | 1,108.3 MB |
| USearch (exp=256) | 99.7% | 0.834ms | 1,200 | 559.0 MB |
| USearch (exp=128, batched) | 99.6% | 0.020ms | 49,647 | 559.0 MB |
| FAISS-Flat (exact) | 100.0% | 24.301ms | 41 | 976.6 MB |
| ScaNN (leaves=707) | 97.6% | 0.438ms | 2,285 | 61.0 MB |

### 1M Faces (44,291 identities) — tiled from 644K real embeddings

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (512b/100c, batched) | 100.0% | 0.198ms | 5,057 | 61.0 MB |
| FaceFlash (512b/100c) | 100.0% | 3.402ms | 294 | 61.0 MB |
| FaceFlash (512b/100c, mmap rerank) | 100.0% | 3.257ms | 307 | 61.0 MB |
| FAISS-BinaryFlat + rerank | 100.0% | 1.711ms | 585 | 61.0 MB |
| FAISS-BinaryHNSW + rerank | 61.7% | 0.127ms | 7,894 | 320.6 MB |
| HNSWLIB (ef=128) | 100.0% | 0.650ms | 1,539 | 2,216.7 MB |
| HNSWLIB (ef=128, batched) | 100.0% | 0.175ms | 5,698 | 2,216.7 MB |
| USearch (exp=256) | 99.9% | 0.943ms | 1,061 | 1,118.3 MB |
| USearch (exp=128, batched) | 98.6% | 0.017ms | 57,457 | 1,118.3 MB |
| FAISS-Flat (exact) | 100.0% | 51.596ms | 19 | 1,953.1 MB |
| ScaNN (leaves=1000) | 98.0% | 0.835ms | 1,198 | 122.1 MB |

</details>

### 1:N identification, 44,290 distinct people

One photo per person in the gallery, identify them from a different photo — the hardest test:

| Method | Rank-1 Accuracy | Memory |
|---|---|---|
| FAISS-Flat (exact ceiling) | 95.8% | 86.5 MB |
| **FaceFlash (512b / 100c)** | **95.8%** | **2.70 MB** |
| FaceFlash (256b / 100c) | 95.6% | 1.35 MB |

The binary codes lose no rank-1 accuracy after rerank. That's an empirical result on ArcFace embeddings, not a general guarantee.

### What a confidence threshold actually buys you

The `confidence` field is raw cosine similarity, not a probability — 0.92 does not mean "92% sure". What a threshold means in error rates, measured on LFW (15K genuine / 200K impostor pairs, `benchmarks/bench_far_frr.py`):

| Threshold | False accept rate | False reject rate |
|---|---|---|
| 0.30 | 0.012% | 3.4% |
| **0.40 (default)** | < 0.001% | 10.1% |
| 0.50 | < 0.001% | 27.2% |

The equal-error rate is ~1.0% at threshold ~0.17. The 0.4 default is deliberately conservative: essentially zero false accepts, at the cost of rejecting ~10% of genuine matches. If your gallery photos are good and a false reject just means "try again", 0.4 is right; if false rejects are expensive, drop toward 0.3. Rerun the script on embeddings from your own population — these rates shift with demographics and photo quality.

### Is the win the compression or the search?

The compression. Two benchmarks prove it: `bench_compression_isolation.py` runs the same PCA+ITQ codes through FaceFlash's Rust kernel and FAISS `IndexBinaryFlat` with identical reranking — same recall, so the kernel does nothing accuracy-wise. And `bench_ann_comparison.py` includes `IndexBinaryFlat` and `IndexBinaryHNSW` rows over the same 64-byte codes, so the compression win and the search-structure cost are separate columns you can read off.

Two results from that worth knowing (measured, same codes, same top-100 rerank):

- **Binary-HNSW sheds recall as the database grows: 95.8% at 100K → 78.0% at 500K → 61.7% at 1M**, while the exhaustive scan holds 100% at every scale. The graph over Hamming space increasingly misses candidates the scan finds. (500K is untiled real data, so this isn't a tiling artifact.) That, not the SIMD kernel, is the argument for scanning.
- **FAISS `IndexBinaryFlat` is actually ~2× faster than FaceFlash's Rust kernel single-query** (0.195ms vs 0.398ms at 100K; 1.71ms vs 3.40ms at 1M) at identical recall. FAISS's binary scan is simply better optimized. FaceFlash's speed edge is the batched path (0.198ms/query effective at 1M) — for single queries, the honest statement is that the win is the compression and the packaging, not the kernel.

### Methodology, briefly

All published numbers are from one full RunPod run (2026-07-02) on MS1MV2 (645,019 embeddings, 44,291 identities), ArcFace w600k_r50, L2-normalized. Ground truth is exact brute-force cosine. Single-query rows are single-threaded; batched rows use all cores on both sides. Memory is measured from saved-index file size where the library supports it (HNSWLIB `save_index`, USearch `save`, FAISS `write_index`); each result row's `memory_note` says whether it was measured or estimated. A dedicated mmap-rerank row measures FaceFlash latency with floats on disk — the mode the RAM number describes. `bash scripts/runpod_ms1m.sh` reproduces everything end-to-end.

---

## Install

```bash
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

Needs a [Rust toolchain](https://rustup.rs) to compile the AVX-512/NEON backend; falls back to NumPy without it. Add `,benchmark` to the extras to get the comparison dependencies. For best accuracy, feed it pre-aligned 112×112 face crops — 5-point alignment (SCRFD/RetinaFace) adds +1.28 accuracy points over a basic center-crop.

## When to use it

- Edge / mobile / IoT — megabytes of RAM instead of hundreds; floats stay on disk
- Multi-tenant servers — 100 galleries × 61 MB = 6 GB of RAM (plus ~200 GB of float stores on disk); HNSW needs its ~220 GB *in RAM*
- 10K–150K galleries — the sweet spot where the scan beats HNSW on latency *and* memory (they tie at ~200K)
- Offline / air-gapped — a Raspberry Pi or cheap VPS, no GPU, no network

Use HNSW instead when you're past ~300K faces and single-query latency matters more than RAM, or when you want years of production hardening.

## How it works

1. ArcFace extracts a 512-d float embedding
2. PCA aligns the quantization with the axes where identity varies most
3. ITQ rotates the bits so each carries maximal information
4. The Rust kernel scans all codes — AVX-512 VPOPCNTDQ counts a whole 512-bit code's set bits in one popcount instruction (each compare is still load + XOR + popcount + accumulate)
5. Exact cosine on the top ~100 Hamming candidates picks the winner

512 bits is the fastest setting because the whole code fits in one AVX-512 register. Config options if you need them:

```python
ff = FaceFlash(n_bits=128, n_candidates=500)   # minimum RAM (16 B/face, 99.4% recall)
ff = FaceFlash(n_bits=256, n_candidates=100)   # half the memory, ~2x the latency
ff = FaceFlash(n_bits=512, n_candidates=100)   # default, fastest
ff.search("query.jpg", n_candidates=200)       # per-query override
```

Past 500K you can trade recall for speed with IVF clustering: `ff.index.build_clusters(n_probe=16)` gives ~5× at ~88% recall, `n_probe=32` gives ~2.8× at ~93%.

```
faceflash/
├── engine.py         high-level API (register, search, verify)
├── detect.py         face detection (SCRFD + Haar fallback)
├── align.py          5-point alignment to the ArcFace template
├── embed.py          ArcFace ONNX embedding (auto-download)
├── index.py          binary index + batched search
└── pca_quantize.py   PCA+ITQ quantizer (the core)
rust/src/lib.rs       AVX-512 VPOPCNTDQ / NEON / scalar POPCNT kernels
```

## Limitations

- Single-query latency at 1M+ is O(N); HNSW is ~5× faster there (0.65ms vs 3.40ms). The batched path ties. FAISS IndexBinaryFlat also scans the same codes ~2× faster single-query.
- The RAM number is not the storage number: 61 MB @ 1M in RAM, ~2 GB of floats on disk for reranking.
- With the SQLite backend, floats never enter RAM (fetched per query, ~+0.2ms). The legacy in-memory index holds all floats in RAM until `save()`/`load()` moves them to mmap'd disk. Batched throughput is lower on the SQLite path (~3K vs ~14K qps at 100K) — use the legacy format for bulk identification jobs.
- The rerank pages ~100 float rows from disk per query — on NVMe this measured indistinguishable from RAM-resident floats (3.26ms vs 3.40ms at 1M); slow storage adds more.
- The single-popcount kernel needs Ice Lake / Zen 4+ / EPYC 9004+; older CPUs fall back to AVX2 or scalar automatically.
- The recall result is specific to face embeddings (low-rank). Don't expect it on arbitrary vectors.

## Reproduce

```bash
# Local quick check (LFW / VGGFace2 100K)
python scripts/extract_lfw_embeddings.py
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# Full suite on any Linux box
bash scripts/runpod_ms1m.sh
```

Or the [Colab notebook](https://colab.research.google.com/github/raghavenderreddygrudhanti/faceflash/blob/main/examples/faceflash_reproduce_colab.ipynb) — 5 minutes on a free CPU, no tokens. It runs a NumPy fallback (no Rust on Colab) that uses the same deterministic (distance, index) tie-breaking as the Rust kernel, so recall matches the table above; memory numbers are hardware-independent either way.

## Contributing

```bash
git clone https://github.com/raghavenderreddygrudhanti/faceflash.git
cd faceflash && python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,benchmark]" && maturin develop --release
python -m pytest tests/   # ~20s
```

Things I haven't gotten to, roughly in order of how much I want them: a DiskANN comparison (the one competitor missing), mobile deployment (ONNX + CoreML), Raspberry Pi / Jetson benchmarks, a WebAssembly build, GPU batched search. (Streaming insertion shipped in v0.2 with the SQLite backend.) Issues are tagged; see [CONTRIBUTING.md](CONTRIBUTING.md). Done work lives in the [CHANGELOG](CHANGELOG.md).

## Credits

FaceFlash doesn't introduce a new embedding model or hashing algorithm — the contribution is the packaging and the measurement. It builds on:

- **PCA+ITQ** — Gong & Lazebnik, *Iterative Quantization*, CVPR 2011 (the core method)
- **ArcFace** — Deng et al., CVPR 2019; weights from [InsightFace](https://github.com/deepinsight/insightface)
- **Detection/alignment** — RetinaFace (CVPR 2020) / SCRFD (ICLR 2022), InsightFace
- **Data** — MS-Celeb-1M (ECCV 2016, MS1MV2 cleanup), LFW (UMass TR 2007)
- **Baselines** — FAISS, HNSW (Malkov & Yashunin), ScaNN, [USearch](https://github.com/unum-cloud/usearch)

## License

MIT — see [LICENSE](LICENSE). If this is useful to you, a star helps others find it.
