# Changelog

All notable changes to FaceFlash are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Model download was broken**: the ArcFace URL returned 401; now points at
  the InsightFace w600k_r50 URL (the weights every benchmark used; the old
  `arcface_r100.onnx` filename was mislabeled w600k_r50). Legacy caches reused.
- **Deterministic tie-breaking** across all Hamming top-k paths (Rust serial /
  parallel / batched and the NumPy fallback) — all backends now return the
  identical candidate set. This removes the ~1% Colab-vs-Rust recall gap.
- **`remove()` kept the float store mmap'd**: it previously materialized the
  whole float array into RAM and left the removed embedding on disk. Now the
  backing .npy is compacted in chunks and the on-disk index stays consistent
  (right-to-be-forgotten includes disk).
- PCA now fits on a random 5,000-row sample instead of the first 5,000
  (enrollment order is identity-correlated via `register_folder`).
- `min_age=0` / `max_age=0` filters were silently ignored (truthiness bug).
- `align_face` returned a crash instead of skipping degenerate landmarks;
  `detect_faces` now filters failed alignments per-face.
- `search`/`search_batch` no longer exclude one database vector from the
  shortlist (`count - 1` clamp leftover).

### Changed
- Benchmarks measure competitor memory from saved-index file sizes instead of
  multiplier estimates (old estimates ran ~30% high), sweep USearch
  `expansion_search` like HNSWLIB's ef, add an mmap-rerank row and
  same-codes FAISS BinaryFlat/BinaryHNSW baselines, and record tiling.
- `embed_batch` does real stacked ONNX inference on CPU/CUDA (looped on
  CoreML, where per-shape compilation makes batching slower).

### Added
- `benchmarks/bench_far_frr.py` — FAR/FRR at cosine thresholds (LFW pairs),
  documenting what the 0.4 default threshold means in error rates.

## [0.1.0] - 2026-06-27

First public release. (Everything below shipped in the initial release; earlier
0.1/0.2/0.3 labels were internal development milestones, never published.)

### Added
- **Core retrieval** — PCA+ITQ binary quantization with a two-phase search
  (Hamming filter → exact cosine rerank). 100% Recall@1 vs exact search at
  36–73× less memory than HNSW (measured saved-index sizes, 512/256-bit configs), on MS1MV2 from 100K to 1M faces.
- **High-level API** (`FaceFlash`) — `register`, `register_folder`, `search`,
  `verify`, plus index management: `remove`, `names`, `save`, `load`, `stats`,
  and Pythonic `len()` / `in` / `repr()`.
- **Low-level index** (`FaceIndex`) — direct embedding add/search, `add_batch`,
  `search_batch` (batched bulk throughput), and `build_clusters` (IVF).
- **Rust SIMD backend** — runtime-dispatched AVX-512 VPOPCNTDQ (x86, ~3× faster
  than scalar), NEON (ARM), and scalar POPCNT fallback. Multi-core, cache-blocked
  batched search (10–17× throughput vs single-query serial on a 128-thread EPYC).
- **5-point alignment** — SCRFD/RetinaFace alignment to the ArcFace template for
  raw photos (+1.28 LFW points over a basic center-crop), with a Haar fallback.
- **Coarse clustering (IVF)** — optional sub-linear scan for large galleries.
- **Packaging** — `pip install faceflash` ships the Rust backend; CPU/GPU
  inference via the `[cpu]` / `[gpu]` extras. Fully type-hinted (PEP 561).
- **Benchmarks** — reproducible suite vs FAISS, HNSWLIB, USearch, and ScaNN at
  100K–1M, with tie-aware recall, a fair batched-throughput comparison, and
  measured (not estimated) FaceFlash memory. 1:N identification: 95.8% rank-1
  on 44,290 identities (ties exact search).

### Known limitations
- Single-query latency at 1M is O(N); HNSW is faster there (batched ties).
- Competitor memory figures are estimated (vectors + index-overhead formula);
  FaceFlash's memory is measured.
- The 1M benchmark scale tiles 645K real embeddings 2× (flagged in results).

[0.1.0]: https://github.com/raghavenderreddygrudhanti/faceflash/releases/tag/v0.1.0
