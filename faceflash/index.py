"""
FaceFlash Index — stores binary face vectors for fast search.

Uses PCA+ITQ binary quantization for high-quality codes,
with Hamming distance search (Rust POPCNT) and cosine reranking.

Two storage backends:

  Legacy (in-memory + .npy directory snapshots)
      idx = FaceIndex(); idx.add(...); idx.save("my_index/"); idx.load("my_index/")
      Floats are mmap'd from disk after save/load.

  SQLite (single .db file, write-through — the v0.2 default for new code)
      idx = FaceIndex(db_path="faces.db")
      Every add() is an INSERT (WAL, crash-safe); binary codes live in one
      contiguous RAM arena rebuilt at open; float embeddings stay on disk and
      the rerank fetches only the ~100 candidate rows per query. Streaming
      insertion works out of the box: PCA is fitted once (after 1,024 faces)
      and new faces are just encode + INSERT + arena append — no refit,
      no file rewrite.
"""

import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
import json

from faceflash.pca_quantize import PCABinaryQuantizer, _POPCOUNT_TABLE, hamming_topk_numpy
from faceflash.cluster import CoarseCluster

# Try Rust backend — packaged as faceflash._core, with a fallback to a
# top-level faceflash_core build (legacy / `maturin develop` dev workflow).
try:
    from faceflash import _core as _rust
    _HAS_RUST = True
except ImportError:
    try:
        import faceflash_core as _rust
        _HAS_RUST = True
    except ImportError:
        _HAS_RUST = False


_DB_SCHEMA_VERSION = 1

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    path       TEXT NOT NULL DEFAULT '',
    code       BLOB NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BLOB
);
"""


class _Arena:
    """Append-friendly 2-D uint8/float32 buffer with geometric growth.

    Replaces the old vstack-per-flush pattern, which copied the entire
    accumulated array on every append (amortized-quadratic). Appends here are
    amortized O(1); `view` exposes the filled prefix as a normal contiguous
    ndarray (C-contiguous axis-0 slices stay contiguous, so the Rust FFI is
    happy with it).
    """

    def __init__(self, width: int, dtype):
        self.width = width
        self.dtype = dtype
        self.buf: Optional[np.ndarray] = None
        self.n = 0

    @property
    def view(self) -> Optional[np.ndarray]:
        if self.buf is None or self.n == 0:
            return None
        return self.buf[:self.n]

    def set(self, arr: Optional[np.ndarray]):
        """Replace contents wholesale (used by load/re-encode/remove)."""
        if arr is None:
            self.buf, self.n = None, 0
        else:
            self.buf = arr  # may be exact-size or mmap; growth copies on next append
            self.n = arr.shape[0]

    def append(self, rows: np.ndarray):
        m = rows.shape[0]
        if m == 0:
            return
        if self.buf is None:
            cap = max(m, 256)
            self.buf = np.empty((cap, self.width), dtype=self.dtype)
            self.buf[:m] = rows
            self.n = m
            return
        cap = self.buf.shape[0]
        if self.n + m > cap or isinstance(self.buf, np.memmap):
            new_cap = max(self.n + m, cap * 2, 256)
            new_buf = np.empty((new_cap, self.width), dtype=self.dtype)
            new_buf[:self.n] = self.buf[:self.n]
            self.buf = new_buf
        self.buf[self.n:self.n + m] = rows
        self.n += m


class FaceIndex:
    """
    Binary face vector index with two-phase search.

    Stores packed binary vectors + float vectors for reranking.
    Uses PCA+ITQ quantization (fitted after accumulating enough data).
    Falls back to random projection before fit threshold is reached.
    """

    def __init__(self, dim: int = 512, n_bits: int = 512,
                 db_path: Optional[str] = None):
        """Create an empty index (or open/create a SQLite-backed one).

        Args:
            dim: embedding dimensionality (ArcFace = 512).
            n_bits: binary code length. 512 (default) = best recall + fastest;
                256 halves memory at ~2x latency. Must be a multiple of 8.
            db_path: path to a SQLite index file. If given, the index is
                write-through: adds/removes persist immediately, codes are
                loaded into a RAM arena at open, and float embeddings stay
                on disk (fetched per-query for the rerank).
        """
        self.labels: List[str] = []
        self.paths: List[str] = []
        self.count = 0
        self.dim = dim
        self.n_bits = n_bits

        self._codes = _Arena(n_bits // 8, np.uint8)
        self._floats = _Arena(dim, np.float32)   # legacy mode only
        self._float_is_mmap = False

        # SQLite backend state
        self._db: Optional[sqlite3.Connection] = None
        self._db_path: Optional[str] = None
        self._ids: List[int] = []                # arena row -> sqlite id

        # Pending buffer — batches encode/INSERT work across many add() calls
        self._pending_embeddings: List[np.ndarray] = []
        self._pending_labels: List[str] = []
        self._pending_paths: List[str] = []

        # PCA+ITQ quantizer (fitted once we have enough data)
        self.quantizer = PCABinaryQuantizer(n_bits=n_bits)
        self._pca_fitted = False
        self._pca_fit_threshold = 1024  # Need > n_bits samples for full-rank SVD

        # Fallback: random projection (used before PCA is fitted).
        # Seeded, so it's identical across processes — codes written to the
        # DB before the fit stay consistent with a fresh process that opens it.
        rng = np.random.default_rng(42)
        self.projection = rng.standard_normal((n_bits, dim)).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)

        # Optional coarse clustering (IVF) for sub-linear scan. None = full scan.
        self.cluster: Optional[CoarseCluster] = None

        if db_path is not None:
            self._open_db(db_path)

    # ── array views (kept as properties so existing call sites still work) ──

    @property
    def vectors(self) -> Optional[np.ndarray]:
        return self._codes.view

    @vectors.setter
    def vectors(self, arr: Optional[np.ndarray]):
        self._codes.set(np.ascontiguousarray(arr) if arr is not None else None)

    @property
    def float_vectors(self) -> Optional[np.ndarray]:
        return self._floats.view

    @float_vectors.setter
    def float_vectors(self, arr: Optional[np.ndarray]):
        self._float_is_mmap = isinstance(arr, np.memmap) or (
            hasattr(arr, 'filename') and getattr(arr, 'filename', None) is not None)
        self._floats.set(arr)

    # ─────────────────────────────────────────────────────────────────────
    # SQLite backend
    # ─────────────────────────────────────────────────────────────────────

    def _open_db(self, path: str):
        """Open (or create) a SQLite-backed index and load the code arena."""
        self._db_path = str(path)
        is_new = not Path(self._db_path).exists() or Path(self._db_path).stat().st_size == 0
        # check_same_thread=False: the CLI's REST server calls into the index
        # from handler threads. All callers that share a connection across
        # threads must serialize access (faceflash.cli does, with one lock).
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        if is_new:
            # A face row is ~2.1 KB (embedding blob dominates). Default 4 KB
            # pages fit ONE row each — half the file would be padding. 16 KB
            # pages fit 7 rows. Must be set before the schema exists.
            self._db.execute("PRAGMA page_size=16384")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        # Let SQLite serve reads through OS-mapped pages — cuts syscall
        # overhead on the per-query candidate fetches.
        self._db.execute("PRAGMA mmap_size=1073741824")
        self._db.executescript(_DB_SCHEMA)

        meta = dict(self._db.execute("SELECT key, value FROM meta").fetchall())
        if meta:
            version = int(meta.get("schema_version", 1))
            if version > _DB_SCHEMA_VERSION:
                raise RuntimeError(
                    f"{path} uses index schema v{version}; this faceflash "
                    f"supports up to v{_DB_SCHEMA_VERSION} — upgrade the package")
            self.dim = int(meta["dim"])
            self.n_bits = int(meta["n_bits"])
            self._codes = _Arena(self.n_bits // 8, np.uint8)
            self._pca_fitted = bool(int(meta.get("pca_fitted", 0)))
            if self._pca_fitted:
                W = np.frombuffer(meta["pca_w"], dtype=np.float32).reshape(
                    self.n_bits, self.dim).copy()
                thr = np.frombuffer(meta["pca_threshold"], dtype=np.float32).copy()
                self.quantizer = PCABinaryQuantizer(n_bits=self.n_bits)
                self.quantizer.W = W
                self.quantizer.threshold = thr
                self.quantizer.fitted = True
        else:
            self._write_meta_static()

        self._rebuild_arena_from_db()
        self.count = len(self._ids)

    def _write_meta_static(self):
        rows = [("schema_version", str(_DB_SCHEMA_VERSION)),
                ("dim", str(self.dim)),
                ("n_bits", str(self.n_bits)),
                ("pca_fitted", "1" if self._pca_fitted else "0")]
        self._db.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", rows)
        self._db.commit()

    def _write_meta_quantizer(self):
        rows = [("pca_fitted", "1" if self._pca_fitted else "0")]
        if self._pca_fitted:
            rows += [("pca_w", self.quantizer.W.astype(np.float32).tobytes()),
                     ("pca_threshold", self.quantizer.threshold.astype(np.float32).tobytes())]
        self._db.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", rows)
        self._db.commit()

    def _rebuild_arena_from_db(self):
        """Load id/label/code columns into RAM (the scan arena). Floats stay
        on disk — this is what keeps the RAM footprint at the code size."""
        rows = self._db.execute(
            "SELECT id, name, path, code FROM faces ORDER BY id").fetchall()
        self._ids = [r[0] for r in rows]
        self.labels = [r[1] for r in rows]
        self.paths = [r[2] for r in rows]
        if rows:
            code_w = self.n_bits // 8
            flat = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.uint8)
            self._codes.set(flat.reshape(len(rows), code_w).copy())
        else:
            self._codes.set(None)

    def _fetch_by_ids(self, unique_ids: List[int]):
        """Fetch embeddings for a set of ids. Returns (matrix, id -> row map).

        Chunked to stay under SQLite's ~999 bound-variable limit, decoded as
        one buffer per chunk (cheaper than per-row frombuffer + np.stack).
        """
        CHUNK = 900
        parts, pos, base = [], {}, 0
        for s in range(0, len(unique_ids), CHUNK):
            blk = unique_ids[s:s + CHUNK]
            qmarks = ",".join("?" * len(blk))
            rows = self._db.execute(
                f"SELECT id, embedding FROM faces WHERE id IN ({qmarks})", blk).fetchall()
            parts.append(np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
                         .reshape(len(rows), self.dim))
            for j, r in enumerate(rows):
                pos[r[0]] = base + j
            base += len(rows)
        mat = parts[0] if len(parts) == 1 else np.vstack(parts)
        return mat, pos

    def _fetch_embeddings(self, arena_indices: np.ndarray) -> np.ndarray:
        """Fetch float embeddings for candidate rows (DB mode rerank path).

        ~100 point lookups by integer primary key per query — the access
        pattern SQLite is best at. Order of the result matches the order of
        `arena_indices` (the rerank maps winners back positionally).
        """
        ids = [self._ids[int(i)] for i in arena_indices]
        mat, pos = self._fetch_by_ids(ids)
        perm = np.fromiter((pos[i] for i in ids), dtype=np.intp, count=len(ids))
        return mat[perm]

    def _embeddings_for(self, arena_indices: np.ndarray) -> np.ndarray:
        """Candidate float rows, whichever backend is active."""
        if self._db is not None:
            return self._fetch_embeddings(arena_indices)
        return self.float_vectors[arena_indices]

    # ─────────────────────────────────────────────────────────────────────
    # Adding faces
    # ─────────────────────────────────────────────────────────────────────

    def _flush_pending(self):
        """Flush buffered embeddings into the main arrays."""
        if not self._pending_embeddings:
            return
        batch = np.array(self._pending_embeddings, dtype=np.float32)
        labels = self._pending_labels[:]
        paths = self._pending_paths[:]
        self._pending_embeddings.clear()
        self._pending_labels.clear()
        self._pending_paths.clear()
        # Use internal _add_batch_raw to avoid double-counting
        self._add_batch_raw(batch, labels, paths)

    def add(self, embedding: np.ndarray, label: str, path: str = ""):
        """Add a single face embedding to the index.

        Buffers internally and flushes in batches for O(n) performance.
        The index is always searchable — pending items are flushed
        automatically before search or when the PCA fit threshold is reached.
        In DB mode the flush is a transactional INSERT (streaming insertion:
        no PCA refit, no file rewrite).
        """
        if embedding.ndim != 1 or embedding.shape[0] != self.dim:
            raise ValueError(f"Expected ({self.dim},) embedding, got {embedding.shape}")

        self._pending_embeddings.append(embedding)
        self._pending_labels.append(label)
        self._pending_paths.append(path)
        self.count += 1

        # Flush when buffer is large enough or PCA threshold reached
        if len(self._pending_embeddings) >= 256 or \
           (not self._pca_fitted and self.count >= self._pca_fit_threshold):
            self._flush_pending()

    def add_batch(self, embeddings: np.ndarray, labels: List[str], paths: Optional[List[str]] = None):
        """Add a batch of face embeddings (more efficient than repeated add())."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (N, {self.dim}) embeddings, got {embeddings.shape}")
        if len(labels) != len(embeddings):
            raise ValueError(f"labels length ({len(labels)}) != embeddings length ({len(embeddings)})")
        if paths is None:
            paths = [""] * len(labels)
        self.count += len(labels)
        self._add_batch_raw(embeddings, labels, paths)

    def _encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        if self._pca_fitted:
            return self.quantizer.encode(embeddings)
        binary = (embeddings @ self.projection.T > 0).astype(np.uint8)
        return np.packbits(binary, axis=1)

    def _add_batch_raw(self, embeddings: np.ndarray, labels: List[str], paths: List[str]):
        """Internal: add batch without incrementing count (used by _flush_pending)."""
        embeddings = embeddings.astype(np.float32, copy=False)
        codes = self._encode_batch(embeddings)

        if self._db is not None:
            # Write-through: one transaction per flush.
            with self._db:
                self._db.executemany(
                    "INSERT INTO faces(name, path, code, embedding) VALUES (?, ?, ?, ?)",
                    [(labels[i], paths[i], codes[i].tobytes(), embeddings[i].tobytes())
                     for i in range(len(labels))])
            new_ids = [r[0] for r in self._db.execute(
                "SELECT id FROM faces ORDER BY id DESC LIMIT ?",
                (len(labels),)).fetchall()][::-1]
            self._ids.extend(new_ids)
        else:
            self._floats.append(embeddings)

        self.labels.extend(labels)
        self.paths.extend(paths)
        self._codes.append(codes)

        # Fit PCA once we have enough data, then re-encode everything
        if not self._pca_fitted and self.count >= self._pca_fit_threshold:
            self._fit_pca()

    # ─────────────────────────────────────────────────────────────────────
    # PCA fit
    # ─────────────────────────────────────────────────────────────────────

    def _all_embeddings(self) -> np.ndarray:
        """All float embeddings (DB mode reads them back; used only by the
        one-time PCA fit/re-encode and by build_clusters)."""
        if self._db is None:
            return self.float_vectors
        rows = self._db.execute(
            "SELECT embedding FROM faces ORDER BY id").fetchall()
        flat = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32)
        return flat.reshape(len(rows), self.dim).copy()

    def _fit_pca(self):
        """Fit PCA+ITQ quantizer on accumulated vectors and re-encode all."""
        all_embs = self._all_embeddings()
        # Random subsample, not the first N rows: enrollment order is
        # correlated (register_folder goes person-by-person), so a prefix
        # would train the quantizer on whoever enrolled first.
        n = all_embs.shape[0]
        n_fit = min(5000, n)
        if n > n_fit:
            rng = np.random.default_rng(0)
            sample = np.sort(rng.choice(n, size=n_fit, replace=False))
            self.quantizer.fit(all_embs[sample])
        else:
            self.quantizer.fit(all_embs)
        self._pca_fitted = True

        # Re-encode all existing vectors with the fitted quantizer
        new_codes = self.quantizer.encode(all_embs)
        self._codes.set(new_codes)

        if self._db is not None:
            with self._db:
                self._db.executemany(
                    "UPDATE faces SET code = ? WHERE id = ?",
                    [(new_codes[i].tobytes(), self._ids[i]) for i in range(n)])
            self._write_meta_quantizer()

    def build_clusters(self, n_clusters: Optional[int] = None, n_probe: int = 8):
        """Build a coarse-clustering (IVF) layer for sub-linear search.

        Partitions faces into ~sqrt(N) buckets; search then scans only the
        `n_probe` closest buckets instead of the whole database. This trades a
        little recall (the match may sit in an unprobed bucket) for a large
        speedup that stops growing with database size. Defaults: ~sqrt(N)
        clusters, 8 probes. Call after all faces are added.

        Note (DB mode): clustering reads all float embeddings into RAM while
        it fits — transient ~2 GB at 1M faces.
        """
        self._flush_pending()
        if self.count == 0:
            raise RuntimeError("Add faces before building clusters")
        self.cluster = CoarseCluster(n_clusters=n_clusters, n_probe=n_probe)
        self.cluster.build(self._all_embeddings())
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────

    def _encode_query(self, query_embedding: np.ndarray) -> np.ndarray:
        if self._pca_fitted:
            return self.quantizer.encode(query_embedding.reshape(1, -1)).ravel()
        query_binary = ((query_embedding @ self.projection.T) > 0).astype(np.uint8)
        return np.packbits(query_binary)

    # Above this many rows, the multi-core scan beats the serial one; below,
    # thread coordination costs more than it saves (measured crossover
    # ~200-300K on an 8-core CPU).
    _PARALLEL_SCAN_MIN_ROWS = 200_000

    def search(self, query_embedding: np.ndarray, k: int = 1,
               n_candidates: Optional[int] = None,
               n_probe: Optional[int] = None,
               parallel: Optional[bool] = None) -> List[Tuple[str, float, int]]:
        """
        Two-phase search: Hamming filter → cosine rerank.
        Uses the Rust SIMD backend when available (much faster than NumPy).

        n_candidates is the search-effort knob, independent of k (the result
        count). More candidates → higher recall + more float reranks. Defaults
        to max(100, k*10), which reaches ≥99% recall in benchmarks.

        parallel: scan across all cores. Default (None) decides by gallery
        size — parallel from ~200K rows up (measured ~1.8x at 1M), serial
        below (thread startup would dominate). Results are identical either
        way (same deterministic tie-breaking).
        """
        # Flush any buffered add() calls before searching
        self._flush_pending()

        if self.vectors is None or self.count == 0:
            return []

        query_packed = self._encode_query(query_embedding)

        # Phase 1: Hamming filter (Rust or NumPy)
        if n_candidates is None:
            n_candidates = max(100, k * 10)
        n_candidates = min(n_candidates, self.count)

        # Coarse clustering: restrict the scan to the closest buckets (sub-linear).
        # Without clustering, `pool` spans the whole database (full scan).
        if self.cluster is not None:
            pool = self.cluster.candidate_pool(query_embedding, n_probe=n_probe)
            scan_vectors = np.ascontiguousarray(self.vectors[pool])
            n_cand = min(n_candidates, len(pool))
        else:
            pool = None
            scan_vectors = self.vectors
            n_cand = n_candidates

        if _HAS_RUST:
            use_parallel = (parallel if parallel is not None
                            else len(scan_vectors) >= self._PARALLEL_SCAN_MIN_ROWS)
            kernel = (_rust.hamming_topk_parallel if use_parallel
                      else _rust.hamming_topk)
            topk = kernel(query_packed, scan_vectors, n_cand)
            local_indices = np.asarray(topk).astype(np.intp)
        else:
            local_indices = hamming_topk_numpy(query_packed, scan_vectors, n_cand)

        # Map bucket-local indices back to global face indices
        candidate_indices = pool[local_indices] if pool is not None else local_indices

        # Phase 2: Cosine rerank (floats fetched from SQLite in DB mode)
        candidates_float = self._embeddings_for(candidate_indices)
        cosine_sims = np.dot(candidates_float, query_embedding)

        sorted_indices = np.argsort(-cosine_sims)[:k]

        results = []
        for idx in sorted_indices:
            orig_idx = candidate_indices[idx]
            results.append((
                self.labels[orig_idx],
                float(cosine_sims[idx]),
                int(orig_idx)
            ))
        return results

    def search_batch(self, query_embeddings: np.ndarray, k: int = 1,
                     n_candidates: Optional[int] = None,
                     parallel: bool = True) -> List[List[Tuple[str, float, int]]]:
        """Search many queries in one shot.

        Equivalent to calling search() per row, but the Hamming phase crosses
        the Python<->Rust boundary once for the whole batch instead of once per
        query. With parallel=True the queries are scanned across CPU cores
        (embarrassingly parallel). This is the throughput path -- use it for
        benchmarks and bulk identification; use search() for single lookups.

        Coarse clustering (IVF) is not applied here: batching targets full-scan
        throughput. Call search() per query if you need the clustered path.
        """
        self._flush_pending()
        if self.vectors is None or self.count == 0:
            return [[] for _ in range(len(query_embeddings))]

        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (M, {self.dim}) queries, got {query_embeddings.shape}")
        m = query_embeddings.shape[0]

        # Encode all queries
        if self._pca_fitted:
            q_packed = self.quantizer.encode(query_embeddings)
        else:
            q_binary = (query_embeddings @ self.projection.T > 0).astype(np.uint8)
            q_packed = np.packbits(q_binary, axis=1)
        q_packed = np.ascontiguousarray(q_packed, dtype=np.uint8)

        if n_candidates is None:
            n_candidates = max(100, k * 10)
        n_cand = min(n_candidates, self.count)

        # Phase 1: one batched Hamming call -> (m, n_cand) candidate indices
        if _HAS_RUST:
            flat = _rust.hamming_topk_batch(q_packed, self.vectors, n_cand, parallel)
            cand = np.asarray(flat).astype(np.intp).reshape(m, -1)
        else:
            cand = np.stack([
                hamming_topk_numpy(q_packed[i], self.vectors, n_cand)
                for i in range(m)
            ])

        # Phase 2: cosine rerank per query (vectorized over candidates).
        # DB mode: ONE bulk fetch for every candidate in the whole batch —
        # per-query SELECTs would multiply the SQLite lookup cost by m.
        if self._db is not None:
            uniq = sorted({self._ids[int(i)] for i in cand.ravel()})
            mat, pos = self._fetch_by_ids(uniq)

            def rows_for(ci):
                perm = np.fromiter((pos[self._ids[int(i)]] for i in ci),
                                   dtype=np.intp, count=len(ci))
                return mat[perm]
        else:
            def rows_for(ci):
                return self.float_vectors[ci]

        results: List[List[Tuple[str, float, int]]] = []
        for i in range(m):
            ci = cand[i]
            sims = rows_for(ci) @ query_embeddings[i]
            order = np.argsort(-sims)[:k]
            results.append([
                (self.labels[ci[j]], float(sims[j]), int(ci[j])) for j in order
            ])
        return results

    def search_binary_only(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[str, int, int]]:
        """Search using only binary codes (fastest, lower accuracy)."""
        self._flush_pending()
        if self.vectors is None:
            return []

        query_packed = self._encode_query(query_embedding)

        if _HAS_RUST:
            topk = _rust.hamming_topk(query_packed, self.vectors, k)
            indices = np.asarray(topk).astype(np.intp)
            # Get distances for the top-k
            xor = np.bitwise_xor(self.vectors[indices], query_packed.reshape(1, -1))
            dists = _POPCOUNT_TABLE[xor].sum(axis=1)
            return [(self.labels[i], int(d), int(i)) for i, d in zip(indices, dists)]
        else:
            xor = np.bitwise_xor(self.vectors, query_packed.reshape(1, -1))
            distances = _POPCOUNT_TABLE[xor].sum(axis=1)
            top_k = np.argsort(distances)[:k]
            return [(self.labels[i], int(distances[i]), int(i)) for i in top_k]

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the index.

        A path ending in .db saves to (or copies) the SQLite format; anything
        else writes the legacy .npy directory. A DB-mode index saving to its
        own path is a no-op (it's already persistent).
        """
        self._flush_pending()
        if str(path).endswith(".db"):
            self._save_db(path)
        else:
            self._save_npy_dir(path)

    def _save_db(self, path: str):
        if self._db is not None:
            if Path(path).resolve() == Path(self._db_path).resolve():
                self._db.commit()
                return
            dest = sqlite3.connect(str(path))
            try:
                self._db.commit()
                self._db.backup(dest)
            finally:
                dest.close()
            return

        # Legacy in-memory index -> export to a fresh .db (migration path)
        if self.count == 0:
            raise RuntimeError("Nothing to save")
        p = Path(path)
        if p.exists():
            p.unlink()
        exported = FaceIndex(dim=self.dim, n_bits=self.n_bits, db_path=str(path))
        exported._pca_fitted = self._pca_fitted
        exported.quantizer = self.quantizer
        exported._write_meta_quantizer()
        embs = np.asarray(self.float_vectors, dtype=np.float32)
        codes = self.vectors
        with exported._db:
            exported._db.executemany(
                "INSERT INTO faces(name, path, code, embedding) VALUES (?, ?, ?, ?)",
                [(self.labels[i], self.paths[i], codes[i].tobytes(), embs[i].tobytes())
                 for i in range(self.count)])
        exported._db.commit()
        exported.close()

    def _save_npy_dir(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        floats = self._all_embeddings() if self._db is not None else self.float_vectors
        np.save(p / "vectors.npy", self.vectors)
        np.save(p / "float_vectors.npy", floats)
        with open(p / "metadata.json", "w") as f:
            json.dump({
                "labels": self.labels,
                "paths": self.paths,
                "count": self.count,
                "pca_fitted": self._pca_fitted,
                "far_calibration": getattr(self, "_calibration", None),
                "extra_meta": getattr(self, "_extra_meta", {}),
            }, f)
        # Persist quantizer if fitted
        if self._pca_fitted:
            self.quantizer.save(str(p / "quantizer"))
        # Persist coarse clusters if built (centroids + per-face assignment)
        if self.cluster is not None and self.cluster.centroids is not None:
            assign = np.empty(self.count, dtype=np.int32)
            for c, members in enumerate(self.cluster.members):
                assign[members] = c
            np.savez(p / "clusters.npz",
                     centroids=self.cluster.centroids,
                     assign=assign,
                     n_probe=self.cluster.n_probe)

    def load(self, path: str):
        """Load an index: a .db file opens the SQLite backend, a directory
        loads the legacy .npy snapshot (floats mmap'd)."""
        if str(path).endswith(".db"):
            self.close()
            self._open_db(path)
            return

        p = Path(path)
        self.vectors = np.load(p / "vectors.npy")
        self.float_vectors = np.load(p / "float_vectors.npy", mmap_mode='r')
        with open(p / "metadata.json") as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.paths = meta["paths"]
        self.count = meta["count"]
        self._pca_fitted = meta.get("pca_fitted", False)
        self._calibration = meta.get("far_calibration")
        self._extra_meta = meta.get("extra_meta", {}) or {}
        # Load quantizer if it was persisted
        quantizer_path = p / "quantizer"
        if quantizer_path.exists() and self._pca_fitted:
            self.quantizer.load(str(quantizer_path))
        # Restore coarse clusters if present
        cl_path = p / "clusters.npz"
        if cl_path.exists():
            d = np.load(cl_path)
            self.cluster = CoarseCluster(n_clusters=int(d["centroids"].shape[0]),
                                         n_probe=int(d["n_probe"]))
            self.cluster.centroids = d["centroids"]
            assign = d["assign"]
            self.cluster.members = [np.where(assign == c)[0].astype(np.intp)
                                    for c in range(self.cluster.centroids.shape[0])]

    def close(self):
        """Close the SQLite connection (DB mode). Safe to call twice."""
        if self._db is not None:
            self._db.commit()
            self._db.close()
            self._db = None
            self._db_path = None

    # ─────────────────────────────────────────────────────────────────────
    # Management
    # ─────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return index statistics."""
        self._flush_pending()
        mem_binary = 0 if self.vectors is None else self.vectors.nbytes
        out = {
            "count": self.count,
            "pca_fitted": self._pca_fitted,
            "rust_backend": _HAS_RUST,
            "binary_memory_mb": round(mem_binary / 1024 / 1024, 2),
        }
        if self._db is not None:
            db_size = Path(self._db_path).stat().st_size if Path(self._db_path).exists() else 0
            out.update({
                "backend": "sqlite",
                "db_path": self._db_path,
                "db_file_mb": round(db_size / 1024 / 1024, 2),
                "float_mode": "sqlite (on-disk, ~100 rows fetched per query)",
                "resident_memory_mb": round(mem_binary / 1024 / 1024, 2),
            })
        else:
            float_bytes = 0 if self.float_vectors is None else self.float_vectors.nbytes
            out.update({
                "backend": "npy",
                "float_storage_mb": round(float_bytes / 1024 / 1024, 2),
                "float_mode": ("mmap (on-disk, paged on demand)"
                               if self._float_is_mmap else "resident (in RAM)"),
                "resident_memory_mb": round(mem_binary / 1024 / 1024, 2) if self._float_is_mmap
                else round((mem_binary + float_bytes) / 1024 / 1024, 2),
                "compression_ratio": round(float_bytes / max(mem_binary, 1), 1),
            })
        return out

    def names(self) -> List[str]:
        """Return the sorted list of distinct labels currently in the index."""
        self._flush_pending()
        return sorted(set(self.labels))

    # ─────────────────────────────────────────────────────────────────────
    # FAR calibration support
    # ─────────────────────────────────────────────────────────────────────

    def sample_impostor_sims(self, n_pairs: int = 100_000, max_rows: int = 5000,
                             seed: int = 0) -> np.ndarray:
        """Cosine similarities of random impostor pairs (different labels)
        drawn from the enrolled gallery. Used to calibrate what a cosine
        threshold means in false-accept rate ON THIS population.

        Fetches at most `max_rows` embeddings (≈10 MB) and forms pairs among
        them, so it stays cheap even at 1M faces.
        """
        self._flush_pending()
        if len(set(self.labels)) < 2:
            raise RuntimeError("Calibration needs at least 2 distinct people")

        rng = np.random.default_rng(seed)
        n = self.count
        rows = (np.sort(rng.choice(n, size=max_rows, replace=False))
                if n > max_rows else np.arange(n))
        embs = self._embeddings_for(rows) if self._db is not None \
            else self.float_vectors[rows]
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        lbls = np.array([self.labels[i] for i in rows])

        sims = []
        want = n_pairs
        while len(sims) < want:
            a = rng.integers(0, len(rows), size=want)
            b = rng.integers(0, len(rows), size=want)
            keep = lbls[a] != lbls[b]
            sims.extend(np.einsum('ij,ij->i', embs[a[keep]], embs[b[keep]]).tolist())
        return np.array(sims[:want], dtype=np.float32)

    def set_meta(self, key: str, value: str):
        """Persist a small engine-level fact with the index (e.g. which
        embedding model enrolled this gallery). DB mode: meta table; legacy:
        in-memory + metadata.json on save."""
        if not hasattr(self, "_extra_meta"):
            self._extra_meta = {}
        self._extra_meta[key] = value
        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (f"x_{key}", value))
            self._db.commit()

    def get_meta(self, key: str) -> Optional[str]:
        if getattr(self, "_extra_meta", {}).get(key) is not None:
            return self._extra_meta[key]
        if self._db is not None:
            row = self._db.execute(
                "SELECT value FROM meta WHERE key = ?", (f"x_{key}",)).fetchone()
            if row:
                v = row[0]
                return v.decode() if isinstance(v, bytes) else str(v)
        return None

    def set_calibration(self, calibration: dict):
        """Persist a FAR->threshold curve (DB meta table, or in-memory +
        metadata.json for the legacy format)."""
        self._calibration = calibration
        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("far_calibration", json.dumps(calibration)))
            self._db.commit()

    def get_calibration(self) -> Optional[dict]:
        if getattr(self, "_calibration", None) is not None:
            return self._calibration
        if self._db is not None:
            row = self._db.execute(
                "SELECT value FROM meta WHERE key='far_calibration'").fetchone()
            if row:
                self._calibration = json.loads(row[0])
                return self._calibration
        return None

    def remove(self, label: str) -> int:
        """Remove every face with `label` from the index. Returns the count removed.

        DB mode: a transactional DELETE — the embedding really leaves the
        database file (right-to-be-forgotten includes disk) — then the RAM
        code arena is rebuilt from the surviving rows. Floats never enter RAM.

        Legacy mode: compacts the in-RAM arrays; if the floats were mmap'd
        from a saved index, the backing .npy is rewritten in chunks so they
        STAY mmap'd (a naive fancy-index would silently pull the whole float
        store into RAM). Any coarse-clustering layer is invalidated either way.
        """
        self._flush_pending()
        if self.count == 0:
            return 0

        if self._db is not None:
            n_removed = self._db.execute(
                "SELECT COUNT(*) FROM faces WHERE name = ?", (label,)).fetchone()[0]
            if n_removed == 0:
                return 0
            with self._db:
                self._db.execute("DELETE FROM faces WHERE name = ?", (label,))
            self._rebuild_arena_from_db()
            self.count = len(self._ids)
            self.cluster = None
            return n_removed

        keep = [i for i, l in enumerate(self.labels) if l != label]
        n_removed = self.count - len(keep)
        if n_removed == 0:
            return 0

        keep_idx = np.asarray(keep, dtype=np.intp)
        if self.vectors is not None:
            self.vectors = np.ascontiguousarray(self.vectors[keep_idx])
        self.labels = [self.labels[i] for i in keep]
        self.paths = [self.paths[i] for i in keep]
        self.count = len(keep)
        self.cluster = None  # membership indices are now stale; rebuild if needed

        if self._float_is_mmap and self.float_vectors is not None:
            self._compact_float_store(keep_idx)
        elif self.float_vectors is not None:
            self.float_vectors = np.ascontiguousarray(self.float_vectors[keep_idx])
        return n_removed

    def _compact_float_store(self, keep_idx: np.ndarray):
        """Rewrite the mmap'd float .npy without the removed rows, in chunks.

        Copies kept rows block-by-block into a new memmap and atomically
        replaces the backing file, so peak RAM stays at one block instead of
        the whole float store. Also rewrites vectors.npy/metadata.json and
        drops any stale clusters.npz so the index directory stays consistent
        with what's in memory.
        """
        import os
        src = self._floats.buf
        fpath = Path(src.filename)
        index_dir = fpath.parent
        tmp = fpath.with_name(fpath.name + ".tmp")

        out = np.lib.format.open_memmap(
            tmp, mode='w+', dtype=np.float32, shape=(len(keep_idx), self.dim))
        chunk = 8192
        for s in range(0, len(keep_idx), chunk):
            blk = keep_idx[s:s + chunk]
            out[s:s + len(blk)] = src[blk]
        out.flush()
        del out
        self._floats.set(None)   # drop the old mmap handle before replacing
        os.replace(tmp, fpath)
        self.float_vectors = np.load(fpath, mmap_mode='r')

        # Keep the rest of the saved index in sync with the compaction
        np.save(index_dir / "vectors.npy", self.vectors)
        with open(index_dir / "metadata.json", "w") as f:
            json.dump({
                "labels": self.labels,
                "paths": self.paths,
                "count": self.count,
                "pca_fitted": self._pca_fitted,
            }, f)
        cl_path = index_dir / "clusters.npz"
        if cl_path.exists():
            cl_path.unlink()        # assignments reference pre-removal indices

    def __len__(self) -> int:
        """Number of faces in the index (flushes any pending adds)."""
        self._flush_pending()
        return self.count

    def __contains__(self, label: str) -> bool:
        """`label in index` — whether any face carries this label."""
        self._flush_pending()
        return label in self.labels

    def __repr__(self) -> str:
        backend = "sqlite" if self._db is not None else "npy"
        return (f"FaceIndex(faces={self.count}, identities={len(set(self.labels))}, "
                f"n_bits={self.n_bits}, backend={backend}, rust={_HAS_RUST})")
