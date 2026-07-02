"""FaceFlash command line — a face-search database without writing Python.

    faceflash register --db faces.db --name Alice alice.jpg
    faceflash register-folder --db faces.db employees/
    faceflash search --db faces.db query.jpg
    faceflash search-all --db faces.db group_photo.jpg
    faceflash verify a.jpg b.jpg
    faceflash calibrate --db faces.db
    faceflash remove --db faces.db --name Bob
    faceflash names --db faces.db
    faceflash stats --db faces.db
    faceflash serve --db faces.db --port 8080

All output is JSON (one object per line), so it pipes into jq cleanly.
The --db file is created on first use; state persists between invocations.
"""

import argparse
import json
import sys


def _engine(args):
    from faceflash.engine import FaceFlash
    return FaceFlash(index_path=args.db)


def _emit(obj):
    print(json.dumps(obj, ensure_ascii=False))


def cmd_register(args):
    ff = _engine(args)
    from faceflash.engine import NoFaceFoundError
    try:
        _emit(ff.register(args.name, args.image,
                          allow_fallback=args.allow_fallback))
    except NoFaceFoundError as e:
        _emit({"registered": False, "error": str(e)})
        return 1
    return 0


def cmd_register_folder(args):
    ff = _engine(args)
    _emit(ff.register_folder(args.folder, progress=not args.quiet))
    return 0


def cmd_search(args):
    ff = _engine(args)
    kwargs = {"k": args.k}
    if args.max_far is not None:
        kwargs["max_far"] = args.max_far
    else:
        kwargs["threshold"] = args.threshold
    _emit(ff.search(args.image, **kwargs))
    return 0


def cmd_search_all(args):
    ff = _engine(args)
    kwargs = {"k": args.k}
    if args.max_far is not None:
        kwargs["max_far"] = args.max_far
    else:
        kwargs["threshold"] = args.threshold
    _emit(ff.search_all(args.image, **kwargs))
    return 0


def cmd_verify(args):
    from faceflash.engine import FaceFlash
    ff = FaceFlash(index_path=args.db)   # db optional for verify
    kwargs = {}
    if args.max_far is not None:
        kwargs["max_far"] = args.max_far
    _emit(ff.verify(args.image1, args.image2, **kwargs))
    return 0


def cmd_calibrate(args):
    ff = _engine(args)
    cal = ff.calibrate(n_pairs=args.pairs)
    _emit({"calibrated": True,
           "n_pairs": cal["n_pairs"],
           "curve": [{"far": f, "threshold": round(t, 4)} for f, t in cal["curve"]]})
    return 0


def cmd_remove(args):
    ff = _engine(args)
    n = ff.remove(args.name)
    _emit({"removed": n, "name": args.name, "index_size": len(ff)})
    return 0


def cmd_names(args):
    ff = _engine(args)
    _emit({"names": ff.names(), "count": len(ff)})
    return 0


def cmd_stats(args):
    ff = _engine(args)
    _emit(ff.stats())
    return 0


def cmd_serve(args):
    """Tiny REST server over the index. Demo-grade: no auth, single-threaded
    writes. Put it behind a reverse proxy for anything real.

      GET  /stats                    index statistics
      GET  /names                    enrolled people
      POST /search?k=1&max_far=…     body = image bytes -> matches
      POST /search_all?k=1           body = image bytes -> per-face matches
      POST /register?name=Alice      body = image bytes -> enroll
      DELETE /person/<name>          remove a person
    """
    import tempfile
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs, unquote

    ff = _engine(args)
    from faceflash.engine import NoFaceFoundError
    # One lock around every engine call: handler threads share one SQLite
    # connection, and SQLite connections are not concurrency-safe. Requests
    # serialize; ThreadingHTTPServer still keeps slow clients from blocking
    # the accept loop.
    lock = threading.Lock()

    def _with_image_tempfile(body, fn):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            tmp.write(body)
            tmp.close()
            return fn(tmp.name)
        finally:
            os.unlink(tmp.name)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, status=200):
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *a):
            if not args.quiet:
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/stats":
                with lock:
                    self._json(ff.stats())
            elif path == "/names":
                with lock:
                    self._json({"names": ff.names(), "count": len(ff)})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not body:
                self._json({"error": "empty body — POST the image bytes"}, 400)
                return
            try:
                if u.path == "/search":
                    kwargs = {"k": int(q.get("k", ["1"])[0])}
                    if "max_far" in q:
                        kwargs["max_far"] = float(q["max_far"][0])
                    elif "threshold" in q:
                        kwargs["threshold"] = float(q["threshold"][0])
                    with lock:
                        self._json(_with_image_tempfile(
                            body, lambda p: ff.search(p, **kwargs)))
                elif u.path == "/search_all":
                    kwargs = {"k": int(q.get("k", ["1"])[0])}
                    if "max_far" in q:
                        kwargs["max_far"] = float(q["max_far"][0])
                    with lock:
                        self._json(_with_image_tempfile(
                            body, lambda p: ff.search_all(p, **kwargs)))
                elif u.path == "/register":
                    name = q.get("name", [None])[0]
                    if not name:
                        self._json({"error": "register needs ?name="}, 400)
                        return
                    with lock:
                        self._json(_with_image_tempfile(
                            body, lambda p: ff.register(name, p)))
                else:
                    self._json({"error": "not found"}, 404)
            except NoFaceFoundError as e:
                self._json({"error": str(e)}, 422)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_DELETE(self):
            path = urlparse(self.path).path
            if path.startswith("/person/"):
                name = unquote(path[len("/person/"):])
                with lock:
                    self._json({"removed": ff.remove(name), "name": name})
            else:
                self._json({"error": "not found"}, 404)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"faceflash serving {args.db} on http://{args.host}:{args.port}  "
          f"({len(ff)} faces)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faceflash",
        description="Face search that fits 1M faces in 61 MB of RAM.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_db(sp, required=True):
        sp.add_argument("--db", required=required, default=None,
                        help="index file (SQLite, created if missing)")

    sp = sub.add_parser("register", help="enroll one face")
    add_db(sp)
    sp.add_argument("--name", required=True)
    sp.add_argument("image")
    sp.add_argument("--allow-fallback", action="store_true",
                    help="enroll even if no face was detected (center crop)")
    sp.set_defaults(fn=cmd_register)

    sp = sub.add_parser("register-folder", help="bulk enroll folder/person/*.jpg")
    add_db(sp)
    sp.add_argument("folder")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_register_folder)

    for name, fn, help_ in (("search", cmd_search, "identify the main face"),
                            ("search-all", cmd_search_all, "identify every face")):
        sp = sub.add_parser(name, help=help_)
        add_db(sp)
        sp.add_argument("image")
        sp.add_argument("-k", type=int, default=1)
        sp.add_argument("--threshold", type=float, default=0.4)
        sp.add_argument("--max-far", type=float, default=None,
                        help="decision threshold as a false-accept rate, e.g. 0.001")
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("verify", help="are these the same person?")
    add_db(sp, required=False)
    sp.add_argument("image1")
    sp.add_argument("image2")
    sp.add_argument("--max-far", type=float, default=None)
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("calibrate", help="measure FAR thresholds on this gallery")
    add_db(sp)
    sp.add_argument("--pairs", type=int, default=100_000)
    sp.set_defaults(fn=cmd_calibrate)

    sp = sub.add_parser("remove", help="delete a person (also from disk)")
    add_db(sp)
    sp.add_argument("--name", required=True)
    sp.set_defaults(fn=cmd_remove)

    sp = sub.add_parser("names", help="list enrolled people")
    add_db(sp)
    sp.set_defaults(fn=cmd_names)

    sp = sub.add_parser("stats", help="index statistics")
    add_db(sp)
    sp.set_defaults(fn=cmd_stats)

    sp = sub.add_parser("serve", help="tiny REST API over the index")
    add_db(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_serve)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
