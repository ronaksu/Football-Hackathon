"""Local-only server for the Model Failure Review page.

Serves review/ over http://127.0.0.1:8765 and persists reviewer feedback to
review/feedback/feedback.json.

Local on purpose and in two senses. It binds to the loopback interface only, so nothing is
reachable from the network; and the data it serves is the open-access IDSSE development set, never
a competition asset. Feedback is recorded for review -- it is NOT wired into training, because
reviewer-supplied labels have to be read before they are trusted.

    python review/server.py
"""

import argparse
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(ROOT, "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")
LOCK = threading.Lock()


def read_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return {}
    with open(FEEDBACK_FILE) as fh:
        try:
            return json.load(fh)
        except ValueError:
            return {}


def write_feedback(data):
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    tmp = FEEDBACK_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, FEEDBACK_FILE)      # atomic, so an interrupted save cannot truncate the file


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        # The exporter rewrites data/*.json in place, so a cached copy means the page quietly
        # shows the numbers from the previous run.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        SimpleHTTPRequestHandler.end_headers(self)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/feedback"):
            with LOCK:
                return self._json(read_feedback())
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/feedback"):
            return self._json({"error": "unknown endpoint"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            return self._json({"error": "payload too large"}, 413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        case_id = str(payload.get("case_id") or "").strip()
        if not case_id:
            return self._json({"error": "case_id required"}, 400)
        with LOCK:
            data = read_feedback()
            entry = data.get(case_id, {})
            for field in ("bookmarked", "verdict", "proposed_label", "comment"):
                if field in payload:
                    entry[field] = payload[field]
            entry["case_id"] = case_id
            entry["updated"] = payload.get("updated")
            # Reviewer input is stored, never promoted. Anything proposed here is a suggestion
            # until it has been read and accepted deliberately.
            entry["used_for_training"] = False
            data[case_id] = entry
            write_feedback(data)
        return self._json({"ok": True, "saved": case_id})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("Model Failure Review  ->  http://127.0.0.1:{}/app.html".format(args.port))
    print("serving {} (loopback only)".format(ROOT))
    print("feedback -> {}".format(FEEDBACK_FILE))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
