from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from syntrive.io import server_registry as sr


@contextmanager
def _health_server(repo_dir: Path, path: str = "/api/health"):
    body = json.dumps({"repo_dir": str(repo_dir.resolve())}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- http.server API
            status = 200 if self.path == path else 404
            self.send_response(status)
            self.end_headers()
            if status == 200:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class TestRecordIO:
    def test_write_read_remove_round_trip(self, tmp_path: Path):
        record = sr.write_record(tmp_path, "webui", 51234, host="0.0.0.0", health_path="/api/v1/health")

        assert sr.record_path(tmp_path, "webui") == tmp_path / ".syntrive" / "servers" / "webui.json"
        assert sr.read_record(tmp_path, "webui") == record
        assert record.local_url == "http://127.0.0.1:51234/"

        sr.remove_record(tmp_path, "webui")
        assert sr.read_record(tmp_path, "webui") is None
        sr.remove_record(tmp_path, "webui")

    def test_corrupt_record_counts_as_absent(self, tmp_path: Path):
        path = sr.record_path(tmp_path, "webui")
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        assert sr.read_record(tmp_path, "webui") is None

    def test_bind_free_port_returns_a_bound_socket(self):
        sock = sr.bind_free_port()
        try:
            assert sock.getsockname()[1] > 0
        finally:
            sock.close()


class TestLiveCheck:
    def test_no_record_means_free(self, tmp_path: Path):
        assert sr.find_live_server(tmp_path, "webui") is None

    def test_record_whose_server_answers_for_this_repo_is_live(self, tmp_path: Path):
        with _health_server(tmp_path, "/api/v1/health") as port:
            sr.write_record(tmp_path, "webui", port, health_path="/api/v1/health")
            live = sr.find_live_server(tmp_path, "webui")
        assert live is not None and live.port == port

    def test_record_left_by_a_dead_server_is_stale(self, tmp_path: Path):
        with _health_server(tmp_path) as port:
            pass
        sr.write_record(tmp_path, "audio_review", port)
        assert sr.find_live_server(tmp_path, "audio_review") is None

    def test_server_answering_for_another_repo_is_not_this_repos_server(self, tmp_path: Path):
        other = tmp_path / "other"
        other.mkdir()
        with _health_server(other) as port:
            sr.write_record(tmp_path, "audio_review", port)
            assert sr.find_live_server(tmp_path, "audio_review") is None

    def test_list_live_servers_for_cross_links(self, tmp_path: Path):
        with _health_server(tmp_path) as live_port:
            sr.write_record(tmp_path, "audio_review", live_port)
            sr.write_record(tmp_path, "webui", 1)
            names = [r.name for r in sr.list_live_servers(tmp_path)]
            excluded = sr.list_live_servers(tmp_path, exclude="audio_review")
        assert names == ["audio_review"]
        assert excluded == []
