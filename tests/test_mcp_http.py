from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client  # noqa: E402

from syntrive.io import server_registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_WINDOWS = sys.platform == "win32"
_URL = re.compile(r"syntrive-mcp \(streamable HTTP\): (http://\S+)")
_TOKEN = re.compile(r"Authorization: Bearer (\S+?)'")


class McpHttpProcess:
    def __init__(self, repo: Path, *extra: str) -> None:
        self.lines: list[str] = []
        self.url = self.token = ""
        ready = threading.Event()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "syntrive.mcp.server", "-r", str(repo), "--transport", "http", "--port", "0", *extra],
            cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _WINDOWS else 0,
        )

        def drain() -> None:
            for line in self.proc.stderr:
                self.lines.append(line.rstrip())
                if match := _URL.search(line):
                    self.url = match.group(1)
                if match := _TOKEN.search(line):
                    self.token = match.group(1)
                if self.url and (self.token or "--host" not in extra):
                    ready.set()
            ready.set()

        threading.Thread(target=drain, daemon=True).start()
        ready.wait(60)

    @property
    def base(self) -> str:
        return self.url.rsplit("/mcp", 1)[0]

    def stop(self) -> int:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT if _WINDOWS else signal.SIGINT)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        return self.proc.returncode

    def log(self) -> str:
        return "\n".join(self.lines[-30:])


@pytest.fixture()
def repo(tmp_path) -> Path:
    from syntrive.db.session import ensure_db_schema

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    ensure_db_schema(repo_dir / "syntrivetts.db")
    return repo_dir


def _call(url: str, tool: str, args: dict | None = None, headers: dict | None = None):
    async def run():
        transport = streamable_http_client(url, http_client=create_mcp_http_client(headers=headers)) if headers else url
        async with Client(transport, read_timeout_seconds=60) as client:
            if tool == "__list__":
                return await client.list_tools()
            return await client.call_tool(tool, args or {})

    return asyncio.run(run())


def _http(url: str, *, method: str = "GET", headers: dict | None = None, body: bytes | None = None) -> int:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_tools_over_http_record_and_single_instance(repo):
    server = McpHttpProcess(repo)
    try:
        assert server.url.startswith("http://127.0.0.1:") and server.url.endswith("/mcp"), server.log()
        record = server_registry.read_record(repo, "mcp")
        assert record is not None and record.health_path == "/api/health" and f":{record.port}/" in server.url
        assert server_registry.find_live_server(repo, "mcp") is not None

        tools = _call(server.url, "__list__").tools
        assert len(tools) == 57
        info = _call(server.url, "repo_info")
        assert not info.is_error and info.structured_content["data"]["transport"] == "http"
        assert info.structured_content["data"]["repo_dir"] == repo.resolve().as_posix()
        switched = _call(server.url, "repo_switch", {"repo_dir": str(repo)})
        assert switched.is_error and "refused: This HTTP server is bound to" in switched.content[0].text

        second = McpHttpProcess(repo)
        assert second.proc.wait(30) == 1
        assert any("already has an MCP HTTP server" in line for line in second.lines), second.log()
    finally:
        server.stop()
    assert any("mcp_http_stopped" in line for line in server.lines), server.log()
    assert not server_registry.record_path(repo, "mcp").exists(), "record left behind:\n" + server.log()


def test_lan_mode_requires_the_bearer_token(repo):
    try:
        probe = server_registry.bind_port("127.0.0.2", 0)
        probe.close()
    except OSError:
        pytest.skip("127.0.0.2 is not a usable loopback address here (macOS by default)")
    server = McpHttpProcess(repo, "--host", "127.0.0.2")
    try:
        assert server.token and server.url.startswith("http://127.0.0.2:"), server.log()
        assert server.token not in (Path(repo) / ".syntrive" / "logs" / "mcp.log").read_text(encoding="utf-8")

        init = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
        post = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        assert _http(server.url, method="POST", headers=post, body=init) == 401
        assert _http(server.url, method="POST", headers={**post, "Authorization": "Bearer wrong"}, body=init) == 401
        assert _http(f"{server.base}/api/health") == 200

        info = _call(server.url, "repo_info", headers={"Authorization": f"Bearer {server.token}"})
        assert not info.is_error and info.structured_content["data"]["transport"] == "http"
    finally:
        server.stop()


def test_bind_port_refuses_a_busy_port():
    taken = server_registry.bind_port("127.0.0.1", 0)
    taken.listen()
    port = taken.getsockname()[1]
    try:
        with pytest.raises(OSError):
            server_registry.bind_port("127.0.0.1", port)
    finally:
        taken.close()
    free = server_registry.bind_port("127.0.0.1", 0)
    assert isinstance(free, socket.socket) and free.getsockname()[1] > 0
    free.close()
