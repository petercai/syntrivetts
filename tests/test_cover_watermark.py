from __future__ import annotations

import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from syntrive.adapters import ffmpeg as ffmpeg_adapter
from syntrive.adapters.image.watermark import watermark_image
from syntrive.services import cover_watermark as cw

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not found on PATH")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import watermark_m4a_covers as tool  # noqa: E402


def _cover(path: Path, fmt: str, size=(500, 750), color=(235, 235, 235)) -> Path:
    Image.new("RGB", size, color).save(path, fmt)
    return path


def _m4a(path: Path, cover: Path | None, *, artist="Author A", album="Book B") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".src_raw.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", str(wav)], check=True)
    result = ffmpeg_adapter.to_m4a(str(wav), str(path), metadata={"artist": artist, "album": album, "title": "Ch 1"},
                                   cover_path=str(cover) if cover else None)
    wav.unlink()
    assert result.ok, result.error
    return path


def _embedded(path: Path) -> Image.Image:
    from mutagen.mp4 import MP4

    return Image.open(BytesIO(bytes(MP4(path).tags["covr"][0])))


def _mark_box(before: Image.Image, after: Image.Image):
    return ImageChops.difference(before.convert("RGB"), after.convert("RGB")).convert("L").point(
        lambda v: 255 if v > 60 else 0).getbbox()


@pytest.mark.parametrize("fmt, color", [("JPEG", (235, 235, 235)), ("PNG", (20, 30, 60))])
def test_image_keeps_size_and_format_and_is_red_bottom_right(tmp_path, fmt, color):
    src = _cover(tmp_path / f"c.{fmt.lower()}", fmt, color=color).read_bytes()
    out = watermark_image(src)
    before, after = Image.open(BytesIO(src)), Image.open(BytesIO(out))
    assert (after.format, after.size) == (fmt, before.size)
    left, top, right, bottom = _mark_box(before, after)
    assert left > 250 and top > 375
    reds = sum(1 for r, g, b in after.convert("RGB").crop((left, top, right, bottom)).getdata()
               if r > 150 and g < 90 and b < 90)
    assert reds > 50


def test_m4a_stamped_once_other_tags_kept(tmp_path):
    from mutagen.mp4 import MP4

    m4a = _m4a(tmp_path / "a.m4a", _cover(tmp_path / "c.jpg", "JPEG"))
    tags_before = {k: v for k, v in MP4(m4a).tags.items() if k != "covr"}
    cover_before = _embedded(m4a)

    dry = cw.watermark_m4a_cover(m4a, dry_run=True)
    assert dry.status == cw.WATERMARKED and not cw.is_marked(MP4(m4a).tags)

    assert cw.watermark_m4a_cover(m4a).status == cw.WATERMARKED
    tags = MP4(m4a).tags
    assert cw.is_marked(tags) and tags[cw.MARK_KEY][0] == cw.MARK_VALUE
    assert {k: v for k, v in tags.items() if k not in ("covr", cw.MARK_KEY)} == tags_before
    assert _mark_box(cover_before, _embedded(m4a)) is not None
    stamped = bytes(tags["covr"][0])
    assert cw.watermark_m4a_cover(m4a).status == cw.ALREADY
    assert bytes(MP4(m4a).tags["covr"][0]) == stamped
    assert ffmpeg_adapter.probe_duration_ms(str(m4a)) > 0


def test_no_cover_and_broken_file_never_raise(tmp_path):
    assert cw.watermark_m4a_cover(_m4a(tmp_path / "plain.m4a", None)).status == cw.NO_COVER
    broken = tmp_path / "broken.m4a"
    broken.write_bytes(b"not an mp4")
    result = cw.watermark_m4a_cover(broken)
    assert result.status == cw.FAILED and result.error and not result.ok


def test_repo_walk_and_tool(tmp_path, monkeypatch):
    from mutagen.mp4 import MP4

    repo = tmp_path / "repo"
    cover = _cover(tmp_path / "c.jpg", "JPEG")
    a = _m4a(repo / "PROCESSING-B" / "audiobooks" / "b_ch0001.m4a", cover)
    b = _m4a(repo / "PROCESSING-B" / "audiobooks" / "sub" / "b_ch0002.m4a", cover)
    c = _m4a(repo / "PROCESSING-C" / "audiobooks" / "c_ch0001.m4a", None, artist="X", album="Y")
    _m4a(repo / "PROCESSING-B" / "chapter_audio" / "ignored.m4a", cover)
    assert cw.find_repo_m4a_files(repo) == sorted([a, b, c])

    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    root = tool.vlc_art_root()
    if sys.platform == "darwin":
        pytest.skip("VLC cache on macOS lives under the real home folder")
    cached = tool.vlc_art_dir(root, "Author A", "Book B")
    cached.mkdir(parents=True)
    (cached / "art.jpg").write_bytes(b"old cover")

    before = {p: p.read_bytes() for p in (a, b, c)}
    assert tool.main(["-r", str(repo), "--dry-run", "--clear-vlc-cache"]) == 0
    assert {p: p.read_bytes() for p in (a, b, c)} == before and cached.is_dir()

    assert tool.main(["-r", str(repo), "--clear-vlc-cache"]) == 0
    assert all(cw.is_marked(MP4(p).tags) for p in (a, b))
    assert c.read_bytes() == before[c]
    assert not cached.exists()
    assert [r.status for r in cw.watermark_repo(repo)] == [cw.ALREADY, cw.ALREADY, cw.NO_COVER]

    (repo / "PROCESSING-B" / "audiobooks" / "bad.m4a").write_bytes(b"junk")
    assert tool.main(["-r", str(repo)]) == 1
    assert tool.main(["-r", str(tmp_path / "missing")]) == 2
