from __future__ import annotations


class Book:
    def __init__(self, book_id, title, author, language, cover_path=None, chapters=None):
        ...

class Chapter:
    def __init__(self, index, title, raw_text=None, cleaned_text=None, segments=None, assets=None):
        ...

class Segment:
    def __init__(self, index, text, tokens=None, audio_path=None, subtitle_path=None):
        ...

class AudioAsset:
    def __init__(self, path, audio_format, duration_ms=None, meta=None):
        ...

class SubtitleAsset:
    def __init__(self, path, subtitle_format, language=None, meta=None):
        ...
