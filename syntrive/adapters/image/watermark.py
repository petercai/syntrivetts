from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont, ImageStat

logger = logging.getLogger(__name__)

RGBA = tuple[int, int, int, int]
_TRANSPARENT: RGBA = (0, 0, 0, 0)


@dataclass(frozen=True)
class WatermarkStyle:
    text: str = "SyntriveTTS"
    angle: float = 45.0
    size_ratio: float = 0.07
    min_size: int = 14
    margin_ratio: float = 0.03
    stroke_ratio: float = 0.08
    halo_px: float = 1.0
    supersample: int = 4
    drop_ratio: float = 0.1
    outline: RGBA = (215, 25, 25, 255)
    halo_on_light: RGBA = (255, 255, 255, 120)
    halo_on_dark: RGBA = (0, 0, 0, 100)
    light_background: float = 150.0


DEFAULT_STYLE = WatermarkStyle()


def _mark(style: WatermarkStyle, size: int, outline: RGBA, halo_fill: RGBA) -> Image.Image:
    ss = max(1, style.supersample)
    font = ImageFont.load_default(size=size * ss)
    stroke = max(1, round(size * ss * style.stroke_ratio))
    halo = stroke + max(1, round(style.halo_px * ss))
    left, top, right, bottom = font.getbbox(style.text, stroke_width=halo)
    pad = 2 * ss
    layer = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), _TRANSPARENT)
    draw = ImageDraw.Draw(layer)
    origin = (pad - left, pad - top)
    draw.text(origin, style.text, font=font, fill=_TRANSPARENT, stroke_width=halo, stroke_fill=halo_fill)
    draw.text(origin, style.text, font=font, fill=_TRANSPARENT, stroke_width=stroke, stroke_fill=outline)
    layer = layer.rotate(style.angle, expand=True, resample=Image.Resampling.BICUBIC)
    return layer.resize((max(1, layer.width // ss), max(1, layer.height // ss)), Image.Resampling.LANCZOS)


def watermark_image(data: bytes, style: WatermarkStyle = DEFAULT_STYLE) -> bytes:
    with Image.open(BytesIO(data)) as src:
        fmt = src.format or "PNG"
        mode = src.mode
        image = src.convert("RGBA")
    width, height = image.size
    short = min(width, height)
    size = max(style.min_size, round(short * style.size_ratio))
    probe = _mark(style, size, style.outline, style.halo_on_dark)
    square_right = (width + short) // 2
    square_bottom = (height + short) // 2
    margin = round(short * style.margin_ratio)
    x = max(0, square_right - margin - probe.width)
    y = max(0, min(height - probe.height, square_bottom - margin - probe.height + round(short * style.drop_ratio)))
    light = ImageStat.Stat(image.convert("L").crop((x, y, x + probe.width, y + probe.height))).mean[0] > style.light_background
    mark = _mark(style, size, style.outline, style.halo_on_light) if light else probe
    image.alpha_composite(mark, (x, y))

    out = BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(out, "JPEG", quality=95, subsampling=0, optimize=True)
    else:
        keep = image if "A" in mode or mode == "P" else image.convert("RGB")
        keep.save(out, fmt)
    logger.debug("watermark_image: format=%s size=%dx%d font_px=%d mark=%dx%d at=(%d,%d) halo=%s",
                 fmt, width, height, size, mark.width, mark.height, x, y, "light" if light else "dark")
    return out.getvalue()
