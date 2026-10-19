# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Draw line/polygon zones and their live counts on a frame."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from autovision.tracking.analytics import LineZone, PolygonZone

__all__ = ["draw_line_zone", "draw_polygon_zone", "draw_zones"]

RGB = tuple[int, int, int]


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=12)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: RGB,
    image_size: tuple[int, int],
) -> None:
    font = _font()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width, height = right - left + 8, bottom - top + 4
    x = min(max(0, int(xy[0])), max(0, image_size[0] - width))
    y = min(max(0, int(xy[1])), max(0, image_size[1] - height))
    draw.rectangle((x, y, x + width, y + height), fill=color)
    draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)


def draw_line_zone(
    image: Image.Image,
    zone: LineZone,
    *,
    color: RGB = (0, 200, 255),
    thickness: int = 2,
) -> Image.Image:
    """Draw a line zone and its cumulative in/out counts."""
    draw = ImageDraw.Draw(image)
    draw.line([zone.start, zone.end], fill=color, width=max(1, int(thickness)))
    radius = 3
    for point in (zone.start, zone.end):
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
        )
    counts = zone.counts
    midpoint = (
        (zone.start[0] + zone.end[0]) / 2.0,
        (zone.start[1] + zone.end[1]) / 2.0 - 18,
    )
    _label(
        draw,
        midpoint,
        f"{zone.name}: in {counts.in_count} out {counts.out_count}",
        color,
        image.size,
    )
    return image


def draw_polygon_zone(
    image: Image.Image,
    zone: PolygonZone,
    *,
    color: RGB = (0, 200, 255),
    thickness: int = 2,
) -> Image.Image:
    """Draw a polygon zone and its current in-zone count."""
    draw = ImageDraw.Draw(image)
    points = [tuple(float(value) for value in point) for point in zone.polygon]
    for start, end in zip(points, points[1:] + points[:1]):
        draw.line([start, end], fill=color, width=max(1, int(thickness)))
    centroid = zone.polygon.mean(axis=0)
    _label(
        draw,
        (float(centroid[0]), float(centroid[1])),
        f"{zone.name}: {zone.counts.total}",
        color,
        image.size,
    )
    return image


def draw_zones(
    image: Image.Image,
    line_zones: Sequence[LineZone] = (),
    polygon_zones: Sequence[PolygonZone] = (),
) -> Image.Image:
    """Draw every configured zone (polygons first, lines on top)."""
    for polygon_zone in polygon_zones:
        image = draw_polygon_zone(image, polygon_zone)
    for line_zone in line_zones:
        image = draw_line_zone(image, line_zone)
    return image
