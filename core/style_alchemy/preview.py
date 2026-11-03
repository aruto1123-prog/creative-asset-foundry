"""絵柄候補の手続き的プレビュー SVG（Style Alchemy Arena のブラインド A/B 判定用）。

ComfyUI 未接続でも選好学習を回せるよう、実験パラメータに連続的に応答するモノクロ漫画風の
サンプル画を決定論的に生成する。

- ``style.line_taper``           → 主線の入り抜き（端ほど細くなる可変幅ストローク）
- ``style.black_fill_ratio``     → 黒ベタ面積（髪・地面・背景の黒塊）
- ``style.shadow_contrast``      → 影のコントラスト（柔らかいハッチング ⇔ 硬いアニメ影）
- ``style.screentone_density``   → 網点トーンの密度と粒径
- ``style.perspective_distortion`` → パース線の本数と消失点の誇張
- ``merge.*`` の平均             → 頭身・目の大きさ（デフォルメ度）
- ``lora.*.strength`` の平均     → 髪の房数・線の黒さ（LoRA の効き）
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape, quoteattr

from core.common import stable_seed

__all__ = ["render_candidate_preview_svg"]


def _unit(params: Mapping[str, float], key: str, default: float = 0.5) -> float:
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(1.0, max(0.0, value))


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _tapered_path(points: Sequence[tuple[float, float]], base_width: float, taper: float) -> str:
    """中心線から入り抜き付きストロークの塗り輪郭パスを作る。"""
    n = len(points)
    if n < 2:
        return ""
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(points):
        prev_x, prev_y = points[max(0, i - 1)]
        next_x, next_y = points[min(n - 1, i + 1)]
        dx, dy = next_x - prev_x, next_y - prev_y
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        t = i / (n - 1)
        profile = (1.0 - taper) + taper * math.sin(math.pi * t)
        half = max(0.25, base_width * (0.35 + 0.65 * profile)) / 2.0
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    outline = left + list(reversed(right))
    return "M" + " L".join(f"{_fmt(x)},{_fmt(y)}" for x, y in outline) + " Z"


def _arc_points(cx: float, cy: float, rx: float, ry: float, start: float, end: float, steps: int = 18) -> list[tuple[float, float]]:
    return [
        (cx + rx * math.cos(start + (end - start) * k / steps), cy + ry * math.sin(start + (end - start) * k / steps))
        for k in range(steps + 1)
    ]


def render_candidate_preview_svg(
    params: Mapping[str, float],
    *,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    label: str | None = None,
) -> str:
    """実験パラメータに応答する決定論的なモノクロ漫画風サンプル SVG を返す。"""
    width = max(64, int(width))
    height = max(64, int(height))
    taper = _unit(params, "style.line_taper")
    black = _unit(params, "style.black_fill_ratio")
    shadow = _unit(params, "style.shadow_contrast")
    tone = _unit(params, "style.screentone_density")
    perspective = _unit(params, "style.perspective_distortion")
    merge_values = [float(v) for k, v in params.items() if k.startswith("merge.") and isinstance(v, (int, float))]
    merge_mean = min(1.0, max(0.0, sum(merge_values) / len(merge_values))) if merge_values else 0.5
    lora_values = [float(v) for k, v in params.items() if k.startswith("lora.") and k.endswith(".strength") and isinstance(v, (int, float))]
    lora_mean = min(1.0, max(0.0, (sum(lora_values) / len(lora_values)) / 1.5)) if lora_values else 0.5
    canonical = json.dumps({k: round(float(v), 6) for k, v in sorted(params.items()) if isinstance(v, (int, float))}, sort_keys=True)
    rng = random.Random(stable_seed("caf-preview", seed, canonical))
    uid = f"p{stable_seed(canonical, seed) % 16**8:08x}"
    w, h = float(width), float(height)
    margin = max(8.0, w * 0.03)
    ink = f"#{int(40 - 35 * lora_mean):02x}{int(40 - 35 * lora_mean):02x}{int(46 - 38 * lora_mean):02x}"
    parts: list[str] = []

    dot_spacing = 10.0 - 6.0 * tone
    dot_radius = 0.5 + 1.9 * tone
    hatch_gap = 9.0 - 6.0 * shadow
    hatch_width = 0.6 + 1.6 * shadow
    parts.append(
        "<defs>"
        f'<pattern id="{uid}-tone" width="{_fmt(dot_spacing)}" height="{_fmt(dot_spacing)}" patternUnits="userSpaceOnUse">'
        f'<circle cx="{_fmt(dot_spacing / 2)}" cy="{_fmt(dot_spacing / 2)}" r="{_fmt(dot_radius)}" fill="{ink}"/></pattern>'
        f'<pattern id="{uid}-hatch" width="{_fmt(hatch_gap)}" height="{_fmt(hatch_gap)}" patternUnits="userSpaceOnUse" patternTransform="rotate(40)">'
        f'<line x1="0" y1="0" x2="0" y2="{_fmt(hatch_gap)}" stroke="{ink}" stroke-width="{_fmt(hatch_width)}"/></pattern>'
        f'<clipPath id="{uid}-frame"><rect x="{_fmt(margin)}" y="{_fmt(margin)}" width="{_fmt(w - 2 * margin)}" height="{_fmt(h - 2 * margin)}"/></clipPath>'
        "</defs>"
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    scene: list[str] = []

    # パース線（消失点の高さと本数がパース歪みに応答）
    vp_x = w * (0.5 + (rng.random() - 0.5) * 0.3)
    vp_y = h * (0.42 - 0.3 * perspective)
    horizon = vp_y + h * 0.08
    line_count = int(8 + 26 * perspective)
    for k in range(line_count):
        angle = math.pi * (0.05 + 0.9 * (k + rng.random() * 0.4) / max(1, line_count))
        reach = max(w, h) * 1.6
        x2 = vp_x + math.cos(angle) * reach * (1.0 if k % 2 else -1.0)
        y2 = vp_y + math.sin(angle) * reach
        scene.append(
            f'<line x1="{_fmt(vp_x)}" y1="{_fmt(vp_y)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" stroke="{ink}" '
            f'stroke-width="{_fmt(0.5 + 0.7 * perspective)}" stroke-opacity="0.45"/>'
        )

    # 地面の網点トーンと黒ベタ
    scene.append(
        f'<rect x="0" y="{_fmt(horizon)}" width="{_fmt(w)}" height="{_fmt(h - horizon)}" fill="url(#{uid}-tone)" fill-opacity="{_fmt(0.25 + 0.75 * tone)}"/>'
    )
    ground_black = h * (0.06 + 0.32 * black)
    scene.append(
        f'<path d="M0,{_fmt(h)} L0,{_fmt(h - ground_black)} Q{_fmt(w * 0.35)},{_fmt(h - ground_black * (1.3 + rng.random() * 0.3))} '
        f'{_fmt(w * 0.6)},{_fmt(h - ground_black * 0.9)} T{_fmt(w)},{_fmt(h - ground_black * 1.1)} L{_fmt(w)},{_fmt(h)} Z" fill="{ink}"/>'
    )
    if black > 0.45:
        block_w = w * (0.12 + 0.3 * (black - 0.45))
        scene.append(
            f'<rect x="{_fmt(w - margin - block_w)}" y="{_fmt(margin)}" width="{_fmt(block_w)}" height="{_fmt(horizon - margin)}" fill="{ink}"/>'
        )

    # 人物（頭身・目の大きさはマージ平均、髪の房数は LoRA 強度に応答）
    head_r = w * (0.095 + 0.05 * merge_mean)
    cx = w * (0.46 + (rng.random() - 0.5) * 0.08)
    cy = h * 0.42
    body_top = cy + head_r * 0.95
    shoulder = head_r * (1.7 - 0.4 * merge_mean)
    body = (
        f"M{_fmt(cx - shoulder)},{_fmt(h - margin)} L{_fmt(cx - shoulder * 0.85)},{_fmt(body_top + head_r * 0.6)} "
        f"Q{_fmt(cx)},{_fmt(body_top - head_r * 0.1)} {_fmt(cx + shoulder * 0.85)},{_fmt(body_top + head_r * 0.6)} "
        f"L{_fmt(cx + shoulder)},{_fmt(h - margin)} Z"
    )
    scene.append(f'<path d="{body}" fill="#ffffff"/>')
    if shadow > 0.7:
        scene.append(
            f'<path d="M{_fmt(cx + shoulder * 0.1)},{_fmt(body_top + head_r * 0.2)} L{_fmt(cx + shoulder)},{_fmt(h - margin)} '
            f'L{_fmt(cx + shoulder * 0.2)},{_fmt(h - margin)} Z" fill="{ink}"/>'
        )
    else:
        scene.append(f'<path d="{body}" fill="url(#{uid}-hatch)" fill-opacity="{_fmt(0.25 + 0.6 * shadow)}"/>')
    scene.append(f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(head_r)}" ry="{_fmt(head_r * 1.12)}" fill="#ffffff"/>')
    if shadow > 0.7:
        scene.append(
            f'<path d="M{_fmt(cx + head_r * 0.15)},{_fmt(cy - head_r * 1.1)} A{_fmt(head_r)},{_fmt(head_r * 1.12)} 0 0 1 '
            f'{_fmt(cx + head_r * 0.15)},{_fmt(cy + head_r * 1.1)} Q{_fmt(cx + head_r * 0.5)},{_fmt(cy)} '
            f'{_fmt(cx + head_r * 0.15)},{_fmt(cy - head_r * 1.1)} Z" fill="{ink}"/>'
        )
    else:
        scene.append(
            f'<ellipse cx="{_fmt(cx + head_r * 0.45)}" cy="{_fmt(cy)}" rx="{_fmt(head_r * 0.55)}" ry="{_fmt(head_r)}" '
            f'fill="url(#{uid}-hatch)" fill-opacity="{_fmt(0.2 + 0.6 * shadow)}"/>'
        )
    tufts = 5 + int(round(6 * lora_mean))
    hair_points: list[str] = []
    for k in range(tufts + 1):
        angle = math.pi * (1.05 + 0.9 * k / tufts)
        radius = head_r * (1.18 + (0.28 if k % 2 else 0.05) + rng.random() * 0.12)
        hair_points.append(f"{_fmt(cx + math.cos(angle) * radius)},{_fmt(cy + math.sin(angle) * radius * 1.05)}")
    hair_fill = ink if black > 0.35 else f"url(#{uid}-hatch)"
    scene.append(
        f'<path d="M{_fmt(cx - head_r * 1.02)},{_fmt(cy + head_r * 0.1)} L{" L".join(hair_points)} '
        f'L{_fmt(cx + head_r * 1.02)},{_fmt(cy + head_r * 0.1)} Q{_fmt(cx)},{_fmt(cy - head_r * 0.35)} '
        f'{_fmt(cx - head_r * 1.02)},{_fmt(cy + head_r * 0.1)} Z" fill="{hair_fill}"/>'
    )
    eye_r = head_r * (0.12 + 0.1 * merge_mean)
    for side in (-1.0, 1.0):
        ex = cx + side * head_r * 0.38
        ey = cy + head_r * 0.18
        scene.append(f'<ellipse cx="{_fmt(ex)}" cy="{_fmt(ey)}" rx="{_fmt(eye_r * 0.8)}" ry="{_fmt(eye_r * 1.2)}" fill="{ink}"/>')
        scene.append(f'<circle cx="{_fmt(ex - eye_r * 0.25)}" cy="{_fmt(ey - eye_r * 0.4)}" r="{_fmt(eye_r * 0.3)}" fill="#ffffff"/>')

    # 入り抜きのある主線（輪郭・肩・髪の流れ）
    base_width = 2.0 + 4.5 * (0.4 + 0.6 * lora_mean)
    strokes = [
        _arc_points(cx, cy, head_r, head_r * 1.12, math.pi * 0.05, math.pi * 0.95),
        _arc_points(cx, cy, head_r, head_r * 1.12, math.pi * 0.95, math.pi * 1.95),
        [(cx - shoulder * 0.85, body_top + head_r * 0.6), (cx - shoulder * 0.5, body_top + head_r * 0.2), (cx, body_top + head_r * 0.05)],
        [(cx + shoulder * 0.85, body_top + head_r * 0.6), (cx + shoulder * 0.5, body_top + head_r * 0.2), (cx, body_top + head_r * 0.05)],
        [(cx - shoulder * 0.85, body_top + head_r * 0.6), (cx - shoulder * 0.95, h * 0.8), (cx - shoulder, h - margin)],
        [(cx + shoulder * 0.85, body_top + head_r * 0.6), (cx + shoulder * 0.95, h * 0.8), (cx + shoulder, h - margin)],
    ]
    for k in range(3):
        offset = (k - 1) * head_r * 0.35
        strokes.append(
            [
                (cx + offset, cy - head_r * 1.2),
                (cx + offset * 1.2 + head_r * 0.1, cy - head_r * 0.6),
                (cx + offset * 1.4 + head_r * 0.2 * (rng.random() - 0.5), cy - head_r * 0.15),
            ]
        )
    for points in strokes:
        path = _tapered_path(points, base_width, taper)
        if path:
            scene.append(f'<path d="{path}" fill="{ink}"/>')

    # 集中線（パースとベタが強いほど増える）
    burst = int(12 * perspective * black)
    for k in range(burst):
        angle = 2.0 * math.pi * k / max(1, burst) + rng.random() * 0.2
        inner = max(w, h) * 0.42
        outer = max(w, h) * 0.75
        scene.append(
            f'<line x1="{_fmt(cx + math.cos(angle) * inner)}" y1="{_fmt(cy + math.sin(angle) * inner)}" '
            f'x2="{_fmt(cx + math.cos(angle) * outer)}" y2="{_fmt(cy + math.sin(angle) * outer)}" stroke="{ink}" stroke-width="1.2"/>'
        )

    parts.append(f'<g clip-path="url(#{uid}-frame)">' + "".join(scene) + "</g>")
    parts.append(
        f'<rect x="{_fmt(margin)}" y="{_fmt(margin)}" width="{_fmt(w - 2 * margin)}" height="{_fmt(h - 2 * margin)}" '
        f'fill="none" stroke="{ink}" stroke-width="{_fmt(3.0 + 3.0 * lora_mean)}"/>'
    )
    if label:
        font = max(12.0, w * 0.05)
        parts.append(
            f'<rect x="{_fmt(w - margin - font * (0.9 + 0.62 * len(label)))}" y="{_fmt(h - margin - font * 1.6)}" '
            f'width="{_fmt(font * (0.9 + 0.62 * len(label)))}" height="{_fmt(font * 1.6)}" fill="#ffffff" stroke="{ink}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{_fmt(w - margin - font * 0.45)}" y="{_fmt(h - margin - font * 0.5)}" text-anchor="end" '
            f'font-family="Inter, Noto Sans JP, sans-serif" font-size="{_fmt(font)}" font-weight="700" fill="{ink}">{escape(label)}</text>'
        )
    raw_title = label if label else "style candidate preview"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label={quoteattr(raw_title)}><title>{escape(raw_title)}</title>' + "".join(parts) + "</svg>"
    )
