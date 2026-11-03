"""ベクター原稿＆SVGネームレンダラー。

1ページを断ち切り込みの viewBox（-bleed 〜 width+bleed）で描画する。

- ガイド: 仕上がり線（トリム）・断ち切り線・基本枠・トンボ（表示切替可）
- コマ枠: 多角形。``bleed_sides`` の辺は断ち切り外まで延長し、その辺には枠線を描かない
- 画像レイヤー: ``image_hrefs`` があれば clipPath でコマ形状に切り抜いて配置、無ければネーム風下描き
  （人物の頭部・胴体の楕円、視線、地平線、集中線、ベタ目安、カメラ注記）
- フキダシ: 通常（楕円＋尾）・叫び（ギザギザ爆発形）・モノローグ（雲形・尾なし）・フラッシュ（ウニフラ放射線）
- 縦書きテキスト: 1文字ずつ配置し、句読点・括弧は縦書き用字形へ置換、長音等は90°回転、小書き仮名は右上へ寄せる
- オノマトペ: 回転・極太・白縁取り
- 視線誘導矢印: 番号付き、逆行は赤（``show_gaze_vectors``）

``layers(storyboard)`` は同じ内容をレイヤー構造 JSON として出力する（他ツールへの受け渡し用）。
"""

from __future__ import annotations

import math
from typing import Any
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field

from core.common import stable_seed
from core.manga_pipeline.geometry import compute_gaze_transitions, polygon_bbox
from core.manga_pipeline.schemas import (
    ANGLE_LABELS_JA,
    Balloon,
    BalloonKind,
    Page,
    PageGeometry,
    Panel,
    PanelShape,
    ShotSize,
    SoundEffect,
    Storyboard,
)
from core.manga_pipeline.text_layout import ROTATE_IN_VERTICAL, SMALL_KANA, VERTICAL_FORMS, fit_font_size

__all__ = ["RenderOptions", "SVGLayoutRenderer", "BALLOON_TEXT_INSET"]

Pt = tuple[float, float]

# 楕円に内接する文字枠の係数（幅比² + 高さ比² ≤ 1 を満たす）
BALLOON_TEXT_INSET: dict[str, float] = {
    BalloonKind.NORMAL.value: 0.68,
    BalloonKind.SHOUT.value: 0.62,
    BalloonKind.FLASH.value: 0.6,
    BalloonKind.MONOLOGUE.value: 0.64,
}


class RenderOptions(BaseModel):
    """描画オプション。"""

    model_config = ConfigDict(extra="ignore")

    show_gaze_vectors: bool = False
    show_panel_labels: bool = True
    show_bleed_guides: bool = True
    frame_stroke: float = Field(4.0, ge=0.0, le=20.0)
    draft_sketch: bool = True
    image_hrefs: dict[str, str] = Field(default_factory=dict)
    font_family: str = "'Noto Sans JP', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', sans-serif"
    background: str = "#ffffff"
    ink: str = "#111111"


def _attr(value: Any) -> str:
    return escape(str(value), {'"': "&quot;", "'": "&apos;"})


def _num(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _points_attr(points: list[Pt]) -> str:
    return " ".join(f"{_num(x)},{_num(y)}" for x, y in points)


class SVGLayoutRenderer:
    """ネーム（Storyboard）の SVG レンダラー。"""

    def __init__(self, options: RenderOptions | None = None) -> None:
        self.options = options or RenderOptions()

    # -- 幾何補助 -----------------------------------------------------------------

    @staticmethod
    def extended_points(panel: Panel, page: Page, geometry: PageGeometry) -> list[Pt]:
        """断ち切り辺を断ち切り外まで延長したコマ多角形。"""
        fx0, fy0, fx1, fy1 = geometry.frame_rect(page.side)
        bx0, by0, bx1, by1 = geometry.bleed_rect()
        sides = set(panel.bleed_sides)
        tolerance = 1.5
        result: list[Pt] = []
        for point in panel.polygon.points:
            x, y = point.x, point.y
            if "top" in sides and abs(y - fy0) <= tolerance:
                y = by0
            if "bottom" in sides and abs(y - fy1) <= tolerance:
                y = by1
            if "left" in sides and abs(x - fx0) <= tolerance:
                x = bx0
            if "right" in sides and abs(x - fx1) <= tolerance:
                x = bx1
            result.append((x, y))
        return result

    @staticmethod
    def stroked_edges(points: list[Pt], geometry: PageGeometry) -> list[tuple[Pt, Pt]]:
        """枠線を描く辺（断ち切り境界上の辺を除く）。"""
        bx0, by0, bx1, by1 = geometry.bleed_rect()
        edges: list[tuple[Pt, Pt]] = []
        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            on_bleed = (
                (abs(a[1] - by0) < 1e-6 and abs(b[1] - by0) < 1e-6)
                or (abs(a[1] - by1) < 1e-6 and abs(b[1] - by1) < 1e-6)
                or (abs(a[0] - bx0) < 1e-6 and abs(b[0] - bx0) < 1e-6)
                or (abs(a[0] - bx1) < 1e-6 and abs(b[0] - bx1) < 1e-6)
            )
            if not on_bleed:
                edges.append((a, b))
        return edges

    # -- 要素描画 -----------------------------------------------------------------

    def _guides(self, geometry: PageGeometry, page: Page) -> str:
        if not self.options.show_bleed_guides:
            return ""
        w, h, b = geometry.width, geometry.height, geometry.bleed
        fx0, fy0, fx1, fy1 = geometry.frame_rect(page.side)
        marks: list[str] = []
        mark = b * 0.8
        for cx, cy in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            dx = -mark if cx == 0.0 else mark
            dy = -mark if cy == 0.0 else mark
            marks.append(
                f'<line class="trim-mark" x1="{_num(cx)}" y1="{_num(cy)}" x2="{_num(cx + dx)}" y2="{_num(cy)}" stroke="#d946ef" stroke-width="0.8"/>'
                f'<line class="trim-mark" x1="{_num(cx)}" y1="{_num(cy)}" x2="{_num(cx)}" y2="{_num(cy + dy)}" stroke="#d946ef" stroke-width="0.8"/>'
            )
        return (
            '<g class="guides">'
            f'<rect class="guide-bleed" x="{_num(-b)}" y="{_num(-b)}" width="{_num(w + 2 * b)}" height="{_num(h + 2 * b)}" fill="none" stroke="#d946ef" stroke-width="1" stroke-dasharray="2 4"/>'
            f'<rect class="guide-trim" x="0" y="0" width="{_num(w)}" height="{_num(h)}" fill="none" stroke="#06b6d4" stroke-width="1.2" stroke-dasharray="8 4"/>'
            f'<rect class="guide-frame" x="{_num(fx0)}" y="{_num(fy0)}" width="{_num(fx1 - fx0)}" height="{_num(fy1 - fy0)}" fill="none" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="4 6"/>'
            + "".join(marks)
            + "</g>"
        )

    def _sketch(self, panel: Panel, page: Page, extended: list[Pt]) -> str:
        """画像が無いコマのネーム風下描き。"""
        x0, y0, x1, y1 = panel.polygon.core_box()
        w, h = x1 - x0, y1 - y0
        ink = self.options.ink
        parts: list[str] = [f'<polygon points="{_points_attr(extended)}" fill="#fbfbfb"/>']
        shot = panel.camera.shot
        if shot in (ShotSize.ELS, ShotSize.FS):
            horizon = y0 + h * (0.62 if panel.camera.angle.value != "high" else 0.35)
            parts.append(f'<line class="sketch-horizon" x1="{_num(x0)}" y1="{_num(horizon)}" x2="{_num(x1)}" y2="{_num(horizon)}" stroke="#9ca3af" stroke-width="1.2"/>')
        if panel.tension >= 0.75:
            cx, cy = x0 + w * 0.5, y0 + h * 0.5
            rng_seed = stable_seed("caf.speedlines", panel.id)
            count = 18
            for k in range(count):
                angle = 2 * math.pi * k / count + (rng_seed % 100) / 100.0
                r_outer = math.hypot(w, h) * 0.6
                r_inner = min(w, h) * (0.32 + 0.08 * ((rng_seed >> k) & 1))
                parts.append(
                    f'<line class="sketch-speedline" x1="{_num(cx + r_inner * math.cos(angle))}" y1="{_num(cy + r_inner * math.sin(angle))}" '
                    f'x2="{_num(cx + r_outer * math.cos(angle))}" y2="{_num(cy + r_outer * math.sin(angle))}" stroke="#d1d5db" stroke-width="1"/>'
                )
        if page.black_ratio_target >= 0.45 and panel.tension >= 0.6:
            beta_h = h * min(0.45, page.black_ratio_target * 0.5)
            parts.append(
                f'<rect class="sketch-beta" x="{_num(x0)}" y="{_num(y1 - beta_h)}" width="{_num(w)}" height="{_num(beta_h)}" fill="#1f2937" opacity="0.18"/>'
            )
        for character in panel.characters:
            px = x0 + character.position.x * w
            py = y0 + character.position.y * h
            head_r = max(6.0, 0.16 * character.scale * min(w, h) * (1.4 if shot in (ShotSize.CU, ShotSize.ECU) else 1.0))
            head_y = py - (0.25 * character.scale * h if shot not in (ShotSize.CU, ShotSize.ECU) else 0.0)
            if shot not in (ShotSize.ECU, ShotSize.CU):
                body_rx = head_r * 1.1
                body_ry = character.scale * h * 0.3
                parts.append(
                    f'<ellipse class="sketch-body" cx="{_num(px)}" cy="{_num(head_y + head_r + body_ry)}" rx="{_num(body_rx)}" ry="{_num(body_ry)}" fill="none" stroke="#6b7280" stroke-width="1.6"/>'
                )
            parts.append(
                f'<circle class="sketch-head" data-character="{_attr(character.character_id)}" cx="{_num(px)}" cy="{_num(head_y)}" r="{_num(head_r)}" fill="#ffffff" stroke="{ink}" stroke-width="1.8"/>'
            )
            gx, gy = character.gaze.x, character.gaze.y
            parts.append(
                f'<line class="sketch-gaze" x1="{_num(px)}" y1="{_num(head_y)}" x2="{_num(px + gx * head_r * 1.8)}" y2="{_num(head_y + gy * head_r * 1.8)}" stroke="#ef4444" stroke-width="1.4"/>'
            )
        if self.options.show_panel_labels:
            label = f"{panel.camera.shot.value}・{ANGLE_LABELS_JA[panel.camera.angle.value]}・{panel.camera.lens_mm}mm"
            parts.append(
                f'<text class="camera-label" x="{_num(x0 + 8)}" y="{_num(y0 + 18)}" font-size="13" fill="#6b7280">{escape(label)}</text>'
            )
        return "".join(parts)

    def _frame(self, panel: Panel, extended: list[Pt], geometry: PageGeometry) -> str:
        if panel.shape == PanelShape.BORDERLESS or self.options.frame_stroke <= 0:
            return f'<path class="panel-frame" data-panel-id="{_attr(panel.id)}" data-edges="0" d="" fill="none"/>'
        edges = self.stroked_edges(extended, geometry)
        d = " ".join(f"M{_num(a[0])},{_num(a[1])} L{_num(b[0])},{_num(b[1])}" for a, b in edges)
        return (
            f'<path class="panel-frame" data-panel-id="{_attr(panel.id)}" data-edges="{len(edges)}" d="{d}" fill="none" '
            f'stroke="{self.options.ink}" stroke-width="{_num(self.options.frame_stroke)}" stroke-linecap="square" stroke-linejoin="miter"/>'
        )

    def _breakout(self, panel: Panel) -> str:
        if not panel.breakout or not panel.characters:
            return ""
        x0, y0, x1, y1 = panel.polygon.core_box()
        main = panel.characters[0]
        px = x0 + main.position.x * (x1 - x0)
        radius = max(16.0, min(70.0, 0.08 * (y1 - y0)))
        cy = y0 + radius * 0.35
        return (
            f'<g class="breakout" data-panel-id="{_attr(panel.id)}">'
            f'<circle cx="{_num(px)}" cy="{_num(cy)}" r="{_num(radius)}" fill="#ffffff" stroke="{self.options.ink}" stroke-width="3"/>'
            f'<path d="M{_num(px - radius * 0.9)},{_num(cy + radius * 0.2)} Q{_num(px)},{_num(cy - radius * 1.6)} {_num(px + radius * 0.9)},{_num(cy + radius * 0.2)}" fill="none" stroke="{self.options.ink}" stroke-width="2"/>'
            "</g>"
        )

    @staticmethod
    def _ellipse_point(cx: float, cy: float, rx: float, ry: float, angle: float) -> Pt:
        return (cx + rx * math.cos(angle), cy + ry * math.sin(angle))

    def _tail(self, balloon: Balloon, width_factor: float = 0.18) -> tuple[str, str]:
        if balloon.tail_target is None:
            return "", ""
        cx, cy = balloon.center.x, balloon.center.y
        rx, ry = balloon.width / 2.0, balloon.height / 2.0
        tx, ty = balloon.tail_target.x, balloon.tail_target.y
        angle = math.atan2((ty - cy) / max(ry, 1e-6), (tx - cx) / max(rx, 1e-6))
        base1 = self._ellipse_point(cx, cy, rx * 0.92, ry * 0.92, angle - width_factor)
        base2 = self._ellipse_point(cx, cy, rx * 0.92, ry * 0.92, angle + width_factor)
        edge = self._ellipse_point(cx, cy, rx, ry, angle)
        dx, dy = tx - edge[0], ty - edge[1]
        distance = math.hypot(dx, dy)
        length = min(distance, 70.0)
        if distance < 1e-6:
            return "", ""
        tip = (edge[0] + dx / distance * length, edge[1] + dy / distance * length)
        outline = f'<polygon class="balloon-tail" points="{_points_attr([base1, tip, base2])}" fill="#ffffff" stroke="{self.options.ink}" stroke-width="2.2" stroke-linejoin="round"/>'
        inner1 = self._ellipse_point(cx, cy, rx * 0.8, ry * 0.8, angle - width_factor * 0.8)
        inner2 = self._ellipse_point(cx, cy, rx * 0.8, ry * 0.8, angle + width_factor * 0.8)
        cover_tip = (edge[0] + dx / distance * length * 0.55, edge[1] + dy / distance * length * 0.55)
        cover = f'<polygon class="balloon-tail-cover" points="{_points_attr([inner1, cover_tip, inner2])}" fill="#ffffff" stroke="none"/>'
        return outline, cover

    def _balloon_shape(self, balloon: Balloon) -> str:
        cx, cy = balloon.center.x, balloon.center.y
        rx, ry = balloon.width / 2.0, balloon.height / 2.0
        ink = self.options.ink
        seed = stable_seed("caf.balloon", balloon.id)
        if balloon.kind == BalloonKind.SHOUT:
            spikes = 22
            points: list[Pt] = []
            for k in range(spikes * 2):
                angle = math.pi * k / spikes
                jitter = 1.0 + (((seed >> (k % 30)) & 7) - 3.5) * 0.012
                radius = (1.12 if k % 2 == 0 else 0.88) * jitter
                points.append(self._ellipse_point(cx, cy, rx * radius, ry * radius, angle))
            outline, cover = self._tail(balloon, 0.14)
            return outline + f'<polygon class="balloon-shape balloon-shout" points="{_points_attr(points)}" fill="#ffffff" stroke="{ink}" stroke-width="2.6" stroke-linejoin="miter"/>' + cover
        if balloon.kind == BalloonKind.MONOLOGUE:
            bumps = 12
            commands: list[str] = []
            for k in range(bumps):
                a0 = 2 * math.pi * k / bumps
                a1 = 2 * math.pi * (k + 1) / bumps
                p0 = self._ellipse_point(cx, cy, rx * 0.94, ry * 0.94, a0)
                p1 = self._ellipse_point(cx, cy, rx * 0.94, ry * 0.94, a1)
                control = self._ellipse_point(cx, cy, rx * 1.16, ry * 1.16, (a0 + a1) / 2)
                if k == 0:
                    commands.append(f"M{_num(p0[0])},{_num(p0[1])}")
                commands.append(f"Q{_num(control[0])},{_num(control[1])} {_num(p1[0])},{_num(p1[1])}")
            commands.append("Z")
            return f'<path class="balloon-shape balloon-monologue" d="{" ".join(commands)}" fill="#ffffff" stroke="{ink}" stroke-width="1.6"/>'
        if balloon.kind == BalloonKind.FLASH:
            lines: list[str] = []
            count = 56
            for k in range(count):
                angle = 2 * math.pi * k / count
                outer = 1.32 + ((seed >> (k % 31)) & 3) * 0.08
                p0 = self._ellipse_point(cx, cy, rx * 1.0, ry * 1.0, angle)
                p1 = self._ellipse_point(cx, cy, rx * outer, ry * outer, angle)
                lines.append(f'<line x1="{_num(p0[0])}" y1="{_num(p0[1])}" x2="{_num(p1[0])}" y2="{_num(p1[1])}" stroke="{ink}" stroke-width="1.3"/>')
            return (
                '<g class="balloon-shape balloon-flash">'
                + "".join(lines)
                + f'<ellipse cx="{_num(cx)}" cy="{_num(cy)}" rx="{_num(rx)}" ry="{_num(ry)}" fill="#ffffff" stroke="none"/>'
                + "</g>"
            )
        outline, cover = self._tail(balloon)
        return outline + f'<ellipse class="balloon-shape balloon-normal" cx="{_num(cx)}" cy="{_num(cy)}" rx="{_num(rx)}" ry="{_num(ry)}" fill="#ffffff" stroke="{ink}" stroke-width="2.2"/>' + cover

    def _vertical_text(self, balloon: Balloon) -> tuple[str, float]:
        columns = balloon.columns or [balloon.text]
        inset = BALLOON_TEXT_INSET.get(balloon.kind.value, 0.68)
        font = fit_font_size(columns, balloon.width, balloon.height, base=22.0, inset=inset)
        col_w = font * 1.12
        line_h = font * 1.02
        max_len = max(len(c) for c in columns)
        right = balloon.center.x + len(columns) * col_w / 2.0
        top = balloon.center.y - max_len * line_h / 2.0
        weight = "700" if balloon.kind == BalloonKind.SHOUT else "500"
        parts: list[str] = [f'<g class="balloon-text" data-balloon-id="{_attr(balloon.id)}" font-size="{_num(font)}" font-weight="{weight}" fill="{self.options.ink}" data-writing-mode="vertical-rl">']
        for ci, column in enumerate(columns):
            x = right - (ci + 0.5) * col_w
            for ji, char in enumerate(column):
                y = top + (ji + 0.5) * line_h
                glyph = VERTICAL_FORMS.get(char, char)
                transform = ""
                dx = dy = 0.0
                if char in ROTATE_IN_VERTICAL:
                    transform = f' transform="rotate(90 {_num(x)} {_num(y)})"'
                if char in SMALL_KANA:
                    dx, dy = font * 0.1, -font * 0.1
                parts.append(
                    f'<text class="vchar" x="{_num(x + dx)}" y="{_num(y + dy)}" text-anchor="middle" dominant-baseline="central"{transform}>{escape(glyph)}</text>'
                )
        parts.append("</g>")
        return "".join(parts), font

    def _sfx(self, sfx: SoundEffect) -> str:
        size = 46.0 * sfx.scale
        x, y = sfx.position.x, sfx.position.y
        return (
            f'<text class="sfx" data-sfx-id="{_attr(sfx.id)}" x="{_num(x)}" y="{_num(y)}" font-size="{_num(size)}" font-weight="900" '
            f'fill="{self.options.ink}" stroke="#ffffff" stroke-width="{_num(size * 0.12)}" paint-order="stroke" '
            f'transform="rotate({_num(sfx.rotation_deg)} {_num(x)} {_num(y)})">{escape(sfx.text)}</text>'
        )

    def _gaze(self, page: Page, id_prefix: str) -> str:
        if not self.options.show_gaze_vectors:
            return ""
        transitions = compute_gaze_transitions(page)
        parts = ['<g class="gaze">']
        for t in transitions:
            color = "#ef4444" if t.is_backtrack else "#2563eb"
            marker = f"{id_prefix}arrow-bad" if t.is_backtrack else f"{id_prefix}arrow"
            parts.append(
                f'<line class="gaze-arrow" data-from="{_attr(t.from_panel)}" data-to="{_attr(t.to_panel)}" x1="{_num(t.start.x)}" y1="{_num(t.start.y)}" '
                f'x2="{_num(t.end.x)}" y2="{_num(t.end.y)}" stroke="{color}" stroke-width="3" stroke-opacity="0.8" marker-end="url(#{marker})"/>'
            )
        for panel in page.ordered_panels():
            x0, y0, x1, y1 = panel.polygon.core_box()
            ex, ey = x1 - 0.2 * (x1 - x0), y0 + 0.2 * (y1 - y0)
            parts.append(
                f'<g class="gaze-order"><circle cx="{_num(ex)}" cy="{_num(ey)}" r="13" fill="#2563eb" opacity="0.85"/>'
                f'<text x="{_num(ex)}" y="{_num(ey)}" font-size="14" fill="#ffffff" text-anchor="middle" dominant-baseline="central">{panel.index_in_page + 1}</text></g>'
            )
        parts.append("</g>")
        return "".join(parts)

    def _defs(self, id_prefix: str) -> str:
        return (
            f'<marker id="{id_prefix}arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#2563eb"/></marker>'
            f'<marker id="{id_prefix}arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#ef4444"/></marker>'
        )

    # -- ページ ---------------------------------------------------------------------

    def _page_content(self, storyboard: Storyboard, page: Page, id_prefix: str) -> tuple[str, str]:
        geometry = storyboard.geometry
        defs: list[str] = [self._defs(id_prefix)]
        body: list[str] = []
        b = geometry.bleed
        body.append(
            f'<rect class="page-background" x="{_num(-b)}" y="{_num(-b)}" width="{_num(geometry.width + 2 * b)}" height="{_num(geometry.height + 2 * b)}" fill="{self.options.background}"/>'
        )
        body.append(self._guides(geometry, page))
        panel_parts: list[str] = []
        frames: list[str] = []
        overlays: list[str] = []
        for panel in page.ordered_panels():
            extended = self.extended_points(panel, page, geometry)
            clip_id = f"{id_prefix}clip-{panel.id}"
            defs.append(f'<clipPath id="{_attr(clip_id)}"><polygon points="{_points_attr(extended)}"/></clipPath>')
            href = self.options.image_hrefs.get(panel.id)
            content = ""
            if href:
                ex0, ey0, ex1, ey1 = polygon_bbox(extended)
                content = (
                    f'<image class="panel-image" href="{_attr(href)}" xlink:href="{_attr(href)}" x="{_num(ex0)}" y="{_num(ey0)}" '
                    f'width="{_num(ex1 - ex0)}" height="{_num(ey1 - ey0)}" preserveAspectRatio="xMidYMid slice"/>'
                )
            elif self.options.draft_sketch:
                content = self._sketch(panel, page, extended)
            else:
                content = f'<polygon points="{_points_attr(extended)}" fill="#ffffff"/>'
            panel_parts.append(
                f'<g class="panel" data-panel-id="{_attr(panel.id)}" data-shot="{panel.camera.shot.value}" data-emphasis="{panel.emphasis}" clip-path="url(#{_attr(clip_id)})">{content}</g>'
            )
            frames.append(self._frame(panel, extended, geometry))
            overlays.append(self._breakout(panel))
            if self.options.show_panel_labels:
                x0, y0, x1, y1 = panel.polygon.core_box()
                overlays.append(
                    f'<text class="panel-label" x="{_num(x1 - 10)}" y="{_num(y0 + 20)}" font-size="15" text-anchor="end" fill="#2563eb">#{panel.reading_order + 1}</text>'
                )
        body.append('<g class="panels">' + "".join(panel_parts) + "</g>")
        body.append('<g class="frames">' + "".join(frames) + "</g>")
        body.append('<g class="overlays">' + "".join(overlays) + "</g>")
        balloons: list[str] = []
        sfx_parts: list[str] = []
        for panel in page.ordered_panels():
            for balloon in sorted(panel.balloons, key=lambda bl: bl.reading_order):
                text, _ = self._vertical_text(balloon)
                balloons.append(
                    f'<g class="balloon" data-balloon-id="{_attr(balloon.id)}" data-kind="{balloon.kind.value}">{self._balloon_shape(balloon)}{text}</g>'
                )
            sfx_parts.extend(self._sfx(sfx) for sfx in panel.sfx)
        body.append('<g class="balloons">' + "".join(balloons) + "</g>")
        body.append('<g class="sfx-layer">' + "".join(sfx_parts) + "</g>")
        body.append(self._gaze(page, id_prefix))
        return "".join(defs), "".join(body)

    def _svg_open(self, width: float, height: float, view_box: str, extra: str = "") -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="{view_box}" '
            f'width="{_num(width)}" height="{_num(height)}" font-family="{_attr(self.options.font_family)}"{extra}>'
        )

    def render_page(self, storyboard: Storyboard, page_index: int) -> str:
        """1ページの SVG。"""
        page = next((p for p in storyboard.pages if p.index == page_index), None)
        if page is None:
            raise IndexError(f"ページ {page_index} は存在しません")
        geometry = storyboard.geometry
        b = geometry.bleed
        width = geometry.width + 2 * b
        height = geometry.height + 2 * b
        defs, body = self._page_content(storyboard, page, f"p{page.index}-")
        extra = f' data-page-index="{page.index}" data-side="{page.side.value}" data-role="{page.page_turn_role.value}"'
        return (
            self._svg_open(width, height, f"{_num(-b)} {_num(-b)} {_num(width)} {_num(height)}", extra)
            + f"<title>{escape(storyboard.title)} p.{page.number}</title>"
            + f"<defs>{defs}</defs>{body}</svg>"
        )

    def render_spread(self, storyboard: Storyboard, right_index: int, left_index: int) -> str:
        """見開き SVG（右ページを右側、左ページを左側に配置）。"""
        pages = {p.index: p for p in storyboard.pages}
        if right_index not in pages or left_index not in pages:
            raise IndexError("見開きのページが存在しません")
        geometry = storyboard.geometry
        b = geometry.bleed
        page_w = geometry.width + 2 * b
        page_h = geometry.height + 2 * b
        parts = [self._svg_open(page_w * 2, page_h, f"0 0 {_num(page_w * 2)} {_num(page_h)}", f' data-spread="{right_index}-{left_index}"')]
        parts.append(f"<title>{escape(storyboard.title)} 見開き p.{pages[right_index].number}-{pages[left_index].number}</title>")
        for index, offset in ((left_index, 0.0), (right_index, page_w)):
            page = pages[index]
            defs, body = self._page_content(storyboard, page, f"s{index}-")
            parts.append(
                f'<svg class="spread-page" data-page-index="{index}" data-side="{page.side.value}" x="{_num(offset)}" y="0" '
                f'width="{_num(page_w)}" height="{_num(page_h)}" viewBox="{_num(-b)} {_num(-b)} {_num(page_w)} {_num(page_h)}">'
                f"<defs>{defs}</defs>{body}</svg>"
            )
        parts.append(f'<line class="spread-gutter" x1="{_num(page_w)}" y1="0" x2="{_num(page_w)}" y2="{_num(page_h)}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="6 6"/>')
        parts.append("</svg>")
        return "".join(parts)

    def render_all(self, storyboard: Storyboard) -> list[str]:
        return [self.render_page(storyboard, page.index) for page in sorted(storyboard.pages, key=lambda p: p.index)]

    # -- レイヤー JSON ----------------------------------------------------------------

    def layers(self, storyboard: Storyboard) -> dict[str, Any]:
        """レイヤー構造 JSON（frame / image / balloon / text / sfx / guide）。"""
        geometry = storyboard.geometry
        pages: list[dict[str, Any]] = []
        for page in sorted(storyboard.pages, key=lambda p: p.index):
            fx0, fy0, fx1, fy1 = geometry.frame_rect(page.side)
            layer_list: list[dict[str, Any]] = [
                {"type": "guide", "name": "trim", "rect": [0.0, 0.0, geometry.width, geometry.height]},
                {"type": "guide", "name": "bleed", "rect": list(geometry.bleed_rect())},
                {"type": "guide", "name": "frame", "rect": [fx0, fy0, fx1, fy1]},
            ]
            for panel in page.ordered_panels():
                extended = self.extended_points(panel, page, geometry)
                layer_list.append(
                    {
                        "type": "frame",
                        "panel_id": panel.id,
                        "reading_order": panel.reading_order,
                        "polygon": [[p.x, p.y] for p in panel.polygon.points],
                        "extended_polygon": [[x, y] for x, y in extended],
                        "bleed_sides": list(panel.bleed_sides),
                        "stroke": self.options.frame_stroke,
                        "stroked_edges": len(self.stroked_edges(extended, geometry)),
                        "shape": panel.shape.value,
                        "breakout": panel.breakout,
                        "emphasis": panel.emphasis,
                        "camera": panel.camera.model_dump(mode="json"),
                    }
                )
                layer_list.append(
                    {
                        "type": "image",
                        "panel_id": panel.id,
                        "href": self.options.image_hrefs.get(panel.id),
                        "clip_polygon": [[x, y] for x, y in extended],
                        "bbox": list(polygon_bbox(extended)),
                    }
                )
                for balloon in sorted(panel.balloons, key=lambda bl: bl.reading_order):
                    font = fit_font_size(
                        balloon.columns or [balloon.text], balloon.width, balloon.height, inset=BALLOON_TEXT_INSET.get(balloon.kind.value, 0.68)
                    )
                    layer_list.append(
                        {
                            "type": "balloon",
                            "panel_id": panel.id,
                            "balloon_id": balloon.id,
                            "kind": balloon.kind.value,
                            "center": [balloon.center.x, balloon.center.y],
                            "width": balloon.width,
                            "height": balloon.height,
                            "tail_target": [balloon.tail_target.x, balloon.tail_target.y] if balloon.tail_target else None,
                            "reading_order": balloon.reading_order,
                        }
                    )
                    layer_list.append(
                        {
                            "type": "text",
                            "panel_id": panel.id,
                            "balloon_id": balloon.id,
                            "speaker_id": balloon.speaker_id,
                            "text": balloon.text,
                            "columns": list(balloon.columns),
                            "writing_mode": "vertical-rl",
                            "font_size": round(font, 2),
                        }
                    )
                for sfx in panel.sfx:
                    layer_list.append(
                        {
                            "type": "sfx",
                            "panel_id": panel.id,
                            "sfx_id": sfx.id,
                            "text": sfx.text,
                            "position": [sfx.position.x, sfx.position.y],
                            "direction": [sfx.direction.x, sfx.direction.y],
                            "rotation_deg": sfx.rotation_deg,
                            "scale": sfx.scale,
                            "style": sfx.style,
                        }
                    )
            pages.append(
                {
                    "index": page.index,
                    "number": page.number,
                    "side": page.side.value,
                    "page_turn_role": page.page_turn_role.value,
                    "spread_id": page.spread_id,
                    "size": {"width": geometry.width, "height": geometry.height, "bleed": geometry.bleed},
                    "layers": layer_list,
                }
            )
        return {"storyboard_id": storyboard.id, "title": storyboard.title, "pages": pages}
