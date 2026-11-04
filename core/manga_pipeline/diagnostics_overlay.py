"""ネーム構造ドクターの診断オーバーレイ SVG。

診断レポートに含まれる正規化ネーム（``report.storyboard``）のページを ``SVGLayoutRenderer`` で描画し、
同じページ座標系の上に次のレイヤーを重ねる。

- 推定強度ヒート: コマを強度色（青→琥珀→赤）で半透明に塗り、数値を添える
- 視線経路: 読み順の遷移矢印（正常=青緑、逆行=赤、交差=橙）とコマ番号
- 所見ピン: 番号付きの円（severity 色）。``<title>`` に所見の要約を入れ、ブラウザのツールチップで読める
- フキダシ読み順: コマ内の台詞の読み順番号

画像から逆解析したレポート（``report.page_images`` あり）は、元のページ画像を背景に敷き、
検出したコマ枠の輪郭の上に同じレイヤーを重ねる。
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict

from core.manga_pipeline.diagnostics_schemas import SEVERITY_LABELS_JA, DiagnosticReport
from core.manga_pipeline.geometry import compute_gaze_transitions, panel_entry_point
from core.manga_pipeline.layout_renderer import RenderOptions, SVGLayoutRenderer
from core.manga_pipeline.schemas import Page, Storyboard

__all__ = [
    "SEVERITY_COLORS",
    "DiagnosticOverlayOptions",
    "intensity_color",
    "render_diagnostic_overlay",
]

SEVERITY_COLORS: dict[str, str] = {
    "critical": "#dc2626",
    "major": "#f97316",
    "minor": "#eab308",
    "info": "#3b82f6",
}

_GAZE_COLORS = {"ok": "#0d9488", "bad": "#dc2626", "cross": "#f97316"}


class DiagnosticOverlayOptions(BaseModel):
    """オーバーレイの表示切替。"""

    model_config = ConfigDict(extra="ignore")

    show_heat: bool = True
    show_gaze: bool = True
    show_pins: bool = True
    show_balloon_order: bool = True
    show_image: bool = True


def _attr(value: object) -> str:
    return escape(str(value), {'"': "&quot;", "'": "&apos;"})


def _num(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def intensity_color(value: float) -> str:
    """強度 0→1 を 青(#2563eb)→琥珀(#f59e0b)→赤(#dc2626) に写像した16進色。"""
    v = min(1.0, max(0.0, float(value)))
    low, mid, high = (37, 99, 235), (245, 158, 11), (220, 38, 38)
    if v <= 0.5:
        a, b, u = low, mid, v / 0.5
    else:
        a, b, u = mid, high, (v - 0.5) / 0.5
    rgb = tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{_num(x)},{_num(y)}" for x, y in points)


def _defs() -> str:
    markers = []
    for key, color in _GAZE_COLORS.items():
        markers.append(
            f'<marker id="diag-arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
        )
    return "<defs>" + "".join(markers) + "</defs>"


def _heat_layer(report: DiagnosticReport, page: Page) -> str:
    metrics = {m.panel_id: m for m in report.panels}
    parts: list[str] = []
    for panel in page.ordered_panels():
        metric = metrics.get(panel.id)
        if metric is None:
            continue
        color = intensity_color(metric.intensity)
        x0, y0, _, _ = panel.polygon.bbox()
        parts.append(
            f'<polygon class="diag-heat" data-panel-id="{_attr(panel.id)}" points="{_points_attr(panel.polygon.tuples())}" '
            f'fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text class="diag-heat-label" x="{_num(x0 + 10)}" y="{_num(y0 + 26)}" font-size="20" font-weight="700" '
            f'fill="{color}" stroke="#ffffff" stroke-width="4" paint-order="stroke">I {metric.intensity:.2f}</text>'
        )
    return '<g class="diag-heat-layer">' + "".join(parts) + "</g>"


def _gaze_layer(page: Page) -> str:
    panels = page.ordered_panels()
    parts: list[str] = []
    if len(panels) > 1:
        for transition in compute_gaze_transitions(page):
            if transition.is_backtrack:
                key = "bad"
            elif transition.crosses:
                key = "cross"
            else:
                key = "ok"
            dash = ' stroke-dasharray="10 6"' if key == "cross" else ""
            parts.append(
                f'<line class="diag-gaze diag-gaze-{key}" data-from="{_attr(transition.from_panel)}" data-to="{_attr(transition.to_panel)}" '
                f'x1="{_num(transition.start.x)}" y1="{_num(transition.start.y)}" x2="{_num(transition.end.x)}" y2="{_num(transition.end.y)}" '
                f'stroke="{_GAZE_COLORS[key]}" stroke-width="4" stroke-opacity="0.9"{dash} marker-end="url(#diag-arrow-{key})"/>'
            )
    for index, panel in enumerate(panels, start=1):
        x, y = panel_entry_point(panel.polygon.tuples())
        parts.append(
            f'<g class="diag-order" data-panel-id="{_attr(panel.id)}"><circle cx="{_num(x)}" cy="{_num(y)}" r="14" fill="#0f172a" fill-opacity="0.8"/>'
            f'<text x="{_num(x)}" y="{_num(y + 5)}" font-size="15" font-weight="700" text-anchor="middle" fill="#ffffff">{index}</text></g>'
        )
    return '<g class="diag-gaze-layer">' + "".join(parts) + "</g>"


def _balloon_order_layer(page: Page) -> str:
    parts: list[str] = []
    for panel in page.ordered_panels():
        for rank, balloon in enumerate(sorted(panel.balloons, key=lambda b: b.reading_order), start=1):
            x = balloon.center.x + balloon.width / 2.0 - 6.0
            y = balloon.center.y - balloon.height / 2.0 + 6.0
            parts.append(
                f'<g class="diag-balloon-order" data-balloon-id="{_attr(balloon.id)}"><circle cx="{_num(x)}" cy="{_num(y)}" r="11" fill="#7c3aed"/>'
                f'<text x="{_num(x)}" y="{_num(y + 4)}" font-size="12" font-weight="700" text-anchor="middle" fill="#ffffff">{rank}</text></g>'
            )
    return '<g class="diag-balloon-order-layer">' + "".join(parts) + "</g>"


def _pins_layer(report: DiagnosticReport, page: Page, storyboard: Storyboard) -> str:
    findings = report.findings_for_page(page.index)
    parts: list[str] = []
    fallback = 0
    width = storyboard.geometry.width
    for finding in findings:
        number = finding.id.lstrip("f").lstrip("0") or "0"
        if finding.anchor is not None:
            x, y = finding.anchor.x, finding.anchor.y
        else:
            x, y = width - 30.0 - 42.0 * fallback, 24.0
            fallback += 1
        color = SEVERITY_COLORS.get(finding.severity, "#64748b")
        label = f"[{SEVERITY_LABELS_JA.get(finding.severity, finding.severity)}] {finding.title} — {finding.detail}"
        parts.append(
            f'<g class="diag-pin" data-finding-id="{_attr(finding.id)}" data-severity="{_attr(finding.severity)}" data-code="{_attr(finding.code)}">'
            f"<title>{escape(label)}</title>"
            f'<circle cx="{_num(x)}" cy="{_num(y)}" r="17" fill="{color}" stroke="#ffffff" stroke-width="3"/>'
            f'<text x="{_num(x)}" y="{_num(y + 6)}" font-size="16" font-weight="800" text-anchor="middle" fill="#ffffff">{_attr(number)}</text></g>'
        )
    return '<g class="diag-pins-layer">' + "".join(parts) + "</g>"


def _image_base(report: DiagnosticReport, storyboard: Storyboard, page: Page, href: str) -> str:
    geometry = storyboard.geometry
    b = geometry.bleed
    width = geometry.width + 2 * b
    height = geometry.height + 2 * b
    outlines = "".join(
        f'<polygon class="diag-panel-outline" data-panel-id="{_attr(panel.id)}" points="{_points_attr(panel.polygon.tuples())}" '
        f'fill="none" stroke="#0f172a" stroke-width="3" stroke-opacity="0.85"/>'
        for panel in page.ordered_panels()
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="{_num(-b)} {_num(-b)} {_num(width)} {_num(height)}" '
        f'width="{_num(width)}" height="{_num(height)}" font-family="\'Noto Sans JP\', \'Hiragino Kaku Gothic ProN\', sans-serif" '
        f'data-page-index="{page.index}" data-source="image">'
        f"<title>{escape(report.title)} p.{page.number}（画像逆解析）</title>"
        f'<rect x="{_num(-b)}" y="{_num(-b)}" width="{_num(width)}" height="{_num(height)}" fill="#ffffff"/>'
        f'<image class="diag-page-image" href="{_attr(href)}" xlink:href="{_attr(href)}" x="0" y="0" '
        f'width="{_num(geometry.width)}" height="{_num(geometry.height)}" preserveAspectRatio="none"/>'
        f'<g class="diag-panel-outlines">{outlines}</g>'
    )


def render_diagnostic_overlay(
    report: DiagnosticReport,
    page_index: int,
    options: DiagnosticOverlayOptions | None = None,
) -> str:
    """診断レポートの指定ページをオーバーレイ付き SVG で描画する。"""
    opts = options or DiagnosticOverlayOptions()
    storyboard = report.storyboard
    if storyboard is None:
        raise ValueError("レポートに正規化ネームが含まれていないためオーバーレイを描画できません")
    page = next((p for p in storyboard.pages if p.index == page_index), None)
    if page is None:
        raise IndexError(f"ページ {page_index} は存在しません")
    href = report.page_images[page_index] if opts.show_image and 0 <= page_index < len(report.page_images) else None
    if href:
        base = _image_base(report, storyboard, page, href)
    else:
        renderer = SVGLayoutRenderer(RenderOptions(show_gaze_vectors=False, show_panel_labels=False, show_bleed_guides=False))
        rendered = renderer.render_page(storyboard, page_index)
        base = rendered[: rendered.rfind("</svg>")]
    layers = [_defs()]
    if opts.show_heat:
        layers.append(_heat_layer(report, page))
    if opts.show_gaze:
        layers.append(_gaze_layer(page))
    if opts.show_balloon_order:
        layers.append(_balloon_order_layer(page))
    if opts.show_pins:
        layers.append(_pins_layer(report, page, storyboard))
    return base + '<g class="diagnostic-overlay">' + "".join(layers) + "</g></svg>"
