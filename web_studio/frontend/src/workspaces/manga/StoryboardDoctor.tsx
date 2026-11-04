// Manga Studio「ネーム構造ドクター」: 既存のネーム／原稿アノテーション／原稿画像を逆解析して5軸で採点し、
// 視線誘導と感情の緩急を劇的にするための処方箋（反実仮想シミュレーションによる期待改善量つき）を提示・適用する。

import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import { api, downloadJson, errorMessage } from "../../api/client";
import type { DiagnosticOverlayQuery, DiagnosticRequest } from "../../api/client";
import type {
  DiagnosticFinding,
  DiagnosticReport,
  DiagnosticSummary,
  ImageDetectionResult,
  Job,
  PageAnnotation,
  PageDetection,
  PageSide,
  Prescription,
  StoryboardSummary,
} from "../../api/types";
import { JobProgress } from "../../components/JobProgress";
import { IntensityCurveChart } from "../../components/charts/IntensityCurveChart";
import { PageHeatStrip } from "../../components/charts/PageHeatStrip";
import { RadarChart } from "../../components/charts/RadarChart";
import { ScoreGauge } from "../../components/charts/ScoreGauge";
import { Badge, Button, Card, EmptyState, ErrorNote, Field, Select, Slider, Stat, Tabs, TextArea, TextInput, Toggle, cx } from "../../components/ui";
import { useAction, useResource } from "../../hooks/useAsync";
import { useStructureOptions } from "../../hooks/useStructures";
import { dateTime, fmt } from "../../lib/format";
import { fileToGrayscalePng, isImageFile, urlToGrayscalePng } from "../../lib/imageTools";
import type { PreparedImage } from "../../lib/imageTools";
import {
  CATEGORY_LABELS,
  DIAGNOSTIC_AXIS_COLORS,
  DIAGNOSTIC_AXIS_LABELS,
  DIAGNOSTIC_AXIS_ORDER,
  DIAGNOSTIC_SEVERITY_LABELS,
  DIAGNOSTIC_SEVERITY_ORDER,
  DIAGNOSTIC_SEVERITY_TONES,
  DIAGNOSTIC_SOURCE_LABELS,
  GRADE_COLORS,
  PRESCRIPTION_LABELS,
  labelOf,
} from "../../lib/labels";
import { annotationTemplateText } from "./annotationTemplate";

type SourceMode = "storyboard" | "image" | "annotation";

export interface DoctorPreset {
  storyboardId: string;
  nonce: number;
}

const SOURCE_TABS: Array<{ key: SourceMode; label: string }> = [
  { key: "storyboard", label: "保存済みネーム" },
  { key: "image", label: "原稿画像" },
  { key: "annotation", label: "アノテーション JSON" },
];

const SIDE_OPTIONS: Array<{ value: PageSide; label: string }> = [
  { value: "right", label: "1ページ目は右ページ（右開き・見開き開始）" },
  { value: "left", label: "1ページ目は左ページ（扉ページ始まり）" },
];

const SAMPLE_PAGES: Array<{ seed: number; variant: string; name: string }> = [
  { seed: 11, variant: "standard", name: "サンプル原稿 p1（標準コマ割り）.png" },
  { seed: 12, variant: "slanted", name: "サンプル原稿 p2（斜めゴマ）.png" },
];

interface ApplyComparison {
  before: number;
  after: number;
  beforeAxes: Record<string, number>;
  savedStoryboardId: string | null;
  applied: number;
}

function axisRecord(report: DiagnosticReport): Record<string, number> {
  const record: Record<string, number> = {};
  for (const axis of report.axes) {
    record[axis.axis] = axis.score;
  }
  return record;
}

function parseAnnotation(text: string): PageAnnotation {
  if (!text.trim()) {
    throw new Error("アノテーション JSON が空です（「雛形を挿入」で例を読み込めます）");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (reason) {
    throw new Error(`アノテーション JSON の構文エラー: ${reason instanceof Error ? reason.message : String(reason)}`);
  }
  if (!parsed || typeof parsed !== "object" || !Array.isArray((parsed as PageAnnotation).pages)) {
    throw new Error("アノテーション JSON には pages 配列が必要です");
  }
  return parsed as PageAnnotation;
}

function DetectionPreview({ image, detection }: { image: PreparedImage; detection: PageDetection | null }) {
  const width = detection?.width ?? image.width;
  const height = detection?.height ?? image.height;
  return (
    <div className="flex flex-col gap-1">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full rounded border border-ink-700 bg-white" role="img" aria-label={`${image.name} の検出結果`}>
        <image href={image.dataUrl} x={0} y={0} width={width} height={height} preserveAspectRatio="none" />
        {detection?.panels.map((panel, index) => (
          <g key={`panel-${index}`}>
            <polygon
              points={panel.polygon.map((point) => `${point.x},${point.y}`).join(" ")}
              fill="rgba(124,156,255,0.12)"
              stroke="#4f6bff"
              strokeWidth={Math.max(2, width / 260)}
            />
            <text x={panel.bbox[0] + 8} y={panel.bbox[1] + width / 28} fontSize={width / 30} fontWeight={700} fill="#1d4ed8" stroke="#ffffff" strokeWidth={width / 300} paintOrder="stroke">
              #{index + 1} ベタ{Math.round(panel.ink_ratio * 100)}%
            </text>
            {panel.balloons.map((balloon, balloonIndex) => (
              <rect
                key={`balloon-${index}-${balloonIndex}`}
                x={balloon.bbox[0]}
                y={balloon.bbox[1]}
                width={Math.max(1, balloon.bbox[2] - balloon.bbox[0])}
                height={Math.max(1, balloon.bbox[3] - balloon.bbox[1])}
                fill="rgba(251,191,36,0.18)"
                stroke="#d97706"
                strokeWidth={Math.max(1.5, width / 400)}
                strokeDasharray="6 4"
              >
                <title>{`フキダシ（推定 ${balloon.estimated_chars} 字・信頼度 ${fmt(balloon.confidence)}）`}</title>
              </rect>
            ))}
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-ink-400">
        <span className="truncate">{image.name}</span>
        <span className="font-mono">
          {image.width}×{image.height}px
        </span>
        {detection && (
          <>
            <Badge tone="accent">{detection.panels.length} コマ</Badge>
            <Badge tone="warning">{detection.panels.reduce((sum, panel) => sum + panel.balloons.length, 0)} フキダシ</Badge>
            <span className="font-mono">しきい値 {fmt(detection.threshold)}</span>
          </>
        )}
      </div>
      {detection && detection.warnings.length > 0 && <p className="text-[11px] text-amber-300">{detection.warnings.join(" / ")}</p>}
    </div>
  );
}

function FindingItem({ finding, onJump, highlighted }: { finding: DiagnosticFinding; onJump: (page: number) => void; highlighted: boolean }) {
  return (
    <div className={cx("rounded-lg border px-3 py-2 text-xs", highlighted ? "border-accent bg-accent-soft/30" : "border-ink-700 bg-ink-850")}>
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={DIAGNOSTIC_SEVERITY_TONES[finding.severity] ?? "neutral"}>{labelOf(DIAGNOSTIC_SEVERITY_LABELS, finding.severity)}</Badge>
        <span className="rounded px-1 text-[11px] font-medium" style={{ color: DIAGNOSTIC_AXIS_COLORS[finding.axis] }}>
          {labelOf(DIAGNOSTIC_AXIS_LABELS, finding.axis)}
        </span>
        <span className="font-mono text-[11px] text-ink-500">{finding.code}</span>
        {finding.page_index !== null && (
          <Button size="sm" variant="ghost" onClick={() => onJump(finding.page_index ?? 0)}>
            p{finding.page_index + 1} を表示
          </Button>
        )}
      </div>
      <p className="mt-1 font-medium text-ink-100">{finding.title}</p>
      <p className="mt-0.5 text-ink-300">{finding.detail}</p>
      <p className="mt-0.5 font-mono text-[11px] text-ink-500">
        {finding.metric}: 観測 {fmt(finding.observed, 3)} → 目標 {fmt(finding.target, 3)}
      </p>
    </div>
  );
}

function PrescriptionItem({
  prescription,
  maxGain,
  checked,
  onToggle,
  onJump,
}: {
  prescription: Prescription;
  maxGain: number;
  checked: boolean;
  onToggle: (id: string) => void;
  onJump: (page: number) => void;
}) {
  const ratio = maxGain > 0 ? Math.max(0, prescription.expected_gain) / maxGain : 0;
  const gains = Object.entries(prescription.axis_gains).filter(([, value]) => Math.abs(value) >= 0.05);
  return (
    <div className={cx("rounded-lg border px-3 py-2 text-xs", checked ? "border-emerald-500/60 bg-emerald-950/30" : "border-ink-700 bg-ink-850")}>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            className="h-4 w-4 accent-[#34d399]"
            checked={checked}
            disabled={!prescription.applicable}
            onChange={() => onToggle(prescription.id)}
            aria-label={`処方 ${prescription.rank} を選択`}
          />
          <Badge tone="accent">#{prescription.rank}</Badge>
          <span className="font-medium text-ink-100">{prescription.title}</span>
        </label>
        <Badge>{labelOf(PRESCRIPTION_LABELS, prescription.operation)}</Badge>
        {!prescription.applicable && <Badge tone="neutral">手動で修正</Badge>}
        {prescription.page_index !== null && (
          <Button size="sm" variant="ghost" onClick={() => onJump(prescription.page_index ?? 0)}>
            p{prescription.page_index + 1}
          </Button>
        )}
        <span className="ml-auto font-mono text-emerald-300">+{prescription.expected_gain.toFixed(2)} 点</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
        <div className="h-full rounded-full bg-emerald-400" style={{ width: `${Math.round(ratio * 100)}%` }} />
      </div>
      <p className="mt-1.5 text-ink-200">{prescription.instruction}</p>
      {gains.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {gains.map(([axis, value]) => (
            <span key={axis} className="rounded border border-ink-700 px-1.5 py-0.5 font-mono text-[10px]" style={{ color: DIAGNOSTIC_AXIS_COLORS[axis] ?? "#c5cbe0" }}>
              {labelOf(DIAGNOSTIC_AXIS_LABELS, axis)} {value >= 0 ? "+" : ""}
              {value.toFixed(1)}
            </span>
          ))}
          <span className="text-[10px] text-ink-500">{prescription.simulated ? "シミュレーション実測" : "推定"}</span>
        </div>
      )}
    </div>
  );
}

export function StoryboardDoctor({
  dataVersion,
  preset,
  onStoryboardSaved,
}: {
  dataVersion: number;
  preset: DoctorPreset | null;
  onStoryboardSaved: (id: string) => void;
}) {
  const structures = useStructureOptions(dataVersion);
  const [storyboardVersion, setStoryboardVersion] = useState(0);
  const storyboards = useResource<StoryboardSummary[]>(() => api.manga.list(), [storyboardVersion, dataVersion]);
  const [historyVersion, setHistoryVersion] = useState(0);
  const history = useResource<DiagnosticSummary[]>(() => api.diagnostics.list(), [historyVersion]);

  const [source, setSource] = useState<SourceMode>("storyboard");
  const [storyboardId, setStoryboardId] = useState("");
  const [images, setImages] = useState<PreparedImage[]>([]);
  const [firstPageSide, setFirstPageSide] = useState<PageSide>("right");
  const [annotationText, setAnnotationText] = useState("");
  const [detection, setDetection] = useState<ImageDetectionResult | null>(null);
  const [targetId, setTargetId] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [simulate, setSimulate] = useState(true);
  const [maxPrescriptions, setMaxPrescriptions] = useState(12);
  const [title, setTitle] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const [report, setReport] = useState<DiagnosticReport | null>(null);
  const [page, setPage] = useState(0);
  const [overlay, setOverlay] = useState<DiagnosticOverlayQuery>({ heat: true, gaze: true, pins: true, order: true, image: true });
  const [zoom, setZoom] = useState(1);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [axisFilter, setAxisFilter] = useState<string>("all");
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [selectedRx, setSelectedRx] = useState<string[]>([]);
  const [saveImproved, setSaveImproved] = useState(true);
  const [comparison, setComparison] = useState<ApplyComparison | null>(null);

  useEffect(() => {
    if (preset) {
      setSource("storyboard");
      setStoryboardId(preset.storyboardId);
    }
  }, [preset]);

  useEffect(() => {
    const items = storyboards.data ?? [];
    if (!storyboardId && items.length > 0) {
      setStoryboardId(items[0].id);
    }
  }, [storyboards.data, storyboardId]);

  useEffect(() => {
    setOverlayError(null);
  }, [report?.id, page, overlay]);

  const storyboardOptions = useMemo(
    () =>
      (storyboards.data ?? []).map((item) => ({
        value: item.id,
        label: `${item.title}（${item.page_count}ページ${item.score !== null ? `・文法 ${item.score.toFixed(0)}` : ""}）`,
      })),
    [storyboards.data],
  );

  const targetOptions = useMemo(() => {
    const items = [...structures.items].sort((a, b) => {
      const rank = (category: string) => (category === "emotional_wave" ? 0 : category === "manga_direction" ? 1 : 2);
      return rank(a.category) - rank(b.category) || a.title.localeCompare(b.title, "ja");
    });
    return [
      { value: "", label: source === "storyboard" ? "自動（ネームの元構造を目標にする）" : "目標なし（視覚的な緩急のみで評価）" },
      ...items.map((item) => ({
        value: item.id,
        label: `${item.category === "emotional_wave" ? "★推奨 " : ""}［${labelOf(CATEGORY_LABELS, item.category)}］${item.title}`,
      })),
    ];
  }, [structures.items, source]);

  const addFiles = useAction(
    async (files: File[]) => {
      const accepted = files.filter(isImageFile);
      if (accepted.length === 0) {
        throw new Error("画像ファイル（PNG / JPEG / WebP など）を選択してください");
      }
      const prepared: PreparedImage[] = [];
      for (const file of accepted) {
        prepared.push(await fileToGrayscalePng(file));
      }
      setImages((current) => [...current, ...prepared].slice(0, 24));
      setDetection(null);
      return prepared.length;
    },
    { successMessage: (count) => `${count} ページ分の画像を読み込みました（グレースケール PNG に変換済み）` },
  );

  const loadSample = useAction(
    async () => {
      const prepared: PreparedImage[] = [];
      for (const sample of SAMPLE_PAGES) {
        prepared.push(await urlToGrayscalePng(api.diagnostics.sampleImageUrl(sample.seed, sample.variant), sample.name));
      }
      setImages(prepared);
      setDetection(null);
      setSource("image");
      return prepared.length;
    },
    { successMessage: (count) => `サンプル原稿画像 ${count} ページを読み込みました` },
  );

  const detect = useAction(
    async () => {
      if (images.length === 0) {
        throw new Error("先に原稿画像を読み込んでください");
      }
      const result = await api.diagnostics.detect({
        images: images.map((image) => image.dataUrl),
        first_page_side: firstPageSide,
        title: title.trim() || undefined,
      });
      setDetection(result);
      return result;
    },
    {
      successMessage: (result) =>
        `${result.pages.length} ページから ${result.pages.reduce((sum, item) => sum + item.panels.length, 0)} コマを検出しました`,
    },
  );

  const diagnose = useAction(
    async () => {
      setInputError(null);
      const body: DiagnosticRequest = {
        target_structure_id: targetId || null,
        use_llm: useLlm,
        simulate,
        max_prescriptions: maxPrescriptions,
        title: title.trim() || undefined,
      };
      if (source === "storyboard") {
        if (!storyboardId) {
          throw new Error("診断するネームを選択してください（Manga Studio の制作タブで生成できます）");
        }
        body.storyboard_id = storyboardId;
      } else if (source === "image") {
        if (images.length === 0) {
          throw new Error("原稿画像を読み込んでください（「サンプル原稿画像で試す」も利用できます）");
        }
        body.images = images.map((image) => image.dataUrl);
        body.first_page_side = firstPageSide;
      } else {
        try {
          body.annotation = parseAnnotation(annotationText);
        } catch (reason) {
          const message = errorMessage(reason);
          setInputError(message);
          throw reason;
        }
      }
      const result = await api.diagnostics.run(body, setJob);
      setReport(result);
      setPage(0);
      setSelectedRx([]);
      setComparison(null);
      setHistoryVersion((value) => value + 1);
      return result;
    },
    { successMessage: (result) => `診断しました（総合 ${result.overall_score.toFixed(1)} 点・評価 ${result.grade}・処方 ${result.prescriptions.length} 件）` },
  );

  const openReport = useAction(async (id: string) => {
    const loaded = await api.diagnostics.get(id);
    setReport(loaded);
    setPage(0);
    setSelectedRx([]);
    setComparison(null);
    return loaded;
  });

  const removeReport = useAction(
    async () => {
      if (!report) {
        return;
      }
      await api.diagnostics.remove(report.id);
      setReport(null);
      setComparison(null);
      setHistoryVersion((value) => value + 1);
    },
    { successMessage: "診断レポートを削除しました" },
  );

  const apply = useAction(
    async () => {
      if (!report) {
        throw new Error("診断レポートがありません");
      }
      if (selectedRx.length === 0) {
        throw new Error("適用する処方を選択してください");
      }
      const beforeAxes = axisRecord(report);
      const response = await api.diagnostics.apply(report.id, { prescription_ids: selectedRx, save_storyboard: saveImproved });
      setComparison({
        before: response.score_before,
        after: response.score_after,
        beforeAxes,
        savedStoryboardId: response.saved_storyboard_id,
        applied: selectedRx.length,
      });
      setReport(response.report);
      setSelectedRx([]);
      setHistoryVersion((value) => value + 1);
      if (response.saved_storyboard_id) {
        setStoryboardVersion((value) => value + 1);
        onStoryboardSaved(response.saved_storyboard_id);
      }
      return response;
    },
    {
      successMessage: (response) =>
        `処方を適用しました（${response.score_before.toFixed(1)} → ${response.score_after.toFixed(1)} 点${response.saved_storyboard_id ? "・改善版ネームを保存" : ""}）`,
    },
  );

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) {
      void addFiles.run(files);
    }
  };

  const pageCount = report ? Math.max(report.pages.length, report.storyboard?.pages.length ?? 0) : 0;
  const maxGain = report ? Math.max(0, ...report.prescriptions.map((item) => item.expected_gain)) : 0;
  const sortedPrescriptions = report ? [...report.prescriptions].sort((a, b) => b.expected_gain - a.expected_gain || a.rank - b.rank) : [];
  const filteredFindings = report
    ? report.findings
        .filter((finding) => axisFilter === "all" || finding.axis === axisFilter)
        .filter((finding) => severityFilter === "all" || finding.severity === severityFilter)
        .sort((a, b) => (DIAGNOSTIC_SEVERITY_ORDER[a.severity] ?? 9) - (DIAGNOSTIC_SEVERITY_ORDER[b.severity] ?? 9) || (a.page_index ?? -1) - (b.page_index ?? -1))
    : [];
  const radarAxes = report
    ? DIAGNOSTIC_AXIS_ORDER.map((key) => {
        const axis = report.axes.find((item) => item.axis === key);
        return { key, label: labelOf(DIAGNOSTIC_AXIS_LABELS, key), value: axis?.score ?? 0, color: DIAGNOSTIC_AXIS_COLORS[key] };
      })
    : [];
  const currentPageMetrics = report?.pages.find((item) => item.page_index === page) ?? null;
  const pageFindings = report ? report.findings.filter((finding) => finding.page_index === page) : [];
  const hasImages = (report?.page_images.length ?? 0) > 0;
  const applicableCount = report ? report.prescriptions.filter((item) => item.applicable).length : 0;

  const toggleRx = (id: string) => setSelectedRx((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Card
          title="ネーム構造ドクター（Storyboard Diagnostics Engine）"
          subtitle="コマ割り・セリフ配置・情報密度を逆解析し、視線誘導と感情の緩急が劇的になる修正箇所を期待改善量つきで処方します"
        >
          <div className="flex flex-col gap-3">
            <Tabs tabs={SOURCE_TABS} active={source} onChange={setSource} size="sm" />
            {source === "storyboard" && (
              <Field label="診断するネーム" hint="Manga Studio の制作タブで生成・保存したネーム（処方の自動適用と改善版の保存が可能）">
                <Select
                  value={storyboardId}
                  options={storyboardOptions.length > 0 ? storyboardOptions : [{ value: "", label: "保存済みネームがありません" }]}
                  onChange={setStoryboardId}
                  ariaLabel="診断するネーム"
                />
              </Field>
            )}
            {source === "image" && (
              <div className="flex flex-col gap-3">
                <div
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  className={cx(
                    "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center text-xs",
                    dragOver ? "border-accent bg-accent-soft/30 text-accent-strong" : "border-ink-600 text-ink-300",
                  )}
                >
                  <p>原稿画像をここへドラッグ&ドロップ（複数ページ可・読み順に並べてください）</p>
                  <p className="text-ink-500">JPEG / PNG / WebP など、ブラウザで表示できる形式を長辺1600px以下のグレースケール PNG に変換して送信します</p>
                  <div className="mt-1 flex flex-wrap justify-center gap-2">
                    <Button size="sm" onClick={() => fileInput.current?.click()} loading={addFiles.running}>
                      ファイルを選択
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => void loadSample.run()} loading={loadSample.running}>
                      サンプル原稿画像で試す
                    </Button>
                    {images.length > 0 && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          setImages([]);
                          setDetection(null);
                        }}
                      >
                        画像をクリア
                      </Button>
                    )}
                  </div>
                  <input
                    ref={fileInput}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    aria-label="原稿画像ファイル"
                    onChange={(event) => {
                      const files = Array.from(event.target.files ?? []);
                      event.target.value = "";
                      if (files.length > 0) {
                        void addFiles.run(files);
                      }
                    }}
                  />
                </div>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <Field label="見開きの開始側">
                    <Select value={firstPageSide} options={SIDE_OPTIONS} onChange={setFirstPageSide} ariaLabel="1ページ目の左右" />
                  </Field>
                  <div className="flex items-end">
                    <Button disabled={images.length === 0} loading={detect.running} onClick={() => void detect.run()}>
                      コマ・フキダシを検出
                    </Button>
                  </div>
                </div>
                {images.length > 0 && (
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-4">
                    {images.map((image, index) => (
                      <DetectionPreview key={`${image.name}-${index}`} image={image} detection={detection?.pages[index] ?? null} />
                    ))}
                  </div>
                )}
                {detection && (
                  <p className="text-[11px] text-ink-400">
                    検出結果を確認し、問題なければ「診断する」を実行してください（診断時にも同じ検出処理を行います）。
                  </p>
                )}
              </div>
            )}
            {source === "annotation" && (
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <Button size="sm" onClick={() => setAnnotationText(annotationTemplateText())}>
                    雛形を挿入
                  </Button>
                  <span className="text-[11px] text-ink-500">
                    既存原稿のコマ枠（bbox / polygon）・フキダシ（台詞または推定字数）・ショット等を記述します。読み順を指定しない場合は段組みから推定します
                  </span>
                </div>
                <TextArea
                  value={annotationText}
                  onChange={setAnnotationText}
                  rows={14}
                  monospace
                  placeholder='{"title": "既存原稿", "first_page_side": "right", "pages": [{"width": 1000, "height": 1414, "panels": [{"bbox": [520, 90, 930, 470], "balloons": []}]}]}'
                  ariaLabel="アノテーション JSON"
                />
                <ErrorNote message={inputError} />
              </div>
            )}
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <Field label="目標とする構造（任意）" hint="目標の緊張曲線と、コマから逆解析した推定強度の相関・DTW距離を評価します">
                <Select value={targetId} options={targetOptions} onChange={setTargetId} ariaLabel="目標構造" />
              </Field>
              <Field label="レポート名（任意）">
                <TextInput value={title} onChange={setTitle} placeholder="未指定なら入力名から生成" ariaLabel="診断レポート名" />
              </Field>
              <Slider label="処方の最大件数" value={maxPrescriptions} min={3} max={24} step={1} onChange={setMaxPrescriptions} format={(value) => `${value} 件`} />
              <div className="flex flex-col justify-end gap-2">
                <Toggle checked={simulate} onChange={setSimulate} label="反実仮想シミュレーションで期待改善量を実測する" />
                <Toggle checked={useLlm} onChange={setUseLlm} label="DeepSeek で編集者コメントを増幅（manga.diagnosis_commentary）" />
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="primary" loading={diagnose.running} onClick={() => void diagnose.run()}>
                診断する
              </Button>
              {diagnose.running && <span className="text-xs text-ink-400">逆解析・5軸採点・処方シミュレーションを実行しています…</span>}
            </div>
            {diagnose.running && <JobProgress job={job} />}
          </div>
        </Card>

        <Card
          title="診断履歴"
          actions={
            <Button size="sm" variant="ghost" onClick={history.reload}>
              更新
            </Button>
          }
        >
          <div className="flex max-h-[520px] flex-col gap-1.5 overflow-y-auto">
            {(history.data ?? []).length === 0 && <EmptyState title="まだ診断がありません" hint="左のフォームから診断を実行してください" />}
            {(history.data ?? []).map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => void openReport.run(item.id)}
                className={cx(
                  "rounded-lg border px-3 py-2 text-left text-xs",
                  report?.id === item.id ? "border-accent bg-accent-soft/40" : "border-ink-700 bg-ink-850 hover:border-ink-500",
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-base font-bold" style={{ color: GRADE_COLORS[item.grade] ?? "#c5cbe0" }}>
                    {item.grade}
                  </span>
                  <span className="truncate font-medium text-ink-100">{item.title}</span>
                </div>
                <div className="mt-0.5 flex flex-wrap gap-2 text-ink-400">
                  <span>{item.overall_score.toFixed(1)} 点</span>
                  <span>{labelOf(DIAGNOSTIC_SOURCE_LABELS, item.source_type)}</span>
                  <span>{item.page_count}ページ</span>
                  <span>所見 {item.findings}</span>
                  <span>処方 {item.prescriptions}</span>
                </div>
                <div className="text-[10px] text-ink-500">{dateTime(item.created_at)}</div>
              </button>
            ))}
          </div>
          <ErrorNote message={history.error} />
        </Card>
      </div>

      {!report && <EmptyState title="診断結果がここに表示されます" hint="保存済みネーム・原稿画像・アノテーション JSON のいずれかを入力して「診断する」を押してください" />}

      {report && (
        <>
          <Card
            title={report.title}
            subtitle={
              <span>
                <span className="font-mono">{report.id}</span>・{labelOf(DIAGNOSTIC_SOURCE_LABELS, report.source_type)}
                {report.target_structure_id ? `・目標 ${report.target_structure_id}` : "・目標なし"}・{dateTime(report.created_at)}
              </span>
            }
            actions={
              <>
                <Button size="sm" onClick={() => downloadJson(`${report.id}.json`, report)}>
                  レポート JSON
                </Button>
                {report.storyboard && (
                  <Button size="sm" variant="ghost" onClick={() => downloadJson(`${report.id}_storyboard.json`, report.storyboard)}>
                    解析ネーム JSON
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="danger"
                  loading={removeReport.running}
                  onClick={() => {
                    if (window.confirm("この診断レポートを削除しますか？")) {
                      void removeReport.run();
                    }
                  }}
                >
                  削除
                </Button>
              </>
            }
          >
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-[200px_minmax(0,1fr)_320px]">
              <div className="flex flex-col items-center gap-2">
                <ScoreGauge score={report.overall_score} threshold={80} label="総合スコア" size={180} />
                <div className="flex items-center gap-2">
                  <span className="text-xs text-ink-400">評価</span>
                  <span className="font-mono text-3xl font-bold" style={{ color: GRADE_COLORS[report.grade] ?? "#e6e9f3" }}>
                    {report.grade}
                  </span>
                </div>
                {report.used_llm && <Badge tone="accent">DeepSeek 編集者コメント</Badge>}
              </div>
              <div className="flex flex-col gap-3">
                <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
                  <Stat label="ページ" value={pageCount} />
                  <Stat label="コマ" value={report.panels.length} />
                  <Stat label="所見" value={report.findings.length} hint={`致命的 ${report.findings.filter((item) => item.severity === "critical").length}`} />
                  <Stat label="処方" value={report.prescriptions.length} hint={`自動適用可 ${applicableCount}`} />
                  <Stat
                    label="目標相関"
                    value={report.intensity_curve.correlation === null ? "-" : report.intensity_curve.correlation.toFixed(2)}
                    hint={report.intensity_curve.dtw_distance === null ? "目標なし" : `DTW ${report.intensity_curve.dtw_distance.toFixed(3)}`}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  {DIAGNOSTIC_AXIS_ORDER.map((key) => {
                    const axis = report.axes.find((item) => item.axis === key);
                    if (!axis) {
                      return null;
                    }
                    return (
                      <div key={key} className="flex flex-col gap-0.5">
                        <div className="flex items-center gap-2 text-xs">
                          <span className="w-28 shrink-0 font-medium" style={{ color: DIAGNOSTIC_AXIS_COLORS[key] }}>
                            {axis.label || labelOf(DIAGNOSTIC_AXIS_LABELS, key)}
                          </span>
                          <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-700">
                            <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(100, axis.score))}%`, backgroundColor: DIAGNOSTIC_AXIS_COLORS[key] }} />
                          </div>
                          <span className="w-12 shrink-0 text-right font-mono text-ink-100">{axis.score.toFixed(1)}</span>
                          <span className="w-12 shrink-0 text-right font-mono text-[10px] text-ink-500">×{axis.weight.toFixed(2)}</span>
                        </div>
                        {axis.summary && <p className="pl-[7.5rem] text-[11px] text-ink-400">{axis.summary}</p>}
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="flex justify-center">
                <RadarChart axes={radarAxes} compare={comparison?.beforeAxes ?? null} compareLabel="適用前" size={300} />
              </div>
            </div>
            {comparison && (
              <div className="mt-3 flex flex-wrap items-center gap-3 rounded-lg border border-emerald-500/40 bg-emerald-950/30 px-3 py-2 text-xs">
                <Badge tone="success">処方 {comparison.applied} 件を適用</Badge>
                <span className="text-ink-200">
                  適用前 <span className="font-mono">{comparison.before.toFixed(1)}</span> 点 → 適用後{" "}
                  <span className="font-mono text-emerald-300">{comparison.after.toFixed(1)}</span> 点（
                  <span className="font-mono">{comparison.after - comparison.before >= 0 ? "+" : ""}
                  {(comparison.after - comparison.before).toFixed(2)}</span>）
                </span>
                {comparison.savedStoryboardId && (
                  <span className="text-ink-300">
                    改善版ネームを保存しました: <span className="font-mono">{comparison.savedStoryboardId}</span>
                  </span>
                )}
              </div>
            )}
          </Card>

          <Card title="ページ別ヒートストリップ" subtitle="得点・情報密度・平均/最大の推定強度（め=めくり直後、引=めくり前の引き）。クリックでオーバーレイのページを切り替え">
            <PageHeatStrip pages={report.pages} selected={page} onSelect={setPage} />
          </Card>

          <Card
            title="推定劇的強度と目標曲線"
            subtitle={
              report.intensity_curve.target
                ? `相関 ${report.intensity_curve.correlation === null ? "-" : report.intensity_curve.correlation.toFixed(3)}・DTW 距離 ${report.intensity_curve.dtw_distance === null ? "-" : report.intensity_curve.dtw_distance.toFixed(3)}（相関が高く DTW が小さいほど意図した感情曲線どおりの画面になっています）`
                : "目標曲線なし（目標構造を選ぶと、意図した感情曲線との一致度を評価します）"
            }
          >
            <IntensityCurveChart curve={report.intensity_curve} panels={report.panels} selectedPage={page} onSelectPage={setPage} />
          </Card>

          <div className="grid grid-cols-1 gap-4 2xl:grid-cols-[minmax(0,1fr)_460px]">
            <Card
              title="診断オーバーレイ"
              subtitle={
                currentPageMetrics
                  ? `${page + 1}ページ（${currentPageMetrics.side === "right" ? "右" : "左"}・${currentPageMetrics.page_turn_role}）・${currentPageMetrics.panel_count}コマ・${currentPageMetrics.char_count}字・逆行 ${currentPageMetrics.backtracks}・交差 ${currentPageMetrics.crossings}・Z字適合 ${(currentPageMetrics.z_flow_conformity * 100).toFixed(0)}%`
                  : undefined
              }
              actions={
                <>
                  <Button size="sm" variant="ghost" disabled={page <= 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
                    ← 前
                  </Button>
                  <span className="font-mono text-xs text-ink-300">
                    {page + 1} / {Math.max(1, pageCount)}
                  </span>
                  <Button size="sm" variant="ghost" disabled={page >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>
                    次 →
                  </Button>
                </>
              }
            >
              <div className="mb-3 flex flex-wrap items-center gap-4">
                <Toggle checked={overlay.heat} onChange={(value) => setOverlay({ ...overlay, heat: value })} label="推定強度ヒート" />
                <Toggle checked={overlay.gaze} onChange={(value) => setOverlay({ ...overlay, gaze: value })} label="視線経路（逆行=赤）" />
                <Toggle checked={overlay.pins} onChange={(value) => setOverlay({ ...overlay, pins: value })} label="所見ピン" />
                <Toggle checked={overlay.order} onChange={(value) => setOverlay({ ...overlay, order: value })} label="フキダシ読み順" />
                {hasImages && <Toggle checked={overlay.image} onChange={(value) => setOverlay({ ...overlay, image: value })} label="元画像" />}
                <div className="w-48">
                  <Slider label="ズーム" value={zoom} min={0.4} max={2} step={0.05} onChange={setZoom} format={(value) => `${Math.round(value * 100)}%`} />
                </div>
              </div>
              <ErrorNote message={overlayError} />
              {pageCount > 0 && (
                <div className="overflow-auto rounded-lg border border-ink-700 bg-ink-800 p-3">
                  <img
                    key={`${report.id}-${page}-${overlay.heat}-${overlay.gaze}-${overlay.pins}-${overlay.order}-${overlay.image}`}
                    src={api.diagnostics.overlaySvgUrl(report.id, page, overlay)}
                    alt={`${page + 1}ページの診断オーバーレイ`}
                    style={{ width: `${560 * zoom}px`, maxWidth: "none" }}
                    className="mx-auto block bg-white"
                    onError={() => setOverlayError("オーバーレイ SVG を取得できませんでした（レポートが削除されたか、バックエンドが応答していません）")}
                  />
                </div>
              )}
              {pageFindings.length > 0 && (
                <div className="mt-3 flex flex-col gap-1">
                  <p className="text-[11px] font-semibold text-ink-400">このページのピン</p>
                  {pageFindings.map((finding, index) => (
                    <p key={finding.id} className="text-[11px] text-ink-300">
                      <span className="mr-1 font-mono text-ink-500">{index + 1}.</span>
                      <Badge tone={DIAGNOSTIC_SEVERITY_TONES[finding.severity] ?? "neutral"}>{labelOf(DIAGNOSTIC_SEVERITY_LABELS, finding.severity)}</Badge> {finding.title}
                    </p>
                  ))}
                </div>
              )}
            </Card>

            <div className="flex flex-col gap-4">
              <Card title="編集者コメント" subtitle={report.used_llm ? "DeepSeek による増幅" : "決定論ベースライン"}>
                {report.commentary ? (
                  <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink-200">{report.commentary}</p>
                ) : (
                  <p className="text-xs text-ink-500">コメントはありません。</p>
                )}
              </Card>
              <Card
                title="処方箋（期待改善量順）"
                subtitle="各処方を複製ネームへ実際に適用して再採点した改善量です。複数選択して一括適用できます"
                actions={
                  <>
                    <Toggle checked={saveImproved} onChange={setSaveImproved} label="改善版ネームを保存" />
                    <Button size="sm" variant="success" disabled={selectedRx.length === 0} loading={apply.running} onClick={() => void apply.run()}>
                      選択した処方を適用（{selectedRx.length}）
                    </Button>
                  </>
                }
              >
                {sortedPrescriptions.length === 0 && <EmptyState title="処方はありません" hint="改善の余地が小さいか、シミュレーションで改善が確認できる修正が見つかりませんでした" />}
                <div className="mb-2 flex flex-wrap gap-2">
                  {sortedPrescriptions.length > 0 && (
                    <>
                      <Button size="sm" variant="ghost" onClick={() => setSelectedRx(sortedPrescriptions.filter((item) => item.applicable).slice(0, 3).map((item) => item.id))}>
                        上位3件を選択
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setSelectedRx([])}>
                        選択解除
                      </Button>
                    </>
                  )}
                </div>
                <div className="flex max-h-[560px] flex-col gap-2 overflow-y-auto">
                  {sortedPrescriptions.map((prescription) => (
                    <PrescriptionItem
                      key={prescription.id}
                      prescription={prescription}
                      maxGain={maxGain}
                      checked={selectedRx.includes(prescription.id)}
                      onToggle={toggleRx}
                      onJump={setPage}
                    />
                  ))}
                </div>
              </Card>
            </div>
          </div>

          <Card
            title={`所見一覧（${filteredFindings.length} / ${report.findings.length}）`}
            subtitle="観測値と目標値の差として定量化された問題点。クリックで該当ページのオーバーレイを表示"
            actions={
              <>
                <Select
                  value={axisFilter}
                  options={[{ value: "all", label: "全ての軸" }, ...DIAGNOSTIC_AXIS_ORDER.map((key) => ({ value: key as string, label: labelOf(DIAGNOSTIC_AXIS_LABELS, key) }))]}
                  onChange={setAxisFilter}
                  ariaLabel="所見の軸フィルタ"
                />
                <Select
                  value={severityFilter}
                  options={[
                    { value: "all", label: "全ての重要度" },
                    ...["critical", "major", "minor", "info"].map((key) => ({ value: key, label: labelOf(DIAGNOSTIC_SEVERITY_LABELS, key) })),
                  ]}
                  onChange={setSeverityFilter}
                  ariaLabel="所見の重要度フィルタ"
                />
              </>
            }
          >
            {filteredFindings.length === 0 && <EmptyState title="該当する所見はありません" />}
            <div className="grid max-h-[560px] grid-cols-1 gap-2 overflow-y-auto lg:grid-cols-2">
              {filteredFindings.map((finding) => (
                <FindingItem key={finding.id} finding={finding} onJump={setPage} highlighted={finding.page_index === page} />
              ))}
            </div>
          </Card>

          <Card title="ページ別の主要指標" subtitle="視線経路長はページ対角線長で正規化。曖昧さは読み順が段組みから一意に決まらないコマ対の比率">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-ink-400">
                  <tr>
                    <th className="py-1 pr-2">ページ</th>
                    <th className="py-1 pr-2">役割</th>
                    <th className="py-1 pr-2">得点</th>
                    <th className="py-1 pr-2">コマ</th>
                    <th className="py-1 pr-2">字数</th>
                    <th className="py-1 pr-2">面積CV</th>
                    <th className="py-1 pr-2">最大面積</th>
                    <th className="py-1 pr-2">平均強度</th>
                    <th className="py-1 pr-2">経路長</th>
                    <th className="py-1 pr-2">逆行</th>
                    <th className="py-1 pr-2">交差</th>
                    <th className="py-1 pr-2">Z字適合</th>
                    <th className="py-1 pr-2">曖昧さ</th>
                    <th className="py-1 pr-2">めくり/引き強度</th>
                  </tr>
                </thead>
                <tbody>
                  {report.pages.map((item) => (
                    <tr
                      key={item.page_index}
                      onClick={() => setPage(item.page_index)}
                      className={cx("cursor-pointer border-t border-ink-800 text-ink-200 hover:bg-ink-850", item.page_index === page && "bg-accent-soft/30")}
                    >
                      <td className="py-1 pr-2 font-mono">
                        {item.page_index + 1}
                        {item.side === "right" ? "右" : "左"}
                      </td>
                      <td className="py-1 pr-2">{item.page_turn_role}</td>
                      <td className="py-1 pr-2 font-mono">{item.score.toFixed(1)}</td>
                      <td className="py-1 pr-2 font-mono">{item.panel_count}</td>
                      <td className="py-1 pr-2 font-mono">{item.char_count}</td>
                      <td className="py-1 pr-2 font-mono">{item.area_cv.toFixed(2)}</td>
                      <td className="py-1 pr-2 font-mono">{item.max_area_ratio.toFixed(2)}</td>
                      <td className="py-1 pr-2 font-mono">{item.mean_intensity.toFixed(2)}</td>
                      <td className="py-1 pr-2 font-mono">{item.gaze_path_length.toFixed(2)}</td>
                      <td className={cx("py-1 pr-2 font-mono", item.backtracks > 0 && "text-rose-300")}>{item.backtracks}</td>
                      <td className={cx("py-1 pr-2 font-mono", item.crossings > 0 && "text-amber-300")}>{item.crossings}</td>
                      <td className="py-1 pr-2 font-mono">{(item.z_flow_conformity * 100).toFixed(0)}%</td>
                      <td className="py-1 pr-2 font-mono">{item.order_ambiguity.toFixed(2)}</td>
                      <td className="py-1 pr-2 font-mono">
                        {item.reveal_strength !== null ? `め ${item.reveal_strength.toFixed(2)}` : ""}
                        {item.hook_strength !== null ? ` 引 ${item.hook_strength.toFixed(2)}` : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
      <ErrorNote message={structures.error ?? storyboards.error} />
    </div>
  );
}

