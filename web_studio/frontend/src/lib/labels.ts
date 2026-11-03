// 表示用の日本語ラベル・配色定義。

import type { BaseWaveChannel, EdgeKind, NodeKind, WaveChannel } from "../api/types";

export const NODE_KIND_LABELS: Record<NodeKind, string> = {
  agent: "主体",
  desire: "欲求",
  lack: "欠落",
  symbol: "象徴",
};

export const NODE_KIND_COLORS: Record<NodeKind, string> = {
  agent: "#818cf8",
  desire: "#fbbf24",
  lack: "#fb7185",
  symbol: "#34d399",
};

export const EDGE_KIND_LABELS: Record<EdgeKind, string> = {
  opposition: "対立",
  dependency: "依存",
  asymmetry: "非対称性",
  power_gradient: "権力勾配",
  betrayal: "裏切り",
  pursuit: "追求",
  embodiment: "体現",
};

export const EDGE_KIND_STYLES: Record<EdgeKind, { color: string; dash: string }> = {
  opposition: { color: "#f87171", dash: "" },
  dependency: { color: "#94a3b8", dash: "" },
  asymmetry: { color: "#38bdf8", dash: "6 4" },
  power_gradient: { color: "#a78bfa", dash: "" },
  betrayal: { color: "#fb923c", dash: "2 4" },
  pursuit: { color: "#fbbf24", dash: "" },
  embodiment: { color: "#34d399", dash: "4 3" },
};

export const WAVE_CHANNEL_LABELS: Record<WaveChannel, string> = {
  tension: "緊張度",
  release: "カタルシス解放度",
  audience_knowledge: "読者認知量",
  character_knowledge: "作中人物認知量",
  cognitive_load: "認知負荷",
  information_asymmetry: "情報非対称性",
};

export const WAVE_CHANNEL_COLORS: Record<WaveChannel, string> = {
  tension: "#f472b6",
  release: "#34d399",
  audience_knowledge: "#a78bfa",
  character_knowledge: "#2dd4bf",
  cognitive_load: "#fbbf24",
  information_asymmetry: "#38bdf8",
};

export const BASE_WAVE_CHANNELS: BaseWaveChannel[] = [
  "tension",
  "release",
  "audience_knowledge",
  "character_knowledge",
  "cognitive_load",
];

export const ALL_WAVE_CHANNELS: WaveChannel[] = [...BASE_WAVE_CHANNELS, "information_asymmetry"];

export const BEAT_TYPE_LABELS: Record<string, string> = {
  hook: "フック",
  premise: "前提の提示",
  setup: "仕込み",
  escalation: "段階的上昇",
  local_optimization_limit: "局所的最適化の限界",
  false_victory: "偽の勝利",
  crisis: "危機",
  revelation: "真相の開示",
  structural_twist: "構造的反転",
  costly_breakthrough: "代償を伴う突破",
  climax: "クライマックス",
  payoff: "回収",
  afterglow: "余韻",
  transition: "転換",
};

export const MECHANISM_LABELS: Record<string, string> = {
  curiosity_gap: "情報ギャップ",
  dramatic_irony: "劇的アイロニー",
  empathy_transfer: "共感移入",
  prediction_error: "予測誤差",
  cognitive_dissonance: "認知的不協和",
  identity_projection: "自己投影",
  moral_judgment: "道徳的判断",
  loss_aversion: "損失回避",
  social_proof: "社会的証明",
  zeigarnik: "未完了効果",
  peak_end: "ピーク・エンド",
  catharsis: "カタルシス",
  scarcity: "希少性",
  reciprocity: "返報性",
  authority: "権威",
  commitment_consistency: "コミットメントと一貫性",
  narrative_transportation: "物語移入",
  anticipation_reward: "予期と報酬",
  contrast_effect: "対比効果",
  self_efficacy: "自己効力感",
};

export const CATEGORY_LABELS: Record<string, string> = {
  narrative_archetype: "物語原型",
  manga_panel_rhythm: "漫画演出リズム",
  persuasion_marketing: "説得・マーケティング",
  custom: "カスタム",
};

export const SOURCE_KIND_LABELS: Record<string, string> = {
  seed: "シード",
  extracted: "抽出",
  merged: "合成",
  mutated: "変異",
  manual: "手動",
};

export const DOMAIN_LABELS: Record<string, string> = {
  manga_plot: "漫画プロット",
  marketing_lp: "LP・セールスファネル",
  article: "考察・解説記事",
};

export const AXIS_LABELS: Record<string, string> = {
  creator_theory: "クリエイター向け：物語構造・漫画演出理論",
  ai_engineering: "AI開発者向け：モデル調合・特徴量制御・データセット設計論",
  business_marketing: "ビジネス向け：企画・LP・マーケティング構造論",
};

export const AXIS_SHORT_LABELS: Record<string, string> = {
  creator_theory: "クリエイター",
  ai_engineering: "AI開発者",
  business_marketing: "ビジネス",
};

export const SHOT_LABELS: Record<string, string> = {
  ECU: "超クローズアップ",
  CU: "クローズアップ",
  MS: "ミディアム",
  FS: "フルショット",
  ELS: "超ロング",
};

export const ANGLE_LABELS: Record<string, string> = {
  eye_level: "アイレベル",
  high: "俯瞰",
  low: "アオリ",
  dutch: "ダッチ",
};

export const BALLOON_LABELS: Record<string, string> = {
  normal: "通常",
  shout: "叫び",
  monologue: "モノローグ",
  flash: "フラッシュ",
};

export const ARCHITECTURE_LABELS: Record<string, string> = {
  sdxl: "SDXL（UNet: BASE/IN/MID/OUT）",
  sd15: "SD1.5（UNet）",
  flux: "FLUX（Double/Single Blocks）",
  anima: "Anima（汎用DiT）",
};

export const TAG_CATEGORY_LABELS: Record<string, string> = {
  style: "絵柄",
  composition: "構図",
  camera_angle: "カメラアングル",
  pose: "ポーズ",
  expression: "表情",
  lighting: "照明",
  background: "背景",
  subject: "被写体",
  quality: "品質タグ",
  meta: "メタ",
  other: "その他",
};

export const SEVERITY_TONES: Record<string, "danger" | "warning" | "info" | "neutral"> = {
  error: "danger",
  high: "danger",
  warning: "warning",
  medium: "warning",
  info: "info",
  low: "info",
};

export function labelOf(map: Record<string, string>, key: string | null | undefined): string {
  if (!key) {
    return "-";
  }
  return map[key] ?? key;
}
