// 数値・日時・波形指標のフォーマットと計算補助。

import type { EmotionalInformationWave, WaveChannel } from "../api/types";

export function fmt(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

export function pct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function bytes(size: number | null | undefined): string {
  if (size === null || size === undefined) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function dateTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("ja-JP", { hour12: false });
}

export function clamp(value: number, lo = 0, hi = 1): number {
  return Math.min(hi, Math.max(lo, value));
}

export function channelValues(wave: EmotionalInformationWave, channel: WaveChannel): number[] {
  if (channel === "information_asymmetry") {
    return wave.audience_knowledge.map((a, i) => a - (wave.character_knowledge[i] ?? 0));
  }
  return wave[channel];
}

export interface WaveSummary {
  peakT: number;
  peakTension: number;
  meanTension: number;
  suspense: number;
  surprise: number;
  catharsis: number;
}

/** core/structure_engine/schemas.py の波形指標と同じ定義で算出する。 */
export function summarizeWave(wave: EmotionalInformationWave): WaveSummary {
  const tension = wave.tension;
  const n = tension.length;
  let peakIndex = 0;
  for (let i = 1; i < n; i += 1) {
    if (tension[i] > tension[peakIndex]) {
      peakIndex = i;
    }
  }
  const asym = channelValues(wave, "information_asymmetry");
  let suspense = 0;
  for (let i = 0; i < n; i += 1) {
    suspense += Math.max(asym[i], 0) * tension[i];
  }
  suspense = n > 0 ? clamp(suspense / n) : 0;
  let surprise = 0;
  for (let i = 0; i < n - 1; i += 1) {
    const jump = Math.max(0, wave.audience_knowledge[i + 1] - wave.audience_knowledge[i]);
    const concealment = Math.max(0, -asym[i]);
    surprise = Math.max(surprise, jump * (0.5 + 0.5 * concealment));
  }
  surprise = clamp(2 * surprise);
  const peak = n > 0 ? tension[peakIndex] : 0;
  let catharsis = 0;
  for (let i = 0; i < n - 1; i += 1) {
    catharsis += Math.max(0, tension[i] - tension[i + 1]) * wave.release[i + 1];
  }
  catharsis = peak > 1e-6 ? clamp(catharsis / peak) : 0;
  const meanTension = n > 0 ? tension.reduce((sum, value) => sum + value, 0) / n : 0;
  return {
    peakT: n > 0 ? wave.t[peakIndex] : 0,
    peakTension: peak,
    meanTension,
    suspense,
    surprise,
    catharsis,
  };
}

export function parseList(text: string): string[] {
  return text
    .split(/[,、\n]/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

export function parseAxisValues(text: string): Array<string | number> {
  return parseList(text).map((item) => {
    const numeric = Number(item);
    return item !== "" && Number.isFinite(numeric) ? numeric : item;
  });
}

export function shortId(id: string): string {
  return id.length > 18 ? `${id.slice(0, 8)}…${id.slice(-6)}` : id;
}

export function randomSessionId(): string {
  const random = Math.random().toString(36).slice(2, 10);
  return `session_${Date.now().toString(36)}_${random}`;
}
