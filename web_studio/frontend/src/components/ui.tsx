// 汎用 UI 部品（ダークモード基調）。

import { useState } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { copyToClipboard } from "../api/client";

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cx("rounded-xl border border-ink-700/80 bg-ink-900/80 shadow-lg shadow-black/20", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-ink-700/70 px-4 py-3">
          <div className="min-w-0">
            {title && <h3 className="text-sm font-semibold tracking-wide text-ink-100">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-400">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "success";

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary: "bg-accent text-ink-950 hover:bg-accent-strong disabled:bg-ink-600 disabled:text-ink-400",
  secondary: "border border-ink-600 bg-ink-800 text-ink-100 hover:border-accent/60 hover:bg-ink-700 disabled:text-ink-500",
  ghost: "text-ink-300 hover:bg-ink-800 hover:text-ink-100 disabled:text-ink-600",
  danger: "border border-rose-500/40 bg-rose-950/60 text-rose-200 hover:bg-rose-900/70 disabled:opacity-50",
  success: "border border-emerald-500/40 bg-emerald-950/60 text-emerald-200 hover:bg-emerald-900/70 disabled:opacity-50",
};

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  className,
  children,
  disabled,
  type = "button",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: "sm" | "md"; loading?: boolean }) {
  return (
    <button
      type={type}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed",
        size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm",
        BUTTON_STYLES[variant],
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading && <Spinner small />}
      {children}
    </button>
  );
}

type Tone = "neutral" | "accent" | "success" | "warning" | "danger" | "info";

const BADGE_STYLES: Record<Tone, string> = {
  neutral: "border-ink-600 bg-ink-800 text-ink-200",
  accent: "border-accent/40 bg-accent-soft/60 text-accent-strong",
  success: "border-emerald-500/40 bg-emerald-950/70 text-emerald-200",
  warning: "border-amber-500/40 bg-amber-950/70 text-amber-200",
  danger: "border-rose-500/40 bg-rose-950/70 text-rose-200",
  info: "border-sky-500/40 bg-sky-950/70 text-sky-200",
};

export function Badge({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={cx("inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium", BADGE_STYLES[tone])}>
      {children}
    </span>
  );
}

export interface TabItem<K extends string> {
  key: K;
  label: ReactNode;
}

export function Tabs<K extends string>({
  tabs,
  active,
  onChange,
  size = "md",
}: {
  tabs: Array<TabItem<K>>;
  active: K;
  onChange: (key: K) => void;
  size?: "sm" | "md";
}) {
  return (
    <div className="flex flex-wrap gap-1 rounded-lg border border-ink-700 bg-ink-850 p-1" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          role="tab"
          aria-selected={tab.key === active}
          onClick={() => onChange(tab.key)}
          className={cx(
            "rounded-md font-medium transition-colors",
            size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
            tab.key === active ? "bg-accent-soft text-accent-strong shadow" : "text-ink-300 hover:bg-ink-800 hover:text-ink-100",
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={cx("flex flex-col gap-1 text-xs", className)}>
      <span className="font-medium text-ink-300">{label}</span>
      {children}
      {hint && <span className="text-[11px] text-ink-500">{hint}</span>}
    </label>
  );
}

const INPUT_CLASS =
  "w-full rounded-lg border border-ink-600 bg-ink-850 px-3 py-2 text-sm text-ink-100 placeholder:text-ink-500 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/60";

export function TextInput({
  value,
  onChange,
  placeholder,
  type = "text",
  className,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: "text" | "password" | "url" | "search";
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
      className={cx(INPUT_CLASS, className)}
    />
  );
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  className,
  ariaLabel,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <input
      type="number"
      value={Number.isFinite(value) ? value : 0}
      min={min}
      max={max}
      step={step}
      aria-label={ariaLabel}
      onChange={(event) => {
        const next = Number(event.target.value);
        if (Number.isFinite(next)) {
          onChange(next);
        }
      }}
      className={cx(INPUT_CLASS, className)}
    />
  );
}

export function TextArea({
  value,
  onChange,
  rows = 6,
  placeholder,
  className,
  monospace = false,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  rows?: number;
  placeholder?: string;
  className?: string;
  monospace?: boolean;
  ariaLabel?: string;
}) {
  return (
    <textarea
      value={value}
      rows={rows}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
      className={cx(INPUT_CLASS, "resize-y leading-relaxed", monospace && "font-mono text-xs", className)}
    />
  );
}

export interface SelectOption<V extends string> {
  value: V;
  label: string;
}

export function Select<V extends string>({
  value,
  options,
  onChange,
  className,
  ariaLabel,
}: {
  value: V;
  options: Array<SelectOption<V>>;
  onChange: (value: V) => void;
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <select
      value={value}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value as V)}
      className={cx(INPUT_CLASS, "cursor-pointer pr-8", className)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Slider({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
  format,
  hint,
}: {
  label: ReactNode;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  hint?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-ink-300">{label}</span>
        <span className="font-mono text-accent-strong">{format ? format(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-700 accent-[#7c9cff]"
      />
      {hint && <span className="text-[11px] text-ink-500">{hint}</span>}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: ReactNode }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 text-left text-xs text-ink-200"
    >
      <span className={cx("relative inline-flex h-5 w-9 shrink-0 rounded-full transition-colors", checked ? "bg-accent" : "bg-ink-600")}>
        <span className={cx("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform", checked ? "translate-x-4" : "translate-x-0.5")} />
      </span>
      <span>{label}</span>
    </button>
  );
}

export function Spinner({ small = false }: { small?: boolean }) {
  return (
    <span
      className={cx("inline-block animate-spin rounded-full border-2 border-current border-t-transparent", small ? "h-3.5 w-3.5" : "h-5 w-5")}
      aria-hidden="true"
    />
  );
}

export function ProgressBar({ value, label, tone = "accent" }: { value: number; label?: ReactNode; tone?: "accent" | "success" | "danger" }) {
  const width = `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`;
  const color = tone === "success" ? "bg-emerald-400" : tone === "danger" ? "bg-rose-400" : "bg-accent";
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <div className="flex items-center justify-between text-xs text-ink-300">
          <span className="truncate">{label}</span>
          <span className="font-mono">{width}</span>
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-ink-700">
        <div className={cx("h-full rounded-full transition-all duration-300", color)} style={{ width }} />
      </div>
    </div>
  );
}

export function EmptyState({ title, hint, action }: { title: ReactNode; hint?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-ink-600 px-6 py-10 text-center">
      <p className="text-sm font-medium text-ink-200">{title}</p>
      {hint && <p className="max-w-md text-xs text-ink-400">{hint}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: ReactNode; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2">
      <div className="text-[11px] text-ink-400">{label}</div>
      <div className="mt-0.5 font-mono text-lg font-semibold text-ink-100">{value}</div>
      {hint && <div className="text-[11px] text-ink-500">{hint}</div>}
    </div>
  );
}

export function ErrorNote({ message }: { message: string | null | undefined }) {
  if (!message) {
    return null;
  }
  return <div className="rounded-lg border border-rose-500/40 bg-rose-950/50 px-3 py-2 text-xs text-rose-200">{message}</div>;
}

export function ScoreBar({ value, max = 1, label, color = "#7c9cff" }: { value: number; max?: number; label?: ReactNode; color?: string }) {
  const ratio = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
  return (
    <div className="flex items-center gap-2">
      {label && <span className="w-28 shrink-0 truncate text-xs text-ink-300">{label}</span>}
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-700">
        <div className="h-full rounded-full" style={{ width: `${ratio * 100}%`, backgroundColor: color }} />
      </div>
      <span className="w-12 shrink-0 text-right font-mono text-xs text-ink-200">{value.toFixed(max > 1 ? 1 : 2)}</span>
    </div>
  );
}

export function CopyButton({ text, label = "コピー" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      onClick={async () => {
        const ok = await copyToClipboard(text);
        setCopied(ok);
        window.setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? "コピー済み" : label}
    </Button>
  );
}

export function KeyValueEditor({
  entries,
  onChange,
  keyPlaceholder = "キー",
  valuePlaceholder = "値",
}: {
  entries: Array<[string, string]>;
  onChange: (entries: Array<[string, string]>) => void;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
}) {
  const update = (index: number, position: 0 | 1, value: string) => {
    const next = entries.map((entry) => [...entry] as [string, string]);
    next[index][position] = value;
    onChange(next);
  };
  return (
    <div className="flex flex-col gap-2">
      {entries.map(([key, value], index) => (
        <div key={index} className="flex items-center gap-2">
          <div className="w-2/5 shrink-0">
            <TextInput value={key} onChange={(next) => update(index, 0, next)} placeholder={keyPlaceholder} className="font-mono text-xs" ariaLabel={`${keyPlaceholder} ${index + 1}`} />
          </div>
          <div className="min-w-0 flex-1">
            <TextInput value={value} onChange={(next) => update(index, 1, next)} placeholder={valuePlaceholder} ariaLabel={`${valuePlaceholder} ${index + 1}`} />
          </div>
          <Button size="sm" variant="ghost" onClick={() => onChange(entries.filter((_, i) => i !== index))} aria-label="行を削除">
            ✕
          </Button>
        </div>
      ))}
      <div>
        <Button size="sm" variant="ghost" onClick={() => onChange([...entries, ["", ""]])}>
          ＋ 行を追加
        </Button>
      </div>
    </div>
  );
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-2 flex items-center justify-between gap-2">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-400">{children}</h4>
      {aside}
    </div>
  );
}

export function TagList({ items, tone = "neutral" }: { items: string[]; tone?: Tone }) {
  if (items.length === 0) {
    return <span className="text-xs text-ink-500">なし</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item, index) => (
        <Badge key={`${item}-${index}`} tone={tone}>
          {item}
        </Badge>
      ))}
    </div>
  );
}
