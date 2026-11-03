// Creative Asset Foundry API 型定義。
// core の Pydantic モデル（model_dump(mode="json")）と web_studio/backend の応答形に対応する。

// ---------------------------------------------------------------------------
// Structure Engine（5階層・普遍的構造オントロジー）
// ---------------------------------------------------------------------------

export type NodeKind = "agent" | "desire" | "lack" | "symbol";
export type EdgeKind =
  | "opposition"
  | "dependency"
  | "asymmetry"
  | "power_gradient"
  | "betrayal"
  | "pursuit"
  | "embodiment";

export interface TensionNode {
  id: string;
  kind: NodeKind;
  role: string;
  label: string;
  description: string;
  weight: number;
  slot: string | null;
}

export interface TensionEdge {
  source: string;
  target: string;
  kind: EdgeKind;
  intensity: number;
  gradient: number;
  label: string;
  activation_t: number | null;
}

export interface DynamicTensionGraph {
  nodes: TensionNode[];
  edges: TensionEdge[];
}

export type BaseWaveChannel =
  | "tension"
  | "release"
  | "audience_knowledge"
  | "character_knowledge"
  | "cognitive_load";
export type WaveChannel = BaseWaveChannel | "information_asymmetry";

export interface EmotionalInformationWave {
  t: number[];
  tension: number[];
  release: number[];
  audience_knowledge: number[];
  character_knowledge: number[];
  cognitive_load: number[];
  information_asymmetry?: number[];
}

export type BeatType =
  | "hook"
  | "premise"
  | "setup"
  | "escalation"
  | "local_optimization_limit"
  | "false_victory"
  | "crisis"
  | "revelation"
  | "structural_twist"
  | "costly_breakthrough"
  | "climax"
  | "payoff"
  | "afterglow"
  | "transition";

export interface StructuralBeat {
  id: string;
  beat_type: BeatType;
  t: number;
  span: number;
  label: string;
  function: string;
  causes: string[];
  tension_delta: number;
  information_shift: number;
  involved_nodes: string[];
  is_turning_point: boolean;
}

export interface StructuralBeatChain {
  beats: StructuralBeat[];
}

export interface CognitiveMechanism {
  id: string;
  category: string;
  name: string;
  description: string;
  trigger: string;
  effect: string;
  theory: string;
  strength: number;
  linked_beats: string[];
  linked_nodes: string[];
}

export interface CognitiveEmpathyMechanics {
  mechanisms: CognitiveMechanism[];
  empathy_anchor: string;
  immersion_drivers: string[];
  emotion_sequence: string[];
}

export interface BlueprintSlot {
  name: string;
  description: string;
  category: string;
  constraints: string[];
  examples: Record<string, string>;
  required: boolean;
}

export interface DomainAgnosticBlueprint {
  template: string;
  slots: BlueprintSlot[];
  invariants: string[];
}

export type StructureCategory = "narrative_archetype" | "manga_panel_rhythm" | "persuasion_marketing" | "custom";
export type SourceKind = "seed" | "extracted" | "merged" | "mutated" | "manual";

export interface StructureMetadata {
  title: string;
  summary: string;
  category: StructureCategory;
  source_kind: SourceKind;
  tags: string[];
  origin_domain: string;
  lineage: string[];
  parent_ids: string[];
  language: string;
  version: number;
  created_at: string;
  notes: string;
}

export interface PanelRhythmProfile {
  page_count: number;
  panels_per_page: number[];
  emphasis_pages: number[];
  page_turn_reveals: number[];
  spread_pages: number[];
  silent_panels_ratio: number;
  pacing_notes: string[];
}

export interface TransferGuide {
  manga: string;
  marketing: string;
  article: string;
}

export interface UniversalStructure {
  id: string;
  metadata: StructureMetadata;
  tension_graph: DynamicTensionGraph;
  wave: EmotionalInformationWave;
  beats: StructuralBeatChain;
  mechanics: CognitiveEmpathyMechanics;
  blueprint: DomainAgnosticBlueprint;
  transfer_guide: TransferGuide;
  rhythm_profile: PanelRhythmProfile | null;
}

export interface StructureSummary {
  id: string;
  title: string;
  summary: string;
  category: string;
  source_kind: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  peak_t: number;
  suspense: number;
  surprise: number;
}

export interface StructureList {
  items: StructureSummary[];
  total: number;
}

export interface FlavorToken {
  surface: string;
  category: "person" | "place" | "organization" | "artifact" | "genre_term" | "title" | "other";
  placeholder: string;
  occurrences: number;
}

export interface DeflavorResult {
  abstracted_text: string;
  tokens: FlavorToken[];
  genre_signals: Record<string, number>;
}

export interface ExtractionResult {
  structure: UniversalStructure;
  deflavor: DeflavorResult;
  harness_run_id: string | null;
  used_llm: boolean;
  leaked_tokens_fixed: string[];
  critique_score: number | null;
}

export interface MergeReport {
  node_mapping: Record<string, string | null>;
  weight_topology: number;
  weight_wave: number;
  dtw_alignment_cost: number;
  fixes_applied: string[];
  issues_remaining: string[];
}

export interface MergeResponse {
  structure: UniversalStructure;
  report: MergeReport;
}

export interface MutateResponse {
  structure: UniversalStructure;
}

export interface MutationInfo {
  type: string;
  label: string;
  description: string;
}

export type TargetDomain = "manga_plot" | "marketing_lp" | "article";

export interface DesignSection {
  index: number;
  beat_id: string;
  beat_type: string;
  t_start: number;
  t_end: number;
  heading: string;
  content: string;
  tension_target: number;
  release_target: number;
  information_reveal: string;
  emotional_goal: string;
  mechanisms: string[];
  notes: string[];
}

export interface FidelityReport {
  wave_correlation: number;
  beat_coverage: number;
  topology_preserved: boolean;
  score: number;
}

export interface TranspiledDesign {
  id: string;
  structure_id: string;
  domain: TargetDomain;
  title: string;
  logline: string;
  theme_variables: Record<string, string>;
  slot_bindings: Record<string, string>;
  rendered_blueprint: string;
  sections: DesignSection[];
  domain_payload: Record<string, unknown>;
  fidelity: FidelityReport;
  created_at: string;
}

export interface SimilarityHit {
  id: string;
  title: string;
  category: string;
  score: number;
  dtw_distance: number | null;
  cosine: number | null;
}

export interface SearchResponse {
  hits: SimilarityHit[];
}

export interface MermaidResponse {
  graph: string;
  beats: string;
}

export interface SeedReport {
  inserted: number;
  updated: number;
  skipped: number;
  total: number;
  by_category: Record<string, number>;
}

export interface SeedSummary {
  total: number;
  by_category: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Style Alchemy
// ---------------------------------------------------------------------------

export type ModelArchitecture = "sdxl" | "sd15" | "flux" | "anima";
export type BlockGrouping = "grouped" | "per_block";

export interface CheckpointRef {
  name: string;
  path?: string;
  architecture?: ModelArchitecture;
}

export interface BlockMergeRecipe {
  architecture: ModelArchitecture;
  method: "weighted_sum" | "add_difference";
  model_a: CheckpointRef;
  model_b: CheckpointRef;
  model_c: CheckpointRef | null;
  block_weights: Record<string, number>;
  default_weight: number;
}

export interface LoRASpec {
  name: string;
  path?: string;
  strength_model?: number;
  strength_clip?: number;
  block_weights?: Record<string, number> | null;
  trigger_words?: string[];
}

export interface SamplerSettings {
  sampler_name: string;
  scheduler: string;
  steps: number;
  cfg: number;
  denoise: number;
  clip_skip: number;
}

export interface StyleRecipe {
  id: string;
  name: string;
  architecture: ModelArchitecture;
  base_checkpoint: CheckpointRef;
  merge: BlockMergeRecipe | null;
  loras: LoRASpec[];
  sampler: SamplerSettings;
  vae: string | null;
  text_encoders: string[];
  text_encoder_type: string | null;
  style_tokens: string[];
  negative_tokens: string[];
  params: Record<string, number>;
  created_at: string;
  notes: string;
}

export interface StyleCandidate {
  id: string;
  label: string;
  params: Record<string, number>;
  recipe: StyleRecipe | null;
  image_ref: string | null;
  source: string;
  created_at: string;
}

export type ComparisonOutcome = "left" | "right" | "tie";

export interface Comparison {
  id: string;
  left_id: string;
  right_id: string;
  outcome: ComparisonOutcome;
  response_time_ms: number | null;
  session_id: string | null;
  created_at: string;
}

export interface RatingEntry {
  candidate_id: string;
  label: string;
  elo: number;
  bt_strength: number;
  bt_elo_scale: number;
  bt_std_error: number;
  games: number;
  wins: number;
  losses: number;
  ties: number;
  win_rate: number;
  rank: number;
}

export interface ArenaPair {
  left: StyleCandidate;
  right: StyleCandidate;
}

export interface FeatureWeight {
  name: string;
  coefficient: number;
  std_coefficient: number;
  importance: number;
  permutation_importance: number;
  ci_low: number;
  ci_high: number;
  direction: "prefer_higher" | "prefer_lower" | "neutral";
}

export interface AestheticPreferenceVector {
  features: FeatureWeight[];
  intercept: number;
  method: "ridge_bt" | "pairwise_logistic";
  regularization: number;
  fit_accuracy: number;
  r2: number | null;
  n_comparisons: number;
  n_candidates: number;
  created_at: string;
}

export interface ParameterDim {
  name: string;
  low: number;
  high: number;
  default: number;
  group: string;
  description: string;
}

export interface ParameterSpace {
  dims: ParameterDim[];
}

export interface Observation {
  params: Record<string, number>;
  score: number;
  candidate_id: string | null;
  created_at?: string;
}

export interface OptimizationSuggestion {
  params: Record<string, number>;
  predicted_mean: number;
  predicted_std: number;
  acquisition: number;
  strategy: string;
  rationale: string;
  rank: number;
}

export interface SuggestResponse {
  suggestions: OptimizationSuggestion[];
  best: Observation | null;
  observations: Observation[] | number;
  registered?: StyleCandidate[];
  space?: ParameterSpace;
  sensitivity?: Record<string, number>;
}

export interface RecipeExportResponse {
  recipe: StyleRecipe;
  merge_config: Record<string, unknown>;
  merge_script: string;
  comfy_workflow: Record<string, unknown>;
}

export interface CaptionRecord {
  image_id: string;
  caption: string;
  path?: string;
}

export interface CleansedCaption {
  image_id: string;
  trigger_token: string;
  caption: string;
  controllable: Record<string, string[]>;
  subject_tags: string[];
  removed_style_tags: string[];
  removed_noise_tags: string[];
  missing_categories: string[];
}

export interface Entanglement {
  tag_a: string;
  category_a: string;
  tag_b: string;
  category_b: string;
  support: number;
  npmi: number;
  conditional_prob: number;
  lift: number;
  severity: "low" | "medium" | "high";
}

export interface TagDeficit {
  category: string;
  tag: string;
  current_count: number;
  current_share: number;
  target_share: number;
  required_additional: number;
}

export interface ShotRequirement {
  conditions: Record<string, string>;
  count: number;
  reason: string;
  priority: number;
}

export interface CooccurrenceMatrix {
  tags: string[];
  counts: number[][];
  npmi: number[][];
}

export interface DatasetGapReport {
  total_images: number;
  tag_counts: Record<string, number>;
  category_distributions: Record<string, Record<string, number>>;
  category_entropy: Record<string, number>;
  normalized_entropy: Record<string, number>;
  cooccurrence: CooccurrenceMatrix;
  entanglements: Entanglement[];
  deficits: TagDeficit[];
  recommended_shots: ShotRequirement[];
  total_additional_images: number;
  balance_score: number;
  warnings: string[];
  created_at: string;
}

export interface DatasetAnalyzeResponse {
  cleansed: CleansedCaption[];
  report: DatasetGapReport;
}

export interface TokenEntry {
  token: string;
  category: string;
  weight: number;
  description: string;
  tags: string[];
}

export type GridAxisKind = "token" | "param" | "lora_weight" | "merge_weight" | "cfg" | "steps" | "seed" | "sampler";

export interface GridAxis {
  name: string;
  kind: GridAxisKind;
  target: string;
  values: Array<string | number>;
}

export interface GridExperiment {
  id: string;
  name: string;
  base_prompt: string;
  negative_prompt: string;
  x_axis: GridAxis;
  y_axis: GridAxis | null;
  z_axis: GridAxis | null;
  base_params: Record<string, number>;
  seed: number;
  status: "draft" | "running" | "completed";
  created_at: string;
}

export interface GridCell {
  id: string;
  experiment_id: string;
  x_index: number;
  y_index: number;
  z_index: number;
  prompt: string;
  negative_prompt: string;
  params: Record<string, number>;
  seed: number;
  sampler_overrides: Record<string, unknown>;
  image_ref: string | null;
  score: number | null;
  notes: string;
}

export interface GridAnalysis {
  experiment_id: string;
  n_scored: number;
  main_effects: Record<string, Array<Record<string, unknown>>>;
  interaction_xy: Array<Array<number | null>> | null;
  best_cell_id: string | null;
  anova: Record<string, number>;
}

export interface GridDetail {
  experiment: GridExperiment;
  cells: GridCell[];
  analysis?: GridAnalysis | null;
}

// ---------------------------------------------------------------------------
// Manga Pipeline
// ---------------------------------------------------------------------------

export interface Point {
  x: number;
  y: number;
}

export interface Polygon {
  points: Point[];
}

export type ShotSize = "ECU" | "CU" | "MS" | "FS" | "ELS";
export type CameraAngle = "eye_level" | "high" | "low" | "dutch";

export interface CameraWork {
  shot: ShotSize;
  angle: CameraAngle;
  lens_mm: number;
  composition: string;
  dutch_degrees: number;
  notes: string;
  horizontal_fov_deg?: number;
}

export interface CharacterProfile {
  id: string;
  role: string;
  name_placeholder: string;
  visual_description: string;
  seed: number | null;
}

export interface CharacterAction {
  character_id: string;
  position: Point;
  scale: number;
  pose: string;
  expression: string;
  gaze: Point;
  action: string;
}

export type BalloonKind = "normal" | "shout" | "monologue" | "flash";

export interface Balloon {
  id: string;
  kind: BalloonKind;
  speaker_id: string | null;
  text: string;
  columns: string[];
  center: Point;
  width: number;
  height: number;
  tail_target: Point | null;
  reading_order: number;
}

export interface SoundEffect {
  id: string;
  text: string;
  position: Point;
  direction: Point;
  scale: number;
  rotation_deg: number;
  style: "impact" | "motion" | "ambient" | "emotion";
}

export interface Panel {
  id: string;
  page_index: number;
  index_in_page: number;
  reading_order: number;
  polygon: Polygon;
  shape: "rect" | "slanted" | "irregular" | "borderless";
  bleed_sides: Array<"top" | "bottom" | "left" | "right">;
  breakout: boolean;
  camera: CameraWork;
  characters: CharacterAction[];
  balloons: Balloon[];
  sfx: SoundEffect[];
  description: string;
  beat_id: string | null;
  secondary_beat_ids?: string[];
  t: number;
  tension: number;
  emphasis: "normal" | "large" | "splash" | "spread";
  is_silent: boolean;
  spread_role?: "none" | "main" | "continuation";
  prompt_hints: string[];
}

export type PageSide = "right" | "left";
export type PageTurnRole = "none" | "hook" | "reveal";

export interface Page {
  index: number;
  number: number;
  side: PageSide;
  panels: Panel[];
  spread_id: string | null;
  page_turn_role: PageTurnRole;
  information_density: number;
  black_ratio_target: number;
  tension_mean: number;
  t_start?: number;
  t_end?: number;
  notes: string[] | string;
}

export interface Spread {
  id: string;
  right_page: number;
  left_page: number;
  kind: "facing" | "double_page_panel";
}

export interface GrammarViolation {
  rule_id: string;
  severity: "error" | "warning" | "info";
  page_index: number | null;
  panel_ids: string[];
  message: string;
  fix_instruction: string;
  penalty: number;
}

export interface GrammarReport {
  score: number;
  passed: boolean;
  violations: GrammarViolation[];
  metrics: Record<string, number>;
  iteration: number;
}

export interface PageGeometry {
  width: number;
  height: number;
  bleed: number;
  margin_top: number;
  margin_bottom: number;
  margin_outer: number;
  margin_inner: number;
  gutter_x: number;
  gutter_y: number;
}

export interface Storyboard {
  id: string;
  title: string;
  structure_id: string;
  theme: Record<string, string>;
  geometry: PageGeometry;
  first_page_side: PageSide;
  pages: Page[];
  spreads: Spread[];
  characters: CharacterProfile[];
  seed: number;
  grammar: GrammarReport | null;
  revision_log: string[];
  created_at: string;
}

export interface StoryboardSummary {
  id: string;
  title: string;
  structure_id: string;
  page_count: number;
  score: number | null;
  passed: boolean | null;
  created_at: string;
}

export interface StoryboardConfig {
  target_pages: number;
  min_panels_per_page: number;
  max_panels_per_page: number;
  first_page_side: PageSide;
  max_chars_per_balloon: number;
  max_chars_per_column: number;
  max_balloons_per_panel: number;
  grammar_threshold: number;
  max_fix_iterations: number;
  seed: number;
  use_llm_dialogue: boolean;
}

export interface RegionMask {
  character_id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  strength: number;
}

export interface CompiledPanelWorkflow {
  panel_id: string;
  page_index: number;
  prompt: Record<string, unknown>;
  seed: number;
  width: number;
  height: number;
  regions: RegionMask[];
  positive_text: string;
  negative_text: string;
  crop?: string;
  filename_prefix?: string;
  warnings?: string[];
}

export interface ComfyCompileResponse {
  workflows: CompiledPanelWorkflow[];
  recipe?: StyleRecipe;
}

// ---------------------------------------------------------------------------
// Asset Publisher
// ---------------------------------------------------------------------------

export type ContentAxis = "creator_theory" | "ai_engineering" | "business_marketing";

export interface Finding {
  claim: string;
  evidence: string;
  metrics: Record<string, number>;
  confidence: number;
}

export interface ResearchLog {
  id?: string;
  kind?: "structure_discovery" | "model_experiment" | "dataset_analysis" | "manga_production" | "mixed";
  title: string;
  summary: string;
  findings: Finding[];
  data?: Record<string, unknown>;
  tags?: string[];
  created_at?: string;
}

export interface OutlineSection {
  heading: string;
  purpose: string;
  key_points: string[];
  figure_idea: string;
}

export interface ContentPlan {
  axis: ContentAxis;
  title: string;
  subtitle: string;
  target_reader: string;
  reader_pain: string;
  core_insight: string;
  angle: string;
  outline: OutlineSection[];
  keywords: string[];
  hook_line: string;
  call_to_action: string;
  estimated_chars: number;
}

export interface ContentMatrix {
  id: string;
  source_id: string;
  source_title: string;
  shared_core: string;
  plans: ContentPlan[];
  cross_links: string[];
  created_at: string;
}

export type BlueprintRole =
  | "conventional_wisdom"
  | "structural_paradox"
  | "hypothesis_evidence"
  | "universal_framework"
  | "practical_action";

export interface FigureSpec {
  kind: "mermaid" | "table" | "none";
  caption: string;
  mermaid_code: string;
}

export interface BlueprintSection {
  role: BlueprintRole;
  heading: string;
  intent: string;
  key_claims: string[];
  evidence: string[];
  figure: FigureSpec;
}

export interface ArticleBlueprint {
  title: string;
  thesis: string;
  target_reader: string;
  axis: ContentAxis;
  sections: BlueprintSection[];
}

export interface DraftSection {
  role: BlueprintRole;
  heading: string;
  body_markdown: string;
  char_count: number;
}

export interface ArticleDraft {
  title: string;
  lead: string;
  sections: DraftSection[];
  closing: string;
}

export interface Replacement {
  original: string;
  replacement: string;
  index: number;
  context: string;
}

export interface RemovedURL {
  kind: string;
  original: string;
  kept_text: string;
}

export interface SanitizationReport {
  first_person_replacements: Replacement[];
  urls_removed: RemovedURL[];
  first_person_unified: boolean;
  url_free: boolean;
  residual_first_person: string[];
  residual_urls: string[];
}

export interface SanitizeResponse {
  text: string;
  report: SanitizationReport;
}

export interface SectionScore {
  role: BlueprintRole;
  heading: string;
  logic_coherence: number;
  abstraction_balance: number;
  persuasiveness: number;
  evidence_density: number;
  total: number;
  feedback: string[];
}

export interface CritiqueScore {
  sections: SectionScore[];
  overall: number;
  passed: boolean;
  threshold: number;
}

export interface PipelineStageRecord {
  stage: string;
  iteration: number;
  duration_ms: number;
  score: number | null;
  notes: string;
}

export interface PublishedArticle {
  id: string;
  title: string;
  axis: ContentAxis;
  source_id: string;
  markdown: string;
  blueprint: ArticleBlueprint;
  draft: ArticleDraft;
  sanitization: SanitizationReport;
  critique_history: CritiqueScore[];
  final_score: number;
  iterations: number;
  stages: PipelineStageRecord[];
  tags: string[];
  created_at: string;
}

export interface ArticleSummary {
  id: string;
  title: string;
  axis: ContentAxis;
  source_id: string;
  final_score: number;
  tags: string[];
  char_count: number;
  created_at: string;
}

export interface ContentGap {
  concepts: string[];
  axis: ContentAxis | null;
  gap_score: number;
  demand: number;
  coverage: number;
  rationale: string;
  suggested_title: string;
  suggested_outline: string[];
}

export interface SeriesChapter {
  index: number;
  title: string;
  article_ids: string[];
  summary: string;
}

export interface SeriesVolume {
  id: string;
  title: string;
  description: string;
  chapters: SeriesChapter[];
  glossary: Record<string, string>;
  created_at: string;
}

export interface SeriesResponse {
  volume: SeriesVolume;
  markdown: string;
}

// ---------------------------------------------------------------------------
// System / Jobs
// ---------------------------------------------------------------------------

export interface Health {
  status: string;
  version?: string;
  llm_provider?: string;
  llm_fallback_active?: boolean;
  llm_model?: string;
  comfy_base_url?: string;
  comfy_mock?: boolean;
  db_path?: string;
  structures?: number;
  seeds_loaded?: number;
}

export type LLMProvider = "auto" | "mock" | "openai_compatible" | "ollama";
export type StructuredOutputMode = "json_schema" | "json_object" | "prompt_only";

export interface HarnessSettings {
  enable_decompose: boolean;
  enable_critique: boolean;
  enable_refine: boolean;
  critique_accept_score: number;
  skip_refine_when_accepted: boolean;
  max_repair_attempts: number;
  max_refine_rounds: number;
  stage_temperatures: Record<string, number>;
}

export interface LLMSettings {
  provider: LLMProvider;
  base_url: string;
  ollama_base_url: string;
  model: string;
  ollama_model: string;
  api_key: string;
  timeout_s: number;
  temperature: number;
  max_tokens: number;
  structured_output: StructuredOutputMode;
  mock_seed: number;
  harness: HarnessSettings;
}

export interface ComfySettings {
  base_url: string;
  timeout_s: number;
  client_id_prefix: string;
  output_prefix: string;
  use_mock: boolean;
}

export interface PublicConfig {
  data_dir: string;
  db_path?: string;
  llm: LLMSettings;
  comfy: ComfySettings;
}

export interface ConnectionTestResult {
  ok: boolean;
  provider?: string;
  model?: string;
  detail?: string;
  models?: string[];
  latency_ms?: number;
  fallback_active?: boolean;
  [key: string]: unknown;
}

export interface DbStats {
  path: string;
  size_bytes: number;
  tables: Record<string, number>;
}

export interface BackupInfo {
  path?: string;
  name?: string;
  filename?: string;
  size_bytes?: number;
  created_at?: string;
}

export interface TelemetryCall {
  id?: number;
  run_id?: string | null;
  task: string;
  stage: string;
  provider?: string | null;
  model?: string | null;
  attempt?: number;
  success: boolean | number;
  error?: string | null;
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
  latency_ms?: number;
  created_at?: string;
}

export interface TelemetryTotals {
  calls: number;
  successes?: number;
  failures?: number;
  success_rate?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
  total_tokens?: number;
  avg_latency_ms?: number;
  max_latency_ms?: number;
}

export interface TelemetryStats {
  totals: TelemetryTotals;
  by_task_stage?: Array<Record<string, unknown>>;
  runs?: Record<string, unknown>;
}

export interface HarnessRunInfo {
  run_id: string;
  task: string;
  objective?: string | null;
  status: string;
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  total_reasoning_tokens?: number;
  total_latency_ms?: number;
  critique_score?: number | null;
  started_at?: string;
  finished_at?: string | null;
}

export interface TelemetryResponse {
  calls: TelemetryCall[];
  stats: TelemetryStats;
  runs: HarnessRunInfo[];
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
  result?: unknown;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface JobRef {
  job_id: string;
}

export interface JobEvent {
  type: "job";
  job_id: string;
  kind: string;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
}
