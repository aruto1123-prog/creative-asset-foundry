"""絵柄と被写体の完全分離（Disentanglement）データセット精製器。

自作 LoRA 学習において「固定して覚えさせたい独自の絵柄（タッチ・空気感）」と
「プロンプトで自在に制御したい要素（構図・カメラアングル・ポーズ・表情・照明・背景密度）」を分離する。

原則:
- 絵柄（style）タグはキャプションから除去し、トリガートークンへ吸収させる（タグが残ると絵柄がタグ側に学習される）。
- 品質・メタ情報（quality / meta）タグはノイズとして除去する。
- 制御要素はカテゴリごとに明示的に残し、欠落カテゴリは要注釈として報告する。

分析:
- 二値出現行列 M（画像×タグ）から共起行列 C = MᵀM、PMI / NPMI、カテゴリ別エントロピーを算出。
- 異カテゴリ間の高 NPMI・高条件付き確率ペアを「概念の癒着（バイアス）」として検出。
- 目標語彙の被覆不足を、追加画像数 A に関する不動点反復 ``a_t = ceil(target·(N+A) − c_t)`` で算出し、
  不足タグ同士と癒着解消を束ねた撮影条件（追加すべき学習画像の条件と枚数）を提案する。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.common import utc_now_iso

__all__ = [
    "TagCategory",
    "CONTROLLABLE_CATEGORIES",
    "DEFAULT_LEXICON",
    "DEFAULT_SYNONYMS",
    "TARGET_VOCABULARY",
    "CaptionRecord",
    "CleansedCaption",
    "CleansedCaptionBatch",
    "TagTaxonomy",
    "Entanglement",
    "TagDeficit",
    "ShotRequirement",
    "CooccurrenceMatrix",
    "DatasetGapReport",
    "DatasetCurator",
]


class TagCategory(str, Enum):
    STYLE = "style"
    COMPOSITION = "composition"
    CAMERA_ANGLE = "camera_angle"
    POSE = "pose"
    EXPRESSION = "expression"
    LIGHTING = "lighting"
    BACKGROUND = "background"
    SUBJECT = "subject"
    QUALITY = "quality"
    META = "meta"
    OTHER = "other"


CONTROLLABLE_CATEGORIES: tuple[TagCategory, ...] = (
    TagCategory.COMPOSITION,
    TagCategory.CAMERA_ANGLE,
    TagCategory.POSE,
    TagCategory.EXPRESSION,
    TagCategory.LIGHTING,
    TagCategory.BACKGROUND,
)

CATEGORY_LABELS_JA: dict[str, str] = {
    TagCategory.STYLE.value: "絵柄",
    TagCategory.COMPOSITION.value: "構図",
    TagCategory.CAMERA_ANGLE.value: "カメラアングル",
    TagCategory.POSE.value: "ポーズ",
    TagCategory.EXPRESSION.value: "表情",
    TagCategory.LIGHTING.value: "照明",
    TagCategory.BACKGROUND.value: "背景",
    TagCategory.SUBJECT.value: "被写体",
    TagCategory.QUALITY.value: "品質",
    TagCategory.META.value: "メタ情報",
    TagCategory.OTHER.value: "その他",
}

DEFAULT_LEXICON: dict[TagCategory, tuple[str, ...]] = {
    TagCategory.COMPOSITION: (
        "close-up", "extreme close-up", "portrait", "upper body", "bust shot", "cowboy shot", "full body",
        "wide shot", "very wide shot", "medium shot", "long shot", "establishing shot", "head out of frame",
        "face focus", "eye focus", "hand focus", "foot focus", "cropped legs", "cropped torso", "centered",
        "rule of thirds", "symmetrical composition", "dynamic composition", "negative space", "split screen",
        "multiple views", "panorama", "framing", "foreshortening", "out of frame", "perspective",
    ),
    TagCategory.CAMERA_ANGLE: (
        "from above", "from below", "from side", "from behind", "eye level", "straight-on", "dutch angle",
        "pov", "over the shoulder", "three-quarter view", "profile", "overhead view", "upside-down",
    ),
    TagCategory.POSE: (
        "standing", "sitting", "kneeling", "on one knee", "lying", "lying on back", "lying on stomach", "on side",
        "crouching", "squatting", "running", "walking", "jumping", "falling", "flying", "floating", "fighting stance",
        "dynamic pose", "action pose", "arms crossed", "hands on hips", "hand on hip", "hand on own chin",
        "hand on own chest", "hands in pockets", "arms up", "arms behind back", "arms behind head", "reaching out",
        "outstretched arm", "pointing", "pointing at viewer", "waving", "peace sign", "clenched fist", "salute",
        "looking back", "looking at viewer", "looking away", "looking down", "looking up", "looking to the side",
        "leaning forward", "leaning back", "hugging", "holding sword", "holding weapon", "holding phone",
        "covering face", "head tilt", "stretching", "dancing", "contrapposto", "back-to-back", "carrying",
    ),
    TagCategory.EXPRESSION: (
        "smile", "grin", "smirk", "gentle smile", "laughing", "crying", "tears", "sad", "angry", "furious",
        "annoyed", "surprised", "shocked", "scared", "frightened", "terrified", "embarrassed", "blush", "serious",
        "expressionless", "determined", "confused", "worried", "nervous", "pout", "happy", "closed eyes",
        "open mouth", "gritted teeth", "wink", "despair", "calm", "sleepy", "disgust", "sneer", "evil smile",
        "panicking", "sweatdrop", "frown", "raised eyebrow", "half-closed eyes", "empty eyes",
    ),
    TagCategory.LIGHTING: (
        "backlighting", "rim light", "dramatic lighting", "soft lighting", "hard lighting", "side lighting",
        "top lighting", "uplighting", "chiaroscuro", "sunlight", "moonlight", "sunset", "sunrise", "golden hour",
        "twilight", "night", "light rays", "spotlight", "silhouette", "lens flare", "neon lights", "candlelight",
        "firelight", "low-key lighting", "high-key lighting", "volumetric lighting", "dappled sunlight",
        "overcast", "cinematic lighting", "glowing", "light particles", "strong shadows",
    ),
    TagCategory.BACKGROUND: (
        "simple background", "white background", "black background", "grey background", "gradient background",
        "transparent background", "two-tone background", "abstract background", "detailed background",
        "blurry background", "depth of field", "bokeh", "outdoors", "indoors", "cityscape", "city", "street",
        "alley", "classroom", "school", "hallway", "bedroom", "room", "office", "library", "cafe", "restaurant",
        "train interior", "station", "forest", "field", "grass", "sky", "cloudy sky", "night sky", "starry sky",
        "ocean", "beach", "river", "mountain", "ruins", "battlefield", "rooftop", "crowd", "scenery", "building",
        "shrine", "temple", "castle", "desert", "snow", "rain", "speed lines", "emphasis lines", "motion lines",
    ),
    TagCategory.STYLE: (
        "monochrome", "greyscale", "lineart", "sketch", "rough sketch", "ink", "inking", "manga", "comic",
        "screentones", "halftone", "crosshatching", "hatching", "thick lineart", "thin lines", "variable line weight",
        "clean lines", "cel shading", "flat color", "watercolor", "oil painting", "painterly", "anime coloring",
        "realistic", "semi-realistic", "chibi", "retro artstyle", "traditional media", "pencil", "graphite",
        "ink wash", "gekiga", "high contrast", "limited palette", "spot color", "black fill", "sumi-e",
        "toon shading", "pixel art", "3d",
    ),
    TagCategory.QUALITY: (
        "masterpiece", "best quality", "high quality", "good quality", "normal quality", "low quality",
        "worst quality", "bad quality", "highres", "absurdres", "incredibly absurdres", "ultra detailed",
        "extremely detailed", "highly detailed", "8k", "4k", "hd", "amazing quality", "very aesthetic",
        "aesthetic", "newest", "lowres", "jpeg artifacts", "bad anatomy", "bad hands", "blurry",
        "score 9", "score 8 up", "score 7 up", "score 6 up", "score 5 up", "score 4 up",
    ),
    TagCategory.META: (
        "signature", "watermark", "artist name", "username", "text", "logo", "copyright name", "dated",
        "english text", "japanese text", "speech bubble", "translated", "commentary request", "twitter username",
        "web address", "patreon username", "sample watermark", "character name", "border", "letterboxed",
        "page number", "scan", "translation request", "source anime", "rating safe", "rating general",
    ),
    TagCategory.SUBJECT: (
        "1girl", "1boy", "2girls", "2boys", "multiple girls", "multiple boys", "solo", "solo focus", "long hair",
        "short hair", "medium hair", "black hair", "brown hair", "blonde hair", "white hair", "silver hair",
        "red hair", "blue hair", "ponytail", "twintails", "braid", "bangs", "blue eyes", "red eyes", "brown eyes",
        "green eyes", "school uniform", "serafuku", "armor", "dress", "skirt", "shirt", "jacket", "coat", "hoodie",
        "glasses", "hat", "gloves", "sword", "gun", "scarf", "bag", "animal ears", "cat ears", "wings", "horns",
        "tail", "hair ornament",
    ),
}

DEFAULT_SYNONYMS: dict[str, str] = {
    # 英語表記揺れ
    "closeup": "close-up", "close up": "close-up", "extreme closeup": "extreme close-up",
    "upperbody": "upper body", "bust": "bust shot", "fullbody": "full body", "full-body": "full body",
    "wideshot": "wide shot", "high angle": "from above", "high-angle": "from above", "high angle shot": "from above",
    "bird's-eye view": "from above", "birds eye view": "from above", "bird's eye view": "from above",
    "low angle": "from below", "low-angle": "from below", "low angle shot": "from below",
    "worm's-eye view": "from below", "worms eye view": "from below", "worm's eye view": "from below",
    "side view": "from side", "back view": "from behind", "rear view": "from behind",
    "tilted angle": "dutch angle", "tilted frame": "dutch angle", "canted angle": "dutch angle",
    "first-person view": "pov", "first person view": "pov",
    "back light": "backlighting", "backlit": "backlighting", "back lit": "backlighting", "back-lit": "backlighting",
    "side light": "side lighting", "rim-light": "rim light", "back lighting": "backlighting", "backlight": "backlighting",
    "rimlight": "rim light", "rim lighting": "rim light", "god rays": "light rays", "sunbeam": "light rays",
    "crying with eyes open": "crying", "streaming tears": "tears", "teary eyes": "tears", "smiling": "smile",
    "laugh": "laughing", "angry face": "angry", "anger": "angry", "surprise": "surprised", "embarassed": "embarrassed",
    "plain background": "simple background", "white bg": "white background", "grayscale": "greyscale",
    "line art": "lineart", "screentone": "screentones", "cross-hatching": "crosshatching", "cross hatching": "crosshatching",
    "speedlines": "speed lines", "focus lines": "emphasis lines",
    # 日本語タグ → 正準英語タグ
    "アップ": "close-up", "顔アップ": "close-up", "超アップ": "extreme close-up", "バストアップ": "upper body",
    "上半身": "upper body", "膝上": "cowboy shot", "全身": "full body", "引き": "wide shot", "ロングショット": "long shot",
    "俯瞰": "from above", "真俯瞰": "from above", "アオリ": "from below", "煽り": "from below", "あおり": "from below",
    "横から": "from side", "背後から": "from behind", "後ろ姿": "from behind", "目線の高さ": "eye level",
    "ダッチアングル": "dutch angle", "主観": "pov", "立ち": "standing", "立ちポーズ": "standing", "座り": "sitting",
    "走る": "running", "走り": "running", "跳躍": "jumping", "ジャンプ": "jumping", "しゃがみ": "crouching",
    "腕組み": "arms crossed", "振り返り": "looking back", "構え": "fighting stance", "寝そべり": "lying",
    "笑顔": "smile", "微笑み": "gentle smile", "泣き顔": "crying", "涙": "tears", "怒り": "angry", "驚き": "surprised",
    "照れ": "embarrassed", "赤面": "blush", "真顔": "serious", "無表情": "expressionless", "絶望": "despair",
    "決意": "determined", "不敵な笑み": "smirk", "逆光": "backlighting", "リムライト": "rim light", "斜光": "side lighting",
    "劇的な照明": "dramatic lighting", "柔らかい光": "soft lighting", "夕焼け": "sunset", "月光": "moonlight", "夜": "night",
    "木漏れ日": "dappled sunlight", "スポットライト": "spotlight", "シルエット": "silhouette", "白背景": "white background",
    "黒背景": "black background", "背景なし": "simple background", "シンプル背景": "simple background", "街": "cityscape",
    "街並み": "cityscape", "教室": "classroom", "森": "forest", "空": "sky", "屋上": "rooftop", "室内": "indoors",
    "屋外": "outdoors", "集中線": "emphasis lines", "スピード線": "speed lines", "モノクロ": "monochrome",
    "白黒": "monochrome", "グレースケール": "greyscale", "線画": "lineart", "ラフ": "rough sketch",
    "スクリーントーン": "screentones", "トーン": "screentones", "網点": "halftone", "ベタ": "black fill", "劇画": "gekiga",
    "水彩": "watercolor", "厚塗り": "painterly", "アニメ塗り": "cel shading", "最高品質": "best quality",
    "高品質": "high quality", "傑作": "masterpiece", "署名": "signature", "サイン": "signature", "透かし": "watermark",
    "ロゴ": "logo", "文字": "text", "吹き出し": "speech bubble",
}

# 理想の漫画絵柄モデルを完成させるために被覆すべき制御要素の目標語彙
TARGET_VOCABULARY: dict[TagCategory, tuple[str, ...]] = {
    TagCategory.COMPOSITION: ("close-up", "upper body", "cowboy shot", "full body", "wide shot"),
    TagCategory.CAMERA_ANGLE: ("from above", "from below", "eye level", "dutch angle", "from side", "from behind"),
    TagCategory.POSE: ("standing", "sitting", "running", "jumping", "crouching", "arms crossed", "looking back", "fighting stance"),
    TagCategory.EXPRESSION: ("smile", "crying", "angry", "surprised", "serious", "embarrassed", "determined", "scared"),
    TagCategory.LIGHTING: ("backlighting", "rim light", "side lighting", "dramatic lighting", "soft lighting", "night"),
    TagCategory.BACKGROUND: ("simple background", "detailed background", "indoors", "outdoors", "cityscape", "speed lines"),
}

_REGEX_RULES: tuple[tuple[re.Pattern[str], TagCategory], ...] = (
    (re.compile(r"^\d+(girl|boy|other)s?$"), TagCategory.SUBJECT),
    (re.compile(r"^score \d+( up)?$"), TagCategory.QUALITY),
    (re.compile(r"(quality|res|detailed)$"), TagCategory.QUALITY),
    (re.compile(r"(watermark|signature|logo|username|text)$"), TagCategory.META),
    (re.compile(r"^from (above|below|side|behind)"), TagCategory.CAMERA_ANGLE),
    (re.compile(r"background$"), TagCategory.BACKGROUND),
    (re.compile(r"(lighting|light)$"), TagCategory.LIGHTING),
    (re.compile(r"shot$"), TagCategory.COMPOSITION),
    (re.compile(r"(hair|eyes|dress|uniform|skirt|shirt|jacket|ribbon|boots|hat|gloves)$"), TagCategory.SUBJECT),
    (re.compile(r"^(holding|hand on|hands on|arms|leaning|looking)"), TagCategory.POSE),
)

_WEIGHT_PATTERN = re.compile(r"^\((.+?):\s*[-+]?\d*\.?\d+\)$")
_SPLIT_PATTERN = re.compile(r"[,、，;；\n]+")


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class CaptionRecord(_Base):
    """データセット1画像分のキャプション。"""

    image_id: str
    caption: str
    path: str = ""


class CleansedCaption(_Base):
    """精製済みキャプション（絵柄タグを除去しトリガーへ吸収、制御要素をカテゴリ別に保持）。"""

    image_id: str
    trigger_token: str
    caption: str
    controllable: dict[str, list[str]] = Field(default_factory=dict)
    subject_tags: list[str] = Field(default_factory=list)
    removed_style_tags: list[str] = Field(default_factory=list)
    removed_noise_tags: list[str] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)


class CleansedCaptionBatch(_Base):
    """LLM 連携クレンジングの出力スキーマ。"""

    captions: list[CleansedCaption]


class TagTaxonomy:
    """タグの正規化と分類（英語 Danbooru 系＋日本語タグ）。"""

    def __init__(
        self,
        lexicon: Mapping[TagCategory, Iterable[str]] | None = None,
        synonyms: Mapping[str, str] | None = None,
        include_defaults: bool = True,
    ) -> None:
        self._category: dict[str, TagCategory] = {}
        self._synonyms: dict[str, str] = {}
        if include_defaults:
            for category, terms in DEFAULT_LEXICON.items():
                for term in terms:
                    self._category[term] = category
            for alias, canonical in DEFAULT_SYNONYMS.items():
                self._synonyms[self._basic(alias)] = canonical
        for category, terms in (lexicon or {}).items():
            for term in terms:
                self._category[self._basic(term)] = TagCategory(category)
        for alias, canonical in (synonyms or {}).items():
            self._synonyms[self._basic(alias)] = self._basic(canonical)
        self._pattern_cache: list[tuple[re.Pattern[str], str]] | None = None

    def __len__(self) -> int:
        return len(self._category)

    @staticmethod
    def _basic(tag: str) -> str:
        text = tag.strip().replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    def normalize(self, tag: str) -> str:
        """小文字化・``_``→空白・括弧重み除去・エスケープ解除・同義語統合。"""
        text = (tag or "").strip()
        if not text:
            return ""
        text = text.replace("\\(", "\x00").replace("\\)", "\x01")
        match = _WEIGHT_PATTERN.match(text)
        if match:
            text = match.group(1)
        while len(text) >= 2 and ((text[0] == "(" and text[-1] == ")") or (text[0] == "[" and text[-1] == "]") or (text[0] == "{" and text[-1] == "}")):
            text = text[1:-1].strip()
        text = text.strip("()[]{} ")
        inner = _WEIGHT_PATTERN.match(f"({text})") if ":" in text else None
        if inner:
            text = inner.group(1)
        text = text.replace("\x00", "(").replace("\x01", ")")
        text = self._basic(text).strip(" .。")
        return self._synonyms.get(text, text)

    def classify(self, tag: str) -> TagCategory:
        normalized = self.normalize(tag)
        if not normalized:
            return TagCategory.OTHER
        category = self._category.get(normalized)
        if category is not None:
            return category
        for pattern, rule_category in _REGEX_RULES:
            if pattern.search(normalized):
                return rule_category
        return TagCategory.OTHER

    def known(self, tag: str) -> bool:
        return self.normalize(tag) in self._category

    def add(self, tag: str, category: TagCategory | str) -> None:
        self._category[self.normalize(tag)] = TagCategory(category)
        self._pattern_cache = None

    def add_synonym(self, alias: str, canonical: str) -> None:
        self._synonyms[self._basic(alias)] = self._basic(canonical)
        self._pattern_cache = None

    def vocabulary(self, category: TagCategory | str) -> list[str]:
        value = TagCategory(category)
        return sorted(term for term, cat in self._category.items() if cat == value)

    def _patterns(self) -> list[tuple[re.Pattern[str], str]]:
        if self._pattern_cache is None:
            candidates = sorted(set(self._category) | set(self._synonyms), key=len, reverse=True)
            compiled: list[tuple[re.Pattern[str], str]] = []
            for term in candidates:
                if len(term) < 2:
                    continue
                if term.isascii():
                    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")
                else:
                    pattern = re.compile(re.escape(term))
                compiled.append((pattern, self._synonyms.get(term, term)))
            self._pattern_cache = compiled
        return self._pattern_cache

    def extract_known_terms(self, text: str) -> list[str]:
        """自然文から既知タグ（同義語含む）を最長一致で抽出する。"""
        consumed = " " + self._basic(text) + " "
        found: list[str] = []
        for pattern, canonical in self._patterns():
            if pattern.search(consumed):
                if canonical not in found:
                    found.append(canonical)
                consumed = pattern.sub(" ", consumed)
        return found


class Entanglement(_Base):
    """異カテゴリ間の概念癒着（例: 泣き顔が常に俯瞰で描かれている）。"""

    tag_a: str
    category_a: str
    tag_b: str
    category_b: str
    support: int
    npmi: float
    conditional_prob: float
    lift: float
    severity: Literal["low", "medium", "high"]


class TagDeficit(_Base):
    """目標語彙の被覆不足。"""

    category: str
    tag: str
    current_count: int
    current_share: float
    target_share: float
    required_additional: int


class ShotRequirement(_Base):
    """追加すべき学習画像の撮影（作画）条件。"""

    conditions: dict[str, str]
    count: int
    reason: str
    priority: float = Field(..., ge=0.0, le=1.0)


class CooccurrenceMatrix(_Base):
    tags: list[str]
    counts: list[list[int]]
    npmi: list[list[float]]


class DatasetGapReport(_Base):
    """データセットのバイアス・ギャップ分析レポート。"""

    total_images: int
    tag_counts: dict[str, int]
    category_distributions: dict[str, dict[str, int]]
    category_entropy: dict[str, float]
    normalized_entropy: dict[str, float]
    cooccurrence: CooccurrenceMatrix
    entanglements: list[Entanglement]
    deficits: list[TagDeficit]
    recommended_shots: list[ShotRequirement]
    total_additional_images: int
    balance_score: float = Field(..., ge=0.0, le=100.0)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


def _entropy(counts: Iterable[int]) -> float:
    values = np.array([c for c in counts if c > 0], dtype=float)
    if values.size == 0:
        return 0.0
    p = values / values.sum()
    return max(0.0, float(-np.sum(p * np.log2(p)))) + 0.0


class DatasetCurator:
    """キャプション精製とデータセット・ギャップ分析。"""

    def __init__(
        self,
        trigger_token: str = "caf_style",
        taxonomy: TagTaxonomy | None = None,
        llm_client: Any | None = None,
        target_min_share: float = 0.12,
        min_count: int = 4,
        target_vocabulary: Mapping[TagCategory | str, Sequence[str]] | None = None,
        max_matrix_tags: int = 40,
        harness_config: Any | None = None,
        max_entanglement_shots: int = 6,
    ) -> None:
        self.trigger_token = trigger_token.strip() or "caf_style"
        self.taxonomy = taxonomy or TagTaxonomy()
        self.llm_client = llm_client
        self.harness_config = harness_config
        self.target_min_share = float(target_min_share)
        self.min_count = max(1, int(min_count))
        self.max_matrix_tags = max(2, int(max_matrix_tags))
        self.max_entanglement_shots = max(0, int(max_entanglement_shots))
        if target_vocabulary is None:
            self.target_vocabulary = {cat: list(terms) for cat, terms in TARGET_VOCABULARY.items()}
        else:
            self.target_vocabulary = {
                TagCategory(cat): [self.taxonomy.normalize(t) for t in terms] for cat, terms in target_vocabulary.items()
            }

    # -- 入力 ----------------------------------------------------------------

    @staticmethod
    def parse_caption_lines(text: str) -> list[CaptionRecord]:
        """1行1画像。``id: tags`` 形式と tags のみ形式（自動採番）の両方を受理する。"""
        records: list[CaptionRecord] = []
        counter = 0
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            counter += 1
            match = re.match(r"^([A-Za-z0-9_\-\.]+)\s*[:：]\s*(.+)$", line)
            if match and re.search(r"\d", match.group(1)) and not re.match(r"^\(.+:\s*[-+]?\d*\.?\d+\)", line):
                records.append(CaptionRecord(image_id=match.group(1), caption=match.group(2).strip()))
            else:
                records.append(CaptionRecord(image_id=f"img_{counter:04d}", caption=line))
        return records

    def _split_tags(self, caption: str) -> list[str]:
        tags: list[str] = []
        for chunk in _SPLIT_PATTERN.split(caption or ""):
            piece = chunk.strip()
            if not piece:
                continue
            normalized = self.taxonomy.normalize(piece)
            if not normalized:
                continue
            if self.taxonomy.known(normalized) or len(normalized.split()) < 4:
                tags.append(normalized)
            else:
                tags.extend(self.taxonomy.extract_known_terms(piece))
        trigger_forms = {self.trigger_token.lower(), self.taxonomy.normalize(self.trigger_token)}
        unique: list[str] = []
        for tag in tags:
            if tag and tag not in unique and tag not in trigger_forms:
                unique.append(tag)
        return unique

    # -- クレンジング --------------------------------------------------------

    def cleanse_record(self, record: CaptionRecord) -> CleansedCaption:
        controllable: dict[str, list[str]] = {cat.value: [] for cat in CONTROLLABLE_CATEGORIES}
        subject: list[str] = []
        removed_style: list[str] = []
        removed_noise: list[str] = []
        for tag in self._split_tags(record.caption):
            category = self.taxonomy.classify(tag)
            if category == TagCategory.STYLE:
                removed_style.append(tag)
            elif category in (TagCategory.QUALITY, TagCategory.META):
                removed_noise.append(tag)
            elif category in CONTROLLABLE_CATEGORIES:
                controllable[category.value].append(tag)
            else:
                subject.append(tag)
        ordered = [self.trigger_token]
        for category in CONTROLLABLE_CATEGORIES:
            ordered.extend(controllable[category.value])
        ordered.extend(subject)
        missing = [cat.value for cat in CONTROLLABLE_CATEGORIES if not controllable[cat.value]]
        return CleansedCaption(
            image_id=record.image_id,
            trigger_token=self.trigger_token,
            caption=", ".join(ordered),
            controllable={key: value for key, value in controllable.items() if value},
            subject_tags=subject,
            removed_style_tags=removed_style,
            removed_noise_tags=removed_noise,
            missing_categories=missing,
        )

    def cleanse(self, records: Sequence[CaptionRecord]) -> list[CleansedCaption]:
        """ルールベース精製（絵柄・品質・メタ除去、トリガー先頭付与、制御要素のカテゴリ整列）。"""
        return [self.cleanse_record(record) for record in records]

    def _normalize_llm_caption(self, item: CleansedCaption, original: CaptionRecord | None) -> CleansedCaption:
        """LLM 出力を分類器で再検証し、絵柄タグ漏れ・トリガー欠落を補正する。"""
        tags: list[str] = []
        for values in item.controllable.values():
            tags.extend(values)
        tags.extend(item.subject_tags)
        if not tags and original is not None:
            return self.cleanse_record(original)
        rebuilt = CaptionRecord(image_id=item.image_id, caption=", ".join(tags))
        result = self.cleanse_record(rebuilt)
        if original is not None:
            baseline = self.cleanse_record(original)
            result.removed_style_tags = sorted(set(result.removed_style_tags) | set(baseline.removed_style_tags) | set(item.removed_style_tags))
            result.removed_noise_tags = sorted(set(result.removed_noise_tags) | set(baseline.removed_noise_tags) | set(item.removed_noise_tags))
        return result

    async def cleanse_with_llm(self, records: Sequence[CaptionRecord]) -> list[CleansedCaption]:
        """DeepSeek 連携クレンジング（4段階ハーネス）。LLM 未設定・失敗時はルール精製へフォールバック。"""
        if self.llm_client is None or not records:
            return self.cleanse(records)
        from core.llm_client import HarnessConfig, LLMError, MultiStageReasoningHarness
        from core.style_alchemy.prompts import (
            CAPTION_CLEANSE_OBJECTIVE,
            CAPTION_CLEANSE_RUBRIC,
            CAPTION_CLEANSE_SYSTEM_PROMPT,
        )

        config = self.harness_config if isinstance(self.harness_config, HarnessConfig) else None
        harness = MultiStageReasoningHarness(self.llm_client, config)
        payload = {
            "trigger_token": self.trigger_token,
            "records": [record.model_dump(mode="json") for record in records],
            "controllable_categories": [cat.value for cat in CONTROLLABLE_CATEGORIES],
        }
        try:
            result = await harness.run(
                task="style.caption_cleanse",
                objective=CAPTION_CLEANSE_OBJECTIVE,
                payload=payload,
                output_model=CleansedCaptionBatch,
                system_prompt=CAPTION_CLEANSE_SYSTEM_PROMPT,
                rubric=CAPTION_CLEANSE_RUBRIC,
                context={"records": payload["records"], "trigger_token": self.trigger_token},
            )
        except LLMError:
            return self.cleanse(records)
        by_id = {record.image_id: record for record in records}
        produced = {item.image_id: item for item in result.output.captions}
        cleansed: list[CleansedCaption] = []
        for record in records:
            item = produced.get(record.image_id)
            cleansed.append(self._normalize_llm_caption(item, by_id[record.image_id]) if item else self.cleanse_record(record))
        return cleansed

    # -- 分析 ----------------------------------------------------------------

    def analyze(self, items: Sequence[CaptionRecord | CleansedCaption]) -> DatasetGapReport:
        """共起・エントロピー・癒着・不足・追加撮影条件を分析する。"""
        cleansed = [item if isinstance(item, CleansedCaption) else self.cleanse_record(item) for item in items]
        total = len(cleansed)
        warnings: list[str] = []
        if total == 0:
            return DatasetGapReport(
                total_images=0, tag_counts={}, category_distributions={}, category_entropy={}, normalized_entropy={},
                cooccurrence=CooccurrenceMatrix(tags=[], counts=[], npmi=[]), entanglements=[], deficits=[],
                recommended_shots=[], total_additional_images=0, balance_score=0.0,
                warnings=["キャプションが1件もありません"],
            )
        if total < 20:
            warnings.append(f"画像数が {total} 枚と少なく、統計的な偏り判定の信頼度が低いです（目安: 20枚以上）")
        tag_category: dict[str, str] = {}
        per_image: list[set[str]] = []
        distributions: dict[str, Counter[str]] = {cat.value: Counter() for cat in CONTROLLABLE_CATEGORIES}
        tag_counts: Counter[str] = Counter()
        for item in cleansed:
            tags: set[str] = set()
            for category, values in item.controllable.items():
                for tag in values:
                    tags.add(tag)
                    tag_category[tag] = category
                    distributions.setdefault(category, Counter())[tag] += 1
            for tag in item.subject_tags:
                tags.add(tag)
                tag_category.setdefault(tag, self.taxonomy.classify(tag).value)
            per_image.append(tags)
            tag_counts.update(tags)
        missing_counter = Counter(cat for item in cleansed for cat in item.missing_categories)
        for category, count in sorted(missing_counter.items()):
            if count == total:
                warnings.append(f"{CATEGORY_LABELS_JA.get(category, category)}のタグが全画像で欠落しています（要注釈）")

        # エントロピー
        entropy: dict[str, float] = {}
        normalized: dict[str, float] = {}
        for category in CONTROLLABLE_CATEGORIES:
            counts = distributions.get(category.value, Counter())
            h = _entropy(counts.values())
            vocab = set(self.target_vocabulary.get(category, [])) | set(counts)
            k = max(2, len(vocab))
            entropy[category.value] = round(h, 6)
            normalized[category.value] = round(min(1.0, h / math.log2(k)), 6)

        # 共起行列
        ranked = sorted(tag_counts, key=lambda t: (-tag_counts[t], t))[: self.max_matrix_tags]
        index = {tag: k for k, tag in enumerate(ranked)}
        matrix = np.zeros((total, len(ranked)), dtype=float)
        for row, tags in enumerate(per_image):
            for tag in tags:
                if tag in index:
                    matrix[row, index[tag]] = 1.0
        counts_matrix = (matrix.T @ matrix).astype(int)
        npmi = np.full(counts_matrix.shape, -1.0)
        marginals = np.diag(counts_matrix).astype(float) / total
        for a in range(len(ranked)):
            for b in range(len(ranked)):
                joint = counts_matrix[a, b] / total
                if a == b:
                    npmi[a, b] = 1.0 if joint > 0 else -1.0
                elif joint > 0:
                    pmi = math.log(joint / (marginals[a] * marginals[b]))
                    denominator = -math.log(joint)
                    npmi[a, b] = 1.0 if denominator <= 1e-12 else max(-1.0, min(1.0, pmi / denominator))
        cooccurrence = CooccurrenceMatrix(
            tags=ranked,
            counts=counts_matrix.tolist(),
            npmi=[[round(float(v), 6) for v in row] for row in npmi],
        )

        # 癒着検出
        controllable_values = {cat.value for cat in CONTROLLABLE_CATEGORIES}
        entanglements: list[Entanglement] = []
        seen_pairs: set[frozenset[str]] = set()
        for a, tag_a in enumerate(ranked):
            for b, tag_b in enumerate(ranked):
                if a == b:
                    continue
                cat_a, cat_b = tag_category.get(tag_a, "other"), tag_category.get(tag_b, "other")
                if cat_a == cat_b or (cat_a not in controllable_values and cat_b not in controllable_values):
                    continue
                pair = frozenset((tag_a, tag_b))
                if pair in seen_pairs:
                    continue
                support = int(counts_matrix[a, b])
                if support < self.min_count:
                    continue
                cond_ab = support / max(1, counts_matrix[a, a])
                cond_ba = support / max(1, counts_matrix[b, b])
                value = float(npmi[a, b])
                if value < 0.5 or max(cond_ab, cond_ba) < 0.8:
                    continue
                seen_pairs.add(pair)
                if cond_ab >= cond_ba:
                    first, second, first_cat, second_cat, cond = tag_a, tag_b, cat_a, cat_b, cond_ab
                else:
                    first, second, first_cat, second_cat, cond = tag_b, tag_a, cat_b, cat_a, cond_ba
                if value >= 0.8 or cond >= 0.95:
                    severity: Literal["low", "medium", "high"] = "high"
                elif value >= 0.65 or cond >= 0.9:
                    severity = "medium"
                else:
                    severity = "low"
                lift = (support / total) / (marginals[a] * marginals[b]) if marginals[a] * marginals[b] > 0 else 0.0
                entanglements.append(
                    Entanglement(
                        tag_a=first, category_a=first_cat, tag_b=second, category_b=second_cat, support=support,
                        npmi=round(value, 6), conditional_prob=round(cond, 6), lift=round(float(lift), 6), severity=severity,
                    )
                )
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        entanglements.sort(
            key=lambda e: (
                severity_rank[e.severity],
                0 if (e.category_a in controllable_values and e.category_b in controllable_values) else 1,
                -e.npmi,
                -e.support,
                e.tag_a,
                e.tag_b,
            )
        )

        # 不足の不動点反復
        targets: dict[tuple[str, str], float] = {}
        for category in CONTROLLABLE_CATEGORIES:
            vocab = self.target_vocabulary.get(category, [])
            if not vocab:
                continue
            share = min(self.target_min_share, 1.0 / len(vocab))
            for tag in vocab:
                targets[(category.value, tag)] = share

        def _required(added_total: int) -> dict[tuple[str, str], int]:
            needed: dict[tuple[str, str], int] = {}
            for (category, tag), share in targets.items():
                current = distributions.get(category, Counter()).get(tag, 0)
                needed[(category, tag)] = max(0, math.ceil(share * (total + added_total) - current - 1e-9))
            return needed

        added = 0
        required = _required(added)
        for _ in range(100):
            per_category: dict[str, int] = defaultdict(int)
            for (category, _tag), amount in required.items():
                per_category[category] += amount
            new_added = max(per_category.values(), default=0)
            if new_added == added:
                break
            added = new_added
            required = _required(added)
        deficits: list[TagDeficit] = []
        for (category, tag), share in sorted(targets.items()):
            current = distributions.get(category, Counter()).get(tag, 0)
            amount = required[(category, tag)]
            if amount > 0:
                deficits.append(
                    TagDeficit(
                        category=category, tag=tag, current_count=current, current_share=round(current / total, 6),
                        target_share=round(share, 6), required_additional=amount,
                    )
                )
        deficits.sort(key=lambda d: (-d.required_additional, d.category, d.tag))

        shots = self._recommend_shots(deficits, entanglements, distributions, total)
        penalty_weights = {"high": 0.15, "medium": 0.08, "low": 0.04}
        penalty = min(0.6, sum(penalty_weights[e.severity] for e in entanglements))
        mean_entropy = float(np.mean([normalized[cat.value] for cat in CONTROLLABLE_CATEGORIES]))
        balance = max(0.0, min(100.0, 100.0 * mean_entropy * (1.0 - penalty)))
        return DatasetGapReport(
            total_images=total,
            tag_counts=dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
            category_distributions={cat: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))) for cat, counter in distributions.items()},
            category_entropy=entropy,
            normalized_entropy=normalized,
            cooccurrence=cooccurrence,
            entanglements=entanglements,
            deficits=deficits,
            recommended_shots=shots,
            total_additional_images=int(added),
            balance_score=round(balance, 3),
            warnings=warnings,
        )

    def _recommend_shots(
        self,
        deficits: Sequence[TagDeficit],
        entanglements: Sequence[Entanglement],
        distributions: Mapping[str, Counter[str]],
        total: int,
    ) -> list[ShotRequirement]:
        """不足タグを異カテゴリ間で束ね、癒着ペアを避けた撮影条件を貪欲に構成する。"""
        remaining: dict[str, dict[str, int]] = defaultdict(dict)
        for deficit in deficits:
            remaining[deficit.category][deficit.tag] = deficit.required_additional
        entangled = {frozenset((e.tag_a, e.tag_b)) for e in entanglements}
        shots: dict[tuple[tuple[str, str], ...], ShotRequirement] = {}
        max_need = max((d.required_additional for d in deficits), default=1)

        def _add(conditions: dict[str, str], count: int, reason: str, priority: float) -> None:
            key = tuple(sorted(conditions.items()))
            if key in shots:
                existing = shots[key]
                shots[key] = existing.model_copy(update={"count": existing.count + count, "priority": max(existing.priority, priority)})
            else:
                shots[key] = ShotRequirement(conditions=dict(conditions), count=count, reason=reason, priority=round(min(1.0, priority), 4))

        for _ in range(200):
            active = {cat: {tag: n for tag, n in tags.items() if n > 0} for cat, tags in remaining.items()}
            active = {cat: tags for cat, tags in active.items() if tags}
            if not active:
                break
            conditions: dict[str, str] = {}
            for category in sorted(active, key=lambda c: -max(active[c].values())):
                for tag in sorted(active[category], key=lambda t: (-active[category][t], t)):
                    if any(frozenset((tag, chosen)) in entangled for chosen in conditions.values()):
                        continue
                    conditions[category] = tag
                    break
            if not conditions:
                break
            count = max(1, min(active[cat][tag] for cat, tag in conditions.items()))
            for category, tag in conditions.items():
                remaining[category][tag] -= count
            labels = "・".join(f"{CATEGORY_LABELS_JA.get(cat, cat)}「{tag}」" for cat, tag in conditions.items())
            need = max(active[cat][tag] for cat, tag in conditions.items())
            _add(conditions, count, f"被覆不足の解消: {labels} を同時に満たす画像を追加", 0.4 + 0.6 * need / max(1, max_need))

        for item in list(entanglements)[: self.max_entanglement_shots]:
            cat_b = item.category_b
            counter_a = distributions.get(item.category_a, Counter())
            current_a = counter_a.get(item.tag_a, 0)
            needed = max(1, math.ceil(item.support / 0.6 - current_a))
            vocab_b = [t for t in self.target_vocabulary.get(TagCategory(cat_b), []) if t != item.tag_b] if cat_b in {c.value for c in CONTROLLABLE_CATEGORIES} else []
            counter_b = distributions.get(cat_b, Counter())
            if vocab_b:
                alternatives = sorted(vocab_b, key=lambda t: (counter_b.get(t, 0), t))[:2]
            else:
                alternatives = [f"{item.tag_b} 以外"]
            per_alt = max(1, math.ceil(needed / len(alternatives)))
            severity_bonus = {"high": 0.35, "medium": 0.2, "low": 0.1}[item.severity]
            for alternative in alternatives:
                _add(
                    {item.category_a: item.tag_a, cat_b: alternative},
                    per_alt,
                    f"癒着の解消: 「{item.tag_a}」が「{item.tag_b}」と同時に出現する確率 {item.conditional_prob:.0%} を60%未満へ下げる",
                    0.55 + severity_bonus,
                )
        return sorted(shots.values(), key=lambda s: (-s.priority, -s.count, sorted(s.conditions.items())))
