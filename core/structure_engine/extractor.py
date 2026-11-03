"""脱フレーバー抽出器（De-flavoring Structure Extractor）。

生テキスト（あらすじ・ネーム・記事・LP原稿）から固有名詞とジャンル特有の装飾を除去し、
5階層の普遍的構造（StructureLayers / UniversalStructure）へ逆コンパイルする。

処理の流れ:
1. ``deflavor_text``: 固有名詞（カタカナ人名・敬称付き人名・主要姓＋名・「」『』内の名称・地名/組織接尾辞・
   英語の大文字始まり語）を検出し、出現順に ``[人物A]`` ``[場所A]`` ``[組織A]`` 等へ置換。
   多語・中黒区切りの名前は構成要素を別名として同一プレースホルダへ束ねる。
   さらにジャンル装飾語（魔法→特殊能力 等）を機能的な抽象語へ置換する。
2. ``heuristic_extract``（決定論的ベースライン）: 文分割 → 文ごとの正規化時刻 → 語彙特徴 →
   ガウスカーネル平滑化（帯域 0.08）で21点波形 → 波形の極値に一致させたビート → 力学グラフ →
   メカニズム → ブループリント → 転用ガイド。
3. ``StructureExtractor.extract``: LLM がある場合は4段階ハーネス（Decompose→Synthesize→Critique→Refine）で増幅し、
   最終結果を固有名詞・プレースホルダ・装飾語で再走査して漏洩を自動置換する。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from core.common import truncate
from core.llm_client import HarnessConfig, HarnessError, LLMClient, MultiStageReasoningHarness
from core.structure_engine.consistency import audit_structure
from core.structure_engine.prompts import EXTRACTION_OBJECTIVE, EXTRACTION_RUBRIC, EXTRACTION_SYSTEM_PROMPT
from core.structure_engine.schemas import (
    WAVE_MIN_POINTS,
    BeatType,
    EmotionalInformationWave,
    SourceKind,
    StructureCategory,
    StructureLayers,
    UniversalStructure,
    new_structure_id,
)

__all__ = [
    "FlavorCategory",
    "FlavorToken",
    "DeflavorResult",
    "ExtractionResult",
    "COMMON_KATAKANA",
    "GENRE_TERMS",
    "GENRE_LEXICON",
    "deflavor_text",
    "split_sentences",
    "heuristic_extract",
    "find_flavor_leaks",
    "scrub_flavor",
    "StructureExtractor",
]

logger = logging.getLogger(__name__)

FlavorCategory = Literal["person", "place", "organization", "artifact", "genre_term", "title", "other"]

_PLACEHOLDER_LABELS: dict[str, str] = {
    "person": "人物",
    "place": "場所",
    "organization": "組織",
    "artifact": "固有物",
    "title": "作品",
    "other": "名称",
}
_GENERIC_WORDS: dict[str, str] = {
    "person": "ある人物",
    "place": "ある場所",
    "organization": "ある組織",
    "artifact": "ある固有物",
    "title": "ある作品",
    "other": "ある存在",
}
PLACEHOLDER_RE = re.compile(r"\[(人物|場所|組織|固有物|作品|名称)([A-Z]{1,2}\d*)\]")

# ---------------------------------------------------------------------------
# 辞書
# ---------------------------------------------------------------------------

COMMON_KATAKANA: frozenset[str] = frozenset(
    """
    システム データ ライバル チーム ルール エネルギー パワー ゲーム スポーツ ビジネス マーケティング サービス ユーザー
    プロジェクト プラン イベント パーティー メンバー リーダー ヒーロー ヒロイン キャラクター ストーリー シーン テーマ
    パターン リズム バランス スピード タイミング チャンス ピンチ ミス ミッション ターゲット コスト リスク メリット
    デメリット プロセス ステップ レベル スキル トレーニング クラス グループ クラブ サークル ホテル レストラン カフェ
    ショップ ビル タワー ドア ベッド テーブル ノート ペン バッグ ケース ボール カード コイン キー ナイフ ガラス ライト
    エンジン ガス オイル ウイルス ワクチン ホラー ミステリー ファンタジー ロマンス コメディ アクション バトル ボス
    パートナー ファン アイドル アーティスト デザイン アイデア イメージ スタイル ポイント ページ トップ ライン ルート
    ゴール スタート ラスト エンド ゲート マップ エリア ゾーン フロア ルーム ホーム クラスメイト ライブ ステージ ショー
    コンテスト レース マッチ トーナメント リーグ チャンピオン ヒント トリック パズル サイン シグナル ノイズ プログラム
    コード エンジニア ドクター ナース マネージャー プロデューサー ディレクター スタッフ オーナー ボランティア ゲスト
    クライアント カスタマー ブランド プロダクト オファー セールス キャンペーン コンテンツ メディア ソーシャル コミュニティ
    ネットワーク モデル フレームワーク パラダイム パラドックス ロジック エビデンス リサーチ テスト データベース
    アルゴリズム ツール アプリ ソフト ハード ミサイル ビーム レーザー ロケット ジェット エレベーター トンネル ビジョン
    メッセージ ニュース テレビ ラジオ カメラ ドラマ メール インターネット ネット スマホ コンピューター パソコン マシン
    エピソード シリーズ プロローグ エピローグ クライマックス ドラマチック ハッピーエンド バッドエンド トラウマ
    コンプレックス プライド ストレス プレッシャー モチベーション コミュニケーション トラブル アクシデント ニュアンス
    タイプ ケア サポート ヘルプ アドバイス フィードバック レビュー ランキング スコア ゲージ メーター グラフ チャート
    リスト メモ サンプル ケーススタディ ワークショップ セミナー レッスン トレーナー コーチ キャプテン マスター ライフ
    ワーク スペース スタジオ オフィス キャンパス ステーション ターミナル ホール ロビー ガレージ アパート マンション
    ストア マーケット モール センター ファーム ジム プール ビーチ パーク ケーキ コーヒー ミルク パン メンター
    ヴィラン ヒール ネーム コマ セリフ フキダシ オノマトペ トーン ベタ ペース ジャンル プロット シナリオ ナレーション
    モノローグ カット アングル ショット フレーム レイアウト ブランディング ファネル コピー ヘッドライン ベネフィット
    ペルソナ インサイト ニーズ ウォンツ プレゼン レポート グラフィック ビデオ ウェビナー ダウンロード ログイン
    アカウント パスワード セキュリティ プライバシー ポリシー ライセンス バージョン アップデート リリース ベータ
    トリガー ループ フロー パイプライン パラメータ ベクトル マトリクス ノード エッジ スロット テンプレート
    ブロック レイヤー マージ ウェイト ランダム シード バッチ グリッド カテゴリ タグ ラベル キャプション
    """.split()
)

# ジャンル装飾語 → 機能的抽象語（最長一致優先で置換）
GENRE_TERMS: dict[str, str] = {
    "王国騎士団": "統治体の武装組織",
    "魔法学園": "育成機関",
    "宮廷魔術師": "権力中枢の特殊技能者",
    "魔法使い": "特殊技能者",
    "魔術師": "特殊技能者",
    "魔導士": "特殊技能者",
    "錬金術師": "変成技術の専門家",
    "騎士団": "武装組織",
    "地下書庫": "秘匿された記録庫",
    "古文書": "記録",
    "宇宙船": "移動手段",
    "宇宙戦艦": "大型の移動手段",
    "アンドロイド": "人工的存在",
    "サイボーグ": "改造された存在",
    "ロボット": "人工的存在",
    "ドラゴン": "強大な脅威",
    "モンスター": "脅威",
    "ゾンビ": "脅威",
    "ゴースト": "超常的存在",
    "ハッカー": "侵入者",
    "ギルド": "職能組織",
    "吸血鬼": "捕食的存在",
    "魔王": "支配的存在",
    "勇者": "選ばれた主体",
    "聖剣": "特別な手段",
    "魔法": "特殊能力",
    "魔術": "特殊能力",
    "魔力": "固有の力",
    "禁術": "禁じられた手段",
    "結界": "障壁",
    "呪い": "呪縛",
    "精霊": "超常的存在",
    "神殿": "聖域",
    "王国": "統治体",
    "王都": "統治の中心地",
    "帝国": "統治体",
    "宮廷": "権力中枢",
    "孤児院": "保護施設",
    "学園": "共同体",
    "騎士": "武装した守護者",
    "宇宙": "広大な外部",
    "惑星": "遠隔の領域",
    "怪物": "脅威",
    "忍者": "隠密",
    "探偵": "調査者",
    "警察": "治安組織",
    "殺人": "重大な加害",
    "企業": "組織",
    "会社": "組織",
    "部活": "活動集団",
    "決闘": "直接対決",
    "試合": "勝負",
    "剣": "武器",
    "刀": "武器",
    "銃": "武器",
    "竜": "強大な脅威",
    "王": "最高権力者",
    "姫": "保護対象の高位者",
    "magic": "special ability",
    "wizard": "specialist",
    "sword": "weapon",
    "kingdom": "polity",
    "empire": "polity",
    "spaceship": "vessel",
    "robot": "artificial being",
    "dragon": "great threat",
    "corporation": "organization",
}

GENRE_LEXICON: dict[str, tuple[str, ...]] = {
    "fantasy": ("魔法", "魔術", "魔力", "剣", "王国", "王都", "騎士", "ドラゴン", "勇者", "魔王", "結界", "禁術", "精霊", "宮廷", "古文書", "magic", "wizard", "kingdom", "dragon", "sword"),
    "sf": ("宇宙", "惑星", "ロボット", "アンドロイド", "人工知能", "サイボーグ", "宇宙船", "未来", "研究所", "spaceship", "robot", "planet", "android", "cyber", "engineer", "grid", "power"),
    "school": ("学園", "学校", "高校", "生徒", "教師", "部活", "放課後", "入学", "school", "student", "class"),
    "mystery": ("事件", "犯人", "探偵", "推理", "殺人", "証拠", "密室", "容疑", "真相", "detective", "murder", "clue", "culprit"),
    "business": ("会社", "企業", "社長", "売上", "顧客", "商品", "市場", "起業", "corporation", "company", "market", "customer", "career", "startup"),
    "romance": ("恋", "愛", "告白", "恋人", "結婚", "デート", "love", "romance", "kiss"),
    "horror": ("恐怖", "怪異", "呪い", "幽霊", "ゾンビ", "化け物", "惨劇", "horror", "ghost", "curse", "zombie"),
    "sports": ("試合", "優勝", "選手", "大会", "練習", "甲子園", "match", "championship", "tournament", "team"),
    "history": ("戦国", "幕府", "武将", "江戸", "王朝", "革命", "戦争", "皇帝", "empire", "dynasty", "war"),
}

_HONORIFICS = "さん|くん|君|ちゃん|様|さま|殿|先生|氏|先輩|教授|博士|社長|部長|隊長|団長"
_HONORIFIC_EXCLUDE = frozenset(
    {"皆", "誰", "王", "姫", "神", "母", "父", "兄", "姉", "弟", "妹", "子", "客", "奥", "お客", "皆様", "貴方", "あなた", "先", "後", "俺", "僕", "私"}
)
_SURNAMES = (
    "佐藤 鈴木 高橋 田中 伊藤 渡辺 山本 中村 小林 加藤 吉田 山田 佐々木 山口 松本 井上 木村 斎藤 清水 山崎 池田 橋本 "
    "阿部 石川 山下 中島 石井 小川 前田 岡田 長谷川 藤田 後藤 近藤 村上 遠藤 青木 坂本 斉藤 福田 太田 西村 藤井 金子 "
    "岡本 藤原 中野 三浦 原田 中川 松田 竹内 小野 田村 中山 和田 石田 森田 上田 内田 柴田 酒井 宮崎 横山 高木 安藤 "
    "宮本 大野 小島 谷口 今井 工藤 高田 増田 丸山 杉山 村田 大塚 新井 小山 平野 藤本 河野 上野 野口 武田 松井 千葉 "
    "岩崎 菅原 木下 久保 佐野 野村 松尾 市川 菊地 杉本 古川 大西 島田 水野 桜井 高野 渡部 吉川 山内 西田 飯田 菊池 "
    "西川 小松 北村 安田 五十嵐 川口 平田 中田 久保田 服部 岩田 土屋 川崎 福島 本田 樋口 秋山 田口 永井 山中 中西 "
    "吉村 川上 大橋 石原 松岡 浜田 馬場 森本 矢野 浅野 松下 星野 大久保 吉岡 小池 野田 荒木 大谷 内藤 松浦 熊谷 黒田 "
    "尾崎 永田 川村 望月 田辺 松村 荒井 大石 中尾 藤川 高山"
).split()
_PLACE_SUFFIXES = (
    "共和国|王国|帝国|大陸|惑星|地方|学園|高校|中学|大学|学院|神殿|遺跡|通り|村|町|市|区|県|府|州|国|島|山|川|湖|海|港|城|都|街|谷|森|砦|塔|駅|領|星"
)
_ORG_SUFFIXES = "株式会社|騎士団|研究所|財団|商会|教団|協会|連合|同盟|機関|部隊|一族|財閥|王家|工房|銀行|商店|ギルド|社|団|隊|党|組|会|軍|家"
_GENERIC_PREFIXES = frozenset(
    {"魔法", "王立", "国立", "私立", "県立", "市立", "第一", "第二", "地下", "秘密", "古代", "聖", "新", "旧", "大", "小", "中央", "故", "本", "全", "各", "同", "自", "他", "某", "東", "西", "南", "北"}
)
_PLACE_PRECEDERS = ("王都", "都市", "首都", "惑星", "大陸", "地方", "王国", "帝国", "国家", "街", "島")
_EN_COMMON = frozenset(
    """
    the a an in on at to of for from by with without within into onto over under after before when while during
    but and or nor so yet if then than as because although though since until unless whether
    she he they it we you i her his their its our my your them him us me
    this that these those there here where who whom whose which what why how
    following betrayed hunted with help in end finally later meanwhile suddenly eventually soon
    one two three first second last next new old young great small big
    dr mr mrs ms prof captain king queen lord lady sir saint
    is are was were be been being have has had do does did will would can could should may might must
    """.split()
)
_EN_PERSON_TITLES = frozenset({"Dr", "Mr", "Mrs", "Ms", "Prof", "Captain", "King", "Queen", "Lord", "Lady", "Sir", "Saint"})
_EN_ORG_SUFFIXES = frozenset(
    {"Corporation", "Corp", "Inc", "Company", "Co", "Group", "Labs", "Lab", "Institute", "Agency", "Guild", "Order", "Academy", "University", "School", "Society", "Industries", "Foundation", "Council", "Empire", "Kingdom"}
)
_EN_PLACE_CUES = re.compile(r"(?:city|town|village|kingdom|planet|country|island|land|realm)\s+of\s+$|(?:\bin|\bat|\bfrom|\bto)\s+$", re.IGNORECASE)

_KATAKANA_RE = re.compile(r"[ァ-ヺー・]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*|(?<=[.])\s+(?=[A-Z\[「『（(])|\n+")
_EN_CAP_SEQ_RE = re.compile(r"\b[A-Z][a-z]+(?:[ \-][A-Z][a-z]+)*\b")

# ---------------------------------------------------------------------------
# 語彙特徴（和英）
# ---------------------------------------------------------------------------

_LEX_TENSION = (
    "戦", "危機", "死", "殺", "失", "奪", "追", "襲", "迫", "脅", "絶望", "恐怖", "対立", "罠", "崩", "焼", "破", "怒",
    "叫", "逃", "窮地", "限界", "壁", "失敗", "犠牲", "代償", "痛", "傷", "闘", "争", "衝突", "爆", "炎", "攻撃", "危険",
    "不安", "焦", "打ち破", "倒", "障壁", "直接対決", "勝負", "重大な加害", "禁じられた手段", "脅威",
    "crisis", "death", "kill", "fight", "war", "betray", "threat", "lose", "lost", "hunt", "attack", "danger", "fear",
    "despair", "fail", "sacrific", "destroy", "burn", "chase", "trap", "expose", "tear",
)
_LEX_RELEASE = (
    "解決", "勝利", "和解", "救", "笑", "安堵", "成功", "達成", "平和", "希望", "癒", "再会", "回復", "喜", "感謝", "自由",
    "新しい", "築", "守", "助け", "光", "頭角", "勝",
    "win", "victory", "peace", "hope", "rescue", "save", "smile", "relief", "success", "free", "build", "heal", "reunite", "rise",
)
_LEX_REVEAL = (
    "判明", "明か", "真実", "真相", "気づ", "知る", "発見", "暴", "告白", "正体", "開示", "分か", "わか", "悟", "見抜",
    "discover", "reveal", "truth", "learn", "realiz", "expose", "uncover", "confess",
)
_LEX_CONCEAL = (
    "秘密", "隠", "偽", "嘘", "謎", "知らな", "欺", "騙", "仮面", "装", "秘匿",
    "hide", "hidden", "secret", "lie", "mask", "conceal", "deceiv", "unknown", "mystery",
)
_LEX_DESIRE = (
    "目指", "夢", "取り戻", "願", "求", "望", "目的", "目標", "ため", "誓", "叶", "奪還",
    "want", "dream", "seek", "hoping", "goal", "rebuild", "reclaim", "restore",
)
_LEX_LACK = (
    "失った", "奪われ", "孤児", "欠", "足りな", "喪失", "亡", "孤独", "故郷を", "家族を", "保護施設",
    "lost", "orphan", "missing", "lack", "grief", "killed",
)
_LEX_OPPOSITION = ("対立", "敵", "ライバル", "戦", "争", "反抗", "直接対決", "追われ", "抗", "打ち破", "rival", "enemy", "against", "oppose", "fight", "hunted")
_LEX_BETRAYAL = ("裏切", "欺", "騙", "寝返", "背信", "betray", "deceiv", "double-cross")
_LEX_DEPENDENCY = ("師", "導", "頼", "信じ", "助け", "仲間", "依存", "教え", "庇護", "mentor", "trust", "rely", "help", "guide", "teach")
_LEX_POWER = ("最高権力者", "統治", "支配", "権力", "上司", "命令", "権力中枢", "組織", "武装組織", "control", "power", "rule", "command", "boss", "seize")
_LEX_UNRESOLVED = ("続く", "未解決", "謎のまま", "残された", "to be continued", "unresolved", "remains a mystery")
_LEX_RECOVERY = ("取り戻", "奪還", "回復", "rebuild", "reclaim", "restore")
_LEX_TOKEN = ("形見", "鍵", "証", "印", "指輪", "記録", "地図", "手紙", "key", "ring", "locket", "letter", "archive", "map")


def _count_terms(sentence: str, terms: Sequence[str]) -> int:
    lowered = sentence.lower()
    total = 0
    for term in terms:
        if term.isascii():
            total += len(re.findall(r"\b" + re.escape(term.lower()), lowered))
        else:
            total += sentence.count(term)
    return total


# ---------------------------------------------------------------------------
# スキーマ
# ---------------------------------------------------------------------------


class FlavorToken(BaseModel):
    """除去されたフレーバー（固有名詞・ジャンル装飾）1件。"""

    surface: str
    category: FlavorCategory
    placeholder: str
    occurrences: int = 0


class DeflavorResult(BaseModel):
    """脱フレーバー結果。"""

    abstracted_text: str
    tokens: list[FlavorToken] = Field(default_factory=list)
    genre_signals: dict[str, float] = Field(default_factory=dict)

    def proper_noun_tokens(self) -> list[FlavorToken]:
        return [token for token in self.tokens if token.category != "genre_term"]

    def placeholders(self) -> list[dict[str, str]]:
        seen: list[dict[str, str]] = []
        for token in self.proper_noun_tokens():
            entry = {"placeholder": token.placeholder, "category": token.category}
            if entry not in seen:
                seen.append(entry)
        return seen


class ExtractionResult(BaseModel):
    """構造抽出の結果。"""

    structure: UniversalStructure
    deflavor: DeflavorResult
    harness_run_id: str | None = None
    used_llm: bool = False
    leaked_tokens_fixed: list[str] = Field(default_factory=list)
    critique_score: float | None = None


# ---------------------------------------------------------------------------
# 固有名詞検出
# ---------------------------------------------------------------------------


class _Candidate:
    __slots__ = ("start", "end", "surface", "category")

    def __init__(self, start: int, end: int, surface: str, category: str) -> None:
        self.start = start
        self.end = end
        self.surface = surface
        self.category = category


def _is_common_katakana(word: str) -> bool:
    stripped = word.strip("・ー")
    if stripped in COMMON_KATAKANA or stripped in GENRE_TERMS:
        return True
    parts = [p for p in stripped.split("・") if p]
    return bool(parts) and all(p in COMMON_KATAKANA or p in GENRE_TERMS for p in parts)


def _detect_brackets(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    for match in re.finditer(r"『([^『』\n]{1,20})』", text):
        inner = match.group(1).strip()
        if inner:
            found.append(_Candidate(match.start(1), match.start(1) + len(inner), inner, "title"))
    for match in re.finditer(r"「([^「」\n]{1,12})」", text):
        inner = match.group(1).strip()
        if inner and not re.search(r"[ぁ-ゖ、。！？!?]", inner):
            found.append(_Candidate(match.start(1), match.start(1) + len(inner), inner, "organization"))
    return found


def _detect_katakana(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    place_suffix_re = re.compile(rf"^(?:{_PLACE_SUFFIXES})")
    org_suffix_re = re.compile(rf"^(?:{_ORG_SUFFIXES})")
    for match in _KATAKANA_RE.finditer(text):
        word = match.group(0).strip("・")
        start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip("・")))
        if len(word.strip("ー")) < 2 or _is_common_katakana(word):
            continue
        end = start + len(word)
        following = text[end : end + 4]
        preceding = text[max(0, start - 3) : start]
        org = org_suffix_re.match(following)
        place = place_suffix_re.match(following)
        if org and not following.startswith("会話"):
            found.append(_Candidate(start, end + len(org.group(0)), word + org.group(0), "organization"))
        elif place:
            found.append(_Candidate(start, end + len(place.group(0)), word + place.group(0), "place"))
        elif any(preceding.endswith(cue) for cue in _PLACE_PRECEDERS):
            found.append(_Candidate(start, end, word, "place"))
        else:
            found.append(_Candidate(start, end, word, "person"))
    return found


def _detect_honorific(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    for match in re.finditer(rf"([一-鿿々]{{1,4}})(?:{_HONORIFICS})", text):
        name = match.group(1)
        if name in _HONORIFIC_EXCLUDE or name in GENRE_TERMS:
            continue
        found.append(_Candidate(match.start(1), match.end(1), name, "person"))
    return found


def _detect_surnames(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    pattern = re.compile(
        "(" + "|".join(sorted(map(re.escape, _SURNAMES), key=len, reverse=True)) + ")"
        r"([ァ-ヺー]{2,6}|[一-鿿]{1,2}(?=[はがをにのともへやで、。\s]))?"
    )
    for match in pattern.finditer(text):
        surname, given = match.group(1), match.group(2) or ""
        before = text[match.start() - 1] if match.start() > 0 else ""
        if before and re.match(r"[一-鿿]", before):
            continue
        found.append(_Candidate(match.start(), match.start() + len(surname) + len(given), surname + given, "person"))
    return found


def _detect_kanji_suffix(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    pattern = re.compile(rf"([一-鿿]{{2,4}})({_PLACE_SUFFIXES}|{_ORG_SUFFIXES})(?![一-鿿])")
    for match in pattern.finditer(text):
        prefix, suffix = match.group(1), match.group(2)
        full = prefix + suffix
        if prefix in _GENERIC_PREFIXES or full in GENRE_TERMS or prefix in GENRE_TERMS:
            continue
        if any(full.endswith(term) and len(term) >= 2 for term in GENRE_TERMS if not term.isascii()):
            continue
        category = "organization" if re.fullmatch(_ORG_SUFFIXES, suffix) else "place"
        if suffix in ("家", "会", "組", "社", "団", "隊", "党", "軍") and len(prefix) < 2:
            continue
        found.append(_Candidate(match.start(), match.end(), full, category))
    return found


def _detect_english(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    sentence_starts = {0}
    for match in re.finditer(r"[.!?。！？]\s+|\n+", text):
        sentence_starts.add(match.end())
    sequences: list[tuple[int, int, str, bool]] = []
    for match in _EN_CAP_SEQ_RE.finditer(text):
        words = match.group(0)
        start = match.start()
        at_start = start in sentence_starts or (start > 0 and text[:start].rstrip() == "")
        sequences.append((start, match.end(), words, at_start))
    mid_tokens: set[str] = set()
    for start, end, words, at_start in sequences:
        parts = re.split(r"[ \-]", words)
        if not at_start:
            mid_tokens.update(p for p in parts if p.lower() not in _EN_COMMON)
    for start, end, words, at_start in sequences:
        parts = re.split(r"[ \-]", words)
        while parts and parts[0].lower() in _EN_COMMON and (at_start or parts[0] in _EN_PERSON_TITLES):
            start += len(parts[0]) + 1
            parts = parts[1:]
            at_start = False
        if not parts:
            continue
        if parts[-1] in _EN_PERSON_TITLES:
            continue
        surface = " ".join(parts)
        if all(p.lower() in _EN_COMMON for p in parts):
            continue
        if at_start and len(parts) == 1 and parts[0] not in mid_tokens:
            continue
        preceding = text[max(0, start - 22) : start]
        title_match = re.search(r"\b(Dr|Mr|Mrs|Ms|Prof|Captain|King|Queen|Lord|Lady|Sir)\.?\s+$", preceding)
        if parts[-1] in _EN_ORG_SUFFIXES and len(parts) >= 2:
            category = "organization"
        elif title_match:
            category = "person"
        elif _EN_PLACE_CUES.search(preceding) and len(parts) >= 2:
            category = "place"
        else:
            category = "person"
        found.append(_Candidate(start, start + len(surface), surface, category))
    return found


def _select_non_overlapping(candidates: list[_Candidate]) -> list[_Candidate]:
    ordered = sorted(candidates, key=lambda c: (c.start, -(c.end - c.start)))
    chosen: list[_Candidate] = []
    last_end = -1
    for candidate in ordered:
        if candidate.start >= last_end:
            chosen.append(candidate)
            last_end = candidate.end
    return chosen


def _letter(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(letters):
        return letters[index]
    return f"{letters[index % len(letters)]}{index // len(letters) + 1}"


def _alias_parts(surface: str) -> list[str]:
    if "・" in surface:
        parts = [p for p in surface.split("・") if len(p.strip("ー")) >= 2]
    elif " " in surface:
        parts = [p for p in surface.split(" ") if len(p) >= 2]
    else:
        return []
    return [p for p in parts if p not in COMMON_KATAKANA and p not in _EN_ORG_SUFFIXES and p.lower() not in _EN_COMMON]


def _replace_surfaces(text: str, mapping: dict[str, str]) -> tuple[str, dict[str, int]]:
    """表層語を最長一致で一括置換し、出現数を返す（ASCII 語は単語境界付き）。"""
    counts = {surface: 0 for surface in mapping}
    if not mapping:
        return text, counts
    parts: list[str] = []
    for surface in sorted(mapping, key=len, reverse=True):
        escaped = re.escape(surface)
        parts.append(rf"\b{escaped}\b" if surface.isascii() else escaped)
    pattern = re.compile("|".join(parts))

    def _sub(match: re.Match[str]) -> str:
        surface = match.group(0)
        counts[surface] = counts.get(surface, 0) + 1
        return mapping[surface]

    return pattern.sub(_sub, text), counts


def _genre_pattern() -> re.Pattern[str]:
    parts: list[str] = []
    for term in sorted(GENRE_TERMS, key=len, reverse=True):
        escaped = re.escape(term)
        if term.isascii():
            parts.append(rf"\b{escaped}s?\b")
        elif len(term) == 1:
            parts.append(rf"(?<![一-鿿]){escaped}(?![一-鿿])")
        else:
            parts.append(escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


_GENRE_RE = _genre_pattern()
_GENRE_LOOKUP = {term.lower(): value for term, value in GENRE_TERMS.items()}


def _genre_signals(text: str) -> dict[str, float]:
    counts = {genre: float(_count_terms(text, terms)) for genre, terms in GENRE_LEXICON.items()}
    total = sum(counts.values())
    if total <= 0:
        return {genre: 0.0 for genre in counts}
    return {genre: round(value / total, 4) for genre, value in counts.items()}


def deflavor_text(text: str) -> DeflavorResult:
    """固有名詞とジャンル装飾を除去した抽象テキストと、除去トークン一覧を返す。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    if not normalized.strip():
        return DeflavorResult(abstracted_text=normalized, tokens=[], genre_signals=_genre_signals(normalized))
    candidates = (
        _detect_brackets(normalized)
        + _detect_katakana(normalized)
        + _detect_honorific(normalized)
        + _detect_surnames(normalized)
        + _detect_kanji_suffix(normalized)
        + _detect_english(normalized)
    )
    chosen = _select_non_overlapping(candidates)
    registry: dict[str, tuple[str, str]] = {}
    aliases: dict[str, str] = {}
    counters: dict[str, int] = {}
    ordered_surfaces: list[str] = []
    for candidate in chosen:
        surface = candidate.surface
        if surface in registry or surface in aliases:
            continue
        parts = _alias_parts(surface)
        reused = next((registry[aliases[p]][1] for p in parts if p in aliases), None)
        reused = reused or next((registry[p][1] for p in parts if p in registry), None)
        if reused is None:
            category = candidate.category
            label = _PLACEHOLDER_LABELS.get(category, _PLACEHOLDER_LABELS["other"])
            index = counters.get(label, 0)
            counters[label] = index + 1
            placeholder = f"[{label}{_letter(index)}]"
        else:
            placeholder = reused
            category = next(cat for cat, ph in registry.values() if ph == placeholder)
        registry[surface] = (category, placeholder)
        ordered_surfaces.append(surface)
        for part in parts:
            if part not in registry and part not in aliases:
                aliases[part] = surface
    mapping: dict[str, str] = {surface: registry[surface][1] for surface in registry}
    for alias, canonical in aliases.items():
        mapping.setdefault(alias, registry[canonical][1])
    replaced, counts = _replace_surfaces(normalized, mapping)
    tokens: list[FlavorToken] = []
    for surface in ordered_surfaces:
        category, placeholder = registry[surface]
        tokens.append(
            FlavorToken(surface=surface, category=category, placeholder=placeholder, occurrences=counts.get(surface, 0))  # type: ignore[arg-type]
        )
    for alias, canonical in aliases.items():
        category, placeholder = registry[canonical]
        tokens.append(
            FlavorToken(surface=alias, category=category, placeholder=placeholder, occurrences=counts.get(alias, 0))  # type: ignore[arg-type]
        )
    genre_counts: dict[str, int] = {}

    def _genre_sub(match: re.Match[str]) -> str:
        matched = match.group(0)
        key = matched.lower()
        if key not in _GENRE_LOOKUP and key.endswith("s"):
            key = key[:-1]
        replacement = _GENRE_LOOKUP.get(key, matched)
        canonical = next((term for term in GENRE_TERMS if term.lower() == key), matched)
        genre_counts[canonical] = genre_counts.get(canonical, 0) + 1
        return replacement

    abstracted = _GENRE_RE.sub(_genre_sub, replaced)
    for term, count in genre_counts.items():
        tokens.append(FlavorToken(surface=term, category="genre_term", placeholder=GENRE_TERMS[term], occurrences=count))
    return DeflavorResult(abstracted_text=abstracted, tokens=tokens, genre_signals=_genre_signals(normalized))


# ---------------------------------------------------------------------------
# ヒューリスティック逆コンパイル
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """和英混在テキストの文分割（句点・感嘆符・疑問符・ピリオド・改行）。"""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _kernel_smooth(times: np.ndarray, values: np.ndarray, grid: np.ndarray, bandwidth: float = 0.08) -> np.ndarray:
    weights = np.exp(-0.5 * ((grid[:, None] - times[None, :]) / bandwidth) ** 2)
    denominator = weights.sum(axis=1)
    denominator[denominator <= 1e-12] = 1.0
    return (weights * values[None, :]).sum(axis=1) / denominator


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _grid_index(grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(grid - t)))


def _ordered_times(values: list[float], gap: float = 0.02, upper: float = 0.99) -> list[float]:
    ordered: list[float] = []
    for value in values:
        candidate = max(value, (ordered[-1] + gap) if ordered else value)
        ordered.append(min(candidate, upper))
    for index in range(len(ordered) - 2, -1, -1):
        if ordered[index] > ordered[index + 1] - gap:
            ordered[index] = max(0.0, ordered[index + 1] - gap)
    return [round(v, 4) for v in ordered]


_SLOT_EXAMPLES: dict[str, dict[str, str]] = {
    "PROTAGONIST": {"manga": "居場所を失った若者", "marketing": "成果が頭打ちの担当者", "article": "定説を信じて努力してきた実践者"},
    "CORE_LACK": {"manga": "帰る場所", "marketing": "伸び悩む成果", "article": "説明できない失敗"},
    "PROTAGONIST_OBSESSIVE_GOAL": {"manga": "失われた場所の奪還", "marketing": "目標数値の達成", "article": "失敗の原因究明"},
    "ANTAGONIST_FORCE": {"manga": "道を阻む強大な対立者", "marketing": "業界の慣習", "article": "根強い定説"},
    "FALSE_MENTOR": {"manga": "育ての親を装う黒幕", "marketing": "常識とされる成功法則", "article": "広く信じられた権威の教え"},
    "STRUCTURAL_IRONY": {"manga": "恩人こそが仇だった", "marketing": "定石の施策こそが成長を止めていた", "article": "定説が失敗を再生産していた"},
    "SYMBOLIC_TOKEN": {"manga": "形見の品", "marketing": "顧客の生の声", "article": "見過ごされていたデータ"},
    "BREAKTHROUGH_COST": {"manga": "安住の居場所", "marketing": "慣れた手法の放棄", "article": "自説の撤回"},
    "ALLY": {"manga": "かつての競争相手", "marketing": "同じ課題を持つ仲間", "article": "異分野の協力者"},
}


def heuristic_extract(
    text: str,
    *,
    source_kind: str = "synopsis",
    title_hint: str | None = None,
    genre_signals: dict[str, float] | None = None,
) -> StructureLayers:
    """LLM なしで動作する決定論的な5階層構造抽出（任意の非空テキストで有効な StructureLayers を返す）。

    genre_signals を与えると（抽象化前テキストのジャンル傾向）、由来傾向タグに用いる。
    """
    if not text or not text.strip():
        raise ValueError("heuristic_extract: 空のテキストからは構造を抽出できません")
    deflavor = deflavor_text(text)
    abstracted = deflavor.abstracted_text
    sentences = split_sentences(abstracted) or [abstracted]
    total_chars = max(1, sum(len(s) for s in sentences))
    times_list: list[float] = []
    cursor = 0
    for sentence in sentences:
        times_list.append((cursor + len(sentence) / 2.0) / total_chars)
        cursor += len(sentence)
    times = np.asarray(times_list, dtype=float)

    def feature(terms: Sequence[str]) -> np.ndarray:
        return np.asarray([float(_count_terms(s, terms)) for s in sentences], dtype=float)

    tension_f = feature(_LEX_TENSION)
    release_f = feature(_LEX_RELEASE)
    reveal_f = feature(_LEX_REVEAL)
    conceal_f = feature(_LEX_CONCEAL)
    opposition_f = feature(_LEX_OPPOSITION)
    betrayal_f = feature(_LEX_BETRAYAL)
    dependency_f = feature(_LEX_DEPENDENCY)
    power_f = feature(_LEX_POWER)
    betrayal_total = float(betrayal_f.sum())
    conceal_total = float(conceal_f.sum())
    unresolved = _count_terms(abstracted, _LEX_UNRESOLVED) > 0
    recovery = _count_terms(abstracted, _LEX_RECOVERY) > 0
    has_lack_words = _count_terms(abstracted, _LEX_LACK) > 0
    has_token_words = _count_terms(abstracted, _LEX_TOKEN) > 0

    seen_placeholders: set[str] = set()
    new_entities = []
    lengths = []
    sentence_people: list[list[str]] = []
    for sentence in sentences:
        found = [m.group(0) for m in PLACEHOLDER_RE.finditer(sentence)]
        new_entities.append(float(len([p for p in dict.fromkeys(found) if p not in seen_placeholders])))
        seen_placeholders.update(found)
        lengths.append(float(len(sentence)))
        sentence_people.append([p for p in dict.fromkeys(found) if p.startswith("[人物")])
    new_entities_arr = np.asarray(new_entities)
    lengths_arr = np.asarray(lengths)

    grid = np.linspace(0.0, 1.0, WAVE_MIN_POINTS)
    prior = 0.22 + 0.62 * grid**1.6
    prior = np.where(grid > 0.9, prior * (1.0 - (grid - 0.9) / 0.1 * 0.6), prior)
    raw_tension = _kernel_smooth(times, tension_f + 1.5 * betrayal_f + 0.7 * opposition_f - 0.6 * release_f, grid)
    tension = 0.5 * _normalize(raw_tension) + 0.5 * prior
    late_max = float(np.max(tension[grid >= 0.6]))
    tension = np.where(grid < 0.6, np.minimum(tension, late_max * 0.95), tension)
    tension[-1] = min(float(tension[-1]), 0.32)
    tension[-2] = min(float(tension[-2]), (float(tension[-3]) + float(tension[-1])) / 2.0 + 0.1)
    tension = np.clip(tension, 0.05, 1.0)

    raw_release = _normalize(_kernel_smooth(times, release_f, grid))
    drops = np.concatenate([[0.0], np.clip(tension[:-1] - tension[1:], 0.0, None)])
    release = np.clip(0.08 + 0.45 * raw_release + 1.8 * drops, 0.0, 1.0)
    release[-1] = max(float(release[-1]), 0.75)
    release[-2] = max(float(release[-2]), 0.5)

    reveal_weights = np.exp(-0.5 * ((grid[:, None] - times[None, :]) / 0.08) ** 2) @ (reveal_f + 0.15)
    cumulative = np.cumsum(reveal_weights)
    audience = 0.1 + 0.9 * cumulative / max(float(cumulative[-1]), 1e-9)
    audience = np.maximum.accumulate(np.clip(audience, 0.0, 1.0))
    conceal_level = _normalize(_kernel_smooth(times, conceal_f, grid)) if conceal_total > 0 else np.zeros_like(grid)
    character = np.clip(audience + 0.35 * conceal_level * (1.0 - audience) + 0.06 * (1.0 - audience), 0.0, 1.0)
    character[-1] = audience[-1]

    load_raw = _kernel_smooth(times, lengths_arr / max(float(lengths_arr.max()), 1.0) * 0.6 + new_entities_arr * 0.25, grid)
    cognitive = np.clip(0.25 + 0.55 * _normalize(load_raw) + 0.1 * prior, 0.0, 1.0)

    def _round(values: np.ndarray) -> list[float]:
        return [round(float(v), 4) for v in np.clip(values, 0.0, 1.0)]

    wave_payload = {
        "t": [round(float(v), 6) for v in grid],
        "tension": _round(tension),
        "release": _round(release),
        "audience_knowledge": _round(audience),
        "character_knowledge": _round(character),
        "cognitive_load": _round(cognitive),
    }
    tension_r = np.asarray(wave_payload["tension"], dtype=float)
    audience_r = np.asarray(wave_payload["audience_knowledge"], dtype=float)

    t_hook = round(min(0.05, max(0.02, float(times[0]) * 0.5)), 4)
    t_premise = 0.15
    window = [i for i, g in enumerate(grid) if 0.35 <= g <= 0.65]
    limit_idx = max(window, key=lambda i: (tension_r[i], -i))
    t_limit = float(grid[limit_idx])
    jumps = np.diff(audience_r)
    twist_candidates = [i for i in range(len(jumps)) if t_limit + 0.05 <= grid[i + 1] <= 0.8]
    if twist_candidates:
        twist_idx = max(twist_candidates, key=lambda i: (jumps[i], -i)) + 1
    else:
        twist_idx = _grid_index(grid, min(0.8, t_limit + 0.1))
    t_twist = float(grid[twist_idx])
    break_window = [i for i, g in enumerate(grid) if t_twist + 0.08 <= g <= 0.95]
    break_idx = max(break_window, key=lambda i: (tension_r[i], -i)) if break_window else _grid_index(grid, min(0.95, t_twist + 0.1))
    # 代償を伴う突破を緊張度の大域最大へ整形する（波形とビートの極値を一致させる）
    global_max = float(np.max(tension_r))
    if tension_r[break_idx] < global_max + 0.04:
        tension_r[break_idx] = min(1.0, global_max + 0.05)
        if break_idx - 1 > twist_idx:
            tension_r[break_idx - 1] = max(tension_r[break_idx - 1], 0.85 * tension_r[break_idx])
    wave_payload["tension"] = [round(float(v), 4) for v in np.clip(tension_r, 0.0, 1.0)]
    t_break = float(grid[break_idx])
    crisis_window = [i for i in range(twist_idx + 1, break_idx) if 0 < i < len(grid)]
    if crisis_window:
        crisis_idx = min(crisis_window, key=lambda i: (tension_r[i], i))
        t_crisis = float(grid[crisis_idx])
    else:
        t_crisis = round((t_twist + t_break) / 2.0, 4)
    t_escalation = round((t_premise + t_limit) / 2.0, 4)
    times_ordered = _ordered_times([t_hook, t_premise, t_escalation, t_limit, t_twist, t_crisis, t_break, 0.97])
    t_hook, t_premise, t_escalation, t_limit, t_twist, t_crisis, t_break, t_after = times_ordered

    people_counts: dict[str, int] = {}
    for people in sentence_people:
        for person in people:
            people_counts[person] = people_counts.get(person, 0) + 1
    first_seen: dict[str, int] = {}
    for position, person in enumerate(p for people in sentence_people for p in people):
        first_seen.setdefault(person, position)
    ranked_people = sorted(people_counts, key=lambda p: (-people_counts[p], first_seen.get(p, 0)))
    others = ranked_people[1:]
    opp_score = {p: 0.0 for p in others}
    dep_score = {p: 0.0 for p in others}
    for index, people in enumerate(sentence_people):
        for person in people:
            if person in opp_score:
                opp_score[person] += opposition_f[index] + betrayal_f[index] + 0.5 * tension_f[index]
                dep_score[person] += dependency_f[index]
    antagonist_ph: str | None = None
    if others:
        if betrayal_total > 0 and max(dep_score.values()) > 0:
            antagonist_ph = max(others, key=lambda p: (dep_score[p], opp_score[p]))
        else:
            antagonist_ph = max(others, key=lambda p: (opp_score[p], people_counts[p]))
    ally_candidates = [p for p in others if p != antagonist_ph]
    has_ally = bool(ally_candidates)
    is_betrayal = betrayal_total > 0
    has_order = any(t.category == "organization" for t in deflavor.tokens) or float(power_f.sum()) >= 2.0
    opposition_strength = min(1.0, 0.6 + 0.08 * float(opposition_f.sum()))

    antagonist_slot = "FALSE_MENTOR" if is_betrayal else "ANTAGONIST_FORCE"
    goal_label = "失われたものの奪還" if recovery else "根源的な目標の達成"
    lack_label = "奪われた拠り所" if has_lack_words else "埋められない欠落"
    nodes: list[dict[str, Any]] = [
        {"id": "protagonist", "kind": "agent", "role": "protagonist", "label": "欠落を抱え目標を追う主体",
         "description": "物語開始以前の欠落を動機として行動し、前提とする方法を疑わずに前進する。", "weight": 0.95, "slot": "PROTAGONIST"},
        {"id": "antagonist", "kind": "agent", "role": "false_mentor" if is_betrayal else "antagonist",
         "label": "導き手を装う対立者" if is_betrayal else "主体と対立する力",
         "description": "主体を導く立場にありながら欠落の真因を隠している。" if is_betrayal else "主体の目標を阻み、構造上の圧力を体現する。",
         "weight": 0.85, "slot": antagonist_slot},
        {"id": "goal", "kind": "desire", "role": "obsessive_goal", "label": goal_label,
         "description": "主体の行動を駆動し続ける執着的な目標。", "weight": 0.9, "slot": "PROTAGONIST_OBSESSIVE_GOAL"},
        {"id": "core_lack", "kind": "lack", "role": "core_lack", "label": lack_label,
         "description": "物語以前に生じ、真因が伏せられている欠落。", "weight": 0.8, "slot": "CORE_LACK"},
        {"id": "token", "kind": "symbol", "role": "symbolic_token",
         "label": "受け継いだ象徴物" if has_token_words else "変化を映す象徴",
         "description": "欠落の真因を示し、結末で新しい意味を帯びる象徴。", "weight": 0.55, "slot": "SYMBOLIC_TOKEN"},
    ]
    if has_ally:
        nodes.append({"id": "ally", "kind": "agent", "role": "ally", "label": "最後の一歩を支える協力者",
                      "description": "当初は距離のある存在だが、危機の後に主体を支える。", "weight": 0.6, "slot": "ALLY"})
    if has_order:
        nodes.append({"id": "order", "kind": "agent", "role": "antagonistic_order", "label": "既存の秩序",
                      "description": "対立者と結びつき、主体の自由な行動を制約する仕組み。", "weight": 0.5, "slot": None})
    edges: list[dict[str, Any]] = [
        {"source": "protagonist", "target": "goal", "kind": "pursuit", "intensity": 0.9, "gradient": 0.0, "label": "執着的追求", "activation_t": None},
        {"source": "token", "target": "core_lack", "kind": "embodiment", "intensity": 0.6, "gradient": 0.0, "label": "欠落の痕跡", "activation_t": t_twist},
        {"source": "core_lack", "target": "protagonist", "kind": "power_gradient", "intensity": 0.6, "gradient": 0.5, "label": "欠落が主体を駆動する圧力", "activation_t": None},
        {"source": "protagonist", "target": "antagonist", "kind": "opposition", "intensity": round(opposition_strength, 4), "gradient": 0.0, "label": "目標をめぐる対立", "activation_t": t_limit},
        {"source": "antagonist", "target": "protagonist", "kind": "power_gradient", "intensity": 0.7, "gradient": 0.6, "label": "構造上の優位", "activation_t": None},
    ]
    if is_betrayal:
        edges.extend([
            {"source": "protagonist", "target": "antagonist", "kind": "dependency", "intensity": 0.8, "gradient": -0.4, "label": "導きへの依存", "activation_t": 0.1},
            {"source": "antagonist", "target": "protagonist", "kind": "betrayal", "intensity": 0.85, "gradient": 0.0, "label": "前提という罠", "activation_t": t_twist},
            {"source": "antagonist", "target": "protagonist", "kind": "asymmetry", "intensity": 0.8, "gradient": 0.75, "label": "真因の秘匿", "activation_t": 0.05},
        ])
    elif conceal_total > 0:
        edges.append({"source": "antagonist", "target": "protagonist", "kind": "asymmetry", "intensity": 0.7, "gradient": 0.6, "label": "情報の秘匿", "activation_t": 0.1})
    if has_ally:
        edges.extend([
            {"source": "protagonist", "target": "ally", "kind": "dependency", "intensity": 0.55, "gradient": 0.0, "label": "危機後の協力", "activation_t": t_crisis},
            {"source": "ally", "target": "antagonist", "kind": "opposition", "intensity": 0.5, "gradient": 0.0, "label": "共闘による対抗", "activation_t": t_break},
        ])
    if has_order:
        edges.append({"source": "protagonist", "target": "order", "kind": "opposition", "intensity": 0.55, "gradient": 0.0, "label": "秩序への反抗", "activation_t": t_break})
        edges.append({"source": "antagonist", "target": "order", "kind": "dependency", "intensity": 0.5, "gradient": 0.0, "label": "秩序との共犯関係", "activation_t": None})

    twist_type = BeatType.STRUCTURAL_TWIST.value if is_betrayal or conceal_total > 0 else BeatType.REVELATION.value
    breakthrough_nodes = ["protagonist", "antagonist", "token"] + (["ally"] if has_ally else [])
    beats: list[dict[str, Any]] = [
        {"id": "hook", "beat_type": "hook", "t": t_hook, "span": 0.05, "label": "日常に走る亀裂",
         "function": "欠落や異変の断片を提示し、受け手に最初の問いを立てる。", "causes": [], "involved_nodes": ["protagonist", "core_lack"], "is_turning_point": False},
        {"id": "premise", "beat_type": "premise", "t": t_premise, "span": 0.08, "label": "行動原理の提示",
         "function": "主体が何を目指し、どの方法を正しいと信じているか（前提）を共有する。", "causes": ["hook"], "involved_nodes": ["protagonist", "antagonist", "goal"], "is_turning_point": False},
        {"id": "escalation", "beat_type": "escalation", "t": t_escalation, "span": 0.1, "label": "前提に沿った前進",
         "function": "前提に従った行動が成果を生み、依存と期待が深まる。", "causes": ["premise"], "involved_nodes": ["protagonist", "goal"], "is_turning_point": False},
        {"id": "limit", "beat_type": "local_optimization_limit", "t": t_limit, "span": 0.06, "label": "局所的最適化の限界",
         "function": "同じ方法の反復では越えられない壁が現れ、前提への違和感が生じる。", "causes": ["escalation"], "involved_nodes": ["protagonist", "antagonist"], "is_turning_point": True},
        {"id": "twist", "beat_type": twist_type, "t": t_twist, "span": 0.05,
         "label": "前提の反転" if twist_type == BeatType.STRUCTURAL_TWIST.value else "核心の開示",
         "function": ("信じていた前提そのものが欠落の原因だったと判明し、関係が反転する。"
                      if twist_type == BeatType.STRUCTURAL_TWIST.value else "隠されていた情報が一挙に開示され、状況の意味が書き換わる。"),
         "causes": ["limit", "premise"], "involved_nodes": ["protagonist", "antagonist", "token"], "is_turning_point": True},
        {"id": "crisis", "beat_type": "crisis", "t": t_crisis, "span": 0.05, "label": "拠り所の喪失",
         "function": "反転の結果、主体は拠り所を失い、自力の判断を迫られる。", "causes": ["twist"], "involved_nodes": ["protagonist", "core_lack"], "is_turning_point": False},
        {"id": "breakthrough", "beat_type": "costly_breakthrough", "t": t_break, "span": 0.06, "label": "代償を伴う突破",
         "function": "最も大切なものを差し出すことで、対立の構造そのものを越える。", "causes": ["crisis", "twist"], "involved_nodes": breakthrough_nodes, "is_turning_point": True},
        {"id": "afterglow", "beat_type": "afterglow", "t": t_after, "span": 0.03,
         "label": "開かれた結末" if unresolved else "変容の余韻",
         "function": ("答えの一部を伏せたまま幕を下ろし、受け手に問いを持ち帰らせる。"
                      if unresolved else "回復ではなく変容としての結末を示し、欠落の意味を書き換える。"),
         "causes": ["breakthrough"], "involved_nodes": ["protagonist", "core_lack"], "is_turning_point": False},
    ]
    wave_preview = EmotionalInformationWave.model_validate(wave_payload)
    for beat in beats:
        h = max(0.06, float(beat["span"]))
        lo, hi = max(0.0, beat["t"] - h), min(1.0, beat["t"] + h)
        beat["tension_delta"] = round(max(-1.0, min(1.0, wave_preview.sample("tension", hi) - wave_preview.sample("tension", lo))), 4)
        beat["information_shift"] = round(
            max(-1.0, min(1.0, wave_preview.sample("information_asymmetry", hi) - wave_preview.sample("information_asymmetry", lo))), 4
        )
    stats = wave_preview.summary()
    mechanisms: list[dict[str, Any]] = [
        {"id": "empathy", "category": "empathy_transfer", "name": "欠落への共感移入",
         "description": "受け手自身の喪失や停滞の経験が主体の欠落に重ねられ、選択を自分事として体験させる。",
         "trigger": "具体的で普遍的な欠落の提示", "effect": "主体の判断への強い感情的関与", "theory": "物語移入理論",
         "strength": 0.7, "linked_beats": ["premise", "crisis"], "linked_nodes": ["protagonist", "core_lack"]},
        {"id": "prediction_error", "category": "prediction_error", "name": "前提の崩壊による予測誤差",
         "description": "受け手が共有していた前提が覆り、それまでの出来事の意味が再解釈される。",
         "trigger": "反転点での核心情報の開示", "effect": "驚愕と再解釈による没入の深化", "theory": "予測符号化理論",
         "strength": round(min(1.0, 0.55 + float(stats["surprise"]) * 0.6), 4), "linked_beats": ["twist"], "linked_nodes": ["antagonist"]},
    ]
    if float(stats["suspense"]) >= 0.05:
        mechanisms.append({"id": "irony", "category": "dramatic_irony", "name": "読者だけが知る危機",
                           "description": "受け手が作中人物より多くを知ることで、平穏な場面にも緊張が生まれる。",
                           "trigger": "人物より先行する情報開示", "effect": "持続的なサスペンス", "theory": "劇的アイロニー",
                           "strength": round(min(1.0, 0.4 + float(stats["suspense"]) * 2.0), 4), "linked_beats": ["escalation"], "linked_nodes": ["protagonist"]})
    if float(stats["catharsis"]) >= 0.35:
        mechanisms.append({"id": "catharsis", "category": "catharsis", "name": "代償の後の解放",
                           "description": "緊張の頂点で代償が支払われ、溜め込まれた緊張が一挙に解放される。",
                           "trigger": "突破直後の緊張度の急落と解放度の急上昇", "effect": "浄化と深い満足感", "theory": "アリストテレスのカタルシス",
                           "strength": round(min(1.0, 0.4 + float(stats["catharsis"]) * 0.5), 4), "linked_beats": ["breakthrough", "afterglow"], "linked_nodes": ["protagonist"]})
    if unresolved:
        mechanisms.append({"id": "zeigarnik", "category": "zeigarnik", "name": "未完了の問い",
                           "description": "解かれない問いが記憶に残り続け、続きを求める動機になる。",
                           "trigger": "結末で一部の答えを伏せる", "effect": "読後の反芻と継続的関心", "theory": "ツァイガルニク効果",
                           "strength": 0.6, "linked_beats": ["afterglow"], "linked_nodes": ["core_lack"]})
    if conceal_total > 0 or len(mechanisms) < 3:
        mechanisms.append({"id": "curiosity", "category": "curiosity_gap", "name": "欠落の真因への問い",
                           "description": "冒頭の断片が答えのない問いを残し、読み進める動機を生む。",
                           "trigger": "原因を伏せた欠落の提示", "effect": "真因を知りたいという持続的な欲求", "theory": "情報ギャップ理論",
                           "strength": 0.65, "linked_beats": ["hook"], "linked_nodes": ["core_lack"]})
    if len(mechanisms) < 3:
        mechanisms.append({"id": "peak_end", "category": "peak_end", "name": "代償の記憶化",
                           "description": "最大の代償の瞬間と結末の余韻が、体験全体の評価を決定づける。",
                           "trigger": "頂点直後の静かな結末", "effect": "長い余韻と再読欲求", "theory": "ピーク・エンドの法則",
                           "strength": 0.55, "linked_beats": ["breakthrough", "afterglow"], "linked_nodes": ["protagonist"]})
    mechanisms = mechanisms[:5]

    slots: list[dict[str, Any]] = [
        {"name": "PROTAGONIST", "description": "欠落を抱え目標を追う主体", "category": "agent"},
        {"name": "CORE_LACK", "description": "物語以前に生じ、真因が伏せられた欠落", "category": "lack"},
        {"name": "PROTAGONIST_OBSESSIVE_GOAL", "description": "主体を駆動する執着的目標", "category": "goal"},
        {"name": antagonist_slot,
         "description": "導き手を装い欠落の真因を隠す存在" if is_betrayal else "主体と対立し構造上の圧力を体現する存在",
         "category": "agent"},
        {"name": "STRUCTURAL_IRONY", "description": "前提そのものが欠落の原因だったという構造的皮肉", "category": "irony"},
        {"name": "SYMBOLIC_TOKEN", "description": "真因を示し結末で意味が変わる象徴", "category": "symbol"},
        {"name": "BREAKTHROUGH_COST", "description": "突破のために差し出す代償", "category": "cost"},
    ]
    if has_ally:
        slots.append({"name": "ALLY", "description": "危機の後に主体を支える協力者", "category": "agent"})
    for slot in slots:
        slot["examples"] = dict(_SLOT_EXAMPLES[slot["name"]])
        slot["constraints"] = []
        slot["required"] = True
    template = (
        "{{PROTAGONIST}} は {{CORE_LACK}} を抱え、{{PROTAGONIST_OBSESSIVE_GOAL}} を追う。"
        f"{{{{{antagonist_slot}}}}} の圧力の下で前進するが、同じ方法の限界に突き当たる。"
        "やがて {{STRUCTURAL_IRONY}} が明らかになり前提は反転する。"
        "{{SYMBOLIC_TOKEN}} に新たな意味を見出し、{{BREAKTHROUGH_COST}} を払って突破する。"
    )
    if has_ally:
        template += "{{ALLY}} の存在が最後の一歩を支える。"
    invariants = [
        f"前提は t≈{t_premise:.2f} までに受け手と共有する",
        f"局所的最適化の限界を t≈{t_limit:.2f} に置く",
        f"構造的反転（核心の開示）は t≈{t_twist:.2f} に置き、それ以前に結論を明かさない",
        f"代償を伴う突破（緊張度ピーク）は t≈{t_break:.2f} に置く",
        "結末の緊張度は 0.35 以下へ緩和する",
    ]

    def page_of(t: float, pages: int = 16) -> int:
        return min(pages, max(1, int(round(t * (pages - 1))) + 1))

    transfer_guide = {
        "manga": (
            f"全16ページ想定。前提の共有を p.{page_of(t_premise)} まで、局所的最適化の限界を p.{page_of(t_limit)} に置く。"
            f"反転（t≈{t_twist:.2f}）はめくり直後の右ページ先頭（p.{page_of(t_twist)} 付近）に大ゴマで置き、直前の左ページ終端に引きを作る。"
            f"代償を伴う突破（t≈{t_break:.2f}）は p.{page_of(t_break)} の見開きまたは1ページ大ゴマ、余韻は最終ページの余白の多い2〜3コマで締める。"
        ),
        "marketing": (
            "ファーストビューで欠落（痛み）を逆説的な問いとして提示し、次に読者が信じる方法（前提）を言語化して共感を得る。"
            f"前提に沿った努力の限界（t≈{t_limit:.2f} 相当）を事例で示した後、前提そのものが原因だったという反転を証拠とともに提示する。"
            "突破の代償（移行の手間・既存手法の放棄）を誠実に示してから、最後に最初の一歩となる CTA を置く。"
        ),
        "article": (
            "通説（前提）の提示 → 通説に沿った実践の限界 → 通説自体が失敗を生むという構造的パラドックス → "
            "代償を伴う新しい方法の提示 → 読者が今日から試せる実践アクション、の順で論証する。"
            f"反転は全体の約 {int(round(t_twist * 100))}% の位置に置く。"
        ),
    }
    twist_word = "反転" if twist_type == BeatType.STRUCTURAL_TWIST.value else "開示"
    title = truncate(title_hint.strip(), 200) if title_hint and title_hint.strip() else f"{goal_label}をめぐる{twist_word}構造"
    summary = (
        f"欠落を抱えた主体が{goal_label}を追い、方法の限界に突き当たった後、t≈{t_twist:.2f} で前提の{twist_word}を迎え、"
        f"t≈{t_break:.2f} で代償を伴う突破に至る構造。"
        f"緊張度ピーク t≈{stats['peak_t']:.2f}、サスペンス係数 {stats['suspense']:.2f}、サプライズ係数 {stats['surprise']:.2f}。"
        f"（入力種別: {source_kind}）"
    )
    tags = ["抽出構造", f"構造的{twist_word}", "代償"]
    if is_betrayal:
        tags.append("裏切り")
    if unresolved:
        tags.append("開かれた結末")
    signals = genre_signals if genre_signals else deflavor.genre_signals
    top_genre = max(signals.items(), key=lambda kv: kv[1]) if signals else ("", 0.0)
    if top_genre[1] > 0:
        tags.append(f"由来傾向:{top_genre[0]}")
    payload = {
        "title": title,
        "summary": summary,
        "tags": tags,
        "tension_graph": {"nodes": nodes, "edges": edges},
        "wave": wave_payload,
        "beats": {"beats": beats},
        "mechanics": {
            "mechanisms": mechanisms,
            "empathy_anchor": (
                "奪われた拠り所を取り戻したいという普遍的な欠落の感覚" if has_lack_words else "目標に向かって努力する者が抱く焦りと希望"
            ),
            "immersion_drivers": _dedupe_list(
                ["欠落への共感", "前提を共有させた上での反転", "代償の重さ"]
                + (["読者だけが知る危機"] if float(stats["suspense"]) >= 0.05 else [])
                + (["真因への問い"] if conceal_total > 0 else [])
            ),
            "emotion_sequence": ["不安", "期待", "高揚", "違和感", "驚愕", "絶望", "決意", "余韻と問い" if unresolved else "余韻"],
        },
        "blueprint": {"template": template, "slots": slots, "invariants": invariants},
        "transfer_guide": transfer_guide,
    }
    return StructureLayers.model_validate(payload)


def _dedupe_list(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# 漏洩検出・置換
# ---------------------------------------------------------------------------

_SKIP_KEYS = frozenset(
    {
        "id", "source", "target", "kind", "beat_type", "category", "slot", "causes", "involved_nodes",
        "linked_beats", "linked_nodes", "lineage", "parent_ids", "created_at", "language", "source_kind",
        "origin_domain", "required",
    }
)


def _leak_patterns(tokens: Sequence[FlavorToken]) -> list[tuple[str, re.Pattern[str], str]]:
    """入力由来トークン＋辞書全体の装飾語（2文字以上）から漏洩検出・置換パターンを作る（最長一致順）。"""
    known = {token.surface for token in tokens}
    extended = list(tokens) + [
        FlavorToken(surface=term, category="genre_term", placeholder=value, occurrences=0)
        for term, value in GENRE_TERMS.items()
        if len(term) >= 2 and term not in known
    ]
    patterns: list[tuple[str, re.Pattern[str], str]] = []
    for token in sorted(extended, key=lambda tok: len(tok.surface), reverse=True):
        surface = token.surface
        if len(surface) < 2:
            continue
        if token.category == "genre_term":
            replacement = token.placeholder
            if surface.isascii():
                pattern = re.compile(rf"\b{re.escape(surface)}s?\b", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(surface))
        else:
            replacement = _GENERIC_WORDS.get(token.category, _GENERIC_WORDS["other"])
            pattern = re.compile(rf"\b{re.escape(surface)}\b" if surface.isascii() else re.escape(surface))
        patterns.append((surface, pattern, replacement))
    return patterns


def find_flavor_leaks(structure: UniversalStructure, tokens: Sequence[FlavorToken]) -> list[str]:
    """構造の可読テキスト（lineage 除外）に残る固有名詞・装飾語・プレースホルダを列挙する。"""
    texts = structure.text_fields()
    leaks: list[str] = []
    for surface, pattern, _replacement in _leak_patterns(tokens):
        if any(pattern.search(text) for text in texts) and surface not in leaks:
            leaks.append(surface)
    for text in texts:
        for match in PLACEHOLDER_RE.finditer(text):
            if match.group(0) not in leaks:
                leaks.append(match.group(0))
    return leaks


def _scrub_value(value: Any, patterns: list[tuple[str, re.Pattern[str], str]], path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SKIP_KEYS or (key == "name" and "slots" in path):
                result[key] = item
            else:
                result[key] = _scrub_value(item, patterns, (*path, key))
        return result
    if isinstance(value, list):
        if path and path[-1] in _SKIP_KEYS:
            return value
        return [_scrub_value(item, patterns, path) for item in value]
    if isinstance(value, str):
        text = value
        for _surface, pattern, replacement in patterns:
            text = pattern.sub(replacement, text)
        return PLACEHOLDER_RE.sub(lambda m: _GENERIC_WORDS.get(_category_of_label(m.group(1)), "ある存在"), text)
    return value


def _category_of_label(label: str) -> str:
    for category, value in _PLACEHOLDER_LABELS.items():
        if value == label:
            return category
    return "other"


def scrub_flavor(structure: UniversalStructure, tokens: Sequence[FlavorToken]) -> UniversalStructure:
    """可読テキスト中の固有名詞・装飾語・プレースホルダを抽象語へ置換した構造を返す（lineage・ID類は不変）。"""
    data = structure.model_dump(mode="json")
    data["wave"].pop("information_asymmetry", None)
    scrubbed = _scrub_value(data, _leak_patterns(tokens), ())
    return UniversalStructure.model_validate(scrubbed)


# ---------------------------------------------------------------------------
# 抽出器
# ---------------------------------------------------------------------------

_SOURCE_DOMAINS = {
    "synopsis": "narrative",
    "storyboard": "manga",
    "name": "manga",
    "article": "article",
    "lp": "marketing",
    "landing_page": "marketing",
}


class StructureExtractor:
    """脱フレーバー構造抽出器（LLM があれば4段階ハーネスで増幅、無ければ決定論的ベースライン）。"""

    def __init__(self, llm_client: LLMClient | None = None, harness_config: HarnessConfig | None = None) -> None:
        self.llm_client = llm_client
        self.harness_config = harness_config

    async def extract(
        self,
        text: str,
        *,
        source_kind: str = "synopsis",
        title_hint: str | None = None,
        category: StructureCategory = StructureCategory.CUSTOM,
    ) -> ExtractionResult:
        if not text or not text.strip():
            raise ValueError("抽出対象のテキストが空です")
        deflavor = deflavor_text(text)
        category_value = StructureCategory(category)
        layers: StructureLayers | None = None
        run_id: str | None = None
        critique_score: float | None = None
        used_llm = False
        if self.llm_client is not None:
            harness = MultiStageReasoningHarness(self.llm_client, self.harness_config or HarnessConfig())

            def post_validate(candidate: StructureLayers) -> list[str]:
                issues: list[str] = []
                probe = UniversalStructure.from_layers(candidate, id="st_validation_probe")
                leaks = find_flavor_leaks(probe, deflavor.tokens)
                if leaks:
                    issues.append(f"固有名詞・装飾語・プレースホルダが残存しています: {', '.join(leaks[:12])}。抽象的な役割語へ置換すること。")
                for issue in audit_structure(probe):
                    if issue.severity == "error":
                        issues.append(issue.message)
                return issues

            try:
                result = await harness.run(
                    task="structure.extract",
                    objective=EXTRACTION_OBJECTIVE,
                    payload={
                        "abstracted_text": deflavor.abstracted_text,
                        "source_kind": source_kind,
                        "placeholders": deflavor.placeholders(),
                        "genre_signals": deflavor.genre_signals,
                    },
                    output_model=StructureLayers,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    rubric=EXTRACTION_RUBRIC,
                    context={
                        "abstracted_text": deflavor.abstracted_text,
                        "source_kind": source_kind,
                        "title_hint": title_hint,
                        "genre_signals": deflavor.genre_signals,
                    },
                    post_validate=post_validate,
                )
            except HarnessError as exc:
                logger.warning("構造抽出ハーネスが失敗したためヒューリスティック抽出へフォールバックします: %s", exc)
            else:
                layers = result.output
                run_id = result.run_id
                used_llm = True
                critique_score = result.critique.overall_score if result.critique is not None else None
        if layers is None:
            layers = heuristic_extract(
                deflavor.abstracted_text,
                source_kind=source_kind,
                title_hint=title_hint,
                genre_signals=deflavor.genre_signals,
            )
        structure = UniversalStructure.from_layers(
            layers,
            id=new_structure_id("ex"),
            source_kind=SourceKind.EXTRACTED,
            category=category_value,
            origin_domain=_SOURCE_DOMAINS.get(source_kind, source_kind or "generic"),
            notes=truncate(
                f"脱フレーバー抽出（入力種別 {source_kind}・除去トークン {len(deflavor.proper_noun_tokens())} 件・"
                f"装飾語 {len(deflavor.tokens) - len(deflavor.proper_noun_tokens())} 種・LLM {'使用' if used_llm else '不使用'}）",
                4000,
            ),
        )
        leaks = find_flavor_leaks(structure, deflavor.tokens)
        if leaks:
            structure = scrub_flavor(structure, deflavor.tokens)
        return ExtractionResult(
            structure=structure,
            deflavor=deflavor,
            harness_run_id=run_id,
            used_llm=used_llm,
            leaked_tokens_fixed=leaks,
            critique_score=critique_score,
        )
