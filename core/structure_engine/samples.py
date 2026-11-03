"""決定論的サンプル構造ビルダー。

各モジュール（Manga Pipeline / Asset Publisher / Web Studio）のテストと UI デモで共通利用する、
スキーマに完全準拠した高密度の純粋骨格を提供する。シードDBとは独立に、常に同一内容を返す。

- ``reversal``   : 構造的反転型（庇護者の裏切りと代償を伴う突破）
- ``inverted``   : 倒叙型（読者だけが真相を知るサスペンス）
- ``persuasion`` : 常識転覆型の論説・LP骨格
"""

from __future__ import annotations

from typing import Any

from core.structure_engine.schemas import (
    EmotionalInformationWave,
    StructureCategory,
    SourceKind,
    UniversalStructure,
)

__all__ = ["SAMPLE_KINDS", "sample_structure", "sample_structures", "sample_text_ja", "sample_text_en"]

SAMPLE_KINDS: tuple[str, ...] = ("reversal", "inverted", "persuasion")


def _reversal_payload() -> dict[str, Any]:
    wave = EmotionalInformationWave.from_keyframes(
        {
            "tension": [(0.0, 0.35), (0.08, 0.5), (0.2, 0.32), (0.38, 0.55), (0.52, 0.48), (0.6, 0.66), (0.72, 0.9), (0.8, 0.62), (0.88, 0.97), (0.95, 0.45), (1.0, 0.2)],
            "release": [(0.0, 0.05), (0.2, 0.18), (0.5, 0.3), (0.58, 0.42), (0.72, 0.06), (0.88, 0.12), (0.95, 0.78), (1.0, 0.9)],
            "audience_knowledge": [(0.0, 0.05), (0.15, 0.2), (0.4, 0.35), (0.6, 0.42), (0.7, 0.48), (0.74, 0.86), (0.9, 0.93), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.08), (0.15, 0.28), (0.4, 0.5), (0.6, 0.62), (0.7, 0.66), (0.74, 0.8), (0.9, 0.95), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.45), (0.15, 0.6), (0.35, 0.5), (0.55, 0.62), (0.74, 0.82), (0.88, 0.7), (1.0, 0.3)],
        }
    )
    return {
        "id": "sample_reversal",
        "metadata": {
            "title": "庇護者の裏切りと代償の突破",
            "summary": "欠落を抱えた主体が、信頼する庇護者の示す最適解に従って前進するが、その最適解こそが欠落の原因だったと判明し、代償を払って自力の道を切り開く構造。",
            "category": StructureCategory.CUSTOM.value,
            "source_kind": SourceKind.MANUAL.value,
            "tags": ["構造的反転", "裏切り", "代償", "自立"],
            "origin_domain": "generic",
            "lineage": ["アリストテレス的逆転（ペリペテイア）", "師の裏切り型"],
        },
        "tension_graph": {
            "nodes": [
                {"id": "seeker", "kind": "agent", "role": "protagonist", "label": "欠落を埋めようとする主体", "description": "失われた均衡を取り戻すことに執着し、庇護者の導きを疑わない。", "weight": 0.95, "slot": "PROTAGONIST"},
                {"id": "guardian", "kind": "agent", "role": "false_mentor", "label": "最適解を示す庇護者", "description": "主体を導くが、自らの支配を維持するために欠落を作り出した張本人。", "weight": 0.85, "slot": "FALSE_MENTOR"},
                {"id": "system", "kind": "agent", "role": "antagonistic_order", "label": "均衡を独占する秩序", "description": "庇護者と共犯関係にある既存の仕組み。", "weight": 0.6},
                {"id": "restoration", "kind": "desire", "role": "obsessive_goal", "label": "失われた均衡の回復", "description": "主体の行動を駆動する執着的目標。", "weight": 0.9, "slot": "PROTAGONIST_OBSESSIVE_GOAL"},
                {"id": "wound", "kind": "lack", "role": "core_lack", "label": "奪われた拠り所", "description": "物語開始以前に生じた欠落。真因は隠蔽されている。", "weight": 0.8, "slot": "CORE_LACK"},
                {"id": "token", "kind": "symbol", "role": "inheritance", "label": "受け継いだ証", "description": "欠落の真因を示す手がかりであり、最後に新しい意味を帯びる。", "weight": 0.55, "slot": "SYMBOLIC_TOKEN"},
            ],
            "edges": [
                {"source": "seeker", "target": "restoration", "kind": "pursuit", "intensity": 0.92, "label": "執着的追求"},
                {"source": "seeker", "target": "guardian", "kind": "dependency", "intensity": 0.85, "gradient": -0.4, "label": "導きへの依存"},
                {"source": "guardian", "target": "seeker", "kind": "power_gradient", "intensity": 0.75, "gradient": 0.7, "label": "保護を装った支配"},
                {"source": "guardian", "target": "seeker", "kind": "asymmetry", "intensity": 0.8, "gradient": 0.85, "label": "真因の秘匿"},
                {"source": "guardian", "target": "seeker", "kind": "betrayal", "intensity": 0.9, "label": "最適解という罠", "activation_t": 0.72},
                {"source": "seeker", "target": "system", "kind": "opposition", "intensity": 0.7, "label": "秩序への反抗"},
                {"source": "guardian", "target": "system", "kind": "dependency", "intensity": 0.6, "label": "共犯関係"},
                {"source": "token", "target": "wound", "kind": "embodiment", "intensity": 0.6, "label": "欠落の痕跡"},
            ],
        },
        "wave": wave.model_dump(mode="json"),
        "beats": {
            "beats": [
                {"id": "hook", "beat_type": "hook", "t": 0.05, "span": 0.06, "label": "欠落の瞬間の断片", "function": "結末を予感させる欠落の断片を提示し、問いを立てる。", "tension_delta": 0.25, "information_shift": -0.05, "involved_nodes": ["seeker", "wound"]},
                {"id": "premise", "beat_type": "premise", "t": 0.15, "span": 0.08, "label": "庇護者の導きという前提", "function": "庇護者の示す最適解に従えば欠落は埋まる、という前提を読者と主体に共有させる。", "causes": ["hook"], "tension_delta": -0.15, "information_shift": -0.1, "involved_nodes": ["seeker", "guardian"]},
                {"id": "escalation", "beat_type": "escalation", "t": 0.38, "span": 0.1, "label": "最適解による前進", "function": "前提に従った行動が成果を生み、依存が深まる。", "causes": ["premise"], "tension_delta": 0.2, "information_shift": -0.12, "involved_nodes": ["seeker", "guardian", "system"]},
                {"id": "limit", "beat_type": "local_optimization_limit", "t": 0.6, "span": 0.06, "label": "局所的最適化の限界", "function": "同じ方法の反復では越えられない壁が現れ、前提への違和感が芽生える。", "causes": ["escalation"], "tension_delta": 0.18, "information_shift": -0.04, "involved_nodes": ["seeker", "system"], "is_turning_point": True},
                {"id": "twist", "beat_type": "structural_twist", "t": 0.72, "span": 0.05, "label": "最適解こそが罠だった", "function": "庇護者が欠落の真因であると判明し、物語の前提が反転する。", "causes": ["limit"], "tension_delta": 0.35, "information_shift": 0.38, "involved_nodes": ["seeker", "guardian", "token"], "is_turning_point": True},
                {"id": "crisis", "beat_type": "crisis", "t": 0.8, "span": 0.05, "label": "拠り所の全喪失", "function": "依存先を失い、主体は自力の判断を迫られる。", "causes": ["twist"], "tension_delta": -0.25, "information_shift": 0.05, "involved_nodes": ["seeker"]},
                {"id": "breakthrough", "beat_type": "costly_breakthrough", "t": 0.88, "span": 0.06, "label": "代償を伴う突破", "function": "証に新しい意味を与え、最も大切なものを差し出して秩序を越える。", "causes": ["crisis"], "tension_delta": 0.35, "information_shift": 0.06, "involved_nodes": ["seeker", "system", "token"], "is_turning_point": True},
                {"id": "afterglow", "beat_type": "afterglow", "t": 0.97, "span": 0.03, "label": "自立の余韻", "function": "回復ではなく変容としての結末を示し、欠落の意味を書き換える。", "causes": ["breakthrough"], "tension_delta": -0.5, "information_shift": 0.02, "involved_nodes": ["seeker", "wound"]},
            ]
        },
        "mechanics": {
            "mechanisms": [
                {"id": "gap", "category": "curiosity_gap", "name": "欠落の真因への問い", "description": "冒頭の断片が答えの無い問いを残し、読み進める動機を生む。", "trigger": "結末の断片を先に見せ、原因を伏せる", "effect": "真因を知りたいという持続的な欲求", "theory": "情報ギャップ理論", "strength": 0.75, "linked_beats": ["hook"], "linked_nodes": ["wound"]},
                {"id": "error", "category": "prediction_error", "name": "前提の崩壊", "description": "読者が共有していた前提が覆り、強い予測誤差が生じる。", "trigger": "信頼の対象が欠落の原因と判明する", "effect": "驚愕と再解釈による深い没入", "theory": "予測符号化理論", "strength": 0.9, "linked_beats": ["twist"], "linked_nodes": ["guardian"]},
                {"id": "empathy", "category": "empathy_transfer", "name": "欠落への共感移入", "description": "読者自身の喪失経験が主体の欠落に重ねられる。", "trigger": "具体的で普遍的な欠落の描写", "effect": "主体の選択を自分事として体験する", "theory": "物語移入理論", "strength": 0.7, "linked_beats": ["premise", "crisis"], "linked_nodes": ["seeker"]},
                {"id": "peak", "category": "peak_end", "name": "代償の記憶化", "description": "最大の代償の瞬間と余韻が体験の評価を決める。", "trigger": "クライマックス直後の静かな結末", "effect": "読後の長い余韻と再読欲求", "theory": "ピーク・エンドの法則", "strength": 0.65, "linked_beats": ["breakthrough", "afterglow"]},
            ],
            "empathy_anchor": "奪われた拠り所を取り戻したいという、誰もが持つ欠落の感覚",
            "immersion_drivers": ["真因への問い", "前提を共有させた上での反転", "代償の重さ"],
            "emotion_sequence": ["不安", "信頼", "高揚", "違和感", "驚愕", "絶望", "決意", "余韻"],
        },
        "blueprint": {
            "template": "{{PROTAGONIST}} は {{CORE_LACK}} を埋めるため {{PROTAGONIST_OBSESSIVE_GOAL}} を追い、{{FALSE_MENTOR}} の示す最適解に依存する。だが {{STRUCTURAL_IRONY}} が明らかになり前提は反転する。{{SYMBOLIC_TOKEN}} に新しい意味を与え、{{BREAKTHROUGH_COST}} を払って自立する。",
            "slots": [
                {"name": "PROTAGONIST", "description": "欠落を抱える主体", "category": "agent", "examples": {"manga": "家族を失った少年", "marketing": "成果が頭打ちの担当者", "article": "定説を信じてきた実践者"}},
                {"name": "CORE_LACK", "description": "物語以前に生じた欠落", "category": "lack", "examples": {"manga": "帰る場所", "marketing": "伸びない売上", "article": "説明できない失敗"}},
                {"name": "PROTAGONIST_OBSESSIVE_GOAL", "description": "主体の執着的目標", "category": "goal", "examples": {"manga": "失われた場所の奪還", "marketing": "目標数値の達成", "article": "失敗の原因究明"}},
                {"name": "FALSE_MENTOR", "description": "最適解を示す庇護者（実は原因）", "category": "agent", "examples": {"manga": "育ての親", "marketing": "業界の常識", "article": "広く信じられた定説"}},
                {"name": "STRUCTURAL_IRONY", "description": "最適解そのものが欠落の原因だったという構造的皮肉", "category": "irony", "examples": {"manga": "育ての親こそが仇だった", "marketing": "常識的施策が成長を止めていた", "article": "定説が失敗を再生産していた"}},
                {"name": "SYMBOLIC_TOKEN", "description": "真因を示す象徴物", "category": "symbol", "examples": {"manga": "形見の鍵", "marketing": "顧客の声", "article": "見過ごされたデータ"}},
                {"name": "BREAKTHROUGH_COST", "description": "突破に必要な代償", "category": "cost", "examples": {"manga": "安住の居場所", "marketing": "慣れた手法の放棄", "article": "自説の撤回"}},
            ],
            "invariants": [
                "前提（最適解）は読者にも共有させた後で反転させる",
                "構造的反転は t≈0.7 前後、代償を伴う突破は反転より後に置く",
                "裏切りの前に十分な依存関係を描く",
                "結末は回復ではなく変容として描く",
            ],
        },
        "transfer_guide": {
            "manga": "全16ページなら反転を見開きめくり直後（右ページ先頭）に置き、直前の左ページ終端で庇護者の微笑を引きに使う。突破は1ページ大ゴマ、余韻は余白の多い2〜3コマで締める。",
            "marketing": "LPでは業界の常識（庇護者）に従った努力の限界を共有し、その常識自体が原因であるという反転をファーストビュー直後ではなく中盤に置き、代償（手法の乗り換えコスト）を正直に提示してCTAへ繋ぐ。",
            "article": "通説の提示→通説に従った実践例→限界→通説そのものが原因というパラドックス→代償を伴う新しい方法→実践ステップの順で論じる。",
        },
    }


def _inverted_payload() -> dict[str, Any]:
    wave = EmotionalInformationWave.from_keyframes(
        {
            "tension": [(0.0, 0.55), (0.1, 0.7), (0.25, 0.5), (0.4, 0.62), (0.55, 0.74), (0.68, 0.66), (0.8, 0.86), (0.9, 0.95), (0.96, 0.5), (1.0, 0.3)],
            "release": [(0.0, 0.1), (0.25, 0.2), (0.5, 0.12), (0.7, 0.18), (0.9, 0.05), (0.96, 0.85), (1.0, 0.7)],
            "audience_knowledge": [(0.0, 0.2), (0.08, 0.85), (0.4, 0.88), (0.7, 0.9), (0.9, 0.96), (1.0, 1.0)],
            "character_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.45, 0.38), (0.65, 0.55), (0.85, 0.8), (0.92, 0.97), (1.0, 1.0)],
            "cognitive_load": [(0.0, 0.5), (0.1, 0.65), (0.3, 0.5), (0.6, 0.66), (0.85, 0.78), (1.0, 0.35)],
        }
    )
    return {
        "id": "sample_inverted",
        "metadata": {
            "title": "全知の読者と追い詰める追跡者",
            "summary": "読者だけが冒頭で真相を知り、真相を隠す主体と真相に迫る追跡者の攻防を、劇的アイロニーによるサスペンスで駆動する構造。",
            "category": StructureCategory.CUSTOM.value,
            "source_kind": SourceKind.MANUAL.value,
            "tags": ["倒叙", "サスペンス", "心理戦"],
            "origin_domain": "generic",
            "lineage": ["倒叙型推理劇"],
        },
        "tension_graph": {
            "nodes": [
                {"id": "concealer", "kind": "agent", "role": "concealer", "label": "真相を隠す主体", "description": "完璧な隠蔽を信じ、平静を装い続ける。", "weight": 0.9, "slot": "CONCEALER"},
                {"id": "pursuer", "kind": "agent", "role": "pursuer", "label": "違和感から迫る追跡者", "description": "小さな矛盾を積み上げて真相へ到達する。", "weight": 0.9, "slot": "PURSUER"},
                {"id": "safety", "kind": "desire", "role": "goal", "label": "露見しない日常", "description": "隠蔽者が守ろうとする現状。", "weight": 0.7, "slot": "PROTECTED_STATUS"},
                {"id": "truth", "kind": "lack", "role": "hidden_truth", "label": "隠された真相", "description": "読者のみが知る出来事。", "weight": 0.85, "slot": "HIDDEN_TRUTH"},
                {"id": "clue", "kind": "symbol", "role": "decisive_clue", "label": "決定的な綻び", "description": "隠蔽者自身が見落とした一点。", "weight": 0.6, "slot": "FATAL_FLAW"},
            ],
            "edges": [
                {"source": "concealer", "target": "safety", "kind": "pursuit", "intensity": 0.85, "label": "現状維持への執着"},
                {"source": "pursuer", "target": "truth", "kind": "pursuit", "intensity": 0.9, "label": "真相への追求"},
                {"source": "concealer", "target": "pursuer", "kind": "asymmetry", "intensity": 0.8, "gradient": 0.75, "label": "情報優位", "activation_t": 0.05},
                {"source": "pursuer", "target": "concealer", "kind": "opposition", "intensity": 0.85, "label": "静かな攻防", "activation_t": 0.25},
                {"source": "concealer", "target": "pursuer", "kind": "power_gradient", "intensity": 0.55, "gradient": 0.4, "label": "社会的地位の優位"},
                {"source": "clue", "target": "truth", "kind": "embodiment", "intensity": 0.75, "label": "真相の痕跡", "activation_t": 0.85},
            ],
        },
        "wave": wave.model_dump(mode="json"),
        "beats": {
            "beats": [
                {"id": "reveal_first", "beat_type": "hook", "t": 0.05, "span": 0.06, "label": "真相の先行開示", "function": "読者にだけ出来事の全貌を見せ、劇的アイロニーを成立させる。", "tension_delta": 0.3, "information_shift": 0.65, "involved_nodes": ["concealer", "truth"]},
                {"id": "facade", "beat_type": "premise", "t": 0.2, "span": 0.08, "label": "平静の仮面", "function": "隠蔽者が日常を装い、読者は綻びを探しながら見守る。", "causes": ["reveal_first"], "tension_delta": -0.15, "information_shift": 0.0, "involved_nodes": ["concealer", "safety"]},
                {"id": "first_probe", "beat_type": "escalation", "t": 0.4, "span": 0.08, "label": "最初の探り", "function": "追跡者が些細な違和感を口にし、攻防が始まる。", "causes": ["facade"], "tension_delta": 0.15, "information_shift": -0.12, "involved_nodes": ["pursuer", "concealer"]},
                {"id": "counter", "beat_type": "false_victory", "t": 0.62, "span": 0.06, "label": "隠蔽者の反撃", "function": "隠蔽者が追跡者を出し抜き、逃げ切りを確信する偽の勝利。", "causes": ["first_probe"], "tension_delta": -0.08, "information_shift": -0.1, "involved_nodes": ["concealer", "pursuer"], "is_turning_point": True},
                {"id": "flaw", "beat_type": "revelation", "t": 0.86, "span": 0.05, "label": "綻びの指摘", "function": "追跡者が決定的な綻びを示し、隠蔽者の確信が崩れる。", "causes": ["counter"], "tension_delta": 0.3, "information_shift": -0.25, "involved_nodes": ["pursuer", "clue", "concealer"], "is_turning_point": True},
                {"id": "confession", "beat_type": "climax", "t": 0.91, "span": 0.04, "label": "仮面の崩落", "function": "隠蔽者が自ら真相を認め、緊張が頂点で解放される。", "causes": ["flaw"], "tension_delta": 0.1, "information_shift": -0.05, "involved_nodes": ["concealer", "pursuer"], "is_turning_point": True},
                {"id": "afterglow", "beat_type": "afterglow", "t": 0.98, "span": 0.02, "label": "静かな幕切れ", "function": "勝者のいない余韻で、隠蔽の動機を反芻させる。", "causes": ["confession"], "tension_delta": -0.55, "information_shift": 0.0, "involved_nodes": ["concealer"]},
            ]
        },
        "mechanics": {
            "mechanisms": [
                {"id": "irony", "category": "dramatic_irony", "name": "全知の読者", "description": "読者だけが真相を知るため、一挙手一投足が二重の意味を帯びる。", "trigger": "冒頭での真相開示", "effect": "持続的なサスペンスと手に汗握る見守り", "theory": "劇的アイロニー", "strength": 0.92, "linked_beats": ["reveal_first", "facade"], "linked_nodes": ["concealer"]},
                {"id": "dissonance", "category": "cognitive_dissonance", "name": "共犯的な不協和", "description": "隠蔽者の視点に寄り添うほど、逃げ切りを願う自分に不協和が生じる。", "trigger": "隠蔽者の内面描写", "effect": "倫理的な緊張と自己観察", "theory": "認知的不協和理論", "strength": 0.6, "linked_beats": ["counter"], "linked_nodes": ["concealer"]},
                {"id": "reward", "category": "anticipation_reward", "name": "綻びの予期", "description": "読者は綻びが露見する瞬間を予期し、その成就で報酬を得る。", "trigger": "決定的な手がかりの事前提示", "effect": "予期が満たされる快感", "theory": "報酬予測誤差", "strength": 0.7, "linked_beats": ["flaw"], "linked_nodes": ["clue"]},
            ],
            "empathy_anchor": "隠し事を抱えて平静を装う緊張という普遍的な経験",
            "immersion_drivers": ["読者だけが知る真相", "知的な攻防", "綻び探し"],
            "emotion_sequence": ["衝撃", "緊張", "期待", "焦燥", "確信", "解放", "余韻"],
        },
        "blueprint": {
            "template": "読者だけが {{HIDDEN_TRUTH}} を知る。{{CONCEALER}} は {{PROTECTED_STATUS}} を守るため平静を装い、{{PURSUER}} は違和感から迫る。偽の勝利の後、{{FATAL_FLAW}} が示され {{DRAMATIC_IRONY_PAYOFF}} が訪れる。",
            "slots": [
                {"name": "HIDDEN_TRUTH", "description": "読者のみが知る真相", "category": "irony", "examples": {"manga": "主人公が犯した罪", "marketing": "既存手法の隠れたコスト", "article": "指標の裏に隠れた事実"}},
                {"name": "CONCEALER", "description": "真相を隠す主体", "category": "agent", "examples": {"manga": "優等生の主人公", "marketing": "現状維持バイアス", "article": "定説の擁護者"}},
                {"name": "PROTECTED_STATUS", "description": "隠蔽者が守る現状", "category": "goal", "examples": {"manga": "平穏な学園生活", "marketing": "慣れた業務フロー", "article": "既存の評価軸"}},
                {"name": "PURSUER", "description": "違和感から迫る追跡者", "category": "agent", "examples": {"manga": "転校生の探偵役", "marketing": "データで問い直す担当者", "article": "検証を重ねる筆者"}},
                {"name": "FATAL_FLAW", "description": "決定的な綻び", "category": "evidence", "examples": {"manga": "時刻の矛盾", "marketing": "解約理由の共通点", "article": "再現できない成功例"}},
                {"name": "DRAMATIC_IRONY_PAYOFF", "description": "劇的アイロニーの回収", "category": "reversal", "examples": {"manga": "自白", "marketing": "乗り換えの決断", "article": "定説の更新"}},
            ],
            "invariants": [
                "真相は冒頭10%以内に読者へ開示する",
                "読者の認知量は全編を通じて作中人物以上に保つ",
                "偽の勝利の後に決定的な綻びを置く",
            ],
        },
        "transfer_guide": {
            "manga": "1ページ目で真相を見せ、以降は隠蔽者の表情アップと追跡者の何気ない一言を交互に置くカットバックで進める。綻びの指摘はめくり直後の大ゴマ、自白は無音の1コマ。",
            "marketing": "冒頭で読者だけに既存手法の隠れたコストを示し、比較検討の過程でその事実が浮かび上がる構成にする。",
            "article": "結論（隠れた事実）を冒頭で示し、なぜ多くの人が気づかないのかを検証の過程として描く。",
        },
    }


def _persuasion_payload() -> dict[str, Any]:
    wave = EmotionalInformationWave.from_keyframes(
        {
            "tension": [(0.0, 0.4), (0.1, 0.55), (0.25, 0.68), (0.35, 0.6), (0.5, 0.78), (0.6, 0.45), (0.75, 0.52), (0.85, 0.66), (0.95, 0.35), (1.0, 0.25)],
            "release": [(0.0, 0.05), (0.3, 0.1), (0.5, 0.08), (0.6, 0.55), (0.8, 0.4), (0.95, 0.75), (1.0, 0.8)],
            "audience_knowledge": [(0.0, 0.1), (0.2, 0.2), (0.45, 0.3), (0.55, 0.7), (0.8, 0.85), (1.0, 0.95)],
            "character_knowledge": [(0.0, 0.5), (0.2, 0.55), (0.45, 0.6), (0.55, 0.75), (0.8, 0.88), (1.0, 0.95)],
            "cognitive_load": [(0.0, 0.3), (0.25, 0.45), (0.5, 0.7), (0.65, 0.6), (0.85, 0.5), (1.0, 0.3)],
        }
    )
    return {
        "id": "sample_persuasion",
        "metadata": {
            "title": "常識転覆と新しい枠組みの提示",
            "summary": "読者が信じる常識に沿った努力が報われない理由を構造的パラドックスとして示し、証拠と新しい枠組みによって行動変容へ導く論説骨格。",
            "category": StructureCategory.PERSUASION_MARKETING.value,
            "source_kind": SourceKind.MANUAL.value,
            "tags": ["論説", "常識転覆", "フレームワーク", "行動変容"],
            "origin_domain": "article",
            "lineage": ["問題提起型論説", "パラダイムシフト型LP"],
        },
        "tension_graph": {
            "nodes": [
                {"id": "reader", "kind": "agent", "role": "reader", "label": "常識に従って努力する読者", "description": "正しいと信じる方法を続けても成果が出ない。", "weight": 0.95, "slot": "TARGET_READER"},
                {"id": "guide", "kind": "agent", "role": "guide", "label": "検証を重ねた語り手", "description": "自らの失敗と実験から新しい枠組みに到達した。", "weight": 0.75, "slot": "GUIDE"},
                {"id": "dogma", "kind": "symbol", "role": "common_belief", "label": "広く信じられた常識", "description": "努力の方向を固定してしまう前提。", "weight": 0.7, "slot": "COMMON_BELIEF"},
                {"id": "result", "kind": "desire", "role": "desired_outcome", "label": "努力が報われる状態", "description": "読者が本当に望む成果。", "weight": 0.85, "slot": "DESIRED_OUTCOME"},
                {"id": "pain", "kind": "lack", "role": "pain", "label": "報われない努力の痛み", "description": "原因が分からない停滞。", "weight": 0.8, "slot": "TARGET_PAIN"},
            ],
            "edges": [
                {"source": "reader", "target": "result", "kind": "pursuit", "intensity": 0.9, "label": "成果への追求"},
                {"source": "reader", "target": "dogma", "kind": "dependency", "intensity": 0.85, "label": "常識への依存"},
                {"source": "dogma", "target": "reader", "kind": "betrayal", "intensity": 0.7, "label": "努力を空転させる前提", "activation_t": 0.5},
                {"source": "guide", "target": "reader", "kind": "asymmetry", "intensity": 0.7, "gradient": 0.6, "label": "検証済みの知見"},
                {"source": "guide", "target": "dogma", "kind": "opposition", "intensity": 0.8, "label": "常識への反証"},
                {"source": "pain", "target": "reader", "kind": "power_gradient", "intensity": 0.6, "gradient": 0.5, "label": "停滞の圧力"},
            ],
        },
        "wave": wave.model_dump(mode="json"),
        "beats": {
            "beats": [
                {"id": "hook", "beat_type": "hook", "t": 0.04, "span": 0.05, "label": "逆説的な問い", "function": "正しい努力ほど報われないという逆説で注意を奪う。", "tension_delta": 0.2, "information_shift": -0.4, "involved_nodes": ["reader", "pain"]},
                {"id": "premise", "beat_type": "premise", "t": 0.15, "span": 0.08, "label": "常識の確認", "function": "読者が信じる常識を丁寧に言語化し共感を得る。", "causes": ["hook"], "tension_delta": 0.08, "information_shift": 0.05, "involved_nodes": ["reader", "dogma"]},
                {"id": "limit", "beat_type": "local_optimization_limit", "t": 0.3, "span": 0.08, "label": "努力の空転", "function": "常識に沿って改善を重ねても頭打ちになる構造を示す。", "causes": ["premise"], "tension_delta": 0.12, "information_shift": 0.05, "involved_nodes": ["reader", "result"]},
                {"id": "paradox", "beat_type": "structural_twist", "t": 0.5, "span": 0.06, "label": "構造的パラドックス", "function": "常識そのものが停滞の原因であると証拠で示す。", "causes": ["limit"], "tension_delta": 0.15, "information_shift": 0.4, "involved_nodes": ["dogma", "guide", "reader"], "is_turning_point": True},
                {"id": "framework", "beat_type": "payoff", "t": 0.65, "span": 0.1, "label": "新しい枠組み", "function": "再現可能なフレームワークとして新しい方法を提示する。", "causes": ["paradox"], "tension_delta": -0.25, "information_shift": 0.12, "involved_nodes": ["guide", "result"], "is_turning_point": True},
                {"id": "cost", "beat_type": "costly_breakthrough", "t": 0.85, "span": 0.06, "label": "乗り換えの代償", "function": "既存の習慣を手放す代償を誠実に示し、それでも得られる価値を比較する。", "causes": ["framework"], "tension_delta": 0.15, "information_shift": 0.05, "involved_nodes": ["reader", "dogma"]},
                {"id": "action", "beat_type": "afterglow", "t": 0.96, "span": 0.04, "label": "最初の一歩", "function": "今日できる最小の行動を示して締めくくる。", "causes": ["cost"], "tension_delta": -0.35, "information_shift": 0.03, "involved_nodes": ["reader", "result"]},
            ]
        },
        "mechanics": {
            "mechanisms": [
                {"id": "dissonance", "category": "cognitive_dissonance", "name": "努力と結果の不協和", "description": "正しいはずの努力が報われない矛盾が、解消を求める動機になる。", "trigger": "常識に従った失敗例の提示", "effect": "新しい説明を受け入れる準備", "theory": "認知的不協和理論", "strength": 0.8, "linked_beats": ["limit", "paradox"], "linked_nodes": ["reader"]},
                {"id": "authority", "category": "authority", "name": "検証による権威", "description": "語り手の失敗と実験の記録が信頼の根拠になる。", "trigger": "具体的なデータと過程の開示", "effect": "主張への信頼", "theory": "精緻化見込みモデル", "strength": 0.6, "linked_beats": ["paradox"], "linked_nodes": ["guide"]},
                {"id": "efficacy", "category": "self_efficacy", "name": "最初の一歩の自己効力感", "description": "小さく具体的な行動提示が実行可能感を高める。", "trigger": "今日できる最小行動の提示", "effect": "行動変容の開始", "theory": "自己効力感理論", "strength": 0.7, "linked_beats": ["action"], "linked_nodes": ["reader"]},
            ],
            "empathy_anchor": "正しくやっているのに報われないという、努力家ほど抱える痛み",
            "immersion_drivers": ["逆説的な問い", "証拠による反転", "再現可能な枠組み"],
            "emotion_sequence": ["引っかかり", "共感", "焦り", "驚き", "納得", "覚悟", "前向きさ"],
        },
        "blueprint": {
            "template": "{{TARGET_READER}} は {{COMMON_BELIEF}} に従い {{DESIRED_OUTCOME}} を目指すが、{{TARGET_PAIN}} から抜け出せない。{{GUIDE}} は {{PARADIGM_SHIFT}} を証拠とともに示し、{{OFFER_COST}} を正直に提示した上で最初の一歩を促す。",
            "slots": [
                {"name": "TARGET_READER", "description": "常識に従って努力する読者", "category": "audience", "examples": {"manga": "努力型の主人公", "marketing": "改善を重ねる担当者", "article": "独学の実践者"}},
                {"name": "COMMON_BELIEF", "description": "広く信じられた常識", "category": "symbol", "examples": {"manga": "師の教え", "marketing": "業界の定石", "article": "定説"}},
                {"name": "DESIRED_OUTCOME", "description": "読者が望む成果", "category": "goal", "examples": {"manga": "勝利", "marketing": "成約率の改善", "article": "再現性のある上達"}},
                {"name": "TARGET_PAIN", "description": "報われない努力の痛み", "category": "lack", "examples": {"manga": "伸び悩み", "marketing": "施策の頭打ち", "article": "停滞期"}},
                {"name": "GUIDE", "description": "検証を重ねた語り手", "category": "agent", "examples": {"manga": "異端の指導者", "marketing": "ブランド", "article": "私"}},
                {"name": "PARADIGM_SHIFT", "description": "常識を覆す新しい枠組み", "category": "reversal", "examples": {"manga": "型を捨てる戦い方", "marketing": "新しい導線設計", "article": "新しいフレームワーク"}},
                {"name": "OFFER_COST", "description": "乗り換えの代償", "category": "cost", "examples": {"manga": "積み上げた型の放棄", "marketing": "移行の手間", "article": "既存習慣の見直し"}},
            ],
            "invariants": [
                "反転（パラドックス）は証拠と共に中盤に置く",
                "代償は隠さず提示し、誠実な比較で判断を委ねる",
                "締めくくりは今日できる最小行動にする",
            ],
        },
        "transfer_guide": {
            "manga": "努力型の主人公が師の教えに従って伸び悩み、異端の指導者に型の限界を突きつけられる修行編として転用する。",
            "marketing": "ファーストビューで逆説的な問い、次に常識の確認と失敗事例、中盤に反証データ、後半に新しい導線と移行コスト、最後にCTAを置く。",
            "article": "通説→通説に沿った失敗→パラドックスと実験データ→フレームワーク→実践アクションの5部構成で書く。",
        },
    }


_BUILDERS = {
    "reversal": _reversal_payload,
    "inverted": _inverted_payload,
    "persuasion": _persuasion_payload,
}


def sample_structure(kind: str = "reversal") -> UniversalStructure:
    """スキーマ完全準拠のサンプル構造を返す（毎回新しいインスタンス）。"""
    if kind not in _BUILDERS:
        raise KeyError(f"未知のサンプル種別です: {kind}（候補: {', '.join(SAMPLE_KINDS)}）")
    return UniversalStructure.model_validate(_BUILDERS[kind]())


def sample_structures() -> list[UniversalStructure]:
    """全サンプル構造。"""
    return [sample_structure(kind) for kind in SAMPLE_KINDS]


def sample_text_ja() -> str:
    """固有名詞・ジャンル装飾を多く含む和文あらすじ（脱フレーバー抽出のテスト入力）。"""
    return (
        "王都アルカディアの孤児院で育った少年レオン・ハートは、師匠である宮廷魔術師グレゴールに導かれ、"
        "奪われた故郷ミスト村を取り戻すために魔法学園へ入学する。"
        "レオンはグレゴールの教えどおりに禁術を学び、ライバルのセリスとの決闘にも勝利して頭角を現す。"
        "しかし学園の地下書庫で母の形見の鍵が開いた古文書から、故郷を焼いたのはグレゴール自身だったという真実が判明する。"
        "信じていた師に裏切られたレオンは全てを失い、王国騎士団にも追われて絶望する。"
        "それでもセリスの助けを得たレオンは、形見の鍵に宿る力を解き放つため自らの魔力を代償として差し出し、"
        "グレゴールと王国の結界を打ち破る。戦いの後、レオンは故郷へは戻らず、新しい居場所を自分の手で築き始める。"
    )


def sample_text_en() -> str:
    """英文あらすじ（多言語入力のテスト用）。"""
    return (
        "In the city of New Avalon, a young engineer named Mara Quinn joins Helix Corporation under the mentorship of "
        "Dr. Victor Hale, hoping to rebuild the power grid that failed and killed her family. Following Hale's methods, "
        "she wins every contest and rises fast. But in the archive she discovers that Hale caused the blackout to seize "
        "control of the city. Betrayed and hunted, Mara loses everything. With help from her rival Theo, she sacrifices "
        "her career to expose the truth and tears down Helix's control. In the end she builds a new grid for everyone."
    )
