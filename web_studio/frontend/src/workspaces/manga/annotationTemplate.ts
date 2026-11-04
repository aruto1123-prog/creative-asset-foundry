// 既存原稿アノテーション（PageAnnotation）の雛形。見開き2ページ（右→左）の典型的なコマ割りを記述した例。

import type { PageAnnotation } from "../../api/types";

export const ANNOTATION_TEMPLATE: PageAnnotation = {
  title: "既存原稿（見開き2ページ）",
  first_page_side: "right",
  notes: "座標はページ左上を原点とするピクセル（width × height のページ座標）。台詞が読めない場合は text を空にして char_count に推定文字数を入れる。",
  pages: [
    {
      width: 1000,
      height: 1414,
      side: "right",
      panels: [
        {
          bbox: [520, 90, 930, 470],
          shot: "ELS",
          angle: "high",
          character_count: 1,
          ink_ratio: 0.18,
          balloons: [{ text: "ここが、すべての始まりの場所だ。", bbox: [820, 110, 910, 300], kind: "monologue" }],
          note: "状況提示の俯瞰ロング",
        },
        {
          bbox: [70, 90, 500, 470],
          shot: "MS",
          angle: "eye_level",
          character_count: 2,
          ink_ratio: 0.22,
          balloons: [
            { text: "本当に戻ってきたのか？", bbox: [380, 110, 470, 300], kind: "normal" },
            { text: "約束を果たしに来ただけだ。", bbox: [120, 130, 210, 330], kind: "normal" },
          ],
        },
        {
          bbox: [70, 500, 930, 900],
          shot: "CU",
          angle: "low",
          character_count: 1,
          ink_ratio: 0.35,
          sfx_count: 1,
          balloons: [{ text: "", char_count: 9, bbox: [760, 520, 900, 760], kind: "shout" }],
          note: "決意の表情を大きく",
        },
        {
          bbox: [520, 930, 930, 1304],
          shot: "MS",
          angle: "eye_level",
          character_count: 2,
          ink_ratio: 0.2,
          balloons: [{ text: "なら、案内しよう。", bbox: [800, 950, 900, 1120], kind: "normal" }],
        },
        {
          bbox: [70, 930, 500, 1304],
          shot: "FS",
          angle: "eye_level",
          character_count: 2,
          ink_ratio: 0.16,
          balloons: [],
          note: "無言で歩き出す（間）",
        },
      ],
    },
    {
      width: 1000,
      height: 1414,
      side: "left",
      panels: [
        {
          bbox: [70, 90, 930, 360],
          shot: "ELS",
          angle: "eye_level",
          character_count: 0,
          ink_ratio: 0.42,
          balloons: [],
        },
        {
          bbox: [520, 390, 930, 760],
          shot: "CU",
          angle: "eye_level",
          character_count: 1,
          ink_ratio: 0.25,
          balloons: [{ text: "……誰かが、ずっと待っていた。", bbox: [820, 410, 910, 640], kind: "monologue" }],
        },
        {
          bbox: [70, 390, 500, 760],
          shot: "MS",
          angle: "dutch",
          character_count: 1,
          ink_ratio: 0.3,
          balloons: [{ text: "その扉を開けるな！", bbox: [380, 410, 480, 620], kind: "shout" }],
        },
        {
          bbox: [70, 790, 930, 1304],
          shot: "ECU",
          angle: "low",
          character_count: 1,
          ink_ratio: 0.55,
          sfx_count: 2,
          balloons: [],
          intensity_hint: 0.9,
          note: "めくり前の引き：扉の向こうの影",
        },
      ],
    },
  ],
};

export function annotationTemplateText(): string {
  return JSON.stringify(ANNOTATION_TEMPLATE, null, 2);
}
