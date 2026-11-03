// UI スモークテスト: ビルド済みスタジオを Playwright（Chromium）で巡回し、主要操作を実行してコンソールエラー0を確認する。
//
// 使い方（バックエンドを mock プロバイダで起動した状態で）:
//   CAF_UI_URL=http://127.0.0.1:8799 node scripts/ui-smoke.mjs
// 任意: CAF_UI_SCREENSHOTS=<保存先ディレクトリ> でワークスペースごとのスクリーンショットを保存する。

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const BASE_URL = process.env.CAF_UI_URL ?? "http://127.0.0.1:8799";
const SHOTS = process.env.CAF_UI_SCREENSHOTS ?? "";
const SAMPLE_TEXT =
  "王都の孤児院で育った少年は、師である宮廷魔術師に導かれ、奪われた故郷を取り戻すために学園へ入学する。" +
  "少年は師の教えどおりに禁術を学び、ライバルとの決闘にも勝利して頭角を現す。" +
  "しかし地下書庫で母の形見の鍵が開いた古文書から、故郷を焼いたのは師自身だったという真実が判明する。" +
  "裏切られた少年は全てを失い絶望するが、ライバルの助けを得て自らの力を代償に差し出し、師と王国の結界を打ち破る。" +
  "戦いの後、少年は故郷へは戻らず、新しい居場所を自分の手で築き始める。";

const consoleErrors = [];
const steps = [];

function log(message) {
  steps.push(message);
  console.log(`[ui-smoke] ${message}`);
}

async function shot(page, name) {
  if (!SHOTS) {
    return;
  }
  mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: false });
}

async function openWorkspace(page, label) {
  await page.locator("aside nav button", { hasText: label }).first().click();
  await page.waitForTimeout(300);
}

async function clickButton(scope, name, options = {}) {
  const exact = options.exact ?? false;
  const target = scope.getByRole("button", { name, exact }).or(scope.getByRole("tab", { name, exact })).first();
  await target.waitFor({ state: "visible", timeout: options.timeout ?? 20000 });
  await target.click();
}

let browser = null;

async function main() {
  browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));
  page.on("dialog", (dialog) => dialog.accept());

  await page.goto(BASE_URL, { waitUntil: "networkidle" });
  await page.getByText("Creative Asset Foundry").first().waitFor();
  await page.getByText("WebSocket 接続中").waitFor({ timeout: 15000 });
  log("シェル表示と WebSocket 接続を確認");

  // ---------------- Workspace 1: Structure Lab ----------------
  const main = page.locator("main");
  await main.getByText("プリセット骨格ブラウザー").waitFor();
  await main.locator("ul li button").first().waitFor({ timeout: 20000 });
  const count = await main.locator("ul li button").count();
  log(`構造ブラウザー: ${count} 件表示`);
  await main.locator("ul li button").nth(1).click();
  await main.getByText("Layer 1: Dynamic Tension Graph").waitFor({ timeout: 20000 });
  await clickButton(main, "L2 感情・情報波形");
  const handle = main.locator("svg[aria-label='感情・情報開示曲線'] circle").nth(8);
  const box = await handle.boundingBox();
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2, box.y - 40, { steps: 5 });
    await page.mouse.up();
  }
  await clickButton(main, "波形を保存");
  await page.getByText("波形を保存しました").waitFor({ timeout: 15000 });
  log("波形ドラッグ編集と保存");
  for (const tab of ["L3 ビート", "L4 認知メカニズム", "L5 ブループリント", "転用ガイド", "Mermaid", "L1 力学グラフ"]) {
    await clickButton(main, tab);
  }
  await shot(page, "1_structure_lab");

  await clickButton(main, "テキストから抽出");
  await main.getByLabel("抽出対象テキスト").fill(SAMPLE_TEXT);
  await clickButton(main, "抽出を実行");
  await main.getByText("抽出結果:").waitFor({ timeout: 60000 });
  log("テキストから5階層構造を抽出");

  await clickButton(main, "合成・変異");
  await clickButton(main, "合成を実行");
  await main.getByText("合成結果を開く").waitFor({ timeout: 30000 });
  await clickButton(main, "変異を実行");
  await main.getByText("変異結果を開く").waitFor({ timeout: 30000 });
  log("構造合成と変異");

  await clickButton(main, "異分野変換");
  await clickButton(main, "漫画プロット へ変換");
  await main.getByText("忠実度スコア").waitFor({ timeout: 30000 });
  log("漫画プロットへ変換");

  await clickButton(main, "類似検索");
  await clickButton(main, "検索", { exact: true });
  await main.getByText(/score \d/).first().waitFor({ timeout: 30000 });
  log("類似構造検索");

  // ---------------- Workspace 2: Style Alchemy Arena ----------------
  await openWorkspace(page, "Style Alchemy Arena");
  const arena = page.locator("main > div:not([hidden])");
  await clickButton(arena, "候補を生成して登録");
  await page.getByText(/候補を \d+ 件登録しました/).waitFor({ timeout: 30000 });
  await arena.locator("img[alt='候補A']").waitFor({ timeout: 20000 });
  await page.locator("h1", { hasText: "Style Alchemy Arena" }).click();
  for (const key of ["ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowLeft"]) {
    await page.keyboard.press(key);
    await page.waitForTimeout(450);
  }
  await arena.getByText(/このセッション [1-9]\d* 判定/).waitFor({ timeout: 15000 });
  log(`A/B 判定キー操作: ${await arena.getByText(/このセッション \d+ 判定/).innerText()}`);
  await shot(page, "2_style_arena");
  await clickButton(arena, "ランキング・選好ベクトル");
  await arena.getByText("Elo / Bradley-Terry ランキング").waitFor();
  await arena.locator("table tbody tr").first().waitFor({ timeout: 20000 });
  log("ランキング表示");
  await clickButton(arena, "ベイズ最適化");
  await clickButton(arena, "次のレシピを提案");
  await arena.getByText("提案レシピ").waitFor({ timeout: 60000 });
  await clickButton(arena, "レシピ／ComfyUI 出力");
  await arena.getByText("マージスクリプト（ローカル実行用 Python）").waitFor({ timeout: 30000 });
  log("ベイズ最適化提案とレシピ出力");
  await clickButton(arena, "データセット分析");
  const captions = [
    "img_001: 1girl, crying, from above, monochrome, screentone, simple background",
    "img_002: 1girl, crying, from above, upper body, sketch",
    "img_003: 1girl, crying, from above, full body, outdoors",
    "img_004: 1girl, smile, from side, close-up, indoors",
    "img_005: 1boy, angry, from below, cowboy shot, dramatic lighting",
    "img_006: 1girl, crying, from above, portrait, white background",
  ].join("\n");
  await arena.getByLabel("キャプション").fill(captions);
  await clickButton(arena, "分析を実行");
  await arena.getByText("概念の癒着（Entanglement）").waitFor({ timeout: 30000 });
  log("データセット共起分析");
  await clickButton(arena, "グリッド実験・トークン辞書");
  await clickButton(arena, "実験を作成");
  await arena.getByText(/^実験: /).waitFor({ timeout: 30000 });
  await arena.getByRole("button", { name: "記録" }).first().click();
  await page.getByText("スコアを記録しました").waitFor({ timeout: 15000 });
  log("グリッド実験の作成とスコア記録");

  // ---------------- Workspace 3: Manga Studio ----------------
  await openWorkspace(page, "Manga Studio");
  const manga = page.locator("main > div:not([hidden])");
  await clickButton(manga, "ネームを生成");
  await manga.getByText("SVG ネームプレビュー").waitFor({ timeout: 90000 });
  const svgImage = manga.locator("img[alt$='ページのネーム']");
  await svgImage.waitFor();
  await page.waitForFunction(() => {
    const img = document.querySelector("img[alt$='ページのネーム']");
    return img instanceof HTMLImageElement && img.complete && img.naturalWidth > 0;
  });
  log("ネーム生成と SVG プレビュー表示");
  await clickButton(manga, "見開き", { exact: true });
  await page.waitForTimeout(500);
  await clickButton(manga, "再検証");
  await page.getByText(/再検証しました/).waitFor({ timeout: 20000 });
  await shot(page, "3_manga_studio");
  await clickButton(manga, "ワークフロー生成");
  await page.getByText(/コマ分の ComfyUI ワークフローを生成しました/).waitFor({ timeout: 30000 });
  await clickButton(manga, "表示中ページを投入");
  await manga.getByText("ComfyUI 実行結果").waitFor({ timeout: 90000 });
  log("ComfyUI ワークフロー生成とモック ComfyUI への投入");

  // ---------------- Workspace 4: Asset Publisher ----------------
  await openWorkspace(page, "Asset Publisher");
  const publisher = page.locator("main > div:not([hidden])");
  await clickButton(publisher, "3軸企画を展開");
  await publisher.getByText("共通の核心洞察:").waitFor({ timeout: 60000 });
  await publisher.getByRole("button", { name: "このプランで執筆" }).first().click();
  await publisher.getByText(/一人称「私」統一（置換/).first().waitFor({ timeout: 120000 });
  await clickButton(publisher, "再検証");
  await page.getByText("再検証: 一人称・URL ともに準拠しています").waitFor({ timeout: 20000 });
  log("3軸企画展開と5段階執筆・コンプライアンス検証");
  await shot(page, "4_asset_publisher");
  await publisher.getByLabel("サニタイズ対象テキスト").fill("僕は https://example.com を見た。俺たちは [詳細](https://example.org) を参照した。");
  await clickButton(publisher, "サニタイズを実行");
  await publisher.getByText(/URL 完全除去（除去 [1-9]/).first().waitFor({ timeout: 20000 });
  log("サニタイズ試験");

  // ---------------- Workspace 5: System ----------------
  await openWorkspace(page, "System & Local AI");
  const system = page.locator("main > div:not([hidden])");
  await system.getByText("ローカル LLM（DeepSeek-V4.1-Flash）").waitFor({ timeout: 20000 });
  await system.getByRole("button", { name: "接続テスト" }).first().click();
  await system.getByText(/接続成功|接続失敗/).first().waitFor({ timeout: 30000 });
  await system.getByRole("button", { name: "保存" }).first().click();
  await page.getByText("LLM 設定を保存し、推論クライアントを再構築しました").waitFor({ timeout: 20000 });
  await clickButton(system, "バックアップ作成");
  await page.getByText(/バックアップを作成しました/).waitFor({ timeout: 20000 });
  log("LLM 接続テスト・設定保存・DB バックアップ");
  await shot(page, "5_system");

  await browser.close();
  const filtered = consoleErrors.filter((text) => !text.includes("favicon"));
  console.log(`[ui-smoke] 完了: ${steps.length} ステップ / コンソールエラー ${filtered.length} 件`);
  if (filtered.length > 0) {
    for (const text of filtered) {
      console.log(`  - ${text}`);
    }
    process.exitCode = 1;
  }
}

main()
  .catch((error) => {
    console.error(`[ui-smoke] 失敗: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
    if (consoleErrors.length > 0) {
      console.error("コンソールエラー:");
      for (const text of consoleErrors) {
        console.error(`  - ${text}`);
      }
    }
    process.exitCode = 1;
  })
  .finally(async () => {
    if (browser) {
      await browser.close().catch(() => undefined);
    }
  });
