// 画像アップロード補助: ブラウザが読める任意形式（JPEG / PNG / WebP / GIF など）の画像を canvas で
// 白背景に合成・長辺を上限まで縮小・グレースケール化し、PNG の data URL として返す。

export interface PreparedImage {
  name: string;
  dataUrl: string;
  width: number;
  height: number;
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("画像を読み込めませんでした（ブラウザが対応していない形式の可能性があります）"));
    image.src = src;
  });
}

/** 画像 URL（object URL / data URL / 同一オリジン URL）をグレースケール PNG の data URL へ変換する。 */
export async function imageSourceToGrayscalePng(src: string, name: string, maxSide = 1600): Promise<PreparedImage> {
  const image = await loadImageElement(src);
  const naturalW = image.naturalWidth || image.width;
  const naturalH = image.naturalHeight || image.height;
  if (!naturalW || !naturalH) {
    throw new Error(`${name}: 画像サイズを取得できませんでした`);
  }
  const scale = Math.min(1, maxSide / Math.max(naturalW, naturalH));
  const width = Math.max(1, Math.round(naturalW * scale));
  const height = Math.max(1, Math.round(naturalH * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("canvas の 2D コンテキストを取得できません");
  }
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height);
  const data = pixels.data;
  for (let i = 0; i < data.length; i += 4) {
    const luma = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    data[i] = luma;
    data[i + 1] = luma;
    data[i + 2] = luma;
    data[i + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  return { name, dataUrl: canvas.toDataURL("image/png"), width, height };
}

/** File（ドラッグ&ドロップ・ファイル選択）をグレースケール PNG へ変換する。 */
export async function fileToGrayscalePng(file: File, maxSide = 1600): Promise<PreparedImage> {
  const url = URL.createObjectURL(file);
  try {
    return await imageSourceToGrayscalePng(url, file.name, maxSide);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** 同一オリジンの画像 URL を取得してグレースケール PNG へ変換する。 */
export async function urlToGrayscalePng(url: string, name: string, maxSide = 1600): Promise<PreparedImage> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`画像を取得できませんでした（HTTP ${response.status}）`);
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    return await imageSourceToGrayscalePng(objectUrl, name, maxSide);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/** ファイル名の拡張子・MIME から画像らしいものだけを残す。 */
export function isImageFile(file: File): boolean {
  if (file.type.startsWith("image/")) {
    return true;
  }
  return /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(file.name);
}
