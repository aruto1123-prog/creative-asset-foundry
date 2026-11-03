// 非同期処理の実行状態（実行中・エラー）とデータ取得の補助フック。

import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "../api/client";
import { useToast } from "./useToast";

/** ボタン操作などの非同期アクション。失敗時はトーストでエラーを表示する。 */
export function useAction<Args extends unknown[], R>(
  action: (...args: Args) => Promise<R>,
  options: { successMessage?: string | ((result: R) => string) } = {},
) {
  const { push } = useToast();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const actionRef = useRef(action);
  actionRef.current = action;
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const run = useCallback(
    async (...args: Args): Promise<R | undefined> => {
      setRunning(true);
      setError(null);
      try {
        const result = await actionRef.current(...args);
        const success = optionsRef.current.successMessage;
        if (success) {
          push("success", typeof success === "function" ? success(result) : success);
        }
        return result;
      } catch (reason) {
        const message = errorMessage(reason);
        setError(message);
        push("error", message);
        return undefined;
      } finally {
        setRunning(false);
      }
    },
    [push],
  );

  return { run, running, error };
}

/** 依存値が変わるたびに取得し直すデータ取得フック。 */
export function useResource<T>(loader: () => Promise<T>, deps: unknown[], enabled = true) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    loaderRef
      .current()
      .then((result) => {
        if (!cancelled) {
          setData(result);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(errorMessage(reason));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [...deps, version, enabled]);

  const reload = useCallback(() => setVersion((value) => value + 1), []);
  return { data, setData, loading, error, reload };
}
