// 構造一覧（セレクト用）の共有取得フック。

import { useCallback, useEffect, useState } from "react";
import { api, errorMessage } from "../api/client";
import type { StructureSummary } from "../api/types";

export function useStructureOptions(refreshKey = 0) {
  const [items, setItems] = useState<StructureSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.structures
      .list({ limit: 500 })
      .then((result) => {
        if (!cancelled) {
          setItems(result.items);
          setError(null);
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
  }, [refreshKey, version]);

  const reload = useCallback(() => setVersion((value) => value + 1), []);
  return { items, loading, error, reload };
}
