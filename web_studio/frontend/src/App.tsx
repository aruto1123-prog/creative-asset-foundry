import { useEffect, useState } from "react";

interface HealthInfo {
  status: string;
  version?: string;
  llm_provider?: string;
}

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`))))
      .then((data: HealthInfo) => setHealth(data))
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  return (
    <main className="flex h-full items-center justify-center">
      <section className="rounded-xl border border-ink-700 bg-ink-900 p-8 shadow-xl">
        <h1 className="text-2xl font-semibold text-accent-strong">Creative Asset Foundry</h1>
        <p className="mt-2 text-sm text-ink-300">
          {health ? `backend: ${health.status} / LLM: ${health.llm_provider ?? "-"}` : error ? `backend 未接続: ${error}` : "接続確認中…"}
        </p>
      </section>
    </main>
  );
}
