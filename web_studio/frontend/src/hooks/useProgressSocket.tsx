// /ws/progress への WebSocket 接続（自動再接続・ping・ジョブ進捗購読）。

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { JobEvent } from "../api/types";

type Listener = (event: JobEvent) => void;

interface ProgressContextValue {
  connected: boolean;
  jobs: Record<string, JobEvent>;
  lastEvent: JobEvent | null;
  subscribe: (listener: Listener) => () => void;
}

const ProgressContext = createContext<ProgressContextValue | null>(null);

function socketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws/progress`;
}

function isJobEvent(value: unknown): value is JobEvent {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return record.type === "job" && typeof record.job_id === "string";
}

export function ProgressSocketProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [jobs, setJobs] = useState<Record<string, JobEvent>>({});
  const [lastEvent, setLastEvent] = useState<JobEvent | null>(null);
  const listeners = useRef(new Set<Listener>());
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let disposed = false;
    let retry = 0;
    let reconnectTimer: number | undefined;
    let pingTimer: number | undefined;

    const connect = () => {
      if (disposed) {
        return;
      }
      let socket: WebSocket;
      try {
        socket = new WebSocket(socketUrl());
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;
      socket.onopen = () => {
        retry = 0;
        setConnected(true);
        pingTimer = window.setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "ping" }));
          }
        }, 20000);
      };
      socket.onmessage = (message: MessageEvent<string>) => {
        let payload: unknown;
        try {
          payload = JSON.parse(message.data);
        } catch {
          return;
        }
        if (!isJobEvent(payload)) {
          return;
        }
        const event = payload;
        setJobs((previous) => ({ ...previous, [event.job_id]: event }));
        setLastEvent(event);
        listeners.current.forEach((listener) => listener(event));
      };
      socket.onclose = () => {
        setConnected(false);
        if (pingTimer !== undefined) {
          window.clearInterval(pingTimer);
          pingTimer = undefined;
        }
        scheduleReconnect();
      };
      socket.onerror = () => {
        socket.close();
      };
    };

    const scheduleReconnect = () => {
      if (disposed) {
        return;
      }
      retry += 1;
      const delay = Math.min(10000, 500 * 2 ** Math.min(retry, 5));
      reconnectTimer = window.setTimeout(connect, delay);
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) {
        window.clearTimeout(reconnectTimer);
      }
      if (pingTimer !== undefined) {
        window.clearInterval(pingTimer);
      }
      const socket = socketRef.current;
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, []);

  const subscribe = useCallback((listener: Listener) => {
    listeners.current.add(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  const value = useMemo(() => ({ connected, jobs, lastEvent, subscribe }), [connected, jobs, lastEvent, subscribe]);
  return <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>;
}

export function useProgressSocket(): ProgressContextValue {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error("useProgressSocket は ProgressSocketProvider の内側で使用してください");
  }
  return context;
}

/** 指定ジョブの最新進捗イベント（WebSocket 経由）。 */
export function useJobProgress(jobId: string | null): JobEvent | null {
  const { jobs } = useProgressSocket();
  if (!jobId) {
    return null;
  }
  return jobs[jobId] ?? null;
}
