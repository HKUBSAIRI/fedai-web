import { useEffect, useState } from 'react';

export interface Envelope<T> { mode: 'postgres' | 'demo'; demo: boolean; data: T }
export type Row = Record<string, unknown>;

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly unavailable = false) { super(message); }
}

export async function api<T>(path: string, params: Record<string, unknown> = {}): Promise<Envelope<T>> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') qs.set(k, String(v));
  const res = await fetch(`/api${path}${qs.size ? `?${qs}` : ''}`);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(body.error ?? `HTTP ${res.status}`, res.status, Boolean(body.unavailable));
  return body as Envelope<T>;
}

export function useApi<T>(path: string | null, params: Record<string, unknown> = {}) {
  const key = path === null ? null : `${path}?${JSON.stringify(params)}`;
  const [state, setState] = useState<{ data?: T; error?: ApiError; loading: boolean }>({ loading: true });
  useEffect(() => {
    if (key === null) return;
    let live = true;
    setState((s) => ({ data: s.data, loading: true }));
    api<T>(path!, params)
      .then((r) => live && setState({ data: r.data, loading: false }))
      .catch((e) => live && setState({ error: e as ApiError, loading: false }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
