import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

// Optional local .env (git-ignored). Real credentials should preferably live in ~/.pgpass.
const envFile = path.join(ROOT, '.env');
if (fs.existsSync(envFile)) process.loadEnvFile(envFile);

export type DataMode = 'auto' | 'demo' | 'postgres';

export interface Config {
  host: '127.0.0.1';
  port: number;
  dataMode: DataMode;
  schema: string;
  statementTimeoutMs: number;
}

export function loadConfig(env = process.env): Config {
  const host = env.FEDAI_HOST ?? '127.0.0.1';
  if (host !== '127.0.0.1') {
    throw new Error('FEDAI_HOST must be 127.0.0.1 — this service must not listen on public interfaces');
  }
  const mode = (env.FEDAI_DATA_MODE ?? 'auto') as DataMode;
  if (!['auto', 'demo', 'postgres'].includes(mode)) throw new Error(`invalid FEDAI_DATA_MODE: ${mode}`);
  const schema = env.FEDAI_SCHEMA ?? 'fedai_corpus';
  if (!/^[a-z_][a-z0-9_]*$/.test(schema)) throw new Error('invalid FEDAI_SCHEMA');
  return {
    host: '127.0.0.1',
    port: Number(env.FEDAI_PORT ?? 8787),
    dataMode: mode,
    schema,
    statementTimeoutMs: Number(env.FEDAI_STATEMENT_TIMEOUT_MS ?? 15000),
  };
}

/** True if any libpq connection setting is present (values are never read for logging). */
export function hasPgSettings(env = process.env): boolean {
  return Boolean(env.PGHOST || env.PGDATABASE);
}
