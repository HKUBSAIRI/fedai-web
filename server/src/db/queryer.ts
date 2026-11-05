import fs from 'node:fs';
import pg from 'pg';
import { assertSelectOnly } from './sqlGuard.js';

// Return DATE columns as plain 'YYYY-MM-DD' strings (no timezone shifting).
pg.types.setTypeParser(1082, (v) => v);

export type Row = Record<string, unknown>;

export interface Queryer {
  query<T extends Row = Row>(sql: string, params?: unknown[]): Promise<T[]>;
  close(): Promise<void>;
}

export type Transport = 'tls-verified' | 'tls-unverified' | 'plaintext-acknowledged';

/**
 * Resolve TLS settings explicitly. node-pg does NOT implement libpq's `prefer` fallback (it treats
 * prefer like require), so we never rely on its env parsing. Plaintext is only allowed with an
 * explicit, separate acknowledgement (FEDAI_ALLOW_PLAINTEXT_DB=yes).
 */
export function resolveSsl(env = process.env): { ssl: pg.PoolConfig['ssl']; transport: Transport } {
  const mode = env.PGSSLMODE ?? 'require';
  if (mode === 'disable') {
    if (env.FEDAI_ALLOW_PLAINTEXT_DB !== 'yes') {
      throw new Error('PGSSLMODE=disable also requires FEDAI_ALLOW_PLAINTEXT_DB=yes (plaintext connections need the owner’s approval)');
    }
    return { ssl: false, transport: 'plaintext-acknowledged' };
  }
  if (mode === 'verify-full' || mode === 'verify-ca') {
    if (!env.PGSSLROOTCERT) throw new Error(`PGSSLMODE=${mode} requires PGSSLROOTCERT (path to the CA certificate)`);
    return {
      ssl: { ca: fs.readFileSync(env.PGSSLROOTCERT, 'utf8'), rejectUnauthorized: true,
        ...(mode === 'verify-ca' ? { checkServerIdentity: () => undefined } : {}) },
      transport: 'tls-verified',
    };
  }
  if (mode === 'prefer' || mode === 'require' || mode === 'allow') {
    return { ssl: { rejectUnauthorized: false }, transport: 'tls-unverified' };
  }
  throw new Error(`Unsupported PGSSLMODE: ${mode}`);
}

/**
 * Every call is guarded and runs on a session whose transactions are read-only (default_transaction_read_only=on,
 * set when the connection opens), with a read-only database role verified at startup. There is no explicit
 * BEGIN READ ONLY / ROLLBACK around each statement: with the database 40 ms away that tripled every query's latency
 * and added no protection beyond the session setting.
 */
export class PgQueryer implements Queryer {
  private pool: pg.Pool;
  readonly transport: Transport;

  constructor(statementTimeoutMs: number, explicit?: pg.PoolConfig) {
    const resolved = explicit ? { ssl: explicit.ssl ?? false, transport: 'plaintext-acknowledged' as Transport } : resolveSsl();
    this.transport = explicit ? (explicit.ssl ? 'tls-unverified' : 'plaintext-acknowledged') : resolved.transport;
    // Connection parameters come exclusively from libpq env vars (PGHOST, PGPORT, PGDATABASE,
    // PGUSER, PGPASSWORD / ~/.pgpass, PGSSLMODE). Nothing is hard-coded or logged.
    this.pool = new pg.Pool({
      ...explicit,
      ssl: resolved.ssl,
      // the read-only role has a small connection limit shared with other users of the role; extra queries wait in the pool
      max: Number(process.env.FEDAI_DB_POOL_MAX ?? 4),
      connectionTimeoutMillis: 8000,
      idleTimeoutMillis: 30000,
      application_name: 'fedai-web-readonly',
      options: `-c default_transaction_read_only=on -c statement_timeout=${statementTimeoutMs}`,
    });
    this.pool.on('error', () => { /* idle client errors are surfaced on next query */ });
  }

  async query<T extends Row = Row>(sql: string, params: unknown[] = []): Promise<T[]> {
    assertSelectOnly(sql);
    const res = await this.pool.query(sql, params);
    return res.rows as T[];
  }

  close(): Promise<void> {
    return this.pool.end();
  }
}
