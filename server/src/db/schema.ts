import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../config.js';
import type { Queryer } from './queryer.js';

/** table name -> ordered column names, discovered from information_schema (never guessed). */
export type SchemaInfo = Map<string, string[]>;

export async function discoverSchema(q: Queryer, schema: string): Promise<SchemaInfo> {
  const rows = await q.query<{ table_name: string; column_name: string }>(
    `SELECT table_name, column_name
       FROM information_schema.columns
      WHERE table_schema = $1
      ORDER BY table_name, ordinal_position`,
    [schema],
  );
  const info: SchemaInfo = new Map();
  for (const r of rows) {
    if (!info.has(r.table_name)) info.set(r.table_name, []);
    info.get(r.table_name)!.push(r.column_name);
  }
  return info;
}

export function hasColumns(info: SchemaInfo, table: string, cols: string[]): boolean {
  const have = info.get(table);
  return Boolean(have && cols.every((c) => have.includes(c)));
}

/**
 * Transcript tables have not been verified yet, so their mapping must be configured explicitly
 * (server/transcript-mapping.json) after running `npm run introspect`. Nothing is auto-guessed.
 */
export interface TranscriptMapping {
  table: string;
  idColumn: string;
  meetingKey: { column: string; references: 'meeting_id' | 'version_id' };
  textColumn: string;
  personIdColumn?: string | null;
  speakerNameColumn?: string | null;
  orderColumn?: string | null;
  sourceColumns?: string[];
}

export type MappingCheck =
  | { ok: true; mapping: TranscriptMapping }
  | { ok: false; reason: string };

export function loadTranscriptMapping(file = path.join(ROOT, 'server/transcript-mapping.json')): TranscriptMapping | null {
  if (!fs.existsSync(file)) return null;
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  return raw && raw.table ? (raw as TranscriptMapping) : null;
}

export function validateTranscriptMapping(info: SchemaInfo, m: TranscriptMapping | null): MappingCheck {
  if (!m) return { ok: false, reason: 'Transcript table not verified: server/transcript-mapping.json is not configured' };
  const cols = info.get(m.table);
  if (!cols) return { ok: false, reason: `Mapped table ${m.table} does not exist in the schema` };
  const needed = [m.idColumn, m.meetingKey.column, m.textColumn, m.personIdColumn, m.speakerNameColumn,
    m.orderColumn, ...(m.sourceColumns ?? [])].filter((c): c is string => Boolean(c));
  const missing = needed.filter((c) => !cols.includes(c));
  if (missing.length) return { ok: false, reason: `Mapped columns do not exist: ${m.table}.${missing.join(', ')}` };
  if (!['meeting_id', 'version_id'].includes(m.meetingKey.references)) {
    return { ok: false, reason: 'meetingKey.references must be meeting_id or version_id' };
  }
  return { ok: true, mapping: m };
}

/** Documented numeric-projection column names (FedAI_SEP projections_true.csv). */
export const PROJECTION_VARS = [
  { key: 'gdp_growth', label: 'GDP growth' },
  { key: 'unemployment_rate', label: 'Unemployment rate' },
  { key: 'pce_inflation', label: 'PCE inflation' },
  { key: 'core_pce_inflation', label: 'Core PCE inflation' },
  { key: 'federal_funds_rate', label: 'Federal funds rate' },
] as const;
