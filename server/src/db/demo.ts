// DEMO MODE ONLY. Seeds a private, local PostgreSQL instance (embedded-postgres) with the table layout
// documented in FedAI_SEP/07_数据库结构与字段字典.md, filled with obviously fictitious rows.
// Nothing here is real FOMC data and none of these numbers may be reported as findings.
import fs from 'node:fs';
import path from 'node:path';
import { randomBytes } from 'node:crypto';
import EmbeddedPostgres from 'embedded-postgres';
import pg from 'pg';
import { ROOT } from '../config.js';
import { PgQueryer } from './queryer.js';
import { loadTranscriptMapping, type TranscriptMapping } from './schema.js';

// Demo uses the SAME transcript mapping file as the real database.
export const DEMO_TRANSCRIPT_MAPPING: TranscriptMapping = loadTranscriptMapping()!;

// Mirrors the real column layout from docs/schema_introspection.md (subset of tables the site reads).
const DDL = `
CREATE SCHEMA fedai_corpus;
CREATE TABLE fedai_corpus.meetings (
  meeting_id uuid PRIMARY KEY, meeting_date date NOT NULL, meeting_end_date date,
  meeting_type text NOT NULL, active_version_id uuid);
CREATE TABLE fedai_corpus.import_versions (
  version_id uuid PRIMARY KEY, meeting_id uuid NOT NULL, fingerprint text NOT NULL, pipeline_version text NOT NULL,
  manifest jsonb NOT NULL, report jsonb NOT NULL, imported_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE fedai_corpus.persons (
  person_id uuid PRIMARY KEY, identity_key text NOT NULL, canonical_name text, display_name text NOT NULL,
  identity_status text NOT NULL);
CREATE TABLE fedai_corpus.meeting_participants (
  version_id uuid NOT NULL, person_id uuid NOT NULL, meeting_role text, affiliation text, source_flags jsonb NOT NULL);
CREATE TABLE fedai_corpus.transcript_turns (
  turn_id uuid PRIMARY KEY, version_id uuid NOT NULL, document_id uuid NOT NULL, person_id uuid NOT NULL,
  sequence_number integer NOT NULL, speaker_raw text NOT NULL, body_raw text NOT NULL, body_clean text NOT NULL,
  source_start integer NOT NULL, body_start integer NOT NULL, source_end integer NOT NULL, source_page integer,
  body_clean_legacy text, cleaner_version text NOT NULL DEFAULT 'layout_only_v2',
  search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(body_clean, ''))) STORED);
CREATE INDEX transcript_turns_search_idx ON fedai_corpus.transcript_turns USING gin (search_tsv);
CREATE TABLE fedai_corpus.review_issues (
  issue_id uuid PRIMARY KEY, version_id uuid NOT NULL, issue_kind text NOT NULL, details jsonb NOT NULL);
CREATE TABLE fedai_corpus.sep_batches (
  batch_id uuid PRIMARY KEY, meeting_id uuid NOT NULL, meeting_version_id uuid NOT NULL,
  source_meeting_start date NOT NULL, source_meeting_end date NOT NULL, source_fingerprint text NOT NULL,
  processing_version text NOT NULL, source_manifest jsonb NOT NULL DEFAULT '{}'::jsonb,
  imported_at timestamptz NOT NULL DEFAULT now(), notes text);
CREATE TABLE fedai_corpus.sep_participants (
  batch_id uuid NOT NULL, participant_number integer NOT NULL, person_id uuid, source_name text NOT NULL,
  source_affiliation text, mapping_status text NOT NULL DEFAULT 'pending', mapping_notes text);
CREATE TABLE fedai_corpus.sep_questions (
  batch_id uuid NOT NULL, question_code text NOT NULL, question_text text, question_order integer);
CREATE TABLE fedai_corpus.sep_answers (
  batch_id uuid NOT NULL, question_code text NOT NULL, participant_number integer NOT NULL,
  response_status text NOT NULL, answer text, reference_doc text, reference_doc_name text,
  source_file_name text NOT NULL, source_record_number integer, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE fedai_corpus.sep_projections (
  batch_id uuid NOT NULL, participant_number integer NOT NULL, projection_horizon text NOT NULL,
  projection_year smallint, is_long_run boolean NOT NULL, gdp_growth numeric, unemployment_rate numeric,
  pce_inflation numeric, core_pce_inflation numeric, federal_funds_rate numeric,
  source_missing_fields jsonb NOT NULL DEFAULT '[]'::jsonb, source_file_name text NOT NULL,
  source_record_number integer, created_at timestamptz NOT NULL DEFAULT now());
`;
// DEMO-only approximations of the team's views (real definitions are not known to this project).
const VIEWS = `
CREATE VIEW fedai_corpus.current_turns AS
  SELECT m.meeting_date, t.turn_id, t.version_id, t.document_id, t.person_id, t.sequence_number, t.speaker_raw,
         t.body_raw, t.body_clean, t.source_start, t.body_start, t.source_end, t.source_page, p.display_name, p.identity_status
    FROM fedai_corpus.transcript_turns t JOIN fedai_corpus.meetings m ON m.active_version_id = t.version_id
    JOIN fedai_corpus.persons p ON p.person_id = t.person_id;
CREATE VIEW fedai_corpus.current_review_issues AS
  SELECT m.meeting_date, r.issue_id, r.version_id, r.issue_kind, r.details
    FROM fedai_corpus.review_issues r JOIN fedai_corpus.meetings m ON m.active_version_id = r.version_id;
CREATE VIEW fedai_corpus.current_sep_projections AS
  SELECT b.source_meeting_start AS meeting_date, pr.* FROM fedai_corpus.sep_projections pr
    JOIN fedai_corpus.sep_batches b ON b.batch_id = pr.batch_id;
`;

/** Small deterministic PRNG so demo content is stable across restarts and in tests. */
function rng(seed: number) {
  return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
}

const WORDS = ['inflation', 'labor market', 'balance sheet', 'forward guidance', 'growth outlook',
  'financial conditions', 'policy rate', 'uncertainty', 'productivity', 'housing'];

/** Returns the full seed script (DDL + fictitious rows) for the throw-away demo instance. */
export function buildDemoSeedSql(): string[] {
  const r = rng(42);
  const pick = <T>(a: T[]) => a[Math.floor(r() * a.length)];
  const persons = Array.from({ length: 14 }, (_, i) => ({ id: randomUUIDDet(r), name: `Demo Member ${String.fromCharCode(65 + i)}`, status: 'resolved' }));
  // Deliberate demo anomalies: an alias with a second UUID, and a non-member staff speaker.
  persons.push({ id: randomUUIDDet(r), name: 'Chair Demo Member A', status: 'alias_candidate' });
  const staff = { id: randomUUIDDet(r), name: 'Demo Staff Speaker', status: 'unresolved' };
  persons.push(staff);
  const ins: string[] = [];
  const lit = (v: unknown) => (v === null ? 'NULL' : typeof v === 'number' ? String(v) : `'${String(v).replace(/'/g, "''")}'`);
  for (const p of persons) {
    ins.push(`INSERT INTO fedai_corpus.persons VALUES (${lit(p.id)}, ${lit('demo:' + p.name.toLowerCase().replace(/ /g, '_'))},
      ${p.status === 'resolved' ? lit(p.name) : 'NULL'}, ${lit(p.name)}, ${lit(p.status)})`);
  }

  const sepMonths = new Set([3, 6, 9, 12]);
  let utter = 0;
  const turns = (vid: string, docId: string, start: string, attendees: typeof persons, n: number, tag: string) => {
    for (let s = 1; s <= n; s++) {
      const p = s === n ? staff : pick(attendees);
      const text = `[DEMO text #${++utter}${tag}] Fictional remark about ${pick(WORDS)} and ${pick(WORDS)}. Not a real FOMC statement.`;
      ins.push(`INSERT INTO fedai_corpus.transcript_turns VALUES (${lit(randomUUIDDet(r))}, ${lit(vid)}, ${lit(docId)}, ${lit(p.id)},
        ${s}, ${lit(p.name.toUpperCase() + '.')}, ${lit(text)}, ${lit(text)}, ${s * 100}, ${s * 100 + 10}, ${s * 100 + 90}, ${1 + s}, NULL, 'layout_only_v2')`);
    }
  };
  for (const year of [2012, 2013, 2014]) {
    for (const month of [1, 3, 4, 6, 7, 9, 10, 12]) {
      const day = 10 + Math.floor(r() * 10);
      const start = `${year}-${String(month).padStart(2, '0')}-${day}`;
      const end = `${year}-${String(month).padStart(2, '0')}-${day + 1}`;
      const mid = randomUUIDDet(r);
      const vid = randomUUIDDet(r);
      ins.push(`INSERT INTO fedai_corpus.meetings VALUES (${lit(mid)}, ${lit(start)}, ${lit(end)}, 'DEMO scheduled', ${lit(vid)})`);
      ins.push(`INSERT INTO fedai_corpus.import_versions VALUES (${lit(vid)}, ${lit(mid)}, 'demo', 'demo-v2', '{}', '{}', now())`);
      const attendees = persons.slice(0, 10 + Math.floor(r() * 4));
      // Roles like the real data: Chair, Vice Chair and Voters recorded; non-voting participants with an affiliation but no role;
      // the last two attendees have neither (staff).
      attendees.forEach((p, i) => {
        const role = i === 0 ? 'Chair' : i === 1 ? 'Vice Chair' : i < 7 ? 'Voter' : null;
        const aff = i < attendees.length - 2 ? 'DEMO Bank' : null;
        ins.push(`INSERT INTO fedai_corpus.meeting_participants VALUES (${lit(vid)}, ${lit(p.id)}, ${lit(role)}, ${lit(aff)}, '{}')`);
      });
      turns(vid, randomUUIDDet(r), start, attendees, 12, '');
      if (month === 1) {
        // Superseded import version: its turns must NOT appear on the site.
        const old = randomUUIDDet(r);
        ins.push(`INSERT INTO fedai_corpus.import_versions VALUES (${lit(old)}, ${lit(mid)}, 'demo-old', 'demo-v1', '{}', '{}', now())`);
        turns(old, randomUUIDDet(r), start, attendees, 5, ' OLD-VERSION');
        ins.push(`INSERT INTO fedai_corpus.review_issues VALUES (${lit(randomUUIDDet(r))}, ${lit(vid)}, 'demo_speaker_unresolved', '{"note":"DEMO"}')`);
      }
      if (year === 2014 && month > 6) continue; // demo coverage gap, visible on Data Quality
      if (!sepMonths.has(month)) continue;

      const bid = randomUUIDDet(r);
      const file = `DEMO_FOMC${end.replace(/-/g, '')}SEPcompilation.pdf`;
      ins.push(`INSERT INTO fedai_corpus.sep_batches VALUES (${lit(bid)}, ${lit(mid)}, ${lit(vid)}, ${lit(start)}, ${lit(end)},
        ${lit('demo-sha256-' + bid.slice(0, 8))}, 'demo-v0', '{}', now(), 'DEMO batch')`);
      const order = [...attendees].sort(() => r() - 0.5);
      order.forEach((p, i) => {
        const unmatched = i === order.length - 1 && month === 12;
        ins.push(`INSERT INTO fedai_corpus.sep_participants VALUES (${lit(bid)}, ${i + 1}, ${unmatched ? 'NULL' : lit(p.id)},
          ${lit(p.name)}, 'DEMO Reserve Bank', ${lit(unmatched ? 'pending' : 'matched')}, ${lit(unmatched ? 'DEMO: to be checked' : null)})`);
      });
      const qs = ['Q1', 'Q2a', 'Q2b', 'Q3', 'Q4'];
      qs.forEach((q, qi) => {
        ins.push(`INSERT INTO fedai_corpus.sep_questions VALUES (${lit(bid)}, ${lit(q)}, ${lit(`[DEMO] Fictional SEP question ${q}: describe the risks around your ${pick(WORDS)} projection.`)}, ${qi + 1})`);
        order.forEach((_, i) => {
          if (q === 'Q4' && i === 0 && month === 9) return; // demo grid gap
          const x = r();
          const status = x < 0.75 ? 'answered' : x < 0.9 ? 'na' : 'missing';
          const answer = status === 'answered' ? `[DEMO answer] Fictional view on ${pick(WORDS)}; risks ${x < 0.4 ? 'balanced' : 'weighted to the downside'}.`
            : status === 'na' ? 'Not applicable.' : null;
          ins.push(`INSERT INTO fedai_corpus.sep_answers VALUES (${lit(bid)}, ${lit(q)}, ${i + 1}, ${lit(status)}, ${lit(answer)},
            NULL, NULL, ${lit(file)}, NULL, now())`);
        });
      });
      order.forEach((_, i) => {
        for (const [k, yr] of [year, year + 1, year + 2, null].entries()) {
          const v = (base: number, spread: number) => Math.round((base + (r() - 0.5) * spread) * 10) / 10;
          const noCore = i === 0 && k === 2;
          ins.push(`INSERT INTO fedai_corpus.sep_projections VALUES (${lit(bid)}, ${i + 1}, ${lit(yr === null ? 'longer_run' : String(yr))},
            ${yr === null ? 'NULL' : yr}, ${yr === null}, ${v(2.4 - k * 0.1, 1)}, ${v(7 - k * 0.6, 0.8)}, ${v(1.6 + k * 0.1, 0.6)},
            ${noCore || yr === null ? 'NULL' : v(1.7 + k * 0.08, 0.5)}, ${v(0.3 + k * 1.1, 1)},
            ${lit(noCore ? '["core_pce_inflation"]' : '[]')}, ${lit(file)}, NULL, now())`);
        }
      });
    }
  }
  return [DDL, ...ins, VIEWS];
}

function randomUUIDDet(r: () => number): string {
  const h = () => Math.floor(r() * 0x10000).toString(16).padStart(4, '0');
  return `${h()}${h()}-${h()}-4${h().slice(1)}-a${h().slice(1)}-${h()}${h()}${h()}`;
}

export const DEMO_PORT = 54329;
const DEMO_DIR = path.join(ROOT, '.demo-pg');

/**
 * Starts a private PostgreSQL instance (project-local binaries, 127.0.0.1 only, data dir under
 * .demo-pg/, recreated on every start), seeds it as an admin role, then returns a PgQueryer
 * connected as a separate SELECT-only role — the same code path used for the real database.
 */
export async function startDemoDatabase(statementTimeoutMs: number, port = DEMO_PORT, dir = DEMO_DIR) {
  fs.rmSync(dir, { recursive: true, force: true });
  const adminPassword = randomBytes(18).toString('hex'); // in-memory only, never written or logged
  const roPassword = randomBytes(18).toString('hex');
  const server = new EmbeddedPostgres({
    databaseDir: path.join(dir, 'data'),
    user: 'demo_admin',
    password: adminPassword,
    port,
    persistent: false,
    postgresFlags: ['-c', 'listen_addresses=127.0.0.1', '-c', 'unix_socket_directories='],
    onLog: () => undefined,
    onError: () => undefined,
  });
  // embedded-postgres creates its own pg client, which would pick up the real DB's PG* env vars.
  const saved = Object.fromEntries(Object.entries(process.env).filter(([k]) => k.startsWith('PG')));
  for (const k of Object.keys(saved)) delete process.env[k];
  try {
    await server.initialise();
    await server.start();
    await server.createDatabase('fedai_demo');
  } finally {
    Object.assign(process.env, saved);
  }

  const admin = new pg.Client({ host: '127.0.0.1', port, user: 'demo_admin', password: adminPassword, database: 'fedai_demo', ssl: false });
  await admin.connect();
  try {
    for (const stmt of buildDemoSeedSql()) await admin.query(stmt);
    await admin.query(`CREATE ROLE demo_readonly LOGIN PASSWORD '${roPassword}'`);
    await admin.query('GRANT CONNECT ON DATABASE fedai_demo TO demo_readonly');
    await admin.query('GRANT USAGE ON SCHEMA fedai_corpus TO demo_readonly');
    await admin.query('GRANT SELECT ON ALL TABLES IN SCHEMA fedai_corpus TO demo_readonly');
  } finally {
    await admin.end();
  }
  const queryer = new PgQueryer(statementTimeoutMs, {
    host: '127.0.0.1', port, user: 'demo_readonly', password: roPassword, database: 'fedai_demo', ssl: false,
  });
  return { queryer, admin: { port, password: adminPassword }, readonly: { password: roPassword }, stop: async () => { await queryer.close(); await server.stop(); } };
}
