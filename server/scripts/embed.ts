/**
 * Generate the transcript passage vectors for the AI Q&A (fedai_corpus.rag_chunks, docs/rag/01_向量表.sql).
 *
 *   npx tsx server/scripts/embed.ts --dry-run    count passages and words only (no Azure calls, no writes)
 *   npx tsx server/scripts/embed.ts              embed new or changed turns and write them
 *
 * Needs AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY and, for writing, an account that may write rag_chunks:
 * FEDAI_RAG_PGUSER / FEDAI_RAG_PGPASSWORD (the website's read-only account cannot). Host, database and TLS come from
 * the usual PG* settings in .env. Safe to re-run: turns already embedded with the same text and model are skipped,
 * turns whose text changed are re-embedded, and passages of turns that are no longer active are removed.
 */
import pg from 'pg';
import '../src/config.js';   // loads .env
import { resolveSsl } from '../src/db/queryer.js';
import { chunkTurn, embed, embeddingInput, loadRagConfig, modelLabel, textHash, toVectorLiteral } from '../src/rag.js';

const DRY = process.argv.includes('--dry-run');
const BATCH = 100;            // passages per embeddings request (~50k tokens)
const PARALLEL = 4;           // requests in flight
const S = process.env.FEDAI_SCHEMA ?? 'fedai_corpus';
if (!/^[a-z_][a-z0-9_]*$/.test(S)) throw new Error('invalid FEDAI_SCHEMA');

const rag = loadRagConfig();
if (!rag && !DRY) throw new Error('Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY');
const model = rag ? modelLabel(rag) : 'dry-run';

const db = new pg.Client({
  ssl: resolveSsl().ssl,
  ...(process.env.FEDAI_RAG_PGUSER ? { user: process.env.FEDAI_RAG_PGUSER, password: process.env.FEDAI_RAG_PGPASSWORD } : {}),
  application_name: 'fedai-web-embed',
});
await db.connect();

interface Turn { turn_id: string; version_id: string; meeting_date: string; person_id: string | null; speaker: string; body: string }
const turns = (await db.query<Turn>(
  `SELECT tr.turn_id::text, tr.version_id::text, m.meeting_date::text, tr.person_id::text,
          COALESCE(p.display_name, tr.speaker_raw, 'Unknown speaker') AS speaker, COALESCE(tr.body_clean, '') AS body
     FROM ${S}.transcript_turns tr
     JOIN ${S}.meetings m ON m.active_version_id = tr.version_id
     LEFT JOIN ${S}.persons p ON p.person_id = tr.person_id
    ORDER BY m.meeting_date, tr.sequence_number`)).rows;

const have = new Map<string, string>();   // turn_id → text_hash|model already stored
if (!DRY) {
  for (const r of (await db.query(`SELECT DISTINCT turn_id::text, text_hash, model FROM ${S}.rag_chunks`)).rows) have.set(r.turn_id, `${r.text_hash}|${r.model}`);
}

interface Passage { turn: Turn; hash: string; index: number; content: string; input: string }
const todo: Passage[] = [];
let skippedShort = 0, unchanged = 0, words = 0;
for (const t of turns) {
  const hash = textHash(t.body);
  const parts = chunkTurn(t.body);
  if (!parts.length) { skippedShort++; continue; }
  if (have.get(t.turn_id) === `${hash}|${model}`) { unchanged++; continue; }
  parts.forEach((content, index) => {
    words += content.split(' ').length;
    todo.push({ turn: t, hash, index, content, input: embeddingInput(t.speaker, t.meeting_date, content) });
  });
}
console.log(`${turns.length} active turns: ${skippedShort} too short, ${unchanged} already embedded, `
  + `${todo.length} passages to embed (~${Math.round(words / 0.75 / 1e6 * 100) / 100}M tokens)`);

if (DRY) { await db.end(); process.exit(0); }

// Write one batch: replace all passages of the batch's turns. A turn's passages always fall in the same batch run,
// so they are grouped by turn first.
const byTurn = new Map<string, Passage[]>();
for (const p of todo) byTurn.set(p.turn.turn_id, [...(byTurn.get(p.turn.turn_id) ?? []), p]);
const batches: Passage[][] = [];
let cur: Passage[] = [];
for (const ps of byTurn.values()) {
  if (cur.length && cur.length + ps.length > BATCH) { batches.push(cur); cur = []; }
  cur.push(...ps);
}
if (cur.length) batches.push(cur);

let done = 0, tokens = 0;
async function run(batch: Passage[]) {
  const { vectors, tokens: t } = await embed(rag!, batch.map((p) => p.input), 120000);
  tokens += t;
  const values: unknown[] = [];
  const rows = batch.map((p, i) => {
    values.push(p.turn.turn_id, p.index, p.turn.version_id, p.turn.meeting_date, p.turn.person_id, p.content, p.hash, model, toVectorLiteral(vectors[i]));
    const b = i * 9;
    return `($${b + 1}::uuid, $${b + 2}, $${b + 3}::uuid, $${b + 4}::date, $${b + 5}::uuid, $${b + 6}, $${b + 7}, $${b + 8}, $${b + 9}::vector)`;
  });
  const ids = [...new Set(batch.map((p) => p.turn.turn_id))];
  const c = await pool.connect();
  try {
    await c.query('BEGIN');
    await c.query(`DELETE FROM ${S}.rag_chunks WHERE turn_id = ANY($1::uuid[])`, [ids]);
    await c.query(`INSERT INTO ${S}.rag_chunks (turn_id, chunk_index, version_id, meeting_date, person_id, content, text_hash, model, embedding)
                   VALUES ${rows.join(', ')}`, values);
    await c.query('COMMIT');
  } catch (e) {
    await c.query('ROLLBACK').catch(() => undefined);
    throw e;
  } finally { c.release(); }
  done += batch.length;
  if (done % 2000 < batch.length || done === todo.length) console.log(`  ${done}/${todo.length} passages, ${tokens.toLocaleString()} tokens`);
}

const pool = new pg.Pool({
  ssl: resolveSsl().ssl, max: PARALLEL,
  ...(process.env.FEDAI_RAG_PGUSER ? { user: process.env.FEDAI_RAG_PGUSER, password: process.env.FEDAI_RAG_PGPASSWORD } : {}),
  application_name: 'fedai-web-embed',
});
const queue = [...batches];
await Promise.all(Array.from({ length: PARALLEL }, async () => { for (let b = queue.shift(); b; b = queue.shift()) await run(b); }));

// Passages of turns that are no longer in an active version (re-imported or deleted)
const stale = await db.query(`DELETE FROM ${S}.rag_chunks c WHERE NOT EXISTS (
    SELECT 1 FROM ${S}.transcript_turns tr JOIN ${S}.meetings m ON m.active_version_id = tr.version_id WHERE tr.turn_id = c.turn_id)`);
const [{ n, t }] = (await db.query(`SELECT COUNT(*)::int AS n, COUNT(DISTINCT turn_id)::int AS t FROM ${S}.rag_chunks`)).rows;
console.log(`Done: ${tokens.toLocaleString()} tokens embedded (≈ $${(tokens / 1e6 * 0.13).toFixed(2)} at $0.13 / 1M for text-embedding-3-large); `
  + `${stale.rowCount} stale passages removed; rag_chunks now has ${n} passages from ${t} turns.`);
await pool.end();
await db.end();
