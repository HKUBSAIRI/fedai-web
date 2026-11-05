/**
 * AI Q&A over the transcripts (RAG): the question is embedded, the closest transcript passages are retrieved from
 * fedai_corpus.rag_chunks (pgvector), and a chat model answers from those passages only, citing them as [1], [2], …
 * Passages are written by server/scripts/embed.ts; table: docs/rag/01_向量表.sql.
 */
import { createHash } from 'node:crypto';

// Passages: ~375 words ≈ 500 tokens, consecutive passages of a long turn overlap by ~40 words (≈ 50 tokens).
// Turns under 15 words ("Thank you. President Evans.") are not embedded (docs/rag/费用估算.md).
export const CHUNK_WORDS = 375;
export const OVERLAP_WORDS = 40;
export const MIN_WORDS = 15;
export const QUESTION_MAX_LEN = 500;

export interface RagConfig {
  endpoint: string;          // e.g. https://<resource>.openai.azure.com
  apiKey: string;
  embedModel: string;        // Azure deployment name
  dimensions: number;        // must match rag_chunks.embedding (vector(1536))
  chatModel: string;         // Azure deployment name
  apiVersion: string;
}

/** Azure OpenAI settings from the environment, or null when the Q&A is not configured. */
export function loadRagConfig(env = process.env): RagConfig | null {
  const endpoint = env.AZURE_OPENAI_ENDPOINT?.replace(/\/+$/, '');
  const apiKey = env.AZURE_OPENAI_API_KEY;
  if (!endpoint || !apiKey) return null;
  return {
    endpoint, apiKey,
    embedModel: env.FEDAI_RAG_EMBED_MODEL ?? 'text-embedding-3-large',
    dimensions: Number(env.FEDAI_RAG_DIMENSIONS ?? 1536),
    chatModel: env.FEDAI_RAG_CHAT_MODEL ?? 'gpt-4.1-mini',
    apiVersion: env.AZURE_OPENAI_API_VERSION ?? '2024-10-21',
  };
}

/** Label stored with every vector so passages made with another model or size are never mixed in a search. */
export const modelLabel = (c: Pick<RagConfig, 'embedModel' | 'dimensions'>) => `${c.embedModel}@${c.dimensions}`;

export const textHash = (s: string) => createHash('md5').update(s).digest('hex');

/** Split one turn into overlapping passages of whole words; short turns give no passage. */
export function chunkTurn(text: string, size = CHUNK_WORDS, overlap = OVERLAP_WORDS, min = MIN_WORDS): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < min) return [];
  if (words.length <= size) return [words.join(' ')];
  const out: string[] = [];
  const step = size - overlap;
  for (let start = 0; start < words.length; start += step) {
    out.push(words.slice(start, start + size).join(' '));
    if (start + size >= words.length) break;
  }
  return out;
}

/** What is embedded: speaker and meeting date first, so questions naming a person or a year find their passages. */
export function embeddingInput(speaker: string, meetingDate: string, passage: string): string {
  return `${speaker} — FOMC meeting ${meetingDate}\n${passage}`;
}

/**
 * pgvector text form of an embedding, passed as a query parameter ($1::vector). pgvector stores float4 (~7 significant
 * digits), so 7 digits keep the stored value and halve what is sent to the database.
 */
export const toVectorLiteral = (v: number[]) => `[${v.map((x) => Number(x.toPrecision(7))).join(',')}]`;

export class RagError extends Error {
  constructor(message: string, readonly status = 502) { super(message); }
}

async function azure<T>(cfg: RagConfig, deployment: string, op: 'embeddings' | 'chat/completions', body: unknown, timeoutMs: number): Promise<T> {
  const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(deployment)}/${op}?api-version=${cfg.apiVersion}`;
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'api-key': cfg.apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (res.ok) return (await res.json()) as T;
    // rate limited or temporarily unavailable: wait as told and retry a few times
    if ((res.status === 429 || res.status >= 500) && attempt < 4) {
      const wait = Number(res.headers.get('retry-after')) || 2 ** attempt * 2;
      await new Promise((r) => setTimeout(r, wait * 1000));
      continue;
    }
    const detail = await res.text().catch(() => '');
    // never echo the key or the endpoint back
    throw new RagError(`Azure OpenAI ${op} failed: HTTP ${res.status} ${detail.slice(0, 200).replace(cfg.apiKey, '<key>')}`);
  }
}

export async function embed(cfg: RagConfig, inputs: string[], timeoutMs = 60000): Promise<{ vectors: number[][]; tokens: number }> {
  const r = await azure<{ data: { index: number; embedding: number[] }[]; usage?: { prompt_tokens?: number } }>(
    cfg, cfg.embedModel, 'embeddings', { input: inputs, dimensions: cfg.dimensions }, timeoutMs);
  const vectors = [...r.data].sort((a, b) => a.index - b.index).map((d) => d.embedding);
  if (vectors.length !== inputs.length) throw new RagError('Azure OpenAI returned a different number of embeddings');
  return { vectors, tokens: r.usage?.prompt_tokens ?? 0 };
}

export interface Passage {
  n: number;                   // citation number shown as [n]
  speaker: string;
  meetingDate: string;
  seq: number | null;
  page: number | null;
  content: string;
}

export const SYSTEM_PROMPT = [
  'You answer questions about Federal Open Market Committee (FOMC) meetings for a research team.',
  'Use ONLY the numbered transcript excerpts provided. Do not use outside knowledge.',
  'Cite the excerpts that support each statement with their numbers in square brackets, e.g. [2] or [1][4].',
  'Name the speaker and the meeting date when you describe what someone said.',
  'If the excerpts do not contain the answer, say so plainly and state what they do cover.',
  'Write the answer in the same language as the question: an English question gets an English answer, a Chinese question a Chinese answer.',
  'Keep quotes from the transcripts in the original English.',
].join('\n');

export function buildUserPrompt(question: string, passages: Passage[]): string {
  const excerpts = passages.map((p) =>
    `[${p.n}] ${p.speaker}, FOMC meeting ${p.meetingDate}${p.seq !== null ? `, turn #${p.seq}` : ''}${p.page !== null ? `, transcript p. ${p.page}` : ''}\n${p.content}`);
  return `Transcript excerpts:\n\n${excerpts.join('\n\n')}\n\nQuestion: ${question}`;
}

export async function answer(cfg: RagConfig, question: string, passages: Passage[], timeoutMs = 90000) {
  const r = await azure<{ choices: { message: { content: string | null } }[]; usage?: { prompt_tokens?: number; completion_tokens?: number } }>(
    cfg, cfg.chatModel, 'chat/completions', {
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: buildUserPrompt(question, passages) }],
      temperature: 0.2,
      max_tokens: 1200,
    }, timeoutMs);
  return { text: r.choices[0]?.message.content ?? '', usage: r.usage ?? null };
}

/**
 * Meeting years named in a question ("in 2010", "2010 年", "between 2008 and 2012") → a date range covering them.
 * Used only when no range was chosen: matching by meaning alone does not respect years.
 */
export function yearsInQuestion(q: string, first = 2007, last = 2020): { dateFrom: string; dateTo: string } | null {
  // a year followed by a letter is not a year ("2010s" is a decade); "2010年" still is
  const ys = [...q.matchAll(/(?<![\dA-Za-z])(20\d\d)(?![\dA-Za-z])/g)].map((m) => Number(m[1])).filter((y) => y >= first && y <= last);
  if (!ys.length) return null;
  return { dateFrom: `${Math.min(...ys)}-01-01`, dateTo: `${Math.max(...ys)}-12-31` };
}

/** Citation numbers used in an answer, in order of first use (to flag which sources were actually cited). */
export function citedNumbers(text: string): number[] {
  const seen: number[] = [];
  for (const m of text.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (!seen.includes(n)) seen.push(n);
  }
  return seen;
}
