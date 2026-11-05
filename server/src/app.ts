import fs from 'node:fs';
import path from 'node:path';
import Fastify, { type FastifyReply } from 'fastify';
import fastifyStatic from '@fastify/static';
import { ROOT } from './config.js';
import { BadRequestError, Repo, UnavailableError } from './repo.js';
import type { RoleCheck } from './db/readonly.js';
import { answer, citedNumbers, embed, modelLabel, QUESTION_MAX_LEN, RagError, toVectorLiteral, yearsInQuestion, type RagConfig } from './rag.js';

export interface AppStatus {
  mode: 'postgres' | 'demo';
  demo: boolean;
  schema: string;
  role: RoleCheck | null;
  realDb: { configured: boolean; connected: boolean; error: string | null };
  availability: Repo['avail'];
  ragConfigured: boolean;   // Azure OpenAI settings present (the Q&A also needs availability.rag)
  tables: string[];
  startedAt: string;
}

/** Strip anything that could identify infrastructure (hosts, IPs, users) from DB errors. */
export function sanitizeDbError(e: unknown): string {
  const err = e as { code?: string; message?: string };
  const msg = (err.message ?? String(e))
    .replace(/\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b/g, '<host>')
    .replace(/(host|user|database|role)\s+"[^"]*"/gi, '$1 "<redacted>"')
    .replace(/getaddrinfo \w+ \S+/g, 'getaddrinfo <host>');
  return err.code ? `${err.code}: ${msg}` : msg;
}

/** At most this many questions are answered at the same time (each one calls Azure OpenAI twice). */
const ASK_CONCURRENCY = 3;

export function buildApp(repo: Repo, status: AppStatus, opts: { logger?: boolean; rag?: RagConfig | null; cache?: boolean } = {}) {
  const app = Fastify({ logger: opts.logger ?? false });
  const wrap = (data: unknown) => ({ mode: status.mode, demo: status.demo, data });

  app.setErrorHandler((err, _req, reply) => {
    if (err instanceof UnavailableError) return reply.code(503).send({ error: err.message, unavailable: true });
    if (err instanceof BadRequestError) return reply.code(400).send({ error: (err as Error).message });
    if (err instanceof RagError) { app.log.error({ err: err.message }, 'AI Q&A failed'); return reply.code(err.status).send({ error: err.message }); }
    if ((err as { statusCode?: number }).statusCode === 400) return reply.code(400).send({ error: (err as Error).message });
    app.log.error({ err: sanitizeDbError(err) }, 'request failed');
    return reply.code(500).send({ error: `Query failed: ${sanitizeDbError(err)}` });
  });

  // Response cache for data GETs. The data changes only when a fix script is run, and the database is ~40 ms away,
  // so repeated page views are served from memory. A database change shows up on the site within the TTL.
  const ttlMs = Number(process.env.FEDAI_CACHE_TTL_S ?? (opts.cache === false ? 0 : 600)) * 1000;
  const cache = new Map<string, { body: string; expires: number }>();
  const CACHE_MAX = 500;
  if (ttlMs > 0) {
    app.addHook('onRequest', async (req, reply) => {
      if (req.method !== 'GET' || !req.url.startsWith('/api/') || req.url === '/api/status') return;
      const hit = cache.get(req.url);
      if (hit && hit.expires > Date.now()) return reply.header('x-cache', 'hit').type('application/json; charset=utf-8').send(hit.body);
    });
    app.addHook('onSend', async (req, reply, payload) => {
      if (req.method === 'GET' && req.url.startsWith('/api/') && req.url !== '/api/status' && reply.statusCode === 200
          && reply.getHeader('x-cache') !== 'hit' && typeof payload === 'string') {
        if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value!);
        cache.set(req.url, { body: payload, expires: Date.now() + ttlMs });
      }
      return payload;
    });
  }

  const notFound = (reply: FastifyReply, what: string) => reply.code(404).send({ error: `${what} not found` });
  type Q = Record<string, string | undefined>;

  app.get('/api/status', async () => wrap(status));
  app.get('/api/overview', async () => wrap(await repo.overview()));
  app.get('/api/meetings/years', async () => wrap(await repo.meetingYears()));
  app.get('/api/meetings/dates', async () => wrap(await repo.meetingDates()));
  app.get<{ Querystring: Q }>('/api/meetings', async (req) => wrap(await repo.listMeetings(req.query)));
  app.get<{ Params: { id: string } }>('/api/meetings/:id', async (req, reply) => {
    const d = await repo.meeting(req.params.id);
    return d ? wrap(d) : notFound(reply, 'Meeting');
  });
  app.get<{ Params: { id: string } }>('/api/meetings/:id/economy', async (req, reply) => {
    const d = await repo.economy(req.params.id);
    return d ? wrap(d) : notFound(reply, 'Meeting');
  });
  app.get('/api/economy', async () => wrap(await repo.economyList()));
  app.get<{ Querystring: Q }>('/api/persons', async (req) => wrap(await repo.listPersons(req.query)));
  app.get<{ Params: { id: string } }>('/api/persons/:id', async (req, reply) => {
    const d = await repo.person(req.params.id);
    return d ? wrap(d) : notFound(reply, 'Person');
  });
  app.get<{ Querystring: Q }>('/api/transcript', async (req) => wrap(await repo.searchTranscript(req.query)));
  app.get<{ Querystring: Q }>('/api/transcript/speakers', async (req) => wrap(await repo.transcriptSpeakers(req.query)));
  app.get<{ Querystring: Q }>('/api/sep', async (req) => wrap(await repo.listSepBatches(req.query)));
  app.get<{ Params: { id: string } }>('/api/sep/:id', async (req, reply) => {
    const d = await repo.sepBatch(req.params.id);
    return d ? wrap(d) : notFound(reply, 'SEP round');
  });
  app.get('/api/quality', async () => wrap(await repo.dataQuality()));
  app.get('/api/dotplot/batches', async () => wrap(await repo.dotplotBatches()));
  app.get('/api/dotplot/persons', async () => wrap(await repo.dotplotPersons()));
  app.get<{ Params: { id: string } }>('/api/dotplot/person/:id', async (req) => wrap(await repo.dotplotPerson(req.params.id)));
  app.get<{ Params: { id: string } }>('/api/dotplot/:id', async (req, reply) => {
    const d = await repo.dotplot(req.params.id);
    return d ? wrap(d) : notFound(reply, 'Rate projections for this meeting');
  });
  // AI Q&A (RAG): embed the question, retrieve the closest transcript passages, answer from them with citations.
  let asking = 0;
  app.post<{ Body: { question?: unknown; dateFrom?: unknown; dateTo?: unknown } | undefined }>('/api/ask', async (req, reply) => {
    const rag = opts.rag;
    if (!rag) return reply.code(503).send({ error: 'AI Q&A is not configured on this server (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY).', unavailable: true });
    const body = req.body ?? {};
    const question = typeof body.question === 'string' ? body.question.trim() : '';
    if (question.length < 3) throw new BadRequestError('Please enter a question');
    if (question.length > QUESTION_MAX_LEN) throw new BadRequestError(`Questions are limited to ${QUESTION_MAX_LEN} characters`);
    if (asking >= ASK_CONCURRENCY) return reply.code(429).send({ error: 'Other questions are being answered; please try again in a moment.' });
    asking++;
    try {
      const { vectors } = await embed(rag, [question]);
      const chosen = Boolean(body.dateFrom || body.dateTo);
      const autoDates = chosen ? null : yearsInQuestion(question);
      const dates = autoDates ?? { dateFrom: body.dateFrom, dateTo: body.dateTo };
      const hits = await repo.ragSearch(toVectorLiteral(vectors[0]), modelLabel(rag), dates);
      const passages = hits.items.map((r, i) => ({
        n: i + 1, speaker: String(r.person_name ?? r.speaker_label ?? 'Unknown speaker'), meetingDate: String(r.meeting_date),
        seq: r.seq === null || r.seq === undefined ? null : Number(r.seq),
        page: r.source_page === null || r.source_page === undefined ? null : Number(r.source_page), content: String(r.content),
      }));
      const a = passages.length ? await answer(rag, question, passages) : { text: 'No transcript passages were found for these filters.', usage: null };
      const cited = citedNumbers(a.text);
      return wrap({
        question, answer: a.text, autoDates, embedModel: modelLabel(rag), chatModel: rag.chatModel, usage: a.usage, sql: hits.sql,
        sources: hits.items.map((r, i) => ({ n: i + 1, cited: cited.includes(i + 1), ...r })),
      });
    } finally {
      asking--;
    }
  });
  app.all('/api/*', async (_req, reply) => notFound(reply, 'API endpoint'));

  // Built front-end (npm run build). In development Vite serves the UI and proxies /api.
  const dist = path.join(ROOT, 'web/dist');
  if (fs.existsSync(dist)) {
    app.register(fastifyStatic, { root: dist, wildcard: false });
    app.setNotFoundHandler((req, reply) => {
      if (req.method === 'GET' && !req.url.startsWith('/api/')) return reply.sendFile('index.html');
      return notFound(reply, 'Page');
    });
  }
  return app;
}
