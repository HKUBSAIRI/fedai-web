import { buildApp, sanitizeDbError, type AppStatus } from './app.js';
import { hasPgSettings, loadConfig } from './config.js';
import { startDemoDatabase, DEMO_TRANSCRIPT_MAPPING } from './db/demo.js';
import { PgQueryer, type Queryer } from './db/queryer.js';
import { verifyReadOnlyRole, type RoleCheck } from './db/readonly.js';
import { discoverSchema, loadTranscriptMapping } from './db/schema.js';
import { Repo } from './repo.js';
import { loadRagConfig } from './rag.js';

const cfg = loadConfig();
const realDb: AppStatus['realDb'] = { configured: hasPgSettings(), connected: false, error: null };
let queryer: Queryer | null = null;
let role: RoleCheck | null = null;
let stopDemo: (() => Promise<void>) | null = null;

if (cfg.dataMode !== 'demo' && (realDb.configured || cfg.dataMode === 'postgres')) {
  const pgq = new PgQueryer(cfg.statementTimeoutMs);
  try {
    role = await verifyReadOnlyRole(pgq, cfg.schema);
    realDb.connected = true;
    if (!role.ok) {
      realDb.error = `Refusing this database account (not read-only): ${role.problems.join('; ')}`;
      await pgq.close();
    } else {
      queryer = pgq;
    }
  } catch (e) {
    realDb.error = `Cannot connect to the real database: ${sanitizeDbError(e)}`;
    await pgq.close().catch(() => undefined);
  }
  if (!queryer && cfg.dataMode === 'postgres') {
    console.error(`[fedai-web] FEDAI_DATA_MODE=postgres but real database is unusable: ${realDb.error}`);
    process.exit(1);
  }
} else if (cfg.dataMode === 'demo') {
  realDb.error = 'FEDAI_DATA_MODE=demo: configured not to connect to the real database (e.g. VPS preview)';
} else {
  realDb.error = 'No database connection configured (PGHOST / PGDATABASE not set)';
}

let mode: AppStatus['mode'] = 'postgres';
let transcriptMapping = loadTranscriptMapping();
if (!queryer) {
  mode = 'demo';
  const demo = await startDemoDatabase(cfg.statementTimeoutMs);
  queryer = demo.queryer;
  stopDemo = demo.stop;
  role = await verifyReadOnlyRole(queryer, cfg.schema);
  transcriptMapping = DEMO_TRANSCRIPT_MAPPING;
}

const info = await discoverSchema(queryer, cfg.schema);
const repo = new Repo(queryer, cfg.schema, info, transcriptMapping, process.env.FEDAI_PROJECTIONS_TABLE ?? 'sep_projections');
const rag = loadRagConfig();
const status: AppStatus = {
  mode, demo: mode === 'demo', schema: cfg.schema, role, realDb,
  availability: repo.avail, ragConfigured: Boolean(rag), tables: [...info.keys()], startedAt: new Date().toISOString(),
};

const app = buildApp(repo, status, { logger: true, rag });
await app.listen({ host: cfg.host, port: cfg.port });
console.log(`[fedai-web] mode=${mode} listening on http://${cfg.host}:${cfg.port}`);
if (realDb.error) console.log(`[fedai-web] real DB: ${realDb.error}`);

const shutdown = async () => {
  await app.close();
  await queryer?.close().catch(() => undefined);
  await stopDemo?.().catch(() => undefined);
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
