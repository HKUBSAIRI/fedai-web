// Read-only schema introspection for the real FedAI database.
// Usage (after configuring PG* env vars or .env + ~/.pgpass):  npm run introspect
// Writes docs/schema_introspection.md. Only SELECTs against information_schema / pg_catalog.
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, ROOT } from '../src/config.js';
import { PgQueryer } from '../src/db/queryer.js';
import { verifyReadOnlyRole } from '../src/db/readonly.js';
import { sanitizeDbError } from '../src/app.js';

const cfg = loadConfig();
const q = new PgQueryer(cfg.statementTimeoutMs);
const out: string[] = [];
const line = (s = '') => out.push(s);

try {
  const role = await verifyReadOnlyRole(q, cfg.schema);
  line(`# FedAI 数据库只读结构盘点`);
  line();
  line(`生成时间：${new Date().toISOString()}；schema：\`${cfg.schema}\`；PostgreSQL ${role.serverVersion}`);
  line();
  line(`只读核验：superuser=${role.superuser}，可写表数=${role.writableTables}，schema CREATE=${role.canCreateInSchema}，会话 read_only=${role.sessionReadOnly} → **${role.ok ? '通过' : '不通过：' + role.problems.join('；')}**`);
  line();
  if (!role.ok) throw new Error('角色不是只读账号，停止盘点。请改用只读账号。');

  const schemas = await q.query<{ schema_name: string }>(
    `SELECT schema_name FROM information_schema.schemata
      WHERE schema_name NOT IN ('pg_catalog', 'information_schema') AND schema_name NOT LIKE 'pg_toast%'
      ORDER BY schema_name`);
  line(`可见 schema：${schemas.map((s) => '`' + s.schema_name + '`').join('、')}`);
  line();

  const tables = await q.query<{ table_name: string; table_type: string; est_rows: string | null }>(
    `SELECT t.table_name, t.table_type, c.reltuples::bigint::text AS est_rows
       FROM information_schema.tables t
       LEFT JOIN pg_catalog.pg_namespace n ON n.nspname = t.table_schema
       LEFT JOIN pg_catalog.pg_class c ON c.relnamespace = n.oid AND c.relname = t.table_name
      WHERE t.table_schema = $1 ORDER BY t.table_name`, [cfg.schema]);
  const cols = await q.query<Record<string, string>>(
    `SELECT table_name, column_name, data_type, is_nullable, column_default
       FROM information_schema.columns WHERE table_schema = $1
      ORDER BY table_name, ordinal_position`, [cfg.schema]);
  const cons = await q.query<Record<string, string>>(
    `SELECT tc.table_name, tc.constraint_name, tc.constraint_type,
            string_agg(DISTINCT kcu.column_name, ', ') AS columns,
            MAX(ccu.table_name) AS ref_table, string_agg(DISTINCT ccu.column_name, ', ') AS ref_columns
       FROM information_schema.table_constraints tc
       LEFT JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_schema = tc.constraint_schema AND kcu.constraint_name = tc.constraint_name
       LEFT JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_schema = tc.constraint_schema AND ccu.constraint_name = tc.constraint_name
              AND tc.constraint_type = 'FOREIGN KEY'
      WHERE tc.table_schema = $1 AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE')
      GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type
      ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name`, [cfg.schema]);

  line(`## 表清单（${tables.length}）`);
  line();
  line('| 表 | 类型 | 估计行数 (pg_class.reltuples，非精确) |');
  line('|---|---|---|');
  for (const t of tables) line(`| \`${t.table_name}\` | ${t.table_type} | ${t.est_rows ?? '—'} |`);
  line();
  for (const t of tables) {
    line(`### ${t.table_name}`);
    line();
    line('| 字段 | 类型 | 可空 | 默认 |');
    line('|---|---|---|---|');
    for (const c of cols.filter((c) => c.table_name === t.table_name)) {
      line(`| ${c.column_name} | ${c.data_type} | ${c.is_nullable} | ${c.column_default ? '`' + c.column_default.replace(/\|/g, '\\|') + '`' : ''} |`);
    }
    const tc = cons.filter((c) => c.table_name === t.table_name);
    if (tc.length) {
      line();
      for (const c of tc) {
        line(`- ${c.constraint_type} \`${c.constraint_name}\` (${c.columns})${c.ref_table ? ` → \`${c.ref_table}\`(${c.ref_columns})` : ''}`);
      }
    }
    line();
  }

  const candidates = tables.filter((t) => /transcript|utter|speech|speak|statement|remark|turn/i.test(t.table_name));
  line('## Transcript 候选表（按表名匹配，仅供人工确认，不会自动使用）');
  line();
  if (!candidates.length) line('未找到名称疑似 Transcript 的表。');
  for (const c of candidates) line(`- \`${c.table_name}\``);
  line();
  line('确认后，按真实字段填写 `server/transcript-mapping.json`（见 README），重启服务。');

  const file = path.join(ROOT, 'docs/schema_introspection.md');
  fs.writeFileSync(file, out.join('\n') + '\n');
  console.log(`written ${path.relative(ROOT, file)} (${tables.length} tables)`);
} catch (e) {
  console.error(`introspection failed: ${sanitizeDbError(e)}`);
  process.exitCode = 1;
} finally {
  await q.close();
}
