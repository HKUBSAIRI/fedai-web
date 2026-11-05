// Connection + read-only role check. Prints no host, password or connection string.
import { hasPgSettings, loadConfig } from '../src/config.js';
import { PgQueryer } from '../src/db/queryer.js';
import { verifyReadOnlyRole } from '../src/db/readonly.js';
import { sanitizeDbError } from '../src/app.js';

const cfg = loadConfig();
if (!hasPgSettings()) { console.log('未配置：.env 中缺少 PGHOST / PGDATABASE'); process.exit(2); }
let q: PgQueryer;
try { q = new PgQueryer(cfg.statementTimeoutMs); } catch (e) { console.log(`配置错误：${(e as Error).message}`); process.exit(4); }
console.log(`传输方式：${q.transport}`);
try {
  const r = await verifyReadOnlyRole(q, cfg.schema);
  console.log(`连接成功：database=${r.database} PostgreSQL ${r.serverVersion}`);
  console.log(`superuser=${r.superuser} 可写表=${r.writableTables} schema CREATE=${r.canCreateInSchema} session read_only=${r.sessionReadOnly}`);
  console.log(r.ok ? '只读核验：通过' : `只读核验：不通过 → ${r.problems.join('；')}（网站将拒绝使用此账号）`);
  process.exitCode = r.ok ? 0 : 3;
} catch (e) {
  console.log(`连接失败：${sanitizeDbError(e)}`);
  process.exitCode = 1;
} finally { await q.close(); }
