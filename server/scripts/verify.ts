// Real-data acceptance check (run on the machine that can reach the real DB):  npm run verify
// Compares what the website API returns with independent read-only SQL, samples 3 meetings and
// 3 persons, and writes docs/真实数据核验报告.md. No passwords, hosts or transcript text are written.
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, ROOT } from '../src/config.js';
import { PgQueryer } from '../src/db/queryer.js';
import { verifyReadOnlyRole } from '../src/db/readonly.js';
import { discoverSchema, loadTranscriptMapping } from '../src/db/schema.js';
import { Repo } from '../src/repo.js';
import { sanitizeDbError } from '../src/app.js';

const cfg = loadConfig();
const q = new PgQueryer(cfg.statementTimeoutMs);
const S = cfg.schema;
const out: string[] = [];
const line = (s = '') => out.push(s);
let pass = 0, fail = 0;
const row = (item: string, site: unknown, sql: unknown, prefix?: string) => {
  const ok = String(site) === String(sql);
  ok ? pass++ : fail++;
  line(`${prefix !== undefined ? `| ${prefix} ` : ''}| ${item} | ${site ?? '—'} | ${sql ?? '—'} | ${ok ? '✓' : '✗ 不一致'} |`);
};
const one = async (sql: string, params: unknown[] = []) => Number(Object.values((await q.query(sql, params))[0])[0]);

try {
  const role = await verifyReadOnlyRole(q, S);
  if (!role.ok) throw new Error(`账号不是只读：${role.problems.join('；')}`);
  const info = await discoverSchema(q, S);
  const repo = new Repo(q, S, info, loadTranscriptMapping(), process.env.FEDAI_PROJECTIONS_TABLE ?? 'sep_projections');

  line('# FedAI 真实数据核验报告');
  line();
  line(`生成时间：${new Date().toISOString()} · 数据库 ${role.database} · PostgreSQL ${role.serverVersion} · 传输方式 ${q.transport}`);
  line(`只读核验：通过（superuser=${role.superuser}，可写表=${role.writableTables}，schema CREATE=${role.canCreateInSchema}）`);
  line();
  line('方法：左列为网站接口（Repo）返回值，右列为独立编写的只读 SQL 结果；全部只执行 SELECT。');
  line();

  // 1. Dashboard cards
  line('## 1. Dashboard 统计卡片');
  line();
  line('| 项目 | 网站 | 独立 SQL | 结果 |');
  line('|---|---|---|---|');
  const ov = await repo.overview();
  const stat = (k: string) => ov.stats.find((s) => s.key === k)?.value;
  // 官员 + 工作人员 = 所有出现在当前版本参会名单上的人
  row('policymakers + staff = 参会名单上的人', Number(stat('policymakers')) + Number(stat('staff')),
    await one(`SELECT COUNT(DISTINCT mp.person_id) FROM ${S}.meeting_participants mp JOIN ${S}.meetings m ON m.active_version_id = mp.version_id`));
  for (const [k, t] of [['meetings', 'meetings'], ['sep_batches', 'sep_batches'],
    ['sep_participants', 'sep_participants'], ['sep_questions', 'sep_questions'], ['sep_answers', 'sep_answers'],
    ['sep_projections', 'sep_projections']] as const) {
    if (info.has(t)) row(`${k}（${t} 行数）`, stat(k), await one(`SELECT COUNT(*) FROM ${S}.${t}`));
  }
  row('utterances（当前版本发言）', stat('utterances'),
    await one(`SELECT COUNT(*) FROM ${S}.transcript_turns t WHERE t.version_id IN (SELECT active_version_id FROM ${S}.meetings)`));
  if (info.has('current_turns')) row('当前版本发言 vs current_turns 视图', stat('utterances'), await one(`SELECT COUNT(*) FROM ${S}.current_turns`));
  const statusSum = ov.responseStatus.rows.reduce((a, r) => a + r.n, 0);
  row('response_status 合计 = sep_answers', statusSum, stat('sep_answers'));
  line();
  line('response_status 分布：' + ov.responseStatus.rows.map((r) => `${r.status}=${r.n}`).join('，'));
  line();

  // 2. Handover figures (2007–2013) vs. live DB
  line('## 2. 与交接资料（2007–2013）对照');
  line();
  line('| 项目 | 交接资料 | 当前数据库 | 结果 |');
  line('|---|---|---|---|');
  const range = `b.source_meeting_start >= DATE '2007-01-01' AND b.source_meeting_start < DATE '2014-01-01'`;
  row('SEP 批次（场）', 26, await one(`SELECT COUNT(*) FROM ${S}.sep_batches b WHERE ${range}`));
  row('本场人员映射', 454, await one(`SELECT COUNT(*) FROM ${S}.sep_participants p JOIN ${S}.sep_batches b ON b.batch_id = p.batch_id WHERE ${range}`));
  row('会议内问题', 189, await one(`SELECT COUNT(*) FROM ${S}.sep_questions p JOIN ${S}.sep_batches b ON b.batch_id = p.batch_id WHERE ${range}`));
  row('叙述问答', 3299, await one(`SELECT COUNT(*) FROM ${S}.sep_answers p JOIN ${S}.sep_batches b ON b.batch_id = p.batch_id WHERE ${range}`));
  line();
  line('SEP 批次按年份：' + (await q.query<{ y: unknown; n: unknown }>(
    `SELECT EXTRACT(YEAR FROM source_meeting_start) AS y, COUNT(*) AS n FROM ${S}.sep_batches GROUP BY 1 ORDER BY 1`))
    .map((r) => `${Number(r.y)}:${Number(r.n)}`).join('，'));
  line();

  // 3. Three meetings
  line('## 3. 抽样会议（最早 / 中位 / 最晚的有 SEP 会议）');
  line();
  const ms = await q.query<{ meeting_id: string; d: string }>(
    `SELECT m.meeting_id::text AS meeting_id, m.meeting_date::text AS d FROM ${S}.meetings m
      WHERE EXISTS (SELECT 1 FROM ${S}.sep_batches b WHERE b.meeting_id = m.meeting_id) ORDER BY m.meeting_date`);
  const pickM = ms.length ? [ms[0], ms[Math.floor(ms.length / 2)], ms[ms.length - 1]] : [];
  line('| 会议 | 项目 | 网站 | 独立 SQL | 结果 |');
  line('|---|---|---|---|---|');
  for (const m of pickM) {
    const d = (await repo.meeting(m.meeting_id))!;
    const r3 = (item: string, a: unknown, b: unknown) => row(item, a, b, m.d);
    r3('参会人数', d.participants.length, await one(
      `SELECT COUNT(*) FROM ${S}.meeting_participants mp JOIN ${S}.meetings m ON m.active_version_id = mp.version_id WHERE m.meeting_id::text = $1`, [m.meeting_id]));
    r3('投票人数（Chair / Vice Chair / Voter）', d.participants.filter((p) => ['Chair', 'Vice Chair', 'Voter'].includes(String(p.role))).length, await one(
      `SELECT COUNT(*) FROM ${S}.meeting_participants mp JOIN ${S}.meetings m ON m.active_version_id = mp.version_id
        WHERE m.meeting_id::text = $1 AND mp.meeting_role IN ('Chair', 'Vice Chair', 'Voter')`, [m.meeting_id]));
    r3('当前版本发言数', d.utteranceCount, await one(
      `SELECT COUNT(*) FROM ${S}.transcript_turns t JOIN ${S}.meetings m ON m.active_version_id = t.version_id WHERE m.meeting_id::text = $1`, [m.meeting_id]));
    const b = d.sepBatches[0];
    const sep = (await repo.sepBatch(String(b.batch_id)))!;
    r3('SEP 人员', sep.participants.length, await one(`SELECT COUNT(*) FROM ${S}.sep_participants WHERE batch_id::text = $1`, [b.batch_id]));
    r3('SEP 问题', sep.questions.length, await one(`SELECT COUNT(*) FROM ${S}.sep_questions WHERE batch_id::text = $1`, [b.batch_id]));
    r3('SEP 问答', sep.answers.length, await one(`SELECT COUNT(*) FROM ${S}.sep_answers WHERE batch_id::text = $1`, [b.batch_id]));
    r3('SEP answered', b.answered, await one(`SELECT COUNT(*) FROM ${S}.sep_answers WHERE batch_id::text = $1 AND response_status = 'answered'`, [b.batch_id]));
    r3('SEP 预测记录', sep.projections?.length ?? null, await one(`SELECT COUNT(*) FROM ${S}.sep_projections WHERE batch_id::text = $1`, [b.batch_id]));
  }
  line();

  // 4. Three persons (most current-version turns)
  line('## 4. 抽样委员（当前版本发言最多的 3 位）');
  line();
  const ps = await q.query<{ person_id: string; name: string }>(
    `SELECT t.person_id::text AS person_id, MAX(p.display_name) AS name FROM ${S}.transcript_turns t
       JOIN ${S}.meetings m ON m.active_version_id = t.version_id JOIN ${S}.persons p ON p.person_id = t.person_id
      GROUP BY t.person_id ORDER BY COUNT(*) DESC LIMIT 3`);
  line('| 人物 | 项目 | 网站 | 独立 SQL | 结果 |');
  line('|---|---|---|---|---|');
  for (const p of ps) {
    const d = (await repo.person(p.person_id))!;
    const list = await repo.listPersons({ q: p.name, pageSize: 100 });
    const li = list.items.find((x) => x.person_id === p.person_id)!;
    const r3 = (item: string, a: unknown, b: unknown) => row(item, a, b, p.name);
    r3('参会场数', d.meetings.length, await one(
      `SELECT COUNT(DISTINCT m.meeting_id) FROM ${S}.meeting_participants mp JOIN ${S}.meetings m ON m.active_version_id = mp.version_id WHERE mp.person_id::text = $1`, [p.person_id]));
    r3('当前版本发言数', li.utterance_count, await one(
      `SELECT COUNT(*) FROM ${S}.transcript_turns t JOIN ${S}.meetings m ON m.active_version_id = t.version_id WHERE t.person_id::text = $1`, [p.person_id]));
    r3('SEP 参与场数', d.sep.length, await one(`SELECT COUNT(*) FROM ${S}.sep_participants WHERE person_id::text = $1`, [p.person_id]));
  }
  line();

  // 6. Federal funds rate dot plot (latest meeting + Powell tracker)
  const dpb = await repo.dotplotBatches().catch(() => null);
  if (dpb && dpb.items.length) {
    line('## 6. 利率点阵图（最近一场 + Powell 轨迹）');
    line();
    line('| 对象 | 项目 | 网站 | 独立 SQL | 结果 |');
    line('|---|---|---|---|---|');
    const last = dpb.items[dpb.items.length - 1];
    const plot = (await repo.dotplot(String(last.batch_id)))!;
    row('预测点数', plot.dots.length, await one(
      `SELECT COUNT(*) FROM ${S}.sep_projections WHERE batch_id::text = $1 AND federal_funds_rate IS NOT NULL`, [last.batch_id]), String(last.meeting_start));
    for (const st of plot.stats) {
      const med = await one(`SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY federal_funds_rate) FROM ${S}.sep_projections
        WHERE batch_id::text = $1 AND projection_horizon = $2 AND federal_funds_rate IS NOT NULL`, [last.batch_id, st.horizon]);
      row(`中位数 ${st.horizon}`, st.median, Math.round(med * 1e4) / 1e4, String(last.meeting_start));
    }
    const [pw] = await q.query<{ person_id: string }>(
      `SELECT person_id::text FROM ${S}.persons WHERE display_name = 'Jerome H. Powell'`);
    if (pw) {
      const tr = await repo.dotplotPerson(pw.person_id);
      row('利率预测条数', tr.items.length, await one(
        `SELECT COUNT(*) FROM ${S}.sep_participants sp JOIN ${S}.sep_projections pr
           ON pr.batch_id = sp.batch_id AND pr.participant_number = sp.participant_number
          WHERE sp.person_id::text = $1 AND pr.federal_funds_rate IS NOT NULL`, [pw.person_id]), 'Jerome H. Powell');
      row('涉及会议数', new Set(tr.items.map((r) => r.batch_id)).size, await one(
        `SELECT COUNT(DISTINCT sp.batch_id) FROM ${S}.sep_participants sp JOIN ${S}.sep_projections pr
           ON pr.batch_id = sp.batch_id AND pr.participant_number = sp.participant_number
          WHERE sp.person_id::text = $1 AND pr.federal_funds_rate IS NOT NULL`, [pw.person_id]), 'Jerome H. Powell');
    }
    line();
  }

  // 7. Data quality summary
  line('## 7. 数据质量（只登记，不修改）');
  line();
  const dq = await repo.dataQuality();
  line('| 检查 | 命中 |');
  line('|---|---|');
  for (const c of dq.checks) line(`| ${c.title} | ${c.error ? '查询失败：' + c.error : c.count} |`);
  for (const c of dq.crossChecks) line(`| ${c.title} | 本站 ${c.ours} / 视图 ${c.theirs} ${c.ours === c.theirs ? '✓' : '✗'} |`);
  line();
  for (const d of dq.distributions) {
    line(`- ${d.title}：` + d.rows.map((r) => Object.values(r).join('=')).join('，'));
  }
  line();
  line(`## 结论：比对 ${pass + fail} 项，一致 ${pass} 项，不一致 ${fail} 项。`);
  line();
  line('注：第 2 节“不一致”可能是交接后有新增/修正导入，需与数据维护者核对，不代表网站错误。');

  const file = process.env.FEDAI_VERIFY_OUT ?? path.join(ROOT, 'docs/真实数据核验报告.md');
  fs.writeFileSync(file, out.join('\n') + '\n');
  console.log(`比对 ${pass + fail} 项：一致 ${pass}，不一致 ${fail}。已写入 ${path.relative(ROOT, file)}`);
} catch (e) {
  console.error(`核验失败：${sanitizeDbError(e)}`);
  process.exitCode = 1;
} finally {
  await q.close();
}
