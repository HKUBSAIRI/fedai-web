// Integration tests: real PostgreSQL (embedded, demo data) + the same read-only code path.
// Every number the API returns is cross-checked against an independent SQL query.
import path from 'node:path';
import os from 'node:os';
import pg from 'pg';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { buildApp, type AppStatus } from '../src/app.js';
import { startDemoDatabase, DEMO_TRANSCRIPT_MAPPING } from '../src/db/demo.js';
import { verifyReadOnlyRole } from '../src/db/readonly.js';
import { discoverSchema, validateTranscriptMapping } from '../src/db/schema.js';
import { PgQueryer } from '../src/db/queryer.js';
import { Repo, dotStats, isOutlier, officialPdfUrl } from '../src/repo.js';

let demo: Awaited<ReturnType<typeof startDemoDatabase>>;
let app: ReturnType<typeof buildApp>;
let repo: Repo;
let appStatus: AppStatus;
const S = 'fedai_corpus';

const get = async (url: string) => {
  const res = await app.inject({ method: 'GET', url });
  return { status: res.statusCode, body: res.json() };
};
const one = async (sql: string, params: unknown[] = []) => Number(Object.values((await demo.queryer.query(sql, params))[0])[0]);

beforeAll(async () => {
  demo = await startDemoDatabase(15000, 54339, path.join(os.tmpdir(), `fedai-web-test-pg-${process.pid}`));
  const info = await discoverSchema(demo.queryer, S);
  repo = new Repo(demo.queryer, S, info, DEMO_TRANSCRIPT_MAPPING);
  appStatus = {
    mode: 'demo', demo: true, schema: S, role: await verifyReadOnlyRole(demo.queryer, S),
    realDb: { configured: false, connected: false, error: 'test' }, availability: repo.avail, ragConfigured: false,
    tables: [...info.keys()], startedAt: new Date().toISOString(),
  };
  app = buildApp(repo, appStatus, { cache: false });
}, 120_000);

afterAll(async () => { await app?.close(); await demo?.stop(); });

describe('read-only safety', () => {
  it('demo app role passes the read-only role check', async () => {
    const r = await verifyReadOnlyRole(demo.queryer, S);
    expect(r).toMatchObject({ ok: true, superuser: false, writableTables: 0, sessionReadOnly: true });
  });

  it('flags a privileged role as NOT read-only', async () => {
    const adminQ = new PgQueryer(5000, { host: '127.0.0.1', port: demo.admin.port, user: 'demo_admin', password: demo.admin.password, database: 'fedai_demo', ssl: false });
    try {
      const r = await verifyReadOnlyRole(adminQ, S);
      expect(r.ok).toBe(false);
      expect(r.problems.length).toBeGreaterThan(0);
    } finally { await adminQ.close(); }
  });

  it('PgQueryer transactions are READ ONLY even for a privileged role', async () => {
    // Bypass the SQL guard on purpose by using a raw client with the same session options.
    const c = new pg.Client({ host: '127.0.0.1', port: demo.admin.port, user: 'demo_admin', password: demo.admin.password,
      database: 'fedai_demo', ssl: false, options: '-c default_transaction_read_only=on' });
    await c.connect();
    try {
      await expect(c.query(`DELETE FROM ${S}.persons`)).rejects.toThrow(/read-only transaction/);
    } finally { await c.end(); }
  });

  it('SQL guard rejects writes before they reach the database', async () => {
    await expect(demo.queryer.query(`DELETE FROM ${S}.persons`)).rejects.toThrow(/only SELECT/);
  });
});

describe('API', () => {
  it('serves repeated data requests from the response cache', async () => {
    const cached = buildApp(repo, appStatus, { cache: true });
    try {
      const first = await cached.inject({ method: 'GET', url: '/api/overview' });
      const second = await cached.inject({ method: 'GET', url: '/api/overview' });
      expect(first.headers['x-cache']).toBeUndefined();
      expect(second.headers['x-cache']).toBe('hit');
      expect(second.body).toBe(first.body);
      const st = await cached.inject({ method: 'GET', url: '/api/status' });
      expect((await cached.inject({ method: 'GET', url: '/api/status' })).headers['x-cache']).toBeUndefined();
      expect(st.statusCode).toBe(200);
    } finally { await cached.close(); }
  });

  it('status reports demo mode explicitly', async () => {
    const { body } = await get('/api/status');
    expect(body.demo).toBe(true);
    expect(body.data.mode).toBe('demo');
  });

  it('overview stats match independent counts', async () => {
    const { body } = await get('/api/overview');
    const stat = (k: string) => body.data.stats.find((s: { key: string }) => s.key === k).value;
    // officials + staff = everyone on an active attendance list
    expect(stat('policymakers') + stat('staff')).toBe(await one(
      `SELECT COUNT(DISTINCT mp.person_id) FROM ${S}.meeting_participants mp JOIN ${S}.meetings m ON m.active_version_id = mp.version_id`));
    for (const [k, t] of [['meetings', 'meetings'], ['sep_batches', 'sep_batches'],
      ['sep_questions', 'sep_questions'], ['sep_answers', 'sep_answers'], ['sep_participants', 'sep_participants']]) {
      expect(stat(k)).toBe(await one(`SELECT COUNT(*) FROM ${S}.${t}`));
    }
    const statusSum = body.data.responseStatus.rows.reduce((a: number, r: { n: number }) => a + r.n, 0);
    expect(statusSum).toBe(stat('sep_answers'));
    const yearSum = body.data.byYear.rows.reduce((a: number, r: { meetings: number }) => a + r.meetings, 0);
    expect(yearSum).toBe(stat('meetings'));
    for (const s of body.data.stats) if (s.value !== null) expect(s.sql).toBeTruthy();
  });

  it('meetings list filters by year and paginates', async () => {
    const { body } = await get('/api/meetings?year=2013&pageSize=5');
    expect(body.data.total).toBe(await one(`SELECT COUNT(*) FROM ${S}.meetings WHERE meeting_date >= '2013-01-01' AND meeting_date < '2014-01-01'`));
    expect(body.data.items).toHaveLength(5);
    expect(body.data.items.every((m: { meeting_date: string }) => m.meeting_date.startsWith('2013'))).toBe(true);
    const p2 = await get('/api/meetings?year=2013&pageSize=5&page=2');
    expect(p2.body.data.items[0].meeting_id).not.toBe(body.data.items[0].meeting_id);
  });

  it('meeting detail participants match meeting_participants for active version', async () => {
    const list = await get('/api/meetings?pageSize=100');
    const withSep = list.body.data.items.find((m: { sep_batch_count: number }) => m.sep_batch_count > 0);
    const { body } = await get(`/api/meetings/${withSep.meeting_id}`);
    expect(body.data.participants.length).toBe(await one(
      `SELECT COUNT(*) FROM ${S}.meeting_participants mp JOIN ${S}.meetings m ON m.active_version_id = mp.version_id WHERE m.meeting_id::text = $1`, [withSep.meeting_id]));
    expect(body.data.sepBatches).toHaveLength(1);
    expect(body.data.participants.length).toBe(withSep.participant_count);
  });

  it('unknown meeting → 404, unknown api → 404', async () => {
    expect((await get('/api/meetings/00000000-0000-0000-0000-000000000000')).status).toBe(404);
    expect((await get('/api/nope')).status).toBe(404);
  });

  it('person search and detail', async () => {
    const { body } = await get('/api/persons?q=member%20b');
    expect(body.data.items.map((p: { person_name: string }) => p.person_name)).toEqual(['Demo Member B']);
    const pid = body.data.items[0].person_id;
    const d = await get(`/api/persons/${pid}`);
    expect(d.body.data.meetings.length).toBe(body.data.items[0].meeting_count);
    expect(d.body.data.sep.length).toBe(await one(`SELECT COUNT(*) FROM ${S}.sep_participants WHERE person_id::text = $1`, [pid]));
  });

  it('person search treats % and _ literally', async () => {
    const { body } = await get('/api/persons?q=%25');
    expect(body.data.total).toBe(0);
  });

  it('transcript keyword search returns only matching rows and source info', async () => {
    const { body } = await get('/api/transcript?q=inflation&pageSize=10');
    expect(body.data.total).toBe(await one(`SELECT COUNT(*) FROM ${S}.transcript_turns t JOIN ${S}.meetings m ON m.active_version_id = t.version_id WHERE t.body_clean ILIKE '%inflation%'`));
    expect(body.data.items.length).toBeLessThanOrEqual(10);
    for (const u of body.data.items) {
      expect(u.text.toLowerCase()).toContain('inflation');
      expect(u.src_source_page).toBeTruthy();
      expect(u.speaker_label).toBeTruthy();
    }
    expect((await get('/api/transcript?q=a')).status).toBe(400);
  });

  it('transcript filter by person', async () => {
    const p = (await get('/api/persons?q=member%20c')).body.data.items[0];
    const { body } = await get(`/api/transcript?personId=${p.person_id}&pageSize=100`);
    expect(body.data.total).toBe(p.utterance_count);
    expect(body.data.items.every((u: { person_id: string }) => u.person_id === p.person_id)).toBe(true);
  });

  it('transcript filter by year, combined with person and keyword', async () => {
    const year = (await get('/api/meetings/years')).body.data.at(-1);
    const turns = `${S}.transcript_turns t JOIN ${S}.meetings m ON m.active_version_id = t.version_id`;
    const inYear = `m.meeting_date >= $1::date AND m.meeting_date < $2::date`;
    const range = [`${year}-01-01`, `${year + 1}-01-01`];
    const { body } = await get(`/api/transcript?year=${year}&pageSize=100`);
    expect(body.data.total).toBe(await one(`SELECT COUNT(*) FROM ${turns} WHERE ${inYear}`, range));
    expect(body.data.items.every((u: { meeting_date: string }) => u.meeting_date.startsWith(String(year)))).toBe(true);

    const sp = (await get(`/api/transcript/speakers?year=${year}`)).body.data.items;
    expect(sp.reduce((a: number, p: { turns: number }) => a + p.turns, 0)).toBe(body.data.total);
    const top = sp[0];
    const both = (await get(`/api/transcript?year=${year}&personId=${top.person_id}&pageSize=100`)).body.data;
    expect(both.total).toBe(top.turns);
    const kw = (await get(`/api/transcript?year=${year}&personId=${top.person_id}&q=the`)).body.data;
    expect(kw.total).toBe(await one(`SELECT COUNT(*) FROM ${turns} WHERE ${inYear} AND t.person_id::text = $3 AND t.body_clean ILIKE '%the%'`,
      [...range, top.person_id]));
    expect((await get('/api/transcript?year=abc')).status).toBe(400);
  });

  it('transcript lists newest meetings first by default, spoken order within a meeting', async () => {
    const isSorted = (items: { meeting_date: string; seq: number }[], dir: 1 | -1) => items.every((u, i) => {
      if (i === 0) return true;
      const prev = items[i - 1];
      return prev.meeting_date === u.meeting_date ? prev.seq <= u.seq : Math.sign(u.meeting_date.localeCompare(prev.meeting_date)) === dir;
    });
    const desc = (await get('/api/transcript?pageSize=100')).body.data;
    expect(desc.order).toBe('desc');
    expect(desc.items[0].meeting_date).toBe((await demo.queryer.query<{ d: string }>(
      `SELECT max(m.meeting_date)::text AS d FROM ${S}.transcript_turns t JOIN ${S}.meetings m ON m.active_version_id = t.version_id`))[0].d);
    expect(isSorted(desc.items, -1)).toBe(true);
    const asc = (await get('/api/transcript?order=asc&pageSize=100')).body.data;
    expect(asc.order).toBe('asc');
    expect(isSorted(asc.items, 1)).toBe(true);
  });

  it('SEP batch detail is internally consistent', async () => {
    const list = await get('/api/sep');
    expect(list.body.data.items.length).toBe(await one(`SELECT COUNT(*) FROM ${S}.sep_batches`));
    const b = list.body.data.items[0];
    expect(b.answered + b.na + b.missing).toBe(b.answers);
    const { body } = await get(`/api/sep/${b.batch_id}`);
    expect(body.data.participants).toHaveLength(b.participants);
    expect(body.data.questions).toHaveLength(b.questions);
    expect(body.data.answers).toHaveLength(b.answers);
    expect(body.data.projectionsStatus.ok).toBe(true);
    expect(body.data.projections.length).toBeGreaterThan(0);
  });

  it('data quality checks detect the planted demo anomalies', async () => {
    const { body } = await get('/api/quality');
    const c = (id: string) => body.data.checks.find((x: { id: string }) => x.id === id);
    for (const chk of body.data.checks) expect(chk.error).toBeUndefined();
    expect(c('sep_unmatched').count).toBe(await one(`SELECT COUNT(*) FROM ${S}.sep_participants WHERE person_id IS NULL OR mapping_status <> 'matched'`));
    expect(c('sep_unmatched').count).toBeGreaterThan(0);
    expect(c('sep_grid_gap').count).toBeGreaterThan(0);
    expect(c('person_same_surname').count).toBeGreaterThan(0);
  });
});

describe('real-schema semantics', () => {
  it('superseded import versions are never shown', async () => {
    const all = await one(`SELECT COUNT(*) FROM ${S}.transcript_turns WHERE body_clean LIKE '%OLD-VERSION%'`);
    expect(all).toBeGreaterThan(0);
    const { body } = await get('/api/transcript?q=OLD-VERSION');
    expect(body.data.total).toBe(0);
    const ov = (await get('/api/overview')).body.data.stats;
    const v = (k: string) => ov.find((s: { key: string }) => s.key === k).value;
    expect(v('utterances')).toBe(await one(`SELECT COUNT(*) FROM ${S}.transcript_turns`) - all);
  });

  it('active-version joins agree with current_* views', async () => {
    const { body } = await get('/api/quality');
    expect(body.data.crossChecks.length).toBeGreaterThan(0);
    for (const c of body.data.crossChecks) { expect(c.error).toBeUndefined(); expect(c.ours).toBe(c.theirs); }
  });

  it('quality reports registered review issues, identity status and missing projection fields', async () => {
    const { body } = await get('/api/quality');
    const ids = body.data.distributions.map((d: { id: string }) => d.id);
    expect(ids).toEqual(expect.arrayContaining(['identity_status', 'mapping_status', 'review_issues']));
    const c = (id: string) => body.data.checks.find((x: { id: string }) => x.id === id);
    expect(c('proj_missing_fields').count).toBeGreaterThan(0);
    expect(c('turn_speaker_not_participant').count).toBeGreaterThan(0);
  });

  it('persons can be searched by display_name and expose canonical_name / identity_status', async () => {
    const { body } = await get('/api/persons?q=staff');
    expect(body.data.items[0]).toMatchObject({ person_name: 'Demo Staff Speaker', canonical_name: null, identity_status: 'unresolved' });
  });

  it('projections carry horizon / long-run fields', async () => {
    const b = (await get('/api/sep')).body.data.items[0];
    const { body } = await get(`/api/sep/${b.batch_id}`);
    const lr = body.data.projections.filter((r: { is_long_run: boolean }) => r.is_long_run);
    expect(lr.length).toBeGreaterThan(0);
    expect(lr[0].projection_year).toBeNull();
  });
});

describe('unverified areas stay disabled', () => {
  it('transcript mapping is rejected when not configured or columns do not exist', async () => {
    const info = await discoverSchema(demo.queryer, S);
    expect(validateTranscriptMapping(info, null).ok).toBe(false);
    expect(validateTranscriptMapping(info, { ...DEMO_TRANSCRIPT_MAPPING, textColumn: 'nonexistent' }).ok).toBe(false);
  });

  it('without projections table the SEP chart is reported as unavailable', async () => {
    const info = await discoverSchema(demo.queryer, S);
    info.delete('sep_projections');
    const r = new Repo(demo.queryer, S, info, null);
    expect(r.avail.projections.ok).toBe(false);
    expect(r.avail.transcript.ok).toBe(false);
    const ov = await r.overview();
    expect(ov.stats.find((s) => s.key === 'utterances')!.value).toBeNull();
    await expect(r.searchTranscript({})).rejects.toThrow(/not verified|not configured|Not configured/);
  });
});

describe('federal funds rate dot plot', () => {
  it('dotStats matches percentile_cont semantics and outlier rule', () => {
    expect(dotStats([4, 1, 3, 2])).toMatchObject({ n: 4, median: 2.5, q1: 1.75, q3: 3.25, iqr: 1.5, min: 1, max: 4 });
    const st = dotStats([1, 1, 1, 1, 1.25]);
    expect(isOutlier(1.25, st)).toBe(false);       // within 0.5pp floor
    expect(isOutlier(1.5, st)).toBe(false);        // exactly on the 0.5pp line is not an outlier
    expect(isOutlier(2, st)).toBe(true);
  });

  it('batch list counts only meetings with rate projections', async () => {
    const { body } = await get('/api/dotplot/batches');
    expect(body.data.items.length).toBe(await one(
      `SELECT COUNT(DISTINCT batch_id) FROM ${S}.sep_projections WHERE federal_funds_rate IS NOT NULL`));
  });

  it('one meeting: dots and medians agree with SQL', async () => {
    const b = (await get('/api/dotplot/batches')).body.data.items[0];
    const { body } = await get(`/api/dotplot/${b.batch_id}`);
    expect(body.data.dots.length).toBe(await one(
      `SELECT COUNT(*) FROM ${S}.sep_projections WHERE batch_id::text = $1 AND federal_funds_rate IS NOT NULL`, [b.batch_id]));
    for (const st of body.data.stats) {
      const med = await one(`SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY federal_funds_rate) FROM ${S}.sep_projections
        WHERE batch_id::text = $1 AND projection_horizon = $2 AND federal_funds_rate IS NOT NULL`, [b.batch_id, st.horizon]);
      expect(st.median).toBeCloseTo(med, 6);
    }
    expect(body.data.dots.every((d: { person_name: unknown }) => d.person_name !== undefined)).toBe(true);
    expect((await get('/api/dotplot/00000000-0000-0000-0000-000000000000')).status).toBe(404);
  });

  it('person tracker: rows and medians agree with SQL', async () => {
    const p = (await get('/api/dotplot/persons')).body.data.items[0];
    const { body } = await get(`/api/dotplot/person/${p.person_id}`);
    expect(body.data.items.length).toBe(await one(
      `SELECT COUNT(*) FROM ${S}.sep_participants sp JOIN ${S}.sep_projections pr
         ON pr.batch_id = sp.batch_id AND pr.participant_number = sp.participant_number
        WHERE sp.person_id::text = $1 AND pr.federal_funds_rate IS NOT NULL`, [p.person_id]));
    const r = body.data.items[0];
    expect(r.median).toBeCloseTo(await one(`SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY federal_funds_rate)
      FROM ${S}.sep_projections WHERE batch_id::text = $1 AND projection_horizon = $2 AND federal_funds_rate IS NOT NULL`,
      [r.batch_id, r.horizon]), 6);
    const lr = body.data.items.find((x: { is_long_run: boolean }) => x.is_long_run);
    if (lr) expect(lr.year_offset).toBeNull();
    expect(body.data.items.some((x: { year_offset: number | null }) => x.year_offset === 0)).toBe(true);
  });
});

describe('meeting roles, transcript filters and PDF links', () => {
  it('meeting list role counts add up to the attendance list', async () => {
    const { body } = await get('/api/meetings?pageSize=100');
    for (const m of body.data.items) expect(m.voter_count + m.nonvoting_count + m.staff_count).toBe(m.participant_count);
  });

  it('meeting detail lists every attendee once, with a role', async () => {
    const list = (await get('/api/meetings?pageSize=1')).body.data.items[0];
    const { body } = await get(`/api/meetings/${list.meeting_id}`);
    expect(body.data.participants.length).toBe(list.participant_count);
    for (const p of body.data.participants) expect(typeof p.role).toBe('string');
  });

  it('person list filters officials vs staff', async () => {
    const all = (await get('/api/persons?kind=all&pageSize=100')).body.data.total;
    const off = (await get('/api/persons?kind=official&pageSize=100')).body.data.total;
    const staff = (await get('/api/persons?kind=staff&pageSize=100')).body.data.total;
    expect(off + staff).toBe(all);
    expect((await get('/api/persons?kind=bogus')).status).toBe(400);
  });

  it('transcript date range, role filter and per-speaker turn numbers', async () => {
    const meetings = (await get('/api/meetings?pageSize=100')).body.data.items;
    const d = String(meetings[meetings.length - 1].meeting_date).slice(0, 10);
    const r = (await get(`/api/transcript?dateFrom=${d}&dateTo=${d}&order=asc&pageSize=100`)).body.data;
    expect(r.items.length).toBeGreaterThan(0);
    for (const u of r.items) {
      expect(String(u.meeting_date).slice(0, 10)).toBe(d);
      expect(u.speaker_turn).toBeGreaterThanOrEqual(1);
      expect(u.speaker_turn).toBeLessThanOrEqual(u.speaker_turns);
    }
    // the k-th turn of a speaker in a meeting is numbered k
    const first = r.items[0];
    const mine = r.items.filter((u: { person_id: string }) => u.person_id === first.person_id);
    mine.forEach((u: { speaker_turn: number }, i: number) => expect(u.speaker_turn).toBe(i + 1));
    expect((await get('/api/transcript?dateFrom=2020-13-45')).status).toBe(400);
    expect((await get('/api/transcript?role=Nobody')).status).toBe(400);
    const voting = (await get('/api/transcript?role=Voting&pageSize=50')).body.data;
    expect(voting.items.length).toBeGreaterThan(0);
    for (const u of voting.items) expect(['Chair', 'Vice Chair', 'Voter']).toContain(u.role);
    const staff = (await get('/api/transcript?role=Staff&pageSize=50')).body.data;
    for (const u of staff.items) expect(u.role).toBe('Staff');
  });

  it('official PDF link is built only from FOMC transcript file names', () => {
    expect(officialPdfUrl('data/FOMC Meetings/Documents/Transcript/2020/FOMC20201216meeting.txt', 12))
      .toBe('https://www.federalreserve.gov/monetarypolicy/files/FOMC20201216meeting.pdf#page=12');
    expect(officialPdfUrl('x/FOMC20200302confcall.txt', null)).toBe('https://www.federalreserve.gov/monetarypolicy/files/FOMC20200302confcall.pdf');
    expect(officialPdfUrl('x/evil.txt', 3)).toBeNull();
    expect(officialPdfUrl(null, 3)).toBeNull();
  });
});

describe('full-text transcript search', () => {
  it('finds turns with all words in any order and ranks by relevance', async () => {
    const r = (await get('/api/transcript?q=' + encodeURIComponent('uncertainty housing') + '&pageSize=50')).body.data;
    expect(r.searchMode).toBe('fulltext');
    expect(r.order).toBe('relevance');
    expect(r.items.length).toBeGreaterThan(0);
    for (const u of r.items) { expect(u.text).toMatch(/uncertaint/i); expect(u.text).toMatch(/housing/i); }
    // -exclude and stemming ("remarks" matches "remark")
    const ex = (await get('/api/transcript?q=' + encodeURIComponent('housing -uncertainty') + '&pageSize=50')).body.data;
    for (const u of ex.items) expect(u.text).not.toMatch(/uncertaint/i);
    expect((await get('/api/transcript?q=remarks&pageSize=1')).body.data.total).toBeGreaterThan(0);
    // explicit date order still works with a keyword
    expect((await get('/api/transcript?q=housing&order=asc')).body.data.order).toBe('asc');
  });
});

describe('AI Q&A', () => {
  it('reports that it is not configured instead of calling Azure, and the demo database has no passages', async () => {
    const res = await app.inject({ method: 'POST', url: '/api/ask', payload: { question: 'Why did Hoenig dissent?' } });
    expect(res.statusCode).toBe(503);
    expect(res.json()).toMatchObject({ unavailable: true });
    expect(repo.avail.rag).toBe(false);
  });
});
