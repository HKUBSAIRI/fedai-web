import type { Queryer, Row } from './db/queryer.js';
import { quoteIdent } from './db/sqlGuard.js';
import { VERIFIED_MISSING_REASONS, VERIFIED_ROWS } from './verified.js';
import {
  hasColumns, PROJECTION_VARS, validateTranscriptMapping,
  type SchemaInfo, type TranscriptMapping,
} from './db/schema.js';

export interface Availability {
  meetings: boolean;
  persons: boolean;
  participants: boolean; // meetings.active_version_id + meeting_participants(version_id, person_id)
  roles: boolean;        // meeting_participants.meeting_role exists → per-meeting role (Chair / Vice Chair / Voter / Participant / Staff)
  sep: boolean;
  transcript: { ok: boolean; reason?: string };
  projections: { ok: boolean; table: string; reason?: string; columns?: string[] };
  rag: boolean;          // rag_chunks (docs/rag/01_向量表.sql) → AI Q&A over the transcripts
}

export interface Stat {
  key: string;
  label: string;
  value: number | string | null;
  definition: string; // how the number is counted
  sql: string | null;
  note?: string;
}

export const MAX_PAGE_SIZE = 100;
export const SEARCH_MAX_LEN = 100;
const COUNT_CAP = 5000;

export function clampPage(page: unknown, pageSize: unknown, def = 25) {
  const ps = Math.min(Math.max(Number(pageSize) || def, 1), MAX_PAGE_SIZE);
  const p = Math.max(Math.floor(Number(page) || 1), 1);
  return { page: p, pageSize: ps, offset: (p - 1) * ps };
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function uuidParam(v: string, name: string): string {
  if (!UUID_RE.test(v)) throw new BadRequestError(`${name} must be a UUID`);
  return v;
}

export function escapeLike(s: string): string {
  return s.replace(/[\\%_]/g, (c) => `\\${c}`);
}

function yearRange(year: unknown): [string, string] | null {
  const y = Number(year);
  if (!Number.isInteger(y) || y < 1900 || y > 2100) return null;
  return [`${y}-01-01`, `${y + 1}-01-01`];
}

/** Per-meeting roles. Chair, Vice Chair and Voter vote; Participant = non-voting FOMC participant; Staff = everyone else. */
export const VOTER_ROLES = ['Chair', 'Vice Chair', 'Voter'];
export const OFFICIAL_ROLES = [...VOTER_ROLES, 'Participant'];
export const ROLES = [...OFFICIAL_ROLES, 'Staff'];
const OFFICIAL_ROLES_SQL = OFFICIAL_ROLES.map((r) => `'${r}'`).join(', ');
const VOTER_ROLES_SQL = VOTER_ROLES.map((r) => `'${r}'`).join(', ');
const ROLE_ORDER_SQL = (col: string) => `CASE ${col} ${ROLES.map((r, i) => `WHEN '${r}' THEN ${i}`).join(' ')} ELSE 9 END`;

const num = (v: unknown) => (v === null || v === undefined ? null : Number(v));

export class Repo {
  readonly avail: Availability;
  private readonly s: string;
  private readonly tm: TranscriptMapping | null;

  constructor(
    private readonly q: Queryer,
    readonly schemaName: string,
    readonly info: SchemaInfo,
    transcriptMapping: TranscriptMapping | null,
    projectionsTable = 'sep_projections',
  ) {
    this.s = quoteIdent(schemaName);
    const tcheck = validateTranscriptMapping(info, transcriptMapping);
    this.tm = tcheck.ok ? tcheck.mapping : null;
    this.avail = {
      meetings: hasColumns(info, 'meetings', ['meeting_id', 'meeting_date']),
      persons: hasColumns(info, 'persons', ['person_id'])
        && (hasColumns(info, 'persons', ['display_name']) || hasColumns(info, 'persons', ['canonical_name'])),
      participants: hasColumns(info, 'meetings', ['meeting_id', 'active_version_id'])
        && hasColumns(info, 'meeting_participants', ['version_id', 'person_id'])
        && hasColumns(info, 'persons', ['person_id']),
      roles: hasColumns(info, 'meetings', ['meeting_id', 'active_version_id'])
        && hasColumns(info, 'meeting_participants', ['version_id', 'person_id', 'meeting_role']),
      sep: hasColumns(info, 'sep_batches', ['batch_id', 'meeting_id', 'meeting_version_id', 'source_meeting_start'])
        && hasColumns(info, 'sep_participants', ['batch_id', 'participant_number', 'person_id', 'source_name', 'mapping_status'])
        && hasColumns(info, 'sep_questions', ['batch_id', 'question_code', 'question_text'])
        && hasColumns(info, 'sep_answers', ['batch_id', 'question_code', 'participant_number', 'response_status', 'answer']),
      transcript: tcheck.ok ? { ok: true } : { ok: false, reason: tcheck.reason },
      projections: this.checkProjections(projectionsTable),
      rag: tcheck.ok && hasColumns(info, 'rag_chunks', ['turn_id', 'chunk_index', 'meeting_date', 'content', 'text_hash', 'model', 'embedding']),
    };
  }

  private checkProjections(table: string): Availability['projections'] {
    const cols = this.info.get(table);
    if (!cols) {
      return { ok: false, table, reason: `Projection table ${this.schemaName}.${table} does not exist, so projection charts are not drawn.` };
    }
    const base = ['batch_id', 'participant_number', 'projection_horizon', 'projection_year', 'is_long_run'];
    const vars = PROJECTION_VARS.filter((v) => cols.includes(v.key));
    if (!base.every((c) => cols.includes(c)) || vars.length === 0) {
      return { ok: false, table, columns: cols, reason: `Table ${table} exists but lacks the expected columns (${base.join(', ')} and at least one projection variable).` };
    }
    return { ok: true, table, columns: cols };
  }

  private t(name: string) { return `${this.s}.${quoteIdent(name)}`; }

  /** Person label: display_name (NOT NULL in FedAI) if present, else canonical_name. */
  private pn(alias: string): string {
    return this.info.get('persons')?.includes('display_name') ? `${alias}.display_name` : `${alias}.canonical_name`;
  }

  private pcol(alias: string, col: string): string {
    return this.info.get('persons')?.includes(col) ? `${alias}.${quoteIdent(col)}` : 'NULL';
  }

  /** Transcript rows restricted to each meeting's ACTIVE import version (older versions excluded). */
  private activeTurnsFrom(alias = 'tr'): string {
    const tm = this.tm!;
    const k = quoteIdent(tm.meetingKey.column);
    const on = tm.meetingKey.references === 'meeting_id' ? `m_act.meeting_id = ${alias}.${k}` : `m_act.active_version_id = ${alias}.${k}`;
    return `${this.t(tm.table)} ${alias} JOIN ${this.t('meetings')} m_act ON ${on}`;
  }

  /**
   * One row per person per meeting (active version) with the meeting role.
   * meeting_role from the database wins. When it is empty, the person is a non-voting 'Participant' if the meeting
   * falls within their tenure as an FOMC official (first..last meeting where they held a role, had an affiliation,
   * or submitted an SEP), otherwise 'Staff'. Once the database stores Participant/Staff, those values are used as is.
   */
  private rolesFrom(): string {
    const M = this.t('meetings'), MP = this.t('meeting_participants');
    const hasAff = this.info.get('meeting_participants')!.includes('affiliation');
    const sep = this.avail.sep
      ? `UNION ALL SELECT sp.person_id, b.source_meeting_start FROM ${this.t('sep_participants')} sp
           JOIN ${this.t('sep_batches')} b ON b.batch_id = sp.batch_id WHERE sp.person_id IS NOT NULL` : '';
    return `(SELECT cur.meeting_id, cur.meeting_date, cur.version_id, cur.person_id, cur.affiliation,
        COALESCE(cur.meeting_role, CASE WHEN cur.meeting_date BETWEEN o.first_d AND o.last_d THEN 'Participant' ELSE 'Staff' END) AS role,
        (cur.meeting_role IS NULL) AS role_inferred
      FROM (SELECT m.meeting_id, m.meeting_date, mp.version_id, mp.person_id, mp.meeting_role,
                   ${hasAff ? 'mp.affiliation' : 'NULL::text'} AS affiliation
              FROM ${M} m JOIN ${MP} mp ON mp.version_id = m.active_version_id) cur
      LEFT JOIN (SELECT person_id, MIN(d) AS first_d, MAX(d) AS last_d FROM (
          SELECT mp.person_id, m.meeting_date AS d FROM ${M} m JOIN ${MP} mp ON mp.version_id = m.active_version_id
           WHERE mp.meeting_role IN (${OFFICIAL_ROLES_SQL})${hasAff ? ' OR mp.affiliation IS NOT NULL' : ''}
          ${sep}) x GROUP BY person_id) o ON o.person_id = cur.person_id)`;
  }

  /** Turns spoken by a person in one meeting version (NULL if the transcript is not keyed by version + person). */
  private turnsOfExpr(versionExpr: string, personExpr: string): string {
    const tm = this.tm;
    if (!tm?.personIdColumn || tm.meetingKey.references !== 'version_id') return 'NULL';
    return `(SELECT COUNT(*) FROM ${this.t(tm.table)} tt WHERE tt.${quoteIdent(tm.meetingKey.column)} = ${versionExpr}
              AND tt.${quoteIdent(tm.personIdColumn)} = ${personExpr})`;
  }

  private require(flag: boolean, what: string) {
    if (!flag) throw new UnavailableError(`${what}: the required tables/columns were not found in this database`);
  }

  // ---------------------------------------------------------------- overview
  async overview() {
    // All queries run in parallel (the database is ~40 ms away); cards keep their order.
    const cards: Promise<Stat>[] = [];
    const add = (key: string, label: string, definition: string, sql: string, note?: string) => {
      cards.push(this.q.query<{ v: unknown }>(sql).then(([r]) => ({ key, label, definition, sql, value: num(r?.v), note })));
    };
    const push = (s: Stat) => cards.push(Promise.resolve(s));
    const M = this.t('meetings');
    if (this.avail.meetings) {
      add('meetings', 'Meetings', 'Rows in meetings (one per meeting_id)', `SELECT COUNT(*) AS v FROM ${M}`);
      const sql = `SELECT MIN(meeting_date)::text AS min_d, MAX(meeting_date)::text AS max_d FROM ${M}`;
      cards.push(this.q.query<{ min_d: string | null; max_d: string | null }>(sql).then(([r]) => ({
        key: 'meeting_range', label: 'Date range', definition: 'Earliest to latest meetings.meeting_date (first day of the meeting)', sql,
        value: r?.min_d ? `${r.min_d.slice(0, 10)} – ${r.max_d!.slice(0, 10)}` : null })));
    }
    if (this.avail.roles) {
      const R = this.rolesFrom();
      add('policymakers', 'FOMC officials', 'Distinct people who were Chair, Vice Chair, Voter or non-voting Participant in at least one meeting',
        `SELECT COUNT(DISTINCT r.person_id) AS v FROM ${R} r WHERE r.role IN (${OFFICIAL_ROLES_SQL})`,
        'The people the project models: Board governors and Reserve Bank presidents (or their alternates).');
      add('staff', 'Staff and other attendees', 'Distinct people who attended meetings but never as an official',
        `SELECT COUNT(DISTINCT r.person_id) AS v FROM ${R} r
          WHERE NOT EXISTS (SELECT 1 FROM ${R} r2 WHERE r2.person_id = r.person_id AND r2.role IN (${OFFICIAL_ROLES_SQL}))`,
        'Secretaries, economists and other staff listed in the attendance roster. Shown on meeting pages, not counted as officials.');
    } else if (this.avail.persons) {
      add('persons', 'People', 'Rows in persons (one per person_id)', `SELECT COUNT(*) AS v FROM ${this.t('persons')}`);
    }
    if (this.tm) {
      add('utterances', 'Transcript turns', `Rows in ${this.tm.table} belonging to each meeting's active import version`,
        `SELECT COUNT(*) AS v FROM ${this.activeTurnsFrom()}`);
    } else {
      push({ key: 'utterances', label: 'Transcript turns', value: null, sql: null, definition: '—', note: this.avail.transcript.reason });
    }
    if (this.avail.sep) {
      add('sep_batches', 'SEP rounds', 'Rows in sep_batches (one per meeting with an SEP)', `SELECT COUNT(*) AS v FROM ${this.t('sep_batches')}`);
      add('sep_participants', 'SEP respondent mappings', 'Rows in sep_participants: anonymous respondent number → person, per SEP round',
        `SELECT COUNT(*) AS v FROM ${this.t('sep_participants')}`);
      add('sep_questions', 'SEP questions', 'Rows in sep_questions, counted per SEP round (the same question in two rounds counts twice)',
        `SELECT COUNT(*) AS v FROM ${this.t('sep_questions')}`);
      if (this.avail.projections.ok) {
        add('sep_projections', 'SEP numeric projections', `Rows in ${this.avail.projections.table}: one per respondent and forecast horizon`,
          `SELECT COUNT(*) AS v FROM ${this.t(this.avail.projections.table)}`);
      }
      add('sep_answers', 'SEP narrative answers', 'Rows in sep_answers: one per respondent and question (answered / na / missing)',
        `SELECT COUNT(*) AS v FROM ${this.t('sep_answers')}`);
    }

    const statusSql = `SELECT response_status AS status, COUNT(*) AS n FROM ${this.t('sep_answers')} GROUP BY response_status ORDER BY response_status`;
    const statusRows = this.avail.sep ? this.q.query<{ status: string; n: unknown }>(statusSql) : Promise.resolve([]);

    const byYear = new Map<number, { year: number; meetings: number; sep_batches: number; sep_answers: number }>();
    const slot = (y: number) => {
      if (!byYear.has(y)) byYear.set(y, { year: y, meetings: 0, sep_batches: 0, sep_answers: 0 });
      return byYear.get(y)!;
    };
    const yearSql: string[] = [];
    if (this.avail.meetings) {
      const sql = `SELECT EXTRACT(YEAR FROM meeting_date) AS y, COUNT(*) AS n FROM ${M} GROUP BY EXTRACT(YEAR FROM meeting_date)`;
      yearSql.push(sql);
    }
    if (this.avail.sep) {
      const sql1 = `SELECT EXTRACT(YEAR FROM source_meeting_start) AS y, COUNT(*) AS n FROM ${this.t('sep_batches')} GROUP BY EXTRACT(YEAR FROM source_meeting_start)`;
      const sql2 = `SELECT EXTRACT(YEAR FROM b.source_meeting_start) AS y, COUNT(*) AS n FROM ${this.t('sep_answers')} a JOIN ${this.t('sep_batches')} b ON b.batch_id = a.batch_id GROUP BY EXTRACT(YEAR FROM b.source_meeting_start)`;
      yearSql.push(sql1, sql2);
    }
    type YN = { y: unknown; n: unknown };
    const [stats, responseRows, meetingYears, sepYears, answerYears] = await Promise.all([
      Promise.all(cards), statusRows,
      this.avail.meetings ? this.q.query<YN>(yearSql[0]) : Promise.resolve([] as YN[]),
      this.avail.sep ? this.q.query<YN>(yearSql[yearSql.length - 2]) : Promise.resolve([] as YN[]),
      this.avail.sep ? this.q.query<YN>(yearSql[yearSql.length - 1]) : Promise.resolve([] as YN[]),
    ]);
    for (const r of meetingYears) if (r.y !== null) slot(Number(r.y)).meetings = Number(r.n);
    for (const r of sepYears) slot(Number(r.y)).sep_batches = Number(r.n);
    for (const r of answerYears) slot(Number(r.y)).sep_answers = Number(r.n);
    const responseStatus = responseRows.map((r) => ({ status: r.status, n: Number(r.n) }));
    return {
      stats,
      responseStatus: { rows: responseStatus, sql: this.avail.sep ? statusSql : null },
      byYear: { rows: [...byYear.values()].sort((a, b) => a.year - b.year), sql: yearSql },
    };
  }

  // ---------------------------------------------------------------- meetings
  async meetingYears(): Promise<number[]> {
    this.require(this.avail.meetings, 'Meetings');
    const rows = await this.q.query<{ y: unknown }>(
      `SELECT DISTINCT EXTRACT(YEAR FROM meeting_date) AS y FROM ${this.t('meetings')} WHERE meeting_date IS NOT NULL ORDER BY 1 DESC`);
    return rows.map((r) => Number(r.y));
  }

  /** Every meeting start date, newest first (for date pickers). */
  async meetingDates(): Promise<string[]> {
    this.require(this.avail.meetings, 'Meetings');
    const rows = await this.q.query<{ d: string }>(`SELECT DISTINCT meeting_date::text AS d FROM ${this.t('meetings')} WHERE meeting_date IS NOT NULL ORDER BY 1 DESC`);
    return rows.map((r) => r.d.slice(0, 10));
  }

  async listMeetings(opts: { year?: unknown; page?: unknown; pageSize?: unknown }) {
    this.require(this.avail.meetings, 'Meetings');
    const { page, pageSize, offset } = clampPage(opts.page, opts.pageSize);
    const params: unknown[] = [];
    let where = '';
    const yr = yearRange(opts.year);
    if (yr) { params.push(yr[0], yr[1]); where = `WHERE m.meeting_date >= $1::date AND m.meeting_date < $2::date`; }
    const hasEnd = this.info.get('meetings')!.includes('meeting_end_date');
    const partCount = this.avail.participants
      ? `(SELECT COUNT(*) FROM ${this.t('meeting_participants')} mp WHERE mp.version_id = m.active_version_id)` : 'NULL';
    const sepCount = this.avail.sep ? `(SELECT COUNT(*) FROM ${this.t('sep_batches')} b WHERE b.meeting_id = m.meeting_id)` : 'NULL';
    const trCount = this.transcriptCountExpr('m');
    const roleJoin = this.avail.roles
      ? `LEFT JOIN (SELECT meeting_id,
            COUNT(*) FILTER (WHERE role IN (${VOTER_ROLES_SQL})) AS voters,
            COUNT(*) FILTER (WHERE role = 'Participant') AS nonvoting,
            COUNT(*) FILTER (WHERE role NOT IN (${OFFICIAL_ROLES_SQL})) AS staff
          FROM ${this.rolesFrom()} r GROUP BY meeting_id) rc ON rc.meeting_id = m.meeting_id` : '';
    const sql = `SELECT m.meeting_id::text AS meeting_id, m.meeting_date::text AS meeting_date,
        ${hasEnd ? 'm.meeting_end_date::text' : 'NULL'} AS meeting_end_date,
        ${partCount} AS participant_count,
        ${roleJoin ? 'rc.voters' : 'NULL'} AS voter_count, ${roleJoin ? 'rc.nonvoting' : 'NULL'} AS nonvoting_count,
        ${roleJoin ? 'rc.staff' : 'NULL'} AS staff_count,
        ${sepCount} AS sep_batch_count, ${trCount} AS utterance_count
      FROM ${this.t('meetings')} m ${roleJoin} ${where}
      ORDER BY m.meeting_date DESC, m.meeting_id
      LIMIT ${pageSize} OFFSET ${offset}`;
    const countSql = `SELECT COUNT(*) AS n FROM ${this.t('meetings')} m ${where}`;
    const [items, [c]] = await Promise.all([this.q.query(sql, params), this.q.query<{ n: unknown }>(countSql, params)]);
    return { items: items.map(numify), total: Number(c.n), page, pageSize, sql };
  }

  private transcriptCountExpr(alias: string): string {
    if (!this.tm) return 'NULL';
    const t = this.t(this.tm.table);
    const k = quoteIdent(this.tm.meetingKey.column);
    const ref = this.tm.meetingKey.references === 'meeting_id' ? `${alias}.meeting_id` : `${alias}.active_version_id`;
    return `(SELECT COUNT(*) FROM ${t} tr WHERE tr.${k} = ${ref})`;
  }

  async meeting(id: string) {
    this.require(this.avail.meetings, 'Meetings');
    const sql = `SELECT * FROM ${this.t('meetings')} m WHERE m.meeting_id::text = $1`;
    const [meeting] = await this.q.query(sql, [id]);
    if (!meeting) return null;

    let participantsP: Promise<Row[]> = Promise.resolve([]);
    let participantsSql: string | null = null;
    if (this.avail.roles) {
      participantsSql = `SELECT r.person_id::text AS person_id, ${this.pn('p')} AS person_name, r.role, r.role_inferred, r.affiliation,
          ${this.turnsOfExpr('r.version_id', 'r.person_id')} AS turns
        FROM ${this.rolesFrom()} r
        LEFT JOIN ${this.t('persons')} p ON p.person_id = r.person_id
        WHERE r.meeting_id::text = $1
        ORDER BY ${ROLE_ORDER_SQL('r.role')}, ${this.pn('p')}`;
      participantsP = this.q.query(participantsSql, [id]).then((rows) => rows.map(numify));
    } else if (this.avail.participants) {
      participantsSql = `SELECT p.person_id::text AS person_id, ${this.pn('p')} AS person_name, mp.*
        FROM ${this.t('meetings')} m
        JOIN ${this.t('meeting_participants')} mp ON mp.version_id = m.active_version_id
        LEFT JOIN ${this.t('persons')} p ON p.person_id = mp.person_id
        WHERE m.meeting_id::text = $1
        ORDER BY ${this.pn('p')}`;
      participantsP = this.q.query(participantsSql, [id]);
    }
    // the three lookups are independent: run them in parallel (the database is ~40 ms away)
    const [participants, sepBatches, utteranceCount] = await Promise.all([
      participantsP,
      this.avail.sep ? this.sepBatchRows('WHERE b.meeting_id::text = $1', [id]) : Promise.resolve([] as Row[]),
      this.tm
        ? this.q.query<{ n: unknown }>(`SELECT ${this.transcriptCountExpr('m')} AS n FROM ${this.t('meetings')} m WHERE m.meeting_id::text = $1`, [id])
            .then(([r]) => Number(r.n))
        : Promise.resolve(null),
    ]);
    return { meeting: stringifyRow(meeting), participants: participants.map(stringifyRow), participantsSql, sepBatches, utteranceCount, sql };
  }

  // ---------------------------------------------------------------- economy
  /**
   * Economic background of one meeting:
   * - indicators: fedai_corpus.meeting_macro_data (values as published the day before the meeting);
   * - context: the newest sep_dual_agent.meeting_macro_contexts revision for the meeting start date, i.e. the AI summary
   *   of the Beige Book and Greenbook / Tealbook available before the meeting, with page citations linked to the Fed site.
   * Either part is null (with a reason) when its tables are missing or not readable.
   */
  async economy(id: string) {
    this.require(this.avail.meetings, 'Meetings');
    const [meeting] = await this.q.query<{ meeting_date: unknown }>(
      `SELECT m.meeting_date FROM ${this.t('meetings')} m WHERE m.meeting_id = $1::uuid`, [uuidParam(id, 'meetingId')]);
    if (!meeting) return null;
    const date = isoDate(meeting.meeting_date);

    let indicators: Row | null = null;
    let indicatorsNote: string | null = null;
    const indicatorsSql = `SELECT * FROM ${this.t('meeting_macro_data')} WHERE meeting_id = $1::uuid`;
    if (hasColumns(this.info, 'meeting_macro_data', ['meeting_id', 'vintage_date'])) {
      [indicators = null] = (await this.q.query(indicatorsSql, [id])).map(stringifyRow);
      if (!indicators) indicatorsNote = 'No indicators stored for this meeting.';
    } else indicatorsNote = 'Table meeting_macro_data is not in this database.';

    const X = (t: string) => `${quoteIdent(MACRO_SCHEMA)}.${quoteIdent(t)}`;
    const contextSql = `SELECT c.cutoff_date, c.status, c.context_json FROM ${X('meeting_macro_contexts')} c
      WHERE c.event_start_date = $1::date ORDER BY c.imported_at DESC LIMIT 1`;
    const docsSql = `SELECT DISTINCT ON (d.source_id) d.source_id::text AS source_id, d.kind, d.title, d.source_path, d.document_date
      FROM ${X('macro_documents')} d WHERE d.source_id = ANY($1::uuid[]) ORDER BY d.source_id, d.imported_at DESC`;
    let context: ReturnType<typeof shapeContext> | null = null;
    let contextNote: string | null = null;
    try {
      const [c] = await this.q.query<{ cutoff_date: unknown; status: unknown; context_json: MacroContext }>(contextSql, [date]);
      if (!c) contextNote = 'No pre-meeting economic summary for this meeting.';
      else {
        const ids = (c.context_json.source_ids ?? []).filter((s) => UUID_RE.test(s));
        const docs = ids.length ? await this.q.query(docsSql, [ids]) : [];
        context = shapeContext(c.context_json, docs, isoDate(c.cutoff_date), String(c.status));
      }
    } catch (e) {
      const code = (e as { code?: string }).code;
      if (code === '42P01' || code === '3F000' || code === '42501') {
        contextNote = code === '42501' ? `The read-only account cannot read schema ${MACRO_SCHEMA}.` : `Schema ${MACRO_SCHEMA} is not in this database.`;
      } else throw e;
    }
    return { meetingDate: date, indicators, indicatorsNote, context, contextNote, sql: [indicatorsSql, contextSql, docsSql].join(';\n\n') };
  }

  /** Indicators of every meeting (meeting_macro_data), with a link to the meeting page when the meeting is in the corpus. */
  async economyList() {
    this.require(hasColumns(this.info, 'meeting_macro_data', ['meeting_date', 'vintage_date']), 'Economic indicators');
    const sql = `SELECT x.*, m.meeting_id::text AS corpus_meeting_id FROM ${this.t('meeting_macro_data')} x
      LEFT JOIN ${this.t('meetings')} m ON m.meeting_date = x.meeting_date ORDER BY x.meeting_date DESC`;
    return { items: (await this.q.query(sql)).map(stringifyRow), sql };
  }

  // ---------------------------------------------------------------- persons
  async listPersons(opts: { q?: unknown; kind?: unknown; page?: unknown; pageSize?: unknown }) {
    this.require(this.avail.persons, 'Participants');
    const { page, pageSize, offset } = clampPage(opts.page, opts.pageSize, 50);
    const params: unknown[] = [];
    const conds: string[] = [];
    const term = typeof opts.q === 'string' ? opts.q.trim().slice(0, SEARCH_MAX_LEN) : '';
    if (term) {
      params.push(`%${escapeLike(term)}%`);
      const searchable = ['display_name', 'canonical_name', 'identity_key'].filter((c) => this.info.get('persons')!.includes(c));
      conds.push(`(${searchable.map((c) => `p.${quoteIdent(c)} ILIKE $1`).join(' OR ')})`);
    }
    const roleAgg = this.avail.roles
      ? `LEFT JOIN (SELECT person_id,
            COUNT(*) FILTER (WHERE role IN (${VOTER_ROLES_SQL})) AS voter_meetings,
            COUNT(*) FILTER (WHERE role IN (${OFFICIAL_ROLES_SQL})) AS official_meetings,
            bool_or(role = 'Chair') AS was_chair
          FROM ${this.rolesFrom()} r GROUP BY person_id) ra ON ra.person_id = p.person_id` : '';
    if (roleAgg && opts.kind === 'official') conds.push('COALESCE(ra.official_meetings, 0) > 0');
    else if (roleAgg && opts.kind === 'staff') conds.push('COALESCE(ra.official_meetings, 0) = 0');
    else if (opts.kind && opts.kind !== 'all') throw new BadRequestError('kind must be official, staff or all');
    const where = conds.length ? `WHERE ${conds.join(' AND ')}` : '';
    const meetingsExpr = this.avail.participants
      ? `(SELECT COUNT(DISTINCT m.meeting_id) FROM ${this.t('meeting_participants')} mp JOIN ${this.t('meetings')} m ON m.active_version_id = mp.version_id WHERE mp.person_id = p.person_id)`
      : 'NULL';
    const sepExpr = this.avail.sep ? `(SELECT COUNT(*) FROM ${this.t('sep_participants')} sp WHERE sp.person_id = p.person_id)` : 'NULL';
    const trExpr = this.tm?.personIdColumn
      ? `(SELECT COUNT(*) FROM ${this.activeTurnsFrom()} WHERE tr.${quoteIdent(this.tm.personIdColumn)} = p.person_id)` : 'NULL';
    const sql = `SELECT p.person_id::text AS person_id, ${this.pn('p')} AS person_name,
        ${this.pcol('p', 'canonical_name')} AS canonical_name, ${this.pcol('p', 'identity_status')} AS identity_status,
        ${meetingsExpr} AS meeting_count, ${roleAgg ? 'COALESCE(ra.official_meetings, 0)' : 'NULL'} AS official_meetings,
        ${roleAgg ? 'COALESCE(ra.voter_meetings, 0)' : 'NULL'} AS voter_meetings,
        ${sepExpr} AS sep_count, ${trExpr} AS utterance_count
      FROM ${this.t('persons')} p ${roleAgg} ${where}
      ORDER BY ${this.pn('p')}, p.person_id
      LIMIT ${pageSize} OFFSET ${offset}`;
    const countSql = `SELECT COUNT(*) AS n FROM ${this.t('persons')} p ${roleAgg} ${where}`;
    const [items, [c]] = await Promise.all([this.q.query(sql, params), this.q.query<{ n: unknown }>(countSql, params)]);
    return { items: items.map(numify), total: Number(c.n), page, pageSize, sql };
  }

  async person(id: string) {
    this.require(this.avail.persons, 'Participants');
    const [person] = await this.q.query(`SELECT * FROM ${this.t('persons')} p WHERE p.person_id::text = $1`, [id]);
    if (!person) return null;
    const hasEnd = this.info.get('meetings')!.includes('meeting_end_date');
    const meetings = this.avail.roles ? (await this.q.query(
      `SELECT r.meeting_id::text AS meeting_id, r.meeting_date::text AS meeting_date,
              ${hasEnd ? 'm.meeting_end_date::text' : 'NULL'} AS meeting_end_date,
              r.role, r.role_inferred, r.affiliation, ${this.turnsOfExpr('r.version_id', 'r.person_id')} AS turns
         FROM ${this.rolesFrom()} r JOIN ${this.t('meetings')} m ON m.meeting_id = r.meeting_id
        WHERE r.person_id::text = $1
        ORDER BY r.meeting_date`, [id])).map(numify)
      : this.avail.participants ? await this.q.query(
      `SELECT DISTINCT m.meeting_id::text AS meeting_id, m.meeting_date::text AS meeting_date,
              ${hasEnd ? 'm.meeting_end_date::text' : 'NULL'} AS meeting_end_date
         FROM ${this.t('meeting_participants')} mp
         JOIN ${this.t('meetings')} m ON m.active_version_id = mp.version_id
        WHERE mp.person_id::text = $1
        ORDER BY meeting_date`, [id]) : [];
    const sep = this.avail.sep ? await this.q.query(
      `SELECT b.batch_id::text AS batch_id, b.source_meeting_start::text AS meeting_start,
              sp.participant_number, sp.source_name, sp.mapping_status,
              (SELECT COUNT(*) FROM ${this.t('sep_answers')} a
                WHERE a.batch_id = sp.batch_id AND a.participant_number = sp.participant_number) AS answers,
              (SELECT COUNT(*) FROM ${this.t('sep_answers')} a
                WHERE a.batch_id = sp.batch_id AND a.participant_number = sp.participant_number
                  AND a.response_status = 'answered') AS answered
         FROM ${this.t('sep_participants')} sp
         JOIN ${this.t('sep_batches')} b ON b.batch_id = sp.batch_id
        WHERE sp.person_id::text = $1
        ORDER BY b.source_meeting_start`, [id]) : [];
    const [sameName] = await this.q.query<{ n: unknown }>(
      `SELECT COUNT(*) AS n FROM ${this.t('persons')} p2
        WHERE lower(${this.pn('p2')}) = lower($1) AND p2.person_id::text <> $2`,
      [person[this.info.get('persons')!.includes('display_name') ? 'display_name' : 'canonical_name'], id]);
    return {
      person: stringifyRow(person),
      personName: person.display_name ?? person.canonical_name ?? null,
      meetings,
      sep: sep.map(numify),
      sameNameOtherIds: Number(sameName.n),
      transcriptByPerson: Boolean(this.tm?.personIdColumn),
    };
  }

  // ---------------------------------------------------------------- transcript
  /** Filters shared by the transcript search and its speaker list: meeting, year, start-date range, speaker role. */
  private transcriptFilters(opts: { meetingId?: unknown; year?: unknown; dateFrom?: unknown; dateTo?: unknown; role?: unknown }, params: unknown[]) {
    const tm = this.tm!;
    const conds: string[] = [];
    let roleJoin = '';
    if (typeof opts.meetingId === 'string' && opts.meetingId) {
      params.push(uuidParam(opts.meetingId, 'meetingId')); conds.push(`m.meeting_id = $${params.length}::uuid`);
    }
    const yr = yearRange(opts.year);
    if (opts.year && !yr) throw new BadRequestError('Invalid year');
    if (yr) {
      params.push(yr[0], yr[1]);
      conds.push(`m.meeting_date >= $${params.length - 1}::date AND m.meeting_date < $${params.length}::date`);
    }
    for (const [key, op] of [['dateFrom', '>='], ['dateTo', '<=']] as const) {
      const v = opts[key];
      if (v === undefined || v === '') continue;
      if (typeof v !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(v) || Number.isNaN(Date.parse(v))) throw new BadRequestError(`${key} must be a date (YYYY-MM-DD)`);
      params.push(v); conds.push(`m.meeting_date ${op} $${params.length}::date`);
    }
    const rolesUsable = this.avail.roles && Boolean(tm.personIdColumn) && tm.meetingKey.references === 'version_id';
    if (rolesUsable) {
      // one hash join with the per-meeting roles (computed once), not a lookup per row
      roleJoin = `LEFT JOIN ${this.rolesFrom()} rf ON rf.version_id = tr.${quoteIdent(tm.meetingKey.column)} AND rf.person_id = tr.${quoteIdent(tm.personIdColumn!)}`;
    }
    if (typeof opts.role === 'string' && opts.role) {
      if (!rolesUsable) throw new BadRequestError('Role filter is not available for this database');
      const roles = opts.role === 'Voting' ? VOTER_ROLES : opts.role === 'Official' ? OFFICIAL_ROLES : [opts.role];
      if (!roles.every((r) => ROLES.includes(r))) throw new BadRequestError(`role must be one of ${['Voting', 'Official', ...ROLES].join(', ')}`);
      params.push(roles); conds.push(`rf.role = ANY($${params.length}::text[])`);
    }
    return { conds, roleJoin };
  }

  private hasFullText(): boolean {
    return Boolean(this.tm) && (this.info.get(this.tm!.table) ?? []).includes('search_tsv');
  }

  private transcriptJoin(): string {
    const tm = this.tm!;
    return tm.meetingKey.references === 'meeting_id'
      ? `m.meeting_id = tr.${quoteIdent(tm.meetingKey.column)}`
      : `m.active_version_id = tr.${quoteIdent(tm.meetingKey.column)}`;
  }

  async searchTranscript(opts: { q?: unknown; personId?: unknown; meetingId?: unknown; year?: unknown; dateFrom?: unknown; dateTo?: unknown; role?: unknown; order?: unknown; page?: unknown; pageSize?: unknown }) {
    if (!this.tm) throw new UnavailableError(this.avail.transcript.reason ?? 'Transcript is not configured');
    const tm = this.tm;
    const { page, pageSize, offset } = clampPage(opts.page, opts.pageSize, 20);
    const params: unknown[] = [];
    const term = typeof opts.q === 'string' ? opts.q.trim().slice(0, SEARCH_MAX_LEN) : '';
    if (term && term.length < 2) throw new BadRequestError('Keyword must be at least 2 characters');
    const conds: string[] = [];
    // Full-text search when the table has the generated tsvector column (docs/transcriptfix/17): all words in any order,
    // "quoted phrases", OR, -exclude, English stemming, ranked by relevance. Otherwise a plain substring match.
    const fts = this.hasFullText();
    let rank = '';
    if (term && fts) {
      params.push(term);
      const tq = `websearch_to_tsquery('english', $${params.length})`;
      conds.push(`tr.search_tsv @@ ${tq}`);
      rank = `ts_rank(tr.search_tsv, ${tq})`;
    } else if (term) { params.push(`%${escapeLike(term)}%`); conds.push(`tr.${quoteIdent(tm.textColumn)} ILIKE $${params.length}`); }
    if (typeof opts.personId === 'string' && opts.personId) {
      if (!tm.personIdColumn) throw new BadRequestError('The transcript mapping has no person_id column, so it cannot be filtered by person');
      // compare as uuid (not ::text) so the (person_id, version_id) index is used
      params.push(uuidParam(opts.personId, 'personId')); conds.push(`tr.${quoteIdent(tm.personIdColumn)} = $${params.length}::uuid`);
    }
    const f = this.transcriptFilters(opts, params);
    conds.push(...f.conds);
    // Newest meetings first by default (as in the meeting list); within a meeting, turns stay in spoken order.
    // With a full-text keyword the default is most relevant first.
    const byRelevance = Boolean(rank) && (opts.order === 'relevance' || !opts.order);
    const dir = opts.order === 'asc' ? 'ASC' : 'DESC';
    const join = this.transcriptJoin();
    const personJoin = tm.personIdColumn && this.avail.persons
      ? `LEFT JOIN ${this.t('persons')} p ON p.person_id = tr.${quoteIdent(tm.personIdColumn)}` : '';
    const where = conds.length ? `WHERE ${conds.join(' AND ')}` : '';
    const extra = (tm.sourceColumns ?? []).map((c) => `tr.${quoteIdent(c)}::text AS ${quoteIdent('src_' + c)}`);
    const order = tm.orderColumn ? `, tr.${quoteIdent(tm.orderColumn)}` : '';
    // Speaker's own turn number within the meeting ("8th of 12"), counted over the whole meeting, not the filtered set.
    const vk = quoteIdent(tm.meetingKey.column);
    const speakerTurns = tm.personIdColumn && tm.orderColumn
      ? `(SELECT COUNT(*) FROM ${this.t(tm.table)} t2 WHERE t2.${vk} = tr.${vk} AND t2.${quoteIdent(tm.personIdColumn)} = tr.${quoteIdent(tm.personIdColumn)}
            AND t2.${quoteIdent(tm.orderColumn)} <= tr.${quoteIdent(tm.orderColumn)}) AS speaker_turn,
         (SELECT COUNT(*) FROM ${this.t(tm.table)} t2 WHERE t2.${vk} = tr.${vk} AND t2.${quoteIdent(tm.personIdColumn)} = tr.${quoteIdent(tm.personIdColumn)}) AS speaker_turns`
      : 'NULL AS speaker_turn, NULL AS speaker_turns';
    // Total turns in the meeting ("Turn #96 of 312")
    const meetingTurns = `(SELECT COUNT(*) FROM ${this.t(tm.table)} t3 WHERE t3.${vk} = tr.${vk}) AS meeting_turns`;
    const roleCol = f.roleJoin ? 'rf.role' : 'NULL';
    const pdf = this.transcriptPdfSql();
    const sql = `SELECT tr.${quoteIdent(tm.idColumn)}::text AS utterance_id,
        m.meeting_id::text AS meeting_id, m.meeting_date::text AS meeting_date,
        ${tm.personIdColumn ? `tr.${quoteIdent(tm.personIdColumn)}::text` : 'NULL'} AS person_id,
        ${personJoin ? this.pn('p') : 'NULL'} AS person_name,
        ${tm.speakerNameColumn ? `tr.${quoteIdent(tm.speakerNameColumn)}::text` : 'NULL'} AS speaker_label,
        ${tm.orderColumn ? `tr.${quoteIdent(tm.orderColumn)}` : 'NULL'} AS seq,
        ${speakerTurns}, ${meetingTurns}, ${roleCol} AS role, ${pdf.cols},
        tr.${quoteIdent(tm.textColumn)} AS text${extra.length ? ',\n        ' + extra.join(', ') : ''}
      FROM ${this.t(tm.table)} tr
      JOIN ${this.t('meetings')} m ON ${join}
      ${f.roleJoin}
      ${personJoin}
      ${pdf.join}
      ${where}
      ORDER BY ${byRelevance ? `${rank} DESC, ` : ''}m.meeting_date ${dir}${order}, tr.${quoteIdent(tm.idColumn)}
      LIMIT ${pageSize + 1} OFFSET ${offset}`;
    const countJoin = typeof opts.role === 'string' && opts.role ? f.roleJoin : '';
    const countSql = `SELECT COUNT(*) AS n FROM (SELECT 1 FROM ${this.t(tm.table)} tr JOIN ${this.t('meetings')} m ON ${join} ${countJoin} ${where} LIMIT ${COUNT_CAP + 1}) c`;
    const [rows, [c]] = await Promise.all([this.q.query(sql, params), this.q.query<{ n: unknown }>(countSql, params)]);
    const n = Number(c.n);
    return {
      items: rows.slice(0, pageSize).map((r) => {
        const { source_file, ...rest } = numify(r);
        return { ...rest, pdf_url: officialPdfUrl(source_file, rest.source_page) };
      }),
      hasMore: rows.length > pageSize,
      total: Math.min(n, COUNT_CAP),
      totalCapped: n > COUNT_CAP,
      page, pageSize, sql,
      order: byRelevance ? 'relevance' : dir === 'ASC' ? 'asc' : 'desc',
      searchMode: fts ? 'fulltext' : 'substring',
      sourceColumns: tm.sourceColumns ?? [],
      table: tm.table,
      roles: this.avail.roles ? ['Voting', 'Official', ...ROLES] : [],
    };
  }

  /** Source document file name and page of each turn, for the link to the official PDF (only if the tables carry them). */
  private transcriptPdfSql(): { cols: string; join: string } {
    const tm = this.tm!;
    const tcols = this.info.get(tm.table) ?? [];
    const page = tcols.includes('source_page') ? 'tr.source_page' : 'NULL';
    if (tcols.includes('document_id') && hasColumns(this.info, 'documents', ['document_id', 'source_path'])) {
      return { cols: `doc.source_path AS source_file, ${page} AS source_page`, join: `LEFT JOIN ${this.t('documents')} doc ON doc.document_id = tr.document_id` };
    }
    return { cols: `NULL AS source_file, ${page} AS source_page`, join: '' };
  }

  /**
   * Transcript passages closest to a question (AI Q&A): cosine distance on rag_chunks.embedding, best passage per turn.
   * Passages of inactive import versions, or of turns whose text changed after embedding (md5 differs), are left out.
   */
  async ragSearch(vector: string, model: string, opts: { dateFrom?: unknown; dateTo?: unknown }, limit = 12) {
    this.require(this.avail.rag, 'AI Q&A (table rag_chunks)');
    const tm = this.tm!;
    const params: unknown[] = [vector, model];
    const conds = ['c.model = $2'];
    for (const [key, op] of [['dateFrom', '>='], ['dateTo', '<=']] as const) {
      const v = opts[key];
      if (typeof v !== 'string' || !v) continue;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(v)) throw new BadRequestError(`${key} must be YYYY-MM-DD`);
      params.push(v); conds.push(`c.meeting_date ${op} $${params.length}::date`);
    }
    const dist = 'c.embedding <=> $1::vector';
    // Unfiltered: the HNSW index finds the nearest passages. With a date filter the index would rank first and filter
    // afterwards (possibly returning too few), so "+ 0" makes the database rank the filtered passages exactly instead.
    const rankBy = conds.length > 1 ? `(${dist}) + 0` : dist;
    const personJoin = tm.personIdColumn && this.avail.persons
      ? `LEFT JOIN ${this.t('persons')} p ON p.person_id = tr.${quoteIdent(tm.personIdColumn)}` : '';
    const pdf = this.transcriptPdfSql();
    const sql = `WITH hits AS (
        SELECT c.turn_id, c.chunk_index, c.content, c.text_hash, ${dist} AS distance
          FROM ${this.t('rag_chunks')} c
         WHERE ${conds.join(' AND ')}
         ORDER BY ${rankBy}
         LIMIT ${limit * 3})
      SELECT h.turn_id::text AS turn_id, h.chunk_index, h.content, h.distance,
             m.meeting_id::text AS meeting_id, m.meeting_date::text AS meeting_date,
             ${tm.personIdColumn ? `tr.${quoteIdent(tm.personIdColumn)}::text` : 'NULL'} AS person_id,
             ${personJoin ? this.pn('p') : 'NULL'} AS person_name,
             ${tm.speakerNameColumn ? `tr.${quoteIdent(tm.speakerNameColumn)}::text` : 'NULL'} AS speaker_label,
             ${tm.orderColumn ? `tr.${quoteIdent(tm.orderColumn)}` : 'NULL'} AS seq, ${pdf.cols}
        FROM hits h
        JOIN ${this.t(tm.table)} tr ON tr.${quoteIdent(tm.idColumn)} = h.turn_id
        JOIN ${this.t('meetings')} m ON ${this.transcriptJoin()}
        ${personJoin}
        ${pdf.join}
       WHERE md5(tr.${quoteIdent(tm.textColumn)}) = h.text_hash
       ORDER BY h.distance`;
    const rows = (await this.q.query(sql, params)).map(numify);
    const seen = new Set<unknown>();
    const items: Row[] = [];
    for (const r of rows) {
      if (seen.has(r.turn_id)) continue;   // one passage per turn: the closest
      seen.add(r.turn_id);
      const { source_file, ...rest } = r;
      items.push({ ...rest, pdf_url: officialPdfUrl(source_file, rest.source_page) });
      if (items.length >= limit) break;
    }
    return { items, sql: sql.replaceAll('$1::vector', "'[…question embedding…]'::vector") };
  }

  /** People who have transcript turns (optionally within the same filters), most turns first — for the speaker filter. */
  async transcriptSpeakers(opts: { year?: unknown; dateFrom?: unknown; dateTo?: unknown; role?: unknown; meetingId?: unknown }) {
    if (!this.tm) throw new UnavailableError(this.avail.transcript.reason ?? 'Transcript is not configured');
    const tm = this.tm;
    if (!tm.personIdColumn || !this.avail.persons) return { items: [], sql: null };
    const params: unknown[] = [];
    const f = this.transcriptFilters(opts, params);
    const sql = `SELECT p.person_id::text AS person_id, ${this.pn('p')} AS person_name, COUNT(*) AS turns
      FROM ${this.t(tm.table)} tr
      JOIN ${this.t('meetings')} m ON ${this.transcriptJoin()}
      ${f.roleJoin}
      JOIN ${this.t('persons')} p ON p.person_id = tr.${quoteIdent(tm.personIdColumn)}
      ${f.conds.length ? `WHERE ${f.conds.join(' AND ')}` : ''}
      GROUP BY 1, 2
      ORDER BY COUNT(*) DESC, 2`;
    return { items: (await this.q.query(sql, params)).map(numify), sql };
  }

  // ---------------------------------------------------------------- SEP
  private async sepBatchRows(where: string, params: unknown[]) {
    const hasEnd = this.info.get('sep_batches')!.includes('source_meeting_end');
    const rows = await this.q.query(
      `SELECT b.batch_id::text AS batch_id, b.meeting_id::text AS meeting_id,
              b.source_meeting_start::text AS meeting_start,
              ${hasEnd ? 'b.source_meeting_end::text' : 'NULL'} AS meeting_end,
              (SELECT COUNT(*) FROM ${this.t('sep_participants')} p WHERE p.batch_id = b.batch_id) AS participants,
              (SELECT COUNT(*) FROM ${this.t('sep_questions')} q WHERE q.batch_id = b.batch_id) AS questions,
              (SELECT COUNT(*) FROM ${this.t('sep_answers')} a WHERE a.batch_id = b.batch_id) AS answers,
              (SELECT COUNT(*) FROM ${this.t('sep_answers')} a WHERE a.batch_id = b.batch_id AND a.response_status = 'answered') AS answered,
              (SELECT COUNT(*) FROM ${this.t('sep_answers')} a WHERE a.batch_id = b.batch_id AND a.response_status = 'na') AS na,
              (SELECT COUNT(*) FROM ${this.t('sep_answers')} a WHERE a.batch_id = b.batch_id AND a.response_status = 'missing') AS missing
         FROM ${this.t('sep_batches')} b ${where}
        ORDER BY b.source_meeting_start`, params);
    return rows.map(numify);
  }

  async listSepBatches(opts: { year?: unknown }) {
    this.require(this.avail.sep, 'SEP');
    const yr = yearRange(opts.year);
    return {
      items: yr
        ? await this.sepBatchRows('WHERE b.source_meeting_start >= $1::date AND b.source_meeting_start < $2::date', yr)
        : await this.sepBatchRows('', []),
      projections: this.avail.projections,
    };
  }

  async sepBatch(id: string) {
    this.require(this.avail.sep, 'SEP');
    const [batch] = await this.sepBatchRows('WHERE b.batch_id::text = $1', [id]);
    if (!batch) return null;
    const pcols = this.info.get('sep_participants')!;
    const participants = await this.q.query(
      `SELECT sp.participant_number, sp.source_name,
              ${pcols.includes('source_affiliation') ? 'sp.source_affiliation' : 'NULL'} AS source_affiliation,
              sp.person_id::text AS person_id, ${this.pn('per')} AS person_name, sp.mapping_status,
              ${pcols.includes('mapping_notes') ? 'sp.mapping_notes' : 'NULL'} AS mapping_notes
         FROM ${this.t('sep_participants')} sp
         LEFT JOIN ${this.t('persons')} per ON per.person_id = sp.person_id
        WHERE sp.batch_id::text = $1
        ORDER BY sp.participant_number`, [id]);
    const qcols = this.info.get('sep_questions')!;
    const questions = await this.q.query(
      `SELECT q.question_code, q.question_text, ${qcols.includes('question_order') ? 'q.question_order' : 'NULL'} AS question_order
         FROM ${this.t('sep_questions')} q WHERE q.batch_id::text = $1
        ORDER BY ${qcols.includes('question_order') ? 'q.question_order NULLS LAST, ' : ''}q.question_code`, [id]);
    const acols = this.info.get('sep_answers')!;
    const opt = (c: string) => (acols.includes(c) ? `a.${quoteIdent(c)}` : 'NULL');
    const answersSql = `SELECT a.question_code, a.participant_number, a.response_status, a.answer,
              ${opt('reference_doc')} AS reference_doc, ${opt('reference_doc_name')} AS reference_doc_name,
              ${opt('source_file_name')} AS source_file_name, ${opt('source_record_number')} AS source_record_number
         FROM ${this.t('sep_answers')} a WHERE a.batch_id::text = $1
        ORDER BY a.question_code, a.participant_number`;
    const answers = await this.q.query(answersSql, [id]);
    let projections: Row[] | null = null;
    let projectionsSql: string | null = null;
    if (this.avail.projections.ok) {
      const cols = this.avail.projections.columns!;
      const vars = PROJECTION_VARS.filter((v) => cols.includes(v.key)).map((v) => `pr.${quoteIdent(v.key)}`);
      const hasMissing = cols.includes('source_missing_fields');
      projectionsSql = `SELECT pr.participant_number, pr.projection_horizon AS horizon, pr.projection_year, pr.is_long_run,
              ${vars.join(', ')}${hasMissing ? ', pr.source_missing_fields' : ''}
         FROM ${this.t(this.avail.projections.table)} pr WHERE pr.batch_id::text = $1
        ORDER BY pr.is_long_run, pr.projection_year NULLS LAST, pr.projection_horizon, pr.participant_number`;
      projections = (await this.q.query(projectionsSql, [id])).map((r) => stringifyRow(numify(r)));
    }
    return {
      batch,
      participants: participants.map(numify),
      questions: questions.map(numify),
      answers: answers.map(numify),
      answersSql,
      projections,
      projectionsSql,
      projectionsStatus: this.avail.projections,
    };
  }

  // ---------------------------------------------------------------- dot plot (federal funds rate)
  private requireFfr() {
    this.require(this.avail.sep && this.avail.projections.ok
      && (this.avail.projections.columns ?? []).includes('federal_funds_rate'), 'Federal funds rate dot plot');
    return this.t(this.avail.projections.table);
  }

  /** SEP batches that contain at least one federal funds rate projection. */
  async dotplotBatches() {
    const P = this.requireFfr();
    const sql = `SELECT b.batch_id::text AS batch_id, b.source_meeting_start::text AS meeting_start,
              COUNT(pr.federal_funds_rate) AS dots, COUNT(DISTINCT pr.participant_number) AS participants
         FROM ${this.t('sep_batches')} b
         JOIN ${P} pr ON pr.batch_id = b.batch_id
        GROUP BY b.batch_id, b.source_meeting_start
       HAVING COUNT(pr.federal_funds_rate) > 0
        ORDER BY b.source_meeting_start`;
    return { items: (await this.q.query(sql)).map(numify), sql };
  }

  /** One meeting's dot plot: every participant's year-end rate projection per horizon, plus per-horizon stats. */
  async dotplot(batchId: string) {
    const P = this.requireFfr();
    const sql = `SELECT pr.projection_horizon AS horizon, pr.projection_year, pr.is_long_run,
              pr.participant_number, pr.federal_funds_rate AS value,
              sp.person_id::text AS person_id, ${this.pn('per')} AS person_name
         FROM ${P} pr
         JOIN ${this.t('sep_participants')} sp ON sp.batch_id = pr.batch_id AND sp.participant_number = pr.participant_number
         LEFT JOIN ${this.t('persons')} per ON per.person_id = sp.person_id
        WHERE pr.batch_id::text = $1 AND pr.federal_funds_rate IS NOT NULL
        ORDER BY pr.is_long_run, pr.projection_year NULLS LAST, pr.federal_funds_rate, pr.participant_number`;
    const dots = (await this.q.query(sql, [batchId])).map((r) => ({ ...numify(r), value: Number(r.value) }) as Row & { value: number });
    if (!dots.length) return null;
    const [batch] = await this.q.query(
      `SELECT b.batch_id::text AS batch_id, b.source_meeting_start::text AS meeting_start, b.meeting_id::text AS meeting_id
         FROM ${this.t('sep_batches')} b WHERE b.batch_id::text = $1`, [batchId]);
    const horizons = [...new Set(dots.map((d) => String(d.horizon)))];
    const stats = horizons.map((h) => {
      const sel = dots.filter((d) => String(d.horizon) === h);
      const st = dotStats(sel.map((d) => d.value));
      return {
        horizon: h, ...st,
        outliers: sel.filter((d) => isOutlier(d.value, st)).map((d) => ({
          participant_number: d.participant_number, person_name: d.person_name, value: d.value })),
      };
    });
    return { batch, dots, stats, sql, outlierRule: OUTLIER_RULE };
  }

  /** People who have at least one federal funds rate projection (for the tracker selector). */
  async dotplotPersons() {
    const P = this.requireFfr();
    const sql = `SELECT sp.person_id::text AS person_id, ${this.pn('per')} AS person_name,
              COUNT(DISTINCT sp.batch_id) AS meetings
         FROM ${this.t('sep_participants')} sp
         JOIN ${P} pr ON pr.batch_id = sp.batch_id AND pr.participant_number = sp.participant_number
         JOIN ${this.t('persons')} per ON per.person_id = sp.person_id
        WHERE pr.federal_funds_rate IS NOT NULL
        GROUP BY sp.person_id, ${this.pn('per')}
        ORDER BY ${this.pn('per')}`;
    return { items: (await this.q.query(sql)).map(numify), sql };
  }

  /**
   * One person's rate projections across meetings, with the committee median of the same meeting and horizon.
   * offset = projection_year - year of meeting (0 = current year end); long run has offset NULL.
   */
  async dotplotPerson(personId: string) {
    const P = this.requireFfr();
    const sql = `WITH mine AS (
           SELECT b.batch_id, b.source_meeting_start, pr.projection_horizon, pr.projection_year, pr.is_long_run,
                  pr.federal_funds_rate AS value
             FROM ${this.t('sep_participants')} sp
             JOIN ${P} pr ON pr.batch_id = sp.batch_id AND pr.participant_number = sp.participant_number
             JOIN ${this.t('sep_batches')} b ON b.batch_id = sp.batch_id
            WHERE sp.person_id::text = $1 AND pr.federal_funds_rate IS NOT NULL
         ), med AS (
           SELECT pr.batch_id, pr.projection_horizon,
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY pr.federal_funds_rate) AS median, COUNT(*) AS n
             FROM ${P} pr
            WHERE pr.federal_funds_rate IS NOT NULL AND pr.batch_id IN (SELECT batch_id FROM mine)
            GROUP BY pr.batch_id, pr.projection_horizon
         )
         SELECT mine.batch_id::text AS batch_id, mine.source_meeting_start::text AS meeting_start,
                mine.projection_horizon AS horizon, mine.is_long_run,
                CASE WHEN mine.is_long_run THEN NULL
                     ELSE mine.projection_year - EXTRACT(YEAR FROM mine.source_meeting_start)::int END AS year_offset,
                mine.value, med.median, med.n
           FROM mine JOIN med ON med.batch_id = mine.batch_id AND med.projection_horizon = mine.projection_horizon
          ORDER BY mine.source_meeting_start, mine.is_long_run, mine.projection_year`;
    const rows = (await this.q.query(sql, [personId])).map((r) => ({
      ...numify(r), value: Number(r.value), median: Number(r.median),
      year_offset: r.year_offset === null ? null : Number(r.year_offset), n: Number(r.n),
    }) as Row & { value: number; median: number; year_offset: number | null; n: number });
    return { items: rows, sql };
  }

  // ---------------------------------------------------------------- data quality
  async dataQuality() {
    const checks: QualityCheck[] = [];
    const S = (n: string) => this.t(n);
    const run = async (c: Omit<QualityCheck, 'count' | 'sample' | 'error'>) => {
      try {
        const [r] = await this.q.query<{ n: unknown }>(`SELECT COUNT(*) AS n FROM (${c.sql}) chk`);
        const sample = await this.q.query(`${c.sql} LIMIT 20`);
        checks.push({ ...c, count: Number(r.n), sample: sample.map(stringifyRow) });
      } catch (e) {
        checks.push({ ...c, count: null, sample: [], error: (e as Error).message });
      }
    };

    if (this.avail.sep) {
      await run({ id: 'sep_unmatched', severity: 'warn', title: 'SEP respondents not matched to a person',
        description: 'sep_participants rows with no person_id or mapping_status other than matched.',
        sql: `SELECT b.source_meeting_start::text AS meeting_start, sp.participant_number, sp.source_name, sp.mapping_status
                FROM ${S('sep_participants')} sp JOIN ${S('sep_batches')} b ON b.batch_id = sp.batch_id
               WHERE sp.person_id IS NULL OR sp.mapping_status <> 'matched'
               ORDER BY b.source_meeting_start, sp.participant_number` });
      if (this.avail.participants) {
        await run({ id: 'sep_person_not_in_meeting', severity: 'error', title: 'SEP respondent missing from the meeting attendance list',
          description: 'sep_participants.person_id is not in meeting_participants for sep_batches.meeting_version_id.',
          sql: `SELECT b.source_meeting_start::text AS meeting_start, sp.participant_number, sp.source_name, sp.person_id::text AS person_id
                  FROM ${S('sep_participants')} sp JOIN ${S('sep_batches')} b ON b.batch_id = sp.batch_id
                 WHERE sp.person_id IS NOT NULL AND NOT EXISTS (
                   SELECT 1 FROM ${S('meeting_participants')} mp
                    WHERE mp.version_id = b.meeting_version_id AND mp.person_id = sp.person_id)
                 ORDER BY b.source_meeting_start, sp.participant_number` });
        await run({ id: 'sep_version_not_active', severity: 'warn', title: 'SEP round points to an inactive meeting version',
          description: 'sep_batches.meeting_version_id differs from meetings.active_version_id (meeting re-imported without updating the SEP round).',
          sql: `SELECT b.source_meeting_start::text AS meeting_start, b.batch_id::text AS batch_id
                  FROM ${S('sep_batches')} b JOIN ${S('meetings')} m ON m.meeting_id = b.meeting_id
                 WHERE b.meeting_version_id IS DISTINCT FROM m.active_version_id
                 ORDER BY b.source_meeting_start` });
      }
      await run({ id: 'sep_status_unknown', severity: 'error', title: 'Unexpected response_status value',
        description: "response_status is not answered / na / missing.",
        sql: `SELECT a.response_status, COUNT(*) AS n FROM ${S('sep_answers')} a
               WHERE a.response_status IS NULL OR a.response_status NOT IN ('answered', 'na', 'missing')
               GROUP BY a.response_status` });
      await run({ id: 'sep_answered_empty', severity: 'error', title: 'Answered but the answer text is empty',
        description: "response_status = 'answered' with an empty or blank answer.",
        sql: `SELECT b.source_meeting_start::text AS meeting_start, a.question_code, a.participant_number
                FROM ${S('sep_answers')} a JOIN ${S('sep_batches')} b ON b.batch_id = a.batch_id
               WHERE a.response_status = 'answered' AND (a.answer IS NULL OR btrim(a.answer) = '')
               ORDER BY b.source_meeting_start, a.question_code, a.participant_number` });
      await run({ id: 'sep_missing_with_text', severity: 'info', title: 'Missing but there is answer text',
        description: "response_status = 'missing' with a non-empty answer; check the source. (na with the text 'Not applicable.' is normal.)",
        sql: `SELECT b.source_meeting_start::text AS meeting_start, a.question_code, a.participant_number, a.answer
                FROM ${S('sep_answers')} a JOIN ${S('sep_batches')} b ON b.batch_id = a.batch_id
               WHERE a.response_status = 'missing' AND a.answer IS NOT NULL AND btrim(a.answer) <> ''
               ORDER BY b.source_meeting_start, a.question_code` });
      if (this.info.get('sep_answers')!.includes('source_file_name')) {
        await run({ id: 'sep_no_source', severity: 'warn', title: 'Answer without a source file name',
          description: 'sep_answers.source_file_name is empty.',
          sql: `SELECT b.source_meeting_start::text AS meeting_start, a.question_code, a.participant_number
                  FROM ${S('sep_answers')} a JOIN ${S('sep_batches')} b ON b.batch_id = a.batch_id
                 WHERE a.source_file_name IS NULL OR btrim(a.source_file_name) = ''
                 ORDER BY b.source_meeting_start` });
      }
      await run({ id: 'sep_grid_gap', severity: 'warn', title: 'Incomplete answer grid',
        description: 'Each round should have questions × respondents answer rows (including na / missing); rounds where it does not.',
        sql: `SELECT b.source_meeting_start::text AS meeting_start, qn.n AS questions, pn.n AS participants,
                     qn.n * pn.n AS expected, COALESCE(an.n, 0) AS actual
                FROM ${S('sep_batches')} b
                JOIN (SELECT batch_id, COUNT(*) AS n FROM ${S('sep_questions')} GROUP BY batch_id) qn ON qn.batch_id = b.batch_id
                JOIN (SELECT batch_id, COUNT(*) AS n FROM ${S('sep_participants')} GROUP BY batch_id) pn ON pn.batch_id = b.batch_id
                LEFT JOIN (SELECT batch_id, COUNT(*) AS n FROM ${S('sep_answers')} GROUP BY batch_id) an ON an.batch_id = b.batch_id
               WHERE qn.n * pn.n <> COALESCE(an.n, 0)
               ORDER BY b.source_meeting_start` });
      await run({ id: 'sep_name_multi_id', severity: 'warn', title: 'One SEP source name mapped to several person_ids',
        description: 'The same source_name maps to different person_ids: possible alias or duplicate UUID. Listed only, never changed automatically.',
        sql: `SELECT sp.source_name, COUNT(DISTINCT sp.person_id) AS distinct_person_ids, COUNT(DISTINCT sp.batch_id) AS meetings
                FROM ${S('sep_participants')} sp
               GROUP BY sp.source_name HAVING COUNT(DISTINCT sp.person_id) > 1
               ORDER BY sp.source_name` });
    }
    if (this.avail.persons) {
      await run({ id: 'person_dup_name', severity: 'warn', title: 'Same name, different person_id',
        description: 'People whose display_name matches (case-insensitive) but whose person_id differs.',
        sql: `SELECT lower(${this.pn('p')}) AS name_key, COUNT(*) AS ids
                FROM ${S('persons')} p GROUP BY lower(${this.pn('p')}) HAVING COUNT(*) > 1
               ORDER BY lower(${this.pn('p')})` });
      checks.push(await this.surnameCheck());
    }
    if (this.avail.participants) {
      await run({ id: 'meeting_no_participants', severity: 'warn', title: 'Meetings with no attendees in the active version',
        description: 'No meeting_participants rows for meetings.active_version_id (or it is empty).',
        sql: `SELECT m.meeting_id::text AS meeting_id, m.meeting_date::text AS meeting_date
                FROM ${S('meetings')} m
               WHERE NOT EXISTS (SELECT 1 FROM ${S('meeting_participants')} mp WHERE mp.version_id = m.active_version_id)
               ORDER BY m.meeting_date` });
    }

    if (this.avail.participants && this.info.get('meeting_participants')!.includes('meeting_role')) {
      await run({ id: 'meeting_role_gap', severity: 'warn', title: 'Meetings without exactly one Chair and one Vice Chair',
        description: 'Counted from meeting_role in the active version. Either a real absence (e.g. the Vice Chair did not attend) or roles not parsed at import; check against the attendance roster.',
        sql: `SELECT m.meeting_date::text AS meeting_date,
                     COUNT(*) FILTER (WHERE mp.meeting_role = 'Chair') AS chairs,
                     COUNT(*) FILTER (WHERE mp.meeting_role = 'Vice Chair') AS vice_chairs,
                     COUNT(*) FILTER (WHERE mp.meeting_role = 'Voter') AS voters
                FROM ${S('meetings')} m
                LEFT JOIN ${S('meeting_participants')} mp ON mp.version_id = m.active_version_id
               GROUP BY m.meeting_date
              HAVING COUNT(*) FILTER (WHERE mp.meeting_role = 'Chair') <> 1
                  OR COUNT(*) FILTER (WHERE mp.meeting_role = 'Vice Chair') <> 1
               ORDER BY m.meeting_date` });
    }
    if (this.tm?.personIdColumn && this.avail.participants && this.tm.meetingKey.references === 'version_id') {
      const tm = this.tm; const pc = quoteIdent(tm.personIdColumn!);
      await run({ id: 'turn_speaker_not_participant', severity: 'warn', title: 'Speaker missing from the attendance list',
        description: `Turns in the active version of ${tm.table} whose person_id is not in meeting_participants for the same version (grouped by meeting and person). Staff or guests, or a mapping problem.`,
        sql: `SELECT m_act.meeting_date::text AS meeting_date, tr.${pc}::text AS person_id,
                     MAX(${this.pn('p')}) AS person_name, COUNT(*) AS turns
                FROM ${this.activeTurnsFrom()}
                LEFT JOIN ${S('persons')} p ON p.person_id = tr.${pc}
               WHERE NOT EXISTS (SELECT 1 FROM ${S('meeting_participants')} mp
                                  WHERE mp.version_id = tr.${quoteIdent(tm.meetingKey.column)}
                                    AND mp.person_id = tr.${pc})
               GROUP BY m_act.meeting_date, tr.${pc}
               ORDER BY m_act.meeting_date, turns DESC` });
    }
    if (this.avail.projections.ok && this.avail.projections.columns!.includes('source_missing_fields')) {
      await run({ id: 'proj_missing_fields', severity: 'info', title: 'SEP projections with documented missing fields',
        description: 'Rows where sep_projections.source_missing_fields is not empty (the source did not provide the variable; not an import error, do not fill in).',
        sql: `SELECT b.source_meeting_start::text AS meeting_start, pr.participant_number, pr.projection_horizon, pr.source_missing_fields::text AS missing
                FROM ${S(this.avail.projections.table)} pr JOIN ${S('sep_batches')} b ON b.batch_id = pr.batch_id
               WHERE jsonb_array_length(pr.source_missing_fields) > 0
               ORDER BY b.source_meeting_start, pr.participant_number` });
    }

    // Hits already checked by hand against the sources (server/src/verified.ts); new hits stay unverified.
    for (const c of checks) {
      if (!c.count) continue;
      try {
        const rule = VERIFIED_ROWS[c.id];
        if (rule) {
          const rows = c.id === 'person_same_surname' ? c.sample : (await this.q.query(c.sql)).map(stringifyRow);
          c.verified = { count: rows.filter(rule.match).length, note: rule.note };
        } else if (c.id === 'proj_missing_fields') {
          const [r] = await this.q.query<{ n: unknown }>(
            `SELECT COUNT(*) AS n FROM ${S(this.avail.projections.table)} pr
              WHERE jsonb_array_length(pr.source_missing_fields) > 0
                AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements(pr.source_missing_fields) e
                                 WHERE NOT (e->>'reason' = ANY($1::text[])))`, [VERIFIED_MISSING_REASONS.reasons]);
          c.verified = { count: Number(r.n), note: VERIFIED_MISSING_REASONS.note };
        }
      } catch { /* leave unverified */ }
    }

    // Distributions (read-only GROUP BY), including the team's own registered review issues.
    const distributions: { id: string; title: string; sql: string; rows: Row[]; error?: string }[] = [];
    const dist = async (id: string, title: string, sql: string) => {
      try { distributions.push({ id, title, sql, rows: (await this.q.query(sql)).map(stringifyRow) }); }
      catch (e) { distributions.push({ id, title, sql, rows: [], error: (e as Error).message }); }
    };
    if (this.info.get('persons')?.includes('identity_status')) {
      await dist('identity_status', 'Identity status (persons.identity_status)',
        `SELECT identity_status, COUNT(*) AS n FROM ${S('persons')} GROUP BY identity_status ORDER BY n DESC`);
    }
    if (this.avail.sep) {
      await dist('mapping_status', 'SEP respondent mapping status (sep_participants.mapping_status)',
        `SELECT mapping_status, COUNT(*) AS n FROM ${S('sep_participants')} GROUP BY mapping_status ORDER BY n DESC`);
    }
    if (hasColumns(this.info, 'current_review_issues', ['issue_kind'])) {
      await dist('review_issues', 'Issues logged by the import pipeline (current_review_issues, by issue_kind)',
        `SELECT issue_kind, COUNT(*) AS n FROM ${S('current_review_issues')} GROUP BY issue_kind ORDER BY n DESC`);
    } else if (hasColumns(this.info, 'review_issues', ['issue_kind', 'version_id']) && this.avail.meetings) {
      await dist('review_issues', 'Issues logged by the import pipeline (review_issues, active versions, by issue_kind)',
        `SELECT ri.issue_kind, COUNT(*) AS n FROM ${S('review_issues')} ri JOIN ${S('meetings')} m ON m.active_version_id = ri.version_id GROUP BY ri.issue_kind ORDER BY n DESC`);
    }

    // Cross-checks: our explicit "active version" joins vs. the team's current_* views.
    const crossChecks: { id: string; title: string; ours: number | null; theirs: number | null; oursSql: string; theirsSql: string; error?: string }[] = [];
    const cross = async (id: string, title: string, oursSql: string, theirsSql: string) => {
      try {
        const [[a], [b]] = await Promise.all([this.q.query<{ n: unknown }>(oursSql), this.q.query<{ n: unknown }>(theirsSql)]);
        crossChecks.push({ id, title, ours: Number(a.n), theirs: Number(b.n), oursSql, theirsSql });
      } catch (e) {
        crossChecks.push({ id, title, ours: null, theirs: null, oursSql, theirsSql, error: (e as Error).message });
      }
    };
    if (this.tm && this.info.has('current_turns')) {
      await cross('turns', 'Transcript turns: this site\'s join vs the current_turns view',
        `SELECT COUNT(*) AS n FROM ${this.activeTurnsFrom()}`, `SELECT COUNT(*) AS n FROM ${S('current_turns')}`);
    }
    if (this.avail.projections.ok && this.info.has('current_sep_projections')) {
      await cross('projections', 'SEP projections: sep_projections vs the current_sep_projections view',
        `SELECT COUNT(*) AS n FROM ${S(this.avail.projections.table)}`, `SELECT COUNT(*) AS n FROM ${S('current_sep_projections')}`);
    }

    // Coverage table: meetings per year vs. SEP batches per year.
    const coverageSql = `SELECT EXTRACT(YEAR FROM m.meeting_date) AS year, COUNT(*) AS meetings,
              COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM ${S('sep_batches')} b WHERE b.meeting_id = m.meeting_id)) AS meetings_with_sep
         FROM ${S('meetings')} m
        GROUP BY EXTRACT(YEAR FROM m.meeting_date)
        ORDER BY 1`;
    const coverage = this.avail.meetings && this.avail.sep ? await this.q.query(coverageSql) : [];

    return {
      checks,
      coverage: coverage.map(numify),
      distributions,
      crossChecks,
      coverageSql: this.avail.meetings && this.avail.sep ? coverageSql : null,
      unavailable: [
        !this.avail.transcript.ok && { area: 'Transcript', reason: this.avail.transcript.reason },
        !this.avail.projections.ok && { area: 'SEP numeric projections', reason: this.avail.projections.reason },
        !this.avail.sep && { area: 'SEP narrative answers', reason: 'the four sep_* tables or required columns were not found' },
        !this.avail.participants && { area: 'Attendees', reason: 'meetings.active_version_id / meeting_participants columns not found' },
      ].filter(Boolean),
    };
  }

  /** Heuristic alias hint (same final name token, different UUIDs). Grouping done in app code. */
  private async surnameCheck(): Promise<QualityCheck> {
    const sql = `SELECT p.person_id::text AS person_id, ${this.pn('p')} AS canonical_name FROM ${this.t('persons')} p ORDER BY ${this.pn('p')}`;
    const rows = await this.q.query<{ person_id: string; canonical_name: string | null }>(sql);
    const groups = new Map<string, string[]>();
    for (const r of rows) {
      const last = (r.canonical_name ?? '').trim().split(/\s+/).pop()?.toLowerCase().replace(/[^a-z\-']/g, '');
      if (!last) continue;
      if (!groups.has(last)) groups.set(last, []);
      groups.get(last)!.push(r.canonical_name!);
    }
    const sample = [...groups.entries()].filter(([, v]) => v.length > 1)
      .map(([surname, names]) => ({ surname, ids: String(names.length), names: names.join(' | ') }));
    return {
      id: 'person_same_surname', severity: 'info', title: 'Possible aliases (same surname, several person_ids; heuristic)',
      description: 'Grouped by the last word of the name (computed in the app). Only a lead for manual review: different people often share a surname, so never merge on this alone.',
      sql: `${sql}  -- grouping: last word of the name, case-insensitive`,
      count: sample.length, sample: sample.slice(0, 50),
    };
  }
}

export interface QualityCheck {
  id: string;
  severity: 'info' | 'warn' | 'error';
  title: string;
  description: string;
  sql: string;
  count: number | null;
  sample: Row[];
  error?: string;
  /** How many of the hits were checked by hand and are correct as stored (see verified.ts). */
  verified?: { count: number; note: string };
}

export class UnavailableError extends Error {}
export class BadRequestError extends Error {}

/** COUNT(*) comes back as string (bigint) from pg; convert known numeric-looking count fields. */
const COUNT_FIELDS = new Set(['meeting_turns', 'official_meetings', 'voter_meetings', 'participant_count', 'voter_count', 'nonvoting_count', 'staff_count', 'speaker_turn', 'speaker_turns', 'source_page', 'sep_batch_count', 'utterance_count', 'meeting_count', 'sep_count',
  'participants', 'questions', 'answers', 'answered', 'na', 'missing', 'participant_number', 'question_order',
  'source_record_number', 'seq', 'year', 'turns', 'projection_year', 'meetings', 'meetings_with_sep', 'gdp_growth', 'unemployment_rate',
  'pce_inflation', 'core_pce_inflation', 'federal_funds_rate']);

function numify(r: Row): Row {
  const o: Row = {};
  for (const [k, v] of Object.entries(r)) {
    o[k] = COUNT_FIELDS.has(k) && v !== null && v !== undefined && v !== '' && !Number.isNaN(Number(v)) ? Number(v) : v;
  }
  return o;
}

/**
 * Link to the transcript PDF on federalreserve.gov, opened at the page of the turn. Our text files were converted from
 * those PDFs page by page ("----- page N -----"), so source_page is the PDF page number.
 */
export function officialPdfUrl(sourceFile: unknown, page: unknown): string | null {
  const name = typeof sourceFile === 'string' ? sourceFile.split('/').pop() ?? '' : '';
  const m = /^(FOMC\d{8}(?:meeting|confcall))\.(?:txt|pdf)$/i.exec(name);
  if (!m) return null;
  const p = Number(page);
  return `https://www.federalreserve.gov/monetarypolicy/files/${m[1]}.pdf${Number.isInteger(p) && p > 0 ? `#page=${p}` : ''}`;
}

/**
 * Link to a Greenbook / Tealbook / Beige Book on federalreserve.gov. The scraper saved each file under the name it has on
 * the Fed site, so the URL follows from the file name (all 302 links checked against the site on 2026-10-27).
 */
export function officialMacroUrl(sourcePath: unknown, documentDate?: unknown, page?: unknown): string | null {
  const fed = 'https://www.federalreserve.gov';
  const name = typeof sourcePath === 'string' ? sourcePath.split('/').pop() ?? '' : '';
  const p = Number(page);
  const at = Number.isInteger(p) && p > 0 ? `#page=${p}` : '';
  if (/^FOMC\d{8}(?:gbpt\d|tealbook[ab])\d{8}\.pdf$/i.test(name)) return `${fed}/monetarypolicy/files/${name}${at}`;
  // Beige Book PDFs moved between folders over the years; the folder depends on the name and the date
  const bb = /^(BeigeBook_|Beigebook_|fullreport)(\d{8})\.pdf$/.exec(name);
  if (bb) {
    const [, prefix, d] = bb;
    const folder = prefix === 'fullreport' ? (d < '20120401' ? `fomc/beigebook/${d.slice(0, 4)}/${d}` : 'monetarypolicy/beigebook/files')
      : prefix === 'BeigeBook_' ? (d < '20151001' ? 'monetarypolicy/beigebook/files' : 'monetarypolicy/files')
      : (d < '20170401' ? 'monetarypolicy/beigebook/files' : 'monetarypolicy/files');
    return `${fed}/${folder}/${name}${at}`;
  }
  // 2007–early 2008 Beige Books were saved as the HTML page
  const dd = documentDate instanceof Date ? documentDate.toISOString().slice(0, 10) : String(documentDate ?? '');
  if (name === 'default.htm' && /^\d{4}-\d{2}-\d{2}$/.test(dd)) return `${fed}/fomc/beigebook/${dd.slice(0, 4)}/${dd.replace(/-/g, '')}/default.htm`;
  return null;
}

/** Schema holding the economic documents and per-meeting summaries (maintained by the research side). */
export const MACRO_SCHEMA = process.env.FEDAI_MACRO_SCHEMA ?? 'sep_dual_agent';

function isoDate(v: unknown): string {
  return v instanceof Date ? v.toISOString().slice(0, 10) : String(v ?? '').slice(0, 10);
}

interface MacroCitation { page?: number; quote?: string }
interface MacroRecord {
  claim?: string; topic?: string; section?: string; scope?: string; source_id?: string; source_kind?: string;
  variable?: string; value?: string; unit?: string; period?: string; basis?: string; citations?: MacroCitation[];
}
interface MacroContext {
  source_ids?: string[];
  macro?: { items?: MacroRecord[]; numerical_table?: MacroRecord[];
    beige_book?: { source_id?: string; districts?: { district?: string; status?: string; items?: MacroRecord[]; numerical_table?: MacroRecord[] }[] } };
  policy?: { items?: MacroRecord[]; numerical_table?: MacroRecord[] };
  coverage?: { warnings?: string[]; missing_types?: string[]; source_details?: { kind?: string; source_id?: string; document_date?: string; age_days_at_event?: number }[] };
}

/** Keep what the page shows from a meeting_macro_contexts.context_json and link every citation to its page on the Fed site. */
export function shapeContext(ctx: MacroContext, docs: Row[], cutoffDate: string, status: string) {
  const byId = new Map(docs.map((d) => [String(d.source_id), d]));
  const sources = (ctx.coverage?.source_details ?? []).map((s) => {
    const d = byId.get(String(s.source_id));
    return { kind: s.kind ?? d?.kind ?? null, title: d?.title ?? null, documentDate: s.document_date ?? (d ? isoDate(d.document_date) : null),
      ageDays: s.age_days_at_event ?? null, url: d ? officialMacroUrl(d.source_path, d.document_date) : null };
  });
  const cite = (r: MacroRecord) => {
    const d = byId.get(String(r.source_id));
    return (r.citations ?? []).map((c) => ({ page: c.page ?? null, quote: c.quote ?? '', url: d ? officialMacroUrl(d.source_path, d.document_date, c.page) : null }));
  };
  const item = (r: MacroRecord) => ({ topic: r.topic ?? null, section: r.section ?? null, scope: r.scope ?? null, sourceKind: r.source_kind ?? null, claim: r.claim ?? '', citations: cite(r) });
  const number = (r: MacroRecord) => ({ variable: r.variable ?? '', value: r.value ?? '', unit: r.unit ?? null, period: r.period ?? null, basis: r.basis ?? null,
    scope: r.scope ?? null, sourceKind: r.source_kind ?? null, citations: cite(r) });
  return {
    cutoffDate, status, sources,
    warnings: ctx.coverage?.warnings ?? [], missingTypes: ctx.coverage?.missing_types ?? [],
    macro: (ctx.macro?.items ?? []).map(item), macroNumbers: (ctx.macro?.numerical_table ?? []).map(number),
    policy: (ctx.policy?.items ?? []).map(item), policyNumbers: (ctx.policy?.numerical_table ?? []).map(number),
    districts: (ctx.macro?.beige_book?.districts ?? []).map((d) => ({ district: d.district ?? '', status: d.status ?? null,
      items: (d.items ?? []).map(item), numbers: (d.numerical_table ?? []).map(number) })),
  };
}

/** Render generic rows safely for JSON (Dates -> ISO, objects -> JSON text). */
function stringifyRow(r: Row): Row {
  const o: Row = {};
  for (const [k, v] of Object.entries(r)) {
    if (v instanceof Date) o[k] = v.toISOString();
    else if (v !== null && typeof v === 'object') o[k] = JSON.stringify(v);
    else o[k] = v;
  }
  return o;
}

/** Dot-plot statistics (quantiles by linear interpolation, same as PostgreSQL percentile_cont). */
export function dotStats(values: number[]) {
  const v = [...values].sort((a, b) => a - b);
  const q = (p: number) => {
    if (!v.length) return NaN;
    const i = (v.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
    return v[lo] + (v[hi] - v[lo]) * (i - lo);
  };
  const round = (x: number) => Math.round(x * 1e4) / 1e4;
  const q1 = q(0.25), q3 = q(0.75);
  return { n: v.length, median: round(q(0.5)), min: v[0], max: v[v.length - 1], q1: round(q1), q3: round(q3), iqr: round(q3 - q1) };
}

export const OUTLIER_RULE = 'More than 1.5 × IQR and more than 0.5 percentage points from the median (points exactly on the line are not outliers)';

export function isOutlier(value: number, st: { median: number; iqr: number }) {
  return Math.abs(value - st.median) > Math.max(1.5 * st.iqr, 0.5) + 1e-9;   // tolerance: rates like 0.13/0.63 are stored rounded
}
