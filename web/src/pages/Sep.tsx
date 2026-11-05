import { useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { CartesianGrid, ComposedChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { useApi, type Row } from '../api';
import { Empty, ErrorBox, Loading, MeetingLink, PersonLink, SqlBox, Status, Val } from '../components';

interface ProjStatus { ok: boolean; table: string; reason?: string; columns?: string[] }

export function Sep() {
  const [sp, setSp] = useSearchParams();
  const year = sp.get('year') ?? '';
  const { data, error, loading } = useApi<{ items: Row[]; projections: ProjStatus }>('/sep', { year });
  const years = useMemo(() => [...new Set((data?.items ?? []).map((b) => String(b.meeting_start).slice(0, 4)))], [data]);
  return (
    <>
      <h1>SEP · Summary of Economic Projections</h1>
      <div className="filters">
        <label>Year <select value={year} onChange={(e) => setSp(e.target.value ? { year: e.target.value } : {})}>
          <option value="">All</option>
          {(year ? [year] : years).map((y) => <option key={y} value={y}>{y}</option>)}
        </select></label>
      </div>
      {data && !data.projections.ok && <div className="notice"><strong>Numeric projections not available. </strong>{data.projections.reason}</div>}
      {error && <ErrorBox error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <table className="data">
          <thead><tr><th>Meeting start</th><th>End</th><th className="num">Respondents</th><th className="num">Questions</th><th className="num">Answers</th><th className="num">answered</th><th className="num">na</th><th className="num">missing</th></tr></thead>
          <tbody>{data.items.map((b) => (
            <tr key={String(b.batch_id)}>
              <td><Link to={`/sep/${b.batch_id}`}>{String(b.meeting_start)}</Link></td><td><Val v={b.meeting_end} /></td>
              <td className="num">{String(b.participants)}</td><td className="num">{String(b.questions)}</td><td className="num">{String(b.answers)}</td>
              <td className="num">{String(b.answered)}</td><td className="num">{String(b.na)}</td><td className="num">{String(b.missing)}</td>
            </tr>))}</tbody>
        </table>
      )}
      {data && !data.items.length && <Empty>No SEP rounds.</Empty>}
    </>
  );
}

interface SepData {
  batch: Row; participants: Row[]; questions: Row[]; answers: Row[]; answersSql: string;
  projections: Row[] | null; projectionsSql: string | null; projectionsStatus: ProjStatus;
}

export function SepDetail() {
  const { id } = useParams();
  const { data, error, loading } = useApi<SepData>(`/sep/${id}`);
  const [openQ, setOpenQ] = useState<string | null>(null);
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Loading />;
  const d = data!;
  const byQ = new Map<string, Row[]>();
  for (const a of d.answers) {
    const k = String(a.question_code);
    if (!byQ.has(k)) byQ.set(k, []);
    byQ.get(k)!.push(a);
  }
  const cell = new Map(d.answers.map((a) => [`${a.question_code}|${a.participant_number}`, a]));
  const pname = new Map(d.participants.map((p) => [Number(p.participant_number), p]));
  return (
    <>
      <p><Link to="/sep">← SEP rounds</Link></p>
      <h1>SEP {String(d.batch.meeting_start)} – {String(d.batch.meeting_end ?? '')}</h1>
      <p>Meeting <MeetingLink id={d.batch.meeting_id} date={d.batch.meeting_start} /> · batch_id <code>{String(d.batch.batch_id)}</code> ·
        {' '}{String(d.batch.participants)} respondents · {String(d.batch.questions)} questions · {String(d.batch.answers)} answers</p>

      <section>
        <h2>Anonymous respondent number → person</h2>
        <table className="data">
          <thead><tr><th className="num">Respondent #</th><th>Name in the source</th><th>Affiliation in the source</th><th>Person</th><th>mapping_status</th><th>Notes</th></tr></thead>
          <tbody>{d.participants.map((p) => (
            <tr key={String(p.participant_number)}>
              <td className="num">{String(p.participant_number)}</td><td>{String(p.source_name)}</td><td><Val v={p.source_affiliation} /></td>
              <td><PersonLink id={p.person_id} name={p.person_name} /></td><td><Status value={p.mapping_status} /></td><td className="small"><Val v={p.mapping_notes} /></td>
            </tr>))}</tbody>
        </table>
      </section>

      <section>
        <h2>Answer status grid</h2>
        <p className="muted small">Rows = questions, columns = respondent numbers. A = answered, N = na, M = missing, blank = no row (a gap in the grid). Click a question to read the answers.</p>
        <div className="matrix-wrap">
          <table className="matrix">
            <thead><tr><th>Question</th>{d.participants.map((p) => <th key={String(p.participant_number)} title={String(p.source_name)}>{String(p.participant_number)}</th>)}</tr></thead>
            <tbody>{d.questions.map((q) => (
              <tr key={String(q.question_code)}>
                <th><button className="linklike" onClick={() => setOpenQ(String(q.question_code))}>{String(q.question_code)}</button></th>
                {d.participants.map((p) => {
                  const a = cell.get(`${q.question_code}|${p.participant_number}`);
                  const s = a ? String(a.response_status) : 'none';
                  return <td key={String(p.participant_number)} className={`m m-${s}`} title={`${q.question_code} · #${p.participant_number} ${p.source_name}: ${s}`}>
                    {s === 'answered' ? 'A' : s === 'na' ? 'N' : s === 'missing' ? 'M' : ''}</td>;
                })}
              </tr>))}</tbody>
          </table>
        </div>
      </section>

      <section>
        <h2>Questions and narrative answers (source text)</h2>
        {d.questions.map((q) => {
          const code = String(q.question_code);
          const open = openQ === code;
          return (
            <div key={code} className="question">
              <button className="q-head" onClick={() => setOpenQ(open ? null : code)}>
                <strong>{code}</strong> {String(q.question_text)} <span className="muted">({byQ.get(code)?.length ?? 0} answers) {open ? '▲' : '▼'}</span>
              </button>
              {open && (
                <table className="data">
                  <thead><tr><th className="num">#</th><th>Person</th><th>Status</th><th>Answer (source text)</th><th>Source</th></tr></thead>
                  <tbody>{(byQ.get(code) ?? []).map((a) => {
                    const p = pname.get(Number(a.participant_number));
                    return (
                      <tr key={String(a.participant_number)}>
                        <td className="num">{String(a.participant_number)}</td>
                        <td>{p ? <PersonLink id={p.person_id} name={p.person_name ?? p.source_name} /> : '—'}</td>
                        <td><Status value={a.response_status} /></td>
                        <td className="answer"><Val v={a.answer} /></td>
                        <td className="small mono">{String(a.source_file_name ?? '')}{a.reference_doc_name ? ` · ${a.reference_doc_name}` : ''}</td>
                      </tr>);
                  })}</tbody>
                </table>
              )}
            </div>
          );
        })}
        <SqlBox sql={d.answersSql} />
      </section>

      <section>
        <h2>Numeric projections</h2>
        {d.projectionsStatus.ok && d.projections
          ? <Projections rows={d.projections} sql={d.projectionsSql} names={pname} />
          : <div className="notice"><strong>Not available. </strong>{d.projectionsStatus.reason}</div>}
      </section>
    </>
  );
}

const VARS = [
  ['gdp_growth', 'GDP growth (Q4/Q4, %)'],
  ['unemployment_rate', 'Unemployment rate (Q4 avg, %)'],
  ['pce_inflation', 'PCE inflation (Q4/Q4, %)'],
  ['core_pce_inflation', 'Core PCE inflation (Q4/Q4, %)'],
  ['federal_funds_rate', 'Federal funds rate (year-end, %)'],
] as const;

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length ? (s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2) : null;
};

/** Small multiples: one variable per chart (shared x = target year), dots = participants, bar mark = median. */
function Projections({ rows, sql, names }: { rows: Row[]; sql: string | null; names: Map<number, Row> }) {
  const [table, setTable] = useState(false);
  // The dot under the pointer, set by the symbol's own mouse events (Recharts' own lookup by x picks the
  // wrong dot when several share a column).
  const [hover, setHover] = useState<{ chart: string; year: string; v: number; p: unknown } | null>(null);
  const who = (n: unknown) => { const r = names.get(Number(n)); return r ? String(r.person_name ?? r.source_name ?? '') : ''; };
  const years = [...new Set(rows.map((r) => String(r.horizon)))]; // SQL orders by is_long_run, projection_year
  const vars = VARS.filter(([k]) => rows.some((r) => r[k] !== null && r[k] !== undefined));
  return (
    <div className="viz-root">
      <div className="filters">
        <span className="muted small">Each dot is one respondent (hover for number, name and value; equal values sit side by side); the bar is the median of this round. Missing values (including those listed in source_missing_fields) are not drawn or filled in.</span>
        <button className="ghost" onClick={() => setTable(!table)}>{table ? 'Chart' : 'Table'}</button>
      </div>
      {table ? (
        <div className="matrix-wrap">
          <table className="data">
            <thead><tr><th className="num">#</th><th>projection_horizon</th>{vars.map(([k]) => <th key={k} className="num">{k}</th>)}<th>source_missing_fields</th></tr></thead>
            <tbody>{rows.map((r, i) => <tr key={i}><td className="num">{String(r.participant_number)}</td><td>{String(r.horizon)}</td>
              {vars.map(([k]) => <td key={k} className="num"><Val v={r[k]} /></td>)}<td className="small mono">{r.source_missing_fields && r.source_missing_fields !== '[]' ? String(r.source_missing_fields) : ''}</td></tr>)}</tbody>
          </table>
        </div>
      ) : (
        <div className="multiples">
          {vars.map(([k, label]) => {
            // x = year index; dots with the same value in one year are spread side by side so none is hidden
            const seen = new Map<string, number>();
            const raw = rows.filter((r) => r[k] !== null && r[k] !== undefined)
              .map((r) => ({ year: String(r.horizon), v: Number(r[k]), p: r.participant_number }));
            const dup = new Map<string, number>();
            for (const x of raw) dup.set(`${x.year}|${x.v}`, (dup.get(`${x.year}|${x.v}`) ?? 0) + 1);
            const pts = raw.map((x) => {
              const key = `${x.year}|${x.v}`; const n = dup.get(key)!; const i = seen.get(key) ?? 0; seen.set(key, i + 1);
              const step = Math.min(0.12, 0.75 / n);
              return { ...x, x: years.indexOf(x.year) + (i - (n - 1) / 2) * step };
            });
            const med = years.map((y, i) => ({ year: y, x: i, v: median(raw.filter((x) => x.year === y).map((x) => x.v)) }))
              .filter((m): m is { year: string; x: number; v: number } => m.v !== null);
            const medOf = (y: string) => med.find((m) => m.year === y)?.v;
            return (
              <div key={k} className="multiple">
                <h3 className="chart-title">{label}</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <ComposedChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--grid)" />
                    <XAxis dataKey="x" type="number" domain={[-0.6, years.length - 0.4]} ticks={years.map((_, i) => i)}
                      tickFormatter={(i: number) => years[i] ?? ''} tickLine={false}
                      axisLine={{ stroke: 'var(--axis)' }} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                    <YAxis dataKey="v" type="number" domain={['auto', 'auto']} tickLine={false} axisLine={false} width={36}
                      tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                    <Tooltip cursor={false} shared={false} content={() => {
                      if (!hover || hover.chart !== k) return null;
                      const m = medOf(hover.year);
                      // narrow charts: identical values overlap, so list everyone who gave this same value
                      const same = raw.filter((x) => x.year === hover.year && x.v === hover.v && x.p !== hover.p);
                      return <div className="tip">#{String(hover.p)} {who(hover.p)} · {hover.year}: <strong>{hover.v}</strong>
                        {m !== undefined && <span className="muted"> (median {m})</span>}
                        {same.length > 0 && <div className="small muted">{same.length} other(s) also at {hover.v}:{' '}
                          {same.map((x) => `#${String(x.p)} ${who(x.p)}`).join(', ')}</div>}</div>;
                    }} />
                    <Scatter data={pts} dataKey="v" fill="var(--series-1)" fillOpacity={0.55} stroke="var(--surface-1)" strokeWidth={1} isAnimationActive={false}
                      onMouseEnter={(pt: { payload?: unknown }) => { const x = pt.payload as typeof pts[number] | undefined; setHover(x ? { chart: k, ...x } : null); }}
                      onMouseLeave={() => setHover(null)} />
                    <Scatter data={med} dataKey="v" isAnimationActive={false}
                      shape={(props: { cx?: number; cy?: number }) => (props.cx === undefined || props.cy === undefined ? <g /> :
                        <rect x={props.cx - 12} y={props.cy - 1.5} width={24} height={3} rx={1.5} fill="var(--text-primary)" pointerEvents="none" />)} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      )}
      <SqlBox sql={sql} />
    </div>
  );
}
